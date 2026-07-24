# =====================================================================
# STEP 0：匯入程式所需套件
# =====================================================================
# pickle：儲存模型、編碼器與模型設定
import pickle
from pathlib import Path

# NumPy、Pandas：資料清理、數值運算與表格處理
import numpy as np
import pandas as pd

# Scikit-learn：建立隨機森林、Ridge、Logistic Regression 與評估指標
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost：建立非線性回歸與三分類模型
from xgboost import XGBClassifier, XGBRegressor


# =====================================================================
# STEP 1：設定專案路徑、分類門檻與共用參數
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Savings Pct：大於 5%＝節省；介於 ±5%＝接近預算；小於 -5%＝超支
CLOSE_BUDGET_LIMIT = 5.0
RANDOM_STATE = 42
N_OOF_SPLITS = 5
META_VALID_RATIO = 0.25

# 三分類名稱，方便輸出報告與後端系統使用
CLASS_NAMES = {0: "節省", 1: "接近預算", 2: "超支"}


# =====================================================================
# STEP 2：建立資料讀取與清洗工具
# =====================================================================
# 2.1 從程式目錄、目前目錄及專案根目錄尋找資料檔
def find_data_file():
    filenames = ["Data_中英欄位.csv", "Data_中英欄位_After_EDA.csv"]
    search_dirs = [SCRIPT_DIR, Path.cwd(), PROJECT_ROOT]
    for directory in search_dirs:
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    searched = "\n".join(str(d / n) for d in search_dirs for n in filenames)
    raise FileNotFoundError(f"找不到資料檔，已搜尋：\n{searched}")


# 2.2 清理一般數值欄位：移除逗號、轉數值並以中位數補缺失值
def clean_numeric(series):
    result = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    median = result.median()
    return result.fillna(0.0 if pd.isna(median) else median)


# 2.3 清理百分比欄位：有百分比符號時轉成 0～1 小數
def clean_percentage(series):
    text = series.astype(str).str.strip()
    has_percent = text.str.contains("%", regex=False, na=False)
    result = pd.to_numeric(text.str.replace("%", "", regex=False), errors="coerce")
    result.loc[has_percent] = result.loc[has_percent] / 100.0
    median = result.median()
    return result.fillna(0.0 if pd.isna(median) else median)


# 2.4 清理 Yes／No、True／False 等二元欄位
def clean_binary(series):
    mapping = {
        "yes": 1.0, "true": 1.0, "y": 1.0, "1": 1.0,
        "no": 0.0, "false": 0.0, "n": 0.0, "0": 0.0,
        "unknown": 0.0,
    }
    text = series.astype(str).str.strip().str.lower()
    result = text.map(mapping).fillna(pd.to_numeric(text, errors="coerce"))
    median = result.median()
    return result.fillna(0.0 if pd.isna(median) else median)


# 2.5 依節省率建立三分類標籤
def assign_savings_category(pct):
    if pct > CLOSE_BUDGET_LIMIT:
        return 0
    if pct >= -CLOSE_BUDGET_LIMIT:
        return 1
    return 2


# 2.6 安全除法：避免除以 0 產生無限值
def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return (
        numerator.div(denominator)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


# =====================================================================
# STEP 3：建立只使用「過去資料」的歷史聚合特徵
# =====================================================================
# 3.1 計算同群組截至前一筆的歷史平均，避免使用當筆答案
def expanding_history(df, group_cols, value_col, output_col, default=0.0):
    df[output_col] = (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(default)
    )


# 3.2 計算每筆資料發生前的歷史訂單次數
def expanding_count(df, group_cols, output_col):
    df[output_col] = df.groupby(group_cols, sort=False).cumcount().astype(float)


# 3.3 為訓練集建立逐筆歷史特徵，並保存完整訓練期統計
def build_training_aggregates(train_df):
    out = train_df.copy()
    out["Is_Over_Budget"] = (
        out["Savings Pct"] < -CLOSE_BUDGET_LIMIT
    ).astype(float)

    # 品項、供應商及部門的歷史資訊
    expanding_history(out, "Item Code_Encoded", "Unit Price", "Item_Avg_Price")
    expanding_count(out, "Item Code_Encoded", "Item_Order_Count")
    expanding_count(out, "Supplier ID_Encoded", "Supplier_Order_Count")
    expanding_history(
        out, "Supplier ID_Encoded", "Savings Pct", "Supplier_Avg_Savings"
    )
    expanding_history(
        out,
        "Supplier ID_Encoded",
        "Is_Over_Budget",
        "Supplier_Over_Budget_Rate",
    )
    expanding_history(
        out,
        "Supplier ID_Encoded",
        "On Time Delivery",
        "Supplier_On_Time_Rate",
    )
    expanding_history(
        out,
        ["Supplier ID_Encoded", "Item Code_Encoded"],
        "Unit Price",
        "Supplier_Item_Avg_Price",
    )
    expanding_history(
        out, "Category_Encoded", "Unit Price", "Category_Avg_Price"
    )
    expanding_history(
        out, "Department_Encoded", "Savings Pct", "Department_Avg_Savings"
    )

    # 是否有歷史資料，以及預算與歷史價格之間的差距
    out["Has_Item_History"] = (out["Item_Order_Count"] > 0).astype(int)
    out["Has_Supplier_History"] = (
        out["Supplier_Order_Count"] > 0
    ).astype(int)
    out["Budget_vs_Avg_Gap"] = safe_divide(
        out["Budget Unit Price"] - out["Item_Avg_Price"],
        out["Item_Avg_Price"],
    )
    out["Budget_vs_Supplier_Item_Gap"] = safe_divide(
        out["Budget Unit Price"] - out["Supplier_Item_Avg_Price"],
        out["Supplier_Item_Avg_Price"],
    )

    # 完整訓練期統計，僅供未來驗證／測試資料查表使用
    state = {
        "global": {
            "unit_price": float(train_df["Unit Price"].mean()),
            "savings": float(train_df["Savings Pct"].mean()),
            "over_rate": float(out["Is_Over_Budget"].mean()),
            "on_time": float(train_df["On Time Delivery"].mean()),
        },
        "item_price": train_df.groupby("Item Code_Encoded")[
            "Unit Price"
        ].mean().to_dict(),
        "item_count": train_df.groupby("Item Code_Encoded").size().to_dict(),
        "supplier_count": train_df.groupby(
            "Supplier ID_Encoded"
        ).size().to_dict(),
        "supplier_savings": train_df.groupby("Supplier ID_Encoded")[
            "Savings Pct"
        ].mean().to_dict(),
        "supplier_over_rate": out.groupby("Supplier ID_Encoded")[
            "Is_Over_Budget"
        ].mean().to_dict(),
        "supplier_on_time": train_df.groupby("Supplier ID_Encoded")[
            "On Time Delivery"
        ].mean().to_dict(),
        "supplier_item_price": train_df.groupby(
            ["Supplier ID_Encoded", "Item Code_Encoded"]
        )["Unit Price"].mean().to_dict(),
        "category_price": train_df.groupby("Category_Encoded")[
            "Unit Price"
        ].mean().to_dict(),
        "department_savings": train_df.groupby("Department_Encoded")[
            "Savings Pct"
        ].mean().to_dict(),
    }
    return out, state


# 3.4 將較早訓練期的統計套用到未來資料，防止時間洩漏
def apply_aggregate_state(future_df, state):
    out = future_df.copy()
    global_value = state["global"]

    out["Item_Avg_Price"] = out["Item Code_Encoded"].map(
        state["item_price"]
    ).fillna(global_value["unit_price"])
    out["Item_Order_Count"] = out["Item Code_Encoded"].map(
        state["item_count"]
    ).fillna(0).astype(float)
    out["Supplier_Order_Count"] = out["Supplier ID_Encoded"].map(
        state["supplier_count"]
    ).fillna(0).astype(float)
    out["Supplier_Avg_Savings"] = out["Supplier ID_Encoded"].map(
        state["supplier_savings"]
    ).fillna(global_value["savings"])
    out["Supplier_Over_Budget_Rate"] = out["Supplier ID_Encoded"].map(
        state["supplier_over_rate"]
    ).fillna(global_value["over_rate"])
    out["Supplier_On_Time_Rate"] = out["Supplier ID_Encoded"].map(
        state["supplier_on_time"]
    ).fillna(global_value["on_time"])

    pair_keys = zip(
        out["Supplier ID_Encoded"].tolist(),
        out["Item Code_Encoded"].tolist(),
    )
    out["Supplier_Item_Avg_Price"] = [
        state["supplier_item_price"].get(key, global_value["unit_price"])
        for key in pair_keys
    ]
    out["Category_Avg_Price"] = out["Category_Encoded"].map(
        state["category_price"]
    ).fillna(global_value["unit_price"])
    out["Department_Avg_Savings"] = out["Department_Encoded"].map(
        state["department_savings"]
    ).fillna(global_value["savings"])

    out["Has_Item_History"] = (out["Item_Order_Count"] > 0).astype(int)
    out["Has_Supplier_History"] = (
        out["Supplier_Order_Count"] > 0
    ).astype(int)
    out["Budget_vs_Avg_Gap"] = safe_divide(
        out["Budget Unit Price"] - out["Item_Avg_Price"],
        out["Item_Avg_Price"],
    )
    out["Budget_vs_Supplier_Item_Gap"] = safe_divide(
        out["Budget Unit Price"] - out["Supplier_Item_Avg_Price"],
        out["Supplier_Item_Avg_Price"],
    )
    return out


# =====================================================================
# STEP 4：建立第一層回歸模型
# =====================================================================
# 4.1 XGBoost 回歸：學習較複雜的非線性關係
def make_xgb_regressor():
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror",
    )


# 4.2 隨機森林回歸：透過多棵樹降低單一模型偏差
def make_rf_regressor():
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# 4.3 Ridge 回歸：提供較容易解釋的線性基準
def make_ridge_regressor():
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )


# =====================================================================
# STEP 5：建立第一層三分類模型
# =====================================================================
# 5.1 XGBoost 三分類：使用樣本權重降低多數類別偏差
def make_xgb_classifier():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )


# 5.2 隨機森林三分類：自動平衡三類權重
def make_rf_classifier():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =====================================================================
# STEP 6：建立 OOF Stacking 工具
# =====================================================================
# 6.1 計算安全的時序折數
def get_time_series_split(n_rows):
    n_splits = min(N_OOF_SPLITS, n_rows - 1)
    if n_splits < 2:
        raise ValueError("訓練資料太少，無法建立 TimeSeriesSplit。")
    return TimeSeriesSplit(n_splits=n_splits)


# 6.2 產生三個回歸模型的時序 OOF 預測
def make_regression_oof(X, y):
    oof = np.full((len(X), 3), np.nan, dtype=float)
    for fold, (fit_idx, valid_idx) in enumerate(
        get_time_series_split(len(X)).split(X), start=1
    ):
        models = [
            make_xgb_regressor(),
            make_rf_regressor(),
            make_ridge_regressor(),
        ]
        for model_idx, model in enumerate(models):
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            oof[valid_idx, model_idx] = model.predict(X.iloc[valid_idx])
        print(f"回歸 OOF 第 {fold} 折完成：驗證 {len(valid_idx)} 筆")
    return oof


# 6.3 產生兩個分類模型的時序 OOF 類別機率
def make_classification_oof(X, y):
    # 每個模型有三類機率，因此共產生 6 個第二層特徵
    oof = np.full((len(X), 6), np.nan, dtype=float)
    for fold, (fit_idx, valid_idx) in enumerate(
        get_time_series_split(len(X)).split(X), start=1
    ):
        X_fit = X.iloc[fit_idx]
        y_fit = y.iloc[fit_idx]
        X_valid = X.iloc[valid_idx]

        # 若早期資料尚未包含全部三類，跳過該折，避免分類器類別維度錯誤
        if len(np.unique(y_fit)) < 3:
            print(f"分類 OOF 第 {fold} 折略過：訓練期尚未包含全部三類")
            continue

        xgb_model = make_xgb_classifier()
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
        xgb_model.fit(X_fit, y_fit, sample_weight=sample_weight)

        rf_model = make_rf_classifier()
        rf_model.fit(X_fit, y_fit)

        oof[valid_idx, 0:3] = xgb_model.predict_proba(X_valid)
        oof[valid_idx, 3:6] = rf_model.predict_proba(X_valid)
        print(f"分類 OOF 第 {fold} 折完成：驗證 {len(valid_idx)} 筆")
    return oof


# 6.4 將連續節省率預測轉成三類提示特徵
def savings_to_class_hint(predicted_savings):
    return np.select(
        [
            predicted_savings > CLOSE_BUDGET_LIMIT,
            predicted_savings >= -CLOSE_BUDGET_LIMIT,
        ],
        [0.0, 1.0],
        default=2.0,
    ).reshape(-1, 1)


# =====================================================================
# STEP 7：建立模型評估工具
# =====================================================================
# 7.1 評估單一回歸模型或融合回歸模型
def regression_metrics(name, y_true, predictions):
    return {
        "model": name,
        "mae": mean_absolute_error(y_true, predictions),
        "r2": r2_score(y_true, predictions),
    }


# 7.2 評估三分類與超支風險
def classification_metrics(name, y_true, predictions, probabilities):
    over_true = (np.asarray(y_true) == 2).astype(int)
    over_pred = (np.asarray(predictions) == 2).astype(int)
    over_probability = probabilities[:, 2]
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "over_precision": precision_score(
            y_true,
            predictions,
            labels=[2],
            average=None,
            zero_division=0,
        )[0],
        "over_recall": recall_score(
            y_true,
            predictions,
            labels=[2],
            average=None,
            zero_division=0,
        )[0],
        "over_f1": f1_score(
            y_true,
            predictions,
            labels=[2],
            average=None,
            zero_division=0,
        )[0],
        "over_pr_auc": average_precision_score(
            over_true, over_probability
        ),
        "over_roc_auc": roc_auc_score(over_true, over_probability),
        "predicted_over_rate": float(over_pred.mean()),
    }


# =====================================================================
# STEP 8：建立 V10 驗證集選模與機率修正工具
# =====================================================================
# 8.1 依指定倍率修正三類機率，再重新正規化為總和 1
def adjust_class_probabilities(probabilities, class_multipliers):
    adjusted = np.asarray(probabilities, dtype=float) * np.asarray(
        class_multipliers, dtype=float
    )
    row_sum = adjusted.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return adjusted / row_sum


# 8.2 融合隨機森林與第二層 Stacking 機率
def blend_probabilities(rf_probabilities, meta_probabilities, meta_weight):
    blended = (
        (1.0 - meta_weight) * np.asarray(rf_probabilities)
        + meta_weight * np.asarray(meta_probabilities)
    )
    return blended / blended.sum(axis=1, keepdims=True)


# 8.3 V10 多目標分數：兼顧整體正確率、三類平衡及超支辨識
def v10_selection_score(metrics):
    return (
        0.35 * metrics["accuracy"]
        + 0.25 * metrics["balanced_accuracy"]
        + 0.20 * metrics["macro_f1"]
        + 0.15 * metrics["over_f1"]
        + 0.05 * metrics["over_recall"]
    )


# 8.4 只使用訓練期的時序驗證資料選擇 V10 設定
def select_v10_policy(meta_X, rf_prob, y, valid_ratio=META_VALID_RATIO):
    n_rows = len(y)
    valid_size = max(100, int(n_rows * valid_ratio))
    split_at = n_rows - valid_size
    if split_at < 100:
        raise ValueError("第二層資料不足，無法再切分時序驗證集。")

    X_fit, X_valid = meta_X[:split_at], meta_X[split_at:]
    y_fit = np.asarray(y)[:split_at]
    y_valid = np.asarray(y)[split_at:]
    rf_valid_prob = np.asarray(rf_prob)[split_at:]

    # 不再只使用 balanced；加入無權重與較溫和的類別權重
    weight_candidates = [
        ("none", None),
        ("mild", {0: 1.0, 1: 1.20, 2: 1.35}),
        ("medium", {0: 1.0, 1: 1.35, 2: 1.70}),
        ("balanced", "balanced"),
    ]
    meta_weight_candidates = [0.00, 0.25, 0.50, 0.75, 1.00]
    near_multiplier_candidates = [0.90, 1.00, 1.10]
    over_multiplier_candidates = [0.45, 0.60, 0.75, 0.90, 1.00]

    rows = []
    best = None
    for weight_name, class_weight in weight_candidates:
        model = LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X_fit, y_fit)
        meta_valid_prob = model.predict_proba(X_valid)

        for meta_weight in meta_weight_candidates:
            raw_blended = blend_probabilities(
                rf_valid_prob, meta_valid_prob, meta_weight
            )
            for near_multiplier in near_multiplier_candidates:
                for over_multiplier in over_multiplier_candidates:
                    multipliers = (1.0, near_multiplier, over_multiplier)
                    final_prob = adjust_class_probabilities(
                        raw_blended, multipliers
                    )
                    final_pred = np.argmax(final_prob, axis=1)
                    metrics = classification_metrics(
                        "validation", y_valid, final_pred, final_prob
                    )
                    score = v10_selection_score(metrics)

                    # 採購實務限制：避免再次用大量誤報換取超支 Recall
                    meets_guardrails = (
                        metrics["accuracy"] >= 0.60
                        and metrics["over_precision"] >= 0.20
                        and metrics["over_recall"] >= 0.30
                        and metrics["predicted_over_rate"] <= 0.35
                    )
                    row = {
                        "class_weight_name": weight_name,
                        "meta_weight": meta_weight,
                        "near_multiplier": near_multiplier,
                        "over_multiplier": over_multiplier,
                        "selection_score": score,
                        "meets_guardrails": meets_guardrails,
                        **metrics,
                    }
                    rows.append(row)

                    rank_key = (
                        int(meets_guardrails),
                        score,
                        metrics["accuracy"],
                        metrics["over_f1"],
                    )
                    if best is None or rank_key > best["rank_key"]:
                        best = {
                            "rank_key": rank_key,
                            "class_weight_name": weight_name,
                            "class_weight": class_weight,
                            "meta_weight": meta_weight,
                            "class_multipliers": multipliers,
                            "validation_metrics": metrics,
                        }

    search_results = pd.DataFrame(rows).sort_values(
        ["meets_guardrails", "selection_score"],
        ascending=[False, False],
    )
    return best, search_results


# =====================================================================
# STEP 9：讀取原始資料
# =====================================================================
print("=" * 78)
print("模型一 V10：驗證集校正的平衡型 Stacking")
print("=" * 78)

file_path = find_data_file()
df = pd.read_csv(file_path, encoding="utf-8-sig")
df.columns = [col.split("（")[0].strip() for col in df.columns]
print(f"成功讀取資料：{file_path}")
print(f"資料筆數：{len(df)}")


# =====================================================================
# STEP 10：執行資料清洗與缺失值處理
# =====================================================================
# 9.1 清理價格、數量、年度及節省率
for col in [
    "Unit Price",
    "Budget Unit Price",
    "Quantity",
    "PO Year",
    "Savings Pct",
]:
    if col in df.columns:
        df[col] = clean_numeric(df[col])

# 9.2 清理準時交貨百分比
if "On Time Delivery" in df.columns:
    df["On Time Delivery"] = clean_percentage(df["On Time Delivery"])

# 9.3 清理二元欄位
for col in ["Maverick Spend", "Single Source Flag"]:
    if col in df.columns:
        df[col] = clean_binary(df[col])

# 9.4 數值缺失值以中位數補值
for col in df.select_dtypes(include=[np.number]).columns:
    median = df[col].median()
    df[col] = df[col].fillna(0.0 if pd.isna(median) else median)

# 9.5 文字缺失值統一補成 Unknown
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].fillna("Unknown")


# =====================================================================
# STEP 11：建立三分類標籤與文字欄位編碼
# =====================================================================
# 10.1 建立 Y：0＝節省、1＝接近預算、2＝超支
df["Savings_Category_True"] = df["Savings Pct"].apply(
    assign_savings_category
)

# 10.2 將文字欄位轉成數字編碼
le_dict = {}
encode_features = [
    "Supplier ID",
    "Item Code",
    "Category",
    "Department",
    "PO Quarter",
    "Supplier Risk",
    "Supplier Status",
    "Preferred Supplier",
]
for col in encode_features:
    if col in df.columns:
        encoder = LabelEncoder()
        df[f"{col}_Encoded"] = encoder.fit_transform(df[col].astype(str))
        le_dict[col] = encoder

print("\n全資料三分類分布：")
print(df["Savings_Category_True"].value_counts().sort_index())


# =====================================================================
# STEP 12：檢查必要欄位
# =====================================================================
required = [
    "Supplier ID_Encoded",
    "Item Code_Encoded",
    "Category_Encoded",
    "Department_Encoded",
    "PO Year",
    "PO Quarter_Encoded",
    "Unit Price",
    "Budget Unit Price",
    "Quantity",
    "Savings Pct",
    "On Time Delivery",
]
missing = [col for col in required if col not in df.columns]
if missing:
    raise KeyError(f"缺少必要欄位：{missing}")


# =====================================================================
# STEP 13：依時間排序並進行 8：2 盲測切分
# =====================================================================
df = df.sort_values(["PO Year", "PO Quarter_Encoded"]).reset_index(drop=True)
split_idx = int(len(df) * 0.8)
raw_train = df.iloc[:split_idx].copy()
raw_test = df.iloc[split_idx:].copy()
if raw_train.empty or raw_test.empty:
    raise ValueError("資料不足，無法完成 8：2 時序切分。")

print(f"\n訓練集：{len(raw_train)} 筆；測試集：{len(raw_test)} 筆")


# =====================================================================
# STEP 14：建立歷史聚合特徵
# =====================================================================
# 測試集僅能使用訓練期統計，不能使用測試期答案
train_df, aggregation_state = build_training_aggregates(raw_train)
test_df = apply_aggregate_state(raw_test, aggregation_state)


# =====================================================================
# STEP 15：設定模型使用的 X 特徵
# =====================================================================
base_features = [
    "Category_Encoded",
    "Department_Encoded",
    "Quantity",
    "Budget Unit Price",
    "Item_Avg_Price",
    "Item_Order_Count",
    "Has_Item_History",
    "Budget_vs_Avg_Gap",
    "Preferred Supplier_Encoded",
    "Supplier Risk_Encoded",
    "Supplier Status_Encoded",
    "Maverick Spend",
    "Single Source Flag",
    "Supplier_Order_Count",
    "Has_Supplier_History",
    "Supplier_Avg_Savings",
    "Supplier_Over_Budget_Rate",
    "Supplier_On_Time_Rate",
    "Supplier_Item_Avg_Price",
    "Budget_vs_Supplier_Item_Gap",
    "Category_Avg_Price",
    "Department_Avg_Savings",
]
missing = [col for col in base_features if col not in train_df.columns]
if missing:
    raise KeyError(f"缺少模型必要特徵：{missing}")

X_train = train_df[base_features].reset_index(drop=True)
X_test = test_df[base_features].reset_index(drop=True)
y_train_reg = train_df["Savings Pct"].reset_index(drop=True)
y_test_reg = test_df["Savings Pct"].reset_index(drop=True)
y_train_cls = train_df["Savings_Category_True"].reset_index(drop=True)
y_test_cls = test_df["Savings_Category_True"].reset_index(drop=True)


# =====================================================================
# STEP 16：產生第一層 OOF 預測
# =====================================================================
# OOF 是第二層融合模型的訓練資料，可避免同筆資料洩漏
reg_oof = make_regression_oof(X_train, y_train_reg)
cls_oof = make_classification_oof(X_train, y_train_cls)

# 只有所有 OOF 欄位均有效的列，才能用來訓練第二層模型
valid_oof_mask = (
    np.isfinite(reg_oof).all(axis=1)
    & np.isfinite(cls_oof).all(axis=1)
)
if valid_oof_mask.sum() < 100:
    raise ValueError("有效 OOF 資料不足，無法訓練第二層 Stacking 模型。")
print(f"\n可供第二層訓練的 OOF 資料：{valid_oof_mask.sum()} 筆")


# =====================================================================
# STEP 17：訓練第二層回歸融合模型
# =====================================================================
# Ridge Meta Model 自動學習三個回歸模型的融合權重
reg_meta_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0),
)
reg_meta_model.fit(
    reg_oof[valid_oof_mask],
    y_train_reg[valid_oof_mask],
)


# =====================================================================
# STEP 18：使用時序驗證集選擇 V10 平衡策略
# =====================================================================
# 第二層分類輸入：
# 1. XGBoost 三類機率
# 2. 隨機森林三類機率
# 3. 融合回歸預測值
# 4. 由融合回歸值轉成的類別提示
oof_reg_blended = np.full(len(X_train), np.nan)
oof_reg_blended[valid_oof_mask] = reg_meta_model.predict(
    reg_oof[valid_oof_mask]
)
meta_cls_X = np.column_stack(
    [
        cls_oof[valid_oof_mask],
        oof_reg_blended[valid_oof_mask],
        savings_to_class_hint(oof_reg_blended[valid_oof_mask]),
    ]
)

meta_y = y_train_cls[valid_oof_mask].to_numpy()
meta_rf_prob = cls_oof[valid_oof_mask, 3:6]
selected_policy, policy_search_results = select_v10_policy(
    meta_cls_X,
    meta_rf_prob,
    meta_y,
)

print("\nV10 驗證集選定策略：")
print(
    f"類別權重={selected_policy['class_weight_name']}；"
    f"Meta 權重={selected_policy['meta_weight']:.2f}；"
    f"類別倍率={selected_policy['class_multipliers']}"
)
print("驗證集指標：")
print(
    pd.Series(selected_policy["validation_metrics"])
    .drop(labels=["model"])
    .round(4)
    .to_string()
)


# =====================================================================
# STEP 19：以全部有效 OOF 資料重訓選定的第二層分類器
# =====================================================================
cls_meta_model = LogisticRegression(
    max_iter=3000,
    class_weight=selected_policy["class_weight"],
    random_state=RANDOM_STATE,
)
cls_meta_model.fit(meta_cls_X, meta_y)


# =====================================================================
# STEP 20：使用完整訓練集重訓所有第一層模型
# =====================================================================
# 18.1 重訓三個回歸模型
xgb_reg_model = make_xgb_regressor()
rf_reg_model = make_rf_regressor()
ridge_reg_model = make_ridge_regressor()

xgb_reg_model.fit(X_train, y_train_reg)
rf_reg_model.fit(X_train, y_train_reg)
ridge_reg_model.fit(X_train, y_train_reg)

# 18.2 重訓兩個三分類模型
xgb_cls_model = make_xgb_classifier()
xgb_sample_weight = compute_sample_weight(
    class_weight="balanced", y=y_train_cls
)
xgb_cls_model.fit(
    X_train,
    y_train_cls,
    sample_weight=xgb_sample_weight,
)

rf_cls_model = make_rf_classifier()
rf_cls_model.fit(X_train, y_train_cls)


# =====================================================================
# STEP 21：產生測試集預測並套用 V10 平衡策略
# =====================================================================
# 19.1 三個回歸模型各自預測
test_reg_base = np.column_stack(
    [
        xgb_reg_model.predict(X_test),
        rf_reg_model.predict(X_test),
        ridge_reg_model.predict(X_test),
    ]
)

# 19.2 第二層 Ridge 融合出最終節省率
test_reg_blended = reg_meta_model.predict(test_reg_base)

# 19.3 兩個分類器分別輸出三類機率
xgb_test_prob = xgb_cls_model.predict_proba(X_test)
rf_test_prob = rf_cls_model.predict_proba(X_test)

# 19.4 將分類機率、融合節省率及回歸類別提示送入第二層分類器
test_meta_cls_X = np.column_stack(
    [
        xgb_test_prob,
        rf_test_prob,
        test_reg_blended,
        savings_to_class_hint(test_reg_blended),
    ]
)
meta_test_prob = cls_meta_model.predict_proba(test_meta_cls_X)
stacking_test_prob = blend_probabilities(
    rf_test_prob,
    meta_test_prob,
    selected_policy["meta_weight"],
)
stacking_test_prob = adjust_class_probabilities(
    stacking_test_prob,
    selected_policy["class_multipliers"],
)
stacking_test_pred = np.argmax(stacking_test_prob, axis=1)


# =====================================================================
# STEP 22：比較單一回歸模型與融合回歸模型
# =====================================================================
regression_results = pd.DataFrame(
    [
        regression_metrics(
            "XGBoost Regressor", y_test_reg, test_reg_base[:, 0]
        ),
        regression_metrics(
            "Random Forest Regressor", y_test_reg, test_reg_base[:, 1]
        ),
        regression_metrics(
            "Ridge Regressor", y_test_reg, test_reg_base[:, 2]
        ),
        regression_metrics(
            "V10 Regression Stacking", y_test_reg, test_reg_blended
        ),
    ]
).sort_values("mae")

print("\n回歸模型比較：")
print(regression_results.round(4).to_string(index=False))


# =====================================================================
# STEP 23：比較單一分類模型與 V10 平衡型融合模型
# =====================================================================
xgb_test_pred = np.argmax(xgb_test_prob, axis=1)
rf_test_pred = np.argmax(rf_test_prob, axis=1)

classification_results = pd.DataFrame(
    [
        classification_metrics(
            "XGBoost Classifier",
            y_test_cls,
            xgb_test_pred,
            xgb_test_prob,
        ),
        classification_metrics(
            "Random Forest Classifier",
            y_test_cls,
            rf_test_pred,
            rf_test_prob,
        ),
        classification_metrics(
            "V10 Balanced Stacking",
            y_test_cls,
            stacking_test_pred,
            stacking_test_prob,
        ),
    ]
).sort_values("macro_f1", ascending=False)

print("\n三分類模型比較：")
print(classification_results.round(4).to_string(index=False))

print("\nV10 最終三分類報告（0=節省、1=接近預算、2=超支）：")
print(classification_report(y_test_cls, stacking_test_pred, zero_division=0))
print("V10 混淆矩陣（列=真實，欄=預測）：")
print(confusion_matrix(y_test_cls, stacking_test_pred, labels=[0, 1, 2]))


# =====================================================================
# STEP 24：建立供採購人員使用的決策結果表
# =====================================================================
decision_df = pd.DataFrame(
    {
        "Actual_Savings_Pct": y_test_reg,
        "Predicted_Savings_Pct": test_reg_blended,
        "Actual_Class": y_test_cls.map(CLASS_NAMES),
        "Predicted_Class": pd.Series(stacking_test_pred).map(CLASS_NAMES),
        "Saving_Probability": stacking_test_prob[:, 0],
        "Near_Budget_Probability": stacking_test_prob[:, 1],
        "Over_Budget_Probability": stacking_test_prob[:, 2],
    }
)

# 依超支風險分數產生採購審核優先級
decision_df["Review_Priority"] = pd.cut(
    decision_df["Over_Budget_Probability"],
    bins=[-np.inf, 0.30, 0.60, np.inf],
    labels=["低風險：正常流程", "中風險：補充資料", "高風險：優先審核"],
)


# =====================================================================
# STEP 25：整理特徵重要性與融合權重
# =====================================================================
xgb_importance_df = pd.DataFrame(
    {
        "Feature": base_features,
        "XGB_Importance": xgb_cls_model.feature_importances_,
        "RF_Importance": rf_cls_model.feature_importances_,
    }
)
xgb_importance_df["Average_Importance"] = xgb_importance_df[
    ["XGB_Importance", "RF_Importance"]
].mean(axis=1)
xgb_importance_df = xgb_importance_df.sort_values(
    "Average_Importance", ascending=False
)

# 取出 Ridge Meta Model 的三個融合係數
reg_meta_ridge = reg_meta_model.named_steps["ridge"]
regression_weight_df = pd.DataFrame(
    {
        "Base_Model": [
            "XGBoost Regressor",
            "Random Forest Regressor",
            "Ridge Regressor",
        ],
        "Standardized_Meta_Coefficient": reg_meta_ridge.coef_,
    }
)

print("\nV10 前 15 名分類特徵重要性：")
print(xgb_importance_df.head(15).round(5).to_string(index=False))
print("\n第二層回歸融合係數：")
print(regression_weight_df.round(5).to_string(index=False))


# =====================================================================
# STEP 26：儲存 V10 模型、設定與評估結果
# =====================================================================
# 使用獨立 V10 資料夾，不覆蓋 V4～V9
model_dir = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "model_1_aggregation_v10_balanced_stacking"
)
model_dir.mkdir(parents=True, exist_ok=True)

artifacts = {
    "xgb_reg_model.pkl": xgb_reg_model,
    "rf_reg_model.pkl": rf_reg_model,
    "ridge_reg_model.pkl": ridge_reg_model,
    "reg_meta_model.pkl": reg_meta_model,
    "xgb_cls_model.pkl": xgb_cls_model,
    "rf_cls_model.pkl": rf_cls_model,
    "cls_meta_model.pkl": cls_meta_model,
    "label_encoders.pkl": le_dict,
    "aggregation_state.pkl": aggregation_state,
    "features_list.pkl": base_features,
    "v10_config.pkl": {
        "version": "V10",
        "method": "Validation-calibrated balanced time-series OOF stacking",
        "close_budget_limit": CLOSE_BUDGET_LIMIT,
        "classes": CLASS_NAMES,
        "regression_base_models": [
            "XGBRegressor",
            "RandomForestRegressor",
            "Ridge",
        ],
        "classification_base_models": [
            "XGBClassifier",
            "RandomForestClassifier",
        ],
        "regression_meta_model": "Ridge",
        "classification_meta_model": "LogisticRegression",
        "selected_class_weight": selected_policy["class_weight_name"],
        "selected_meta_weight": selected_policy["meta_weight"],
        "selected_class_multipliers": selected_policy["class_multipliers"],
        "selection_score_weights": {
            "accuracy": 0.35,
            "balanced_accuracy": 0.25,
            "macro_f1": 0.20,
            "over_f1": 0.15,
            "over_recall": 0.05,
        },
        "note": "本版本屬於多模型融合／集成學習，不是多模態學習。",
    },
}

for filename, artifact in artifacts.items():
    with (model_dir / filename).open("wb") as file:
        pickle.dump(artifact, file)

# 另存 CSV，方便直接使用 Excel 查看
regression_results.to_csv(
    model_dir / "regression_model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)
classification_results.to_csv(
    model_dir / "classification_model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)
decision_df.to_csv(
    model_dir / "test_procurement_decisions.csv",
    index=False,
    encoding="utf-8-sig",
)
xgb_importance_df.to_csv(
    model_dir / "feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)
regression_weight_df.to_csv(
    model_dir / "regression_stacking_weights.csv",
    index=False,
    encoding="utf-8-sig",
)
policy_search_results.to_csv(
    model_dir / "v10_validation_policy_search.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"\nV10 模型與結果已另存：{model_dir}")
print("=" * 78)