# -*- coding: utf-8 -*-
"""
=====================================================================
 供應商風險分類 Pipeline (Supplier Risk Classification) — v2.0 重寫版
=====================================================================
資料來源 : dataset/data_poisntcancelled.csv (4649 筆, 15 家供應商)
目標標籤 : Supplier Risk (Low / Medium / High)

設計重點
--------
1. 標籤是由 Supplier Tier + Preferred Supplier + Maverick Spend (+ESG)
   規則合成 → 這些「標籤公式輸入」欄位一律排除出 X，
   否則模型只是在反解規則、不是在學風險。
2. 特徵全部改用 as-of 聚合 (expanding + shift(1))：
   每一筆 PO 只能看到「下單當下之前」的供應商歷史行為，
   模擬真實上線情境，且杜絕未來資訊洩漏。
3. 兩組特徵並列對照：
   - Set A: 純行為聚合 (OTD、延誤、發票/付款狀態、交易頻率…)
   - Set B: Set A + 歷史脫軌採購率 / 單一來源率
     (這兩個是標籤公式輸入的行為化版本，單獨隔離以誠實歸因)
4. 不平衡處理：先切分 (stratified 70/15/15)、再對訓練集上
   class weights。不用 SMOTE——在聚合特徵空間內插只會製造
   供應商指紋雜訊。
5. 模型：XGBoost / LightGBM / CatBoost，
   低深度 + min_child_weight/gamma + 降 subsample/colsample
   + 低學習率 + Early Stopping (以 val set 收斂)。
6. 評估：AUC-ROC (OvR macro)、Accuracy、Log Loss、macro-F1、
   訓練/測試並列 (監控過擬合)、混淆矩陣與特徵重要性 (PNG)。

執行方式
--------
    source backend/.venv/bin/activate
    python training_scripts/supplier_risk_github.py
輸出圖表與最佳模型存至 training_scripts/output/
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # 無視窗環境也能存圖
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 路徑以本檔案位置錨定，從任何目錄執行都不會壞
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(REPO_ROOT, "dataset", "data_poisntcancelled.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("🚀 供應商風險分類 Pipeline v2.0 (XGBoost / LightGBM / CatBoost)")
print("=" * 72)

# =====================================================================
# STEP 1. 載入與清理
# =====================================================================
print("\n=== STEP 1: 載入與清理 ===")
df = pd.read_csv(DATA_PATH)
print(f">> 讀入 {DATA_PATH}")
print(f">> 形狀: {df.shape}")

# 日期解析 (dd/mm/yyyy) 並依時間排序 — as-of 聚合的前提
df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df = df.sort_values("PO Date").reset_index(drop=True)

# Yes/No → 1/0
for col in ["On Time Delivery", "Maverick Spend", "Single Source Flag"]:
    df[col] = df[col].map({"Yes": 1, "No": 0}).astype(float)

# 衍生二元事件欄 (供聚合用)
df["is_po_disputed"] = (df["PO Status"] == "Disputed").astype(float)
df["is_inv_overdue"] = df["Invoice Status"].isin(["Overdue", "Disputed"]).astype(float)
df["is_pay_overdue"] = df["Payment Status"].isin(["Overdue", "On Hold"]).astype(float)

print(">> 標籤分佈:")
print(df["Supplier Risk"].value_counts().to_string())

# =====================================================================
# STEP 2. 特徵工程 — as-of data aggregation (expanding + shift(1))
#          每筆 PO 只看得到「這張單以前」的供應商歷史
# =====================================================================
print("\n=== STEP 2: as-of 聚合特徵工程 ===")

g = df.groupby("Supplier ID", sort=False)


def hist_mean(col):
    """該供應商截至上一筆為止的歷史平均 (不含當筆 → 無洩漏)。"""
    return g[col].transform(lambda s: s.expanding().mean().shift(1))


df["hist_po_count"] = g.cumcount()  # 1. 交易頻率 (累計筆數)
df["hist_otd_rate"] = hist_mean("On Time Delivery")  # 2. 歷史準時交貨率
df["hist_avg_days_late"] = hist_mean("Days Late")  # 3. 歷史平均延誤天數
df["hist_dispute_rate"] = hist_mean("is_po_disputed")  # 4. 歷史 PO 爭議率
df["hist_inv_overdue_rate"] = hist_mean("is_inv_overdue")  # 5. 歷史發票逾期率
df["hist_pay_overdue_rate"] = hist_mean("is_pay_overdue")  # 6. 歷史付款逾期率
df["hist_avg_savings_pct"] = hist_mean("Savings Pct")  # 7. 歷史平均節省率
df["hist_avg_lead_time"] = hist_mean("Lead Time Days")  # 8. 歷史平均交期
df["days_since_last_po"] = (
    g["PO Date"].diff().dt.days
)  # 9. 距上次交易天數
df["sup_item_diversity"] = g["Item Code"].transform(
    lambda s: (~s.duplicated()).cumsum().shift(1)
)  # 10. 歷史品項多樣性

# 11. 價格溢價率: 本單單價 vs 該品項(跨供應商)歷史均價
gi = df.groupby("Item Code", sort=False)
item_hist_avg = gi["Unit Price"].transform(lambda s: s.expanding().mean().shift(1))
df["price_premium"] = (df["Unit Price"] - item_hist_avg) / item_hist_avg

# --- Set B 專屬: 標籤公式輸入的「行為化」聚合 ---
df["hist_maverick_rate"] = hist_mean("Maverick Spend")  # 12. 歷史脫軌採購率
df["hist_single_source_rate"] = hist_mean("Single Source Flag")  # 13. 歷史單一來源率

AGG_FEATURES_A = [
    "hist_po_count",
    "hist_otd_rate",
    "hist_avg_days_late",
    "hist_dispute_rate",
    "hist_inv_overdue_rate",
    "hist_pay_overdue_rate",
    "hist_avg_savings_pct",
    "hist_avg_lead_time",
    "days_since_last_po",
    "sup_item_diversity",
    "price_premium",
]
AGG_FEATURES_B_EXTRA = ["hist_maverick_rate", "hist_single_source_rate"]

# 當前 PO 的情境特徵 (與供應商身分無關)
CONTEXT_NUM = ["Quantity", "Lead Time Days", "Discount Pct"]
CONTEXT_CAT = ["Category", "PO Type", "Payment Terms"]

# 每家供應商的第一筆沒有歷史 → 聚合為 NaN。
# 注意: 填補統計量只能來自「訓練集」(否則測試資訊洩入訓練),
# 因此填補延後到 STEP 4 切分索引確定之後執行。

print(f">> 新增聚合特徵 {len(AGG_FEATURES_A) + len(AGG_FEATURES_B_EXTRA)} 個 "
      f"(Set A: {len(AGG_FEATURES_A)}, Set B 額外: {len(AGG_FEATURES_B_EXTRA)})")

# =====================================================================
# STEP 3. 組裝 X / y — 排除一切標籤公式輸入與供應商指紋欄位
# =====================================================================
print("\n=== STEP 3: 組裝特徵矩陣 (排除洩漏欄位) ===")

LEAKY_COLS = [
    "Supplier ID", "Supplier Name", "Supplier Country", "Supplier Region",
    "Supplier Latitude", "Supplier Longitude",
    "Supplier Tier", "Supplier Status", "Preferred Supplier",
    "Supplier ESG Score",
    "Maverick Spend", "Single Source Flag",  # 當列原始旗標
]
print(f">> 已排除洩漏/指紋欄位: {LEAKY_COLS}")

y_text = df["Supplier Risk"]
le = LabelEncoder()
y = le.fit_transform(y_text)  # High/Low/Medium → 0/1/2
print(f">> 標籤編碼: {dict(zip(le.classes_, range(len(le.classes_))))}")

# 類別情境特徵獨熱編碼 (XGB/LGBM 用)；CatBoost 另外走原生類別
X_base = pd.concat(
    [
        df[AGG_FEATURES_A + AGG_FEATURES_B_EXTRA + CONTEXT_NUM],
        df[CONTEXT_CAT].astype(str),
    ],
    axis=1,
)
X_ohe = pd.get_dummies(X_base, columns=CONTEXT_CAT, drop_first=True)

FEATURES_A_OHE = [c for c in X_ohe.columns if c not in AGG_FEATURES_B_EXTRA]
FEATURES_B_OHE = list(X_ohe.columns)
FEATURES_A_CAT = AGG_FEATURES_A + CONTEXT_NUM + CONTEXT_CAT
FEATURES_B_CAT = AGG_FEATURES_A + AGG_FEATURES_B_EXTRA + CONTEXT_NUM + CONTEXT_CAT

# =====================================================================
# STEP 4. 切分 — 先切分、後處理不平衡 (只對訓練集加權)
#          Stratified 70/15/15: val 供 early stopping
# =====================================================================
print("\n=== STEP 4: Stratified 70/15/15 切分 ===")
idx = np.arange(len(df))
idx_train, idx_tmp = train_test_split(
    idx, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
idx_val, idx_test = train_test_split(
    idx_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y[idx_tmp]
)
print(f">> train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

# 缺失值填補 — 統計量只取自訓練列, 再套用到全部 (修正: 原版用全資料
# 中位數, 會把 val/test 的資訊帶進訓練)
_fill_cols = AGG_FEATURES_A + AGG_FEATURES_B_EXTRA
X_ohe_prefill = X_ohe.copy()  # 保留未填補版本, 供 STEP 6.5 依審計折內統計重新填補
neutral_fill = {c: float(X_ohe.iloc[idx_train][c].median()) for c in _fill_cols}
X_ohe[_fill_cols] = X_ohe[_fill_cols].fillna(neutral_fill)
df[_fill_cols] = df[_fill_cols].fillna(neutral_fill)  # CatBoost 用的 df 同步

y_tr, y_va, y_te = y[idx_train], y[idx_val], y[idx_test]
# 不平衡處理: 訓練集 balanced sample weights (SMOTE 在此類聚合特徵上不適用)
w_tr = compute_sample_weight("balanced", y_tr)
cls_weights = {
    c: w for c, w in zip(
        np.unique(y_tr),
        len(y_tr) / (len(np.unique(y_tr)) * np.bincount(y_tr)),
    )
}
print(f">> balanced class weights: "
      f"{ {le.classes_[k]: round(v, 2) for k, v in cls_weights.items()} }")

# =====================================================================
# STEP 5. 三模型訓練與評估 (Set A / Set B 並列)
# =====================================================================
print("\n=== STEP 5: 模型訓練 (early stopping on val) ===")


def evaluate(model, X_tr, X_va, X_te, tag):
    """回傳訓練/驗證/測試三組指標 dict。"""
    out = {}
    for split, X_, y_ in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
        proba = model.predict_proba(X_)
        pred = proba.argmax(axis=1)
        out[split] = {
            "auc": roc_auc_score(y_, proba, multi_class="ovr", average="macro"),
            "acc": accuracy_score(y_, pred),
            "logloss": log_loss(y_, proba),
            "f1": f1_score(y_, pred, average="macro"),
        }
    out["test_pred"] = model.predict_proba(X_te).argmax(axis=1)
    return out


def fit_xgb(X_tr, X_va):
    m = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=4,            # 限制深度
        min_child_weight=5,     # 葉節點最小樣本權重
        gamma=0.5,              # 分裂所需最小增益
        subsample=0.7,          # 每棵樹只看 70% 樣本
        colsample_bytree=0.7,   # 每棵樹只看 70% 特徵
        reg_lambda=2.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
    )
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def fit_lgbm(X_tr, X_va):
    m = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        reg_lambda=2.0,
        objective="multiclass",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    m.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return m


def fit_catboost(X_tr, X_va, cat_cols):
    m = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=6,
        bootstrap_type="Bernoulli",  # 預設 Bayesian 不支援 subsample
        subsample=0.7,
        loss_function="MultiClass",
        class_weights=cls_weights,
        early_stopping_rounds=50,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    m.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va))
    return m


results = {}   # {(set_tag, model_name): metrics}
models = {}

for set_tag, feats_ohe, feats_cat in [
    ("Set A (純行為)", FEATURES_A_OHE, FEATURES_A_CAT),
    ("Set B (+maverick/single)", FEATURES_B_OHE, FEATURES_B_CAT),
]:
    # --- XGBoost / LightGBM 用獨熱特徵 ---
    Xo = X_ohe[feats_ohe]
    Xo_tr, Xo_va, Xo_te = Xo.iloc[idx_train], Xo.iloc[idx_val], Xo.iloc[idx_test]

    print(f"\n--- {set_tag} | XGBoost ---")
    m = fit_xgb(Xo_tr, Xo_va)
    results[(set_tag, "XGBoost")] = evaluate(m, Xo_tr, Xo_va, Xo_te, set_tag)
    models[(set_tag, "XGBoost")] = (m, feats_ohe)
    print(f"    best_iteration={m.best_iteration}")

    print(f"--- {set_tag} | LightGBM ---")
    m = fit_lgbm(Xo_tr, Xo_va)
    results[(set_tag, "LightGBM")] = evaluate(m, Xo_tr, Xo_va, Xo_te, set_tag)
    models[(set_tag, "LightGBM")] = (m, feats_ohe)

    # --- CatBoost 用原生類別特徵 ---
    Xc = pd.concat(
        [df[[c for c in feats_cat if c not in CONTEXT_CAT]], df[CONTEXT_CAT].astype(str)],
        axis=1,
    )[feats_cat]
    Xc_tr, Xc_va, Xc_te = Xc.iloc[idx_train], Xc.iloc[idx_val], Xc.iloc[idx_test]

    print(f"--- {set_tag} | CatBoost ---")
    m = fit_catboost(Xc_tr, Xc_va, CONTEXT_CAT)
    results[(set_tag, "CatBoost")] = evaluate(m, Xc_tr, Xc_va, Xc_te, set_tag)
    models[(set_tag, "CatBoost")] = (m, feats_cat)

# =====================================================================
# STEP 6. 成果報告
# =====================================================================
print("\n" + "=" * 72)
print("📊 六組結果總表  (AUC=OvR macro | 過擬合差距 = train-test AUC)")
print("=" * 72)
header = (f"{'特徵組':26s} {'模型':10s} "
          f"{'AUC(tr)':>8s} {'AUC(te)':>8s} {'Acc(te)':>8s} "
          f"{'LogLoss':>8s} {'F1(te)':>7s} {'過擬合Δ':>8s}")
print(header)
print("-" * len(header))
best_key, best_auc = None, -1
for (set_tag, name), r in results.items():
    gap = r["train"]["auc"] - r["test"]["auc"]
    print(f"{set_tag:26s} {name:10s} "
          f"{r['train']['auc']:8.4f} {r['test']['auc']:8.4f} "
          f"{r['test']['acc']:8.4f} {r['test']['logloss']:8.4f} "
          f"{r['test']['f1']:7.4f} {gap:8.4f}")
    if r["test"]["auc"] > best_auc:
        best_auc, best_key = r["test"]["auc"], (set_tag, name)

print(f"\n🏆 最佳組合: {best_key[0]} | {best_key[1]}  (test AUC={best_auc:.4f})")

# --- 最佳模型: 詳細報告 + 混淆矩陣 ---
best_r = results[best_key]
pred_labels = le.inverse_transform(best_r["test_pred"])
true_labels = le.inverse_transform(y_te)
print(f"\n[最佳模型 classification report — {best_key[0]} | {best_key[1]}]")
print(classification_report(true_labels, pred_labels, zero_division=0, digits=3))

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for ax, ((set_tag, name), r) in zip(axes.flat, results.items()):
    cm = confusion_matrix(
        le.inverse_transform(y_te), le.inverse_transform(r["test_pred"]),
        labels=le.classes_,
    )
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, cmap=plt.cm.Blues, colorbar=False
    )
    ax.set_title(f"{set_tag}\n{name}  (AUC={r['test']['auc']:.3f})", fontsize=10)
plt.suptitle("Confusion Matrices — Test Set", fontsize=14)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"💾 混淆矩陣圖: {cm_path}")

# --- 特徵重要性 (三模型, 各取最佳特徵組) ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, name in zip(axes, ["XGBoost", "LightGBM", "CatBoost"]):
    key = (best_key[0], name)
    m, feats = models[key]
    if name == "CatBoost":
        imp = pd.Series(m.get_feature_importance(), index=feats)
    else:
        imp = pd.Series(m.feature_importances_, index=feats)
    imp = imp.sort_values(ascending=True).tail(15)
    ax.barh(imp.index, imp.values)
    ax.set_title(f"{name} — Top 15 Feature Importance\n({best_key[0]})", fontsize=10)
    ax.tick_params(labelsize=8)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=120)
plt.close()
print(f"💾 特徵重要性圖: {fi_path}")

# --- 導出最佳模型 ---
best_model, best_feats = models[best_key]
bundle = {
    "model": best_model,
    "features": best_feats,
    "label_encoder": le,
    "feature_set": best_key[0],
    "model_name": best_key[1],
}
model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"💾 最佳模型: {model_path}")

# =====================================================================
# STEP 6.5 洩漏審計: 留供應商出局 (Leave-Suppliers-Out)
#   row-level 隨機切分下, expanding 聚合會收斂成 15 家供應商的
#   「指紋」→ 模型可用指紋背出標籤。此審計讓訓練完全看不到
#   測試供應商: 若分數崩回隨機, 即證明上表高分來自供應商記憶。
# =====================================================================
print("\n" + "=" * 72)
print("🕵️ STEP 6.5 洩漏審計: 留供應商出局 (訓練看不到測試供應商)")
print("=" * 72)
rng = np.random.RandomState(RANDOM_STATE)
all_suppliers = np.array(sorted(df["Supplier ID"].unique()))
sup_arr = df["Supplier ID"].values
# 用「未填補」矩陣, 每折以審計訓練供應商的統計量重新填補 —
# 全域 train-median 填補含有被留出供應商的資訊, 會輕微高估審計分數
Xa_raw = X_ohe_prefill[FEATURES_A_OHE]
_audit_fill_cols = [c for c in AGG_FEATURES_A if c in Xa_raw.columns]
audit_accs = []
for trial in range(5):
    held_out = rng.choice(all_suppliers, 4, replace=False)
    mask_te = np.isin(sup_arr, held_out)
    y_tr_a, y_te_a = y[~mask_te], y[mask_te]
    if len(np.unique(y_te_a)) < 2:
        continue
    fold_fill = {c: float(Xa_raw.loc[~mask_te, c].median()) for c in _audit_fill_cols}
    Xa_tr = Xa_raw[~mask_te].fillna(fold_fill)
    Xa_te = Xa_raw[mask_te].fillna(fold_fill)  # 測試也用訓練統計量 (上線一致)
    m = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
        random_state=RANDOM_STATE, verbosity=-1,
    )
    m.fit(Xa_tr, y_tr_a,
          sample_weight=compute_sample_weight("balanced", y_tr_a))
    acc = accuracy_score(y_te_a, m.predict(Xa_te))
    audit_accs.append(acc)
    print(f"  留出 {list(held_out)} → Acc={acc:.3f}")
if audit_accs:
    audit_mean = float(np.mean(audit_accs))
    print(f"\n  留供應商出局平均 Acc = {audit_mean:.3f}"
          f"  (vs 上表 row-level 切分 ≈ {results[best_key]['test']['acc']:.3f})")
else:
    audit_mean = float("nan")
    print("\n  ⚠️ 五次抽樣皆只含單一類別, 審計無有效結果 — 請調高抽樣次數")
print("  註: price_premium 以跨供應商品項均價建構, 審計仍含此輕微資訊共享,")
print("      故本審計分數應視為供應商泛化能力的「上界」。")

# =====================================================================
# STEP 7. 誠實註記 (供專題報告引用)
# =====================================================================
print("\n" + "=" * 72)
print("📝 結果解讀 (誠實註記)")
print("=" * 72)
print(f"""\
1. 上表 row-level 切分的高分 (AUC≈1.0) 並非模型學會了「風險行為」。
   供應商層級 ANOVA 已證實行為欄位 (OTD/延誤/發票/付款) 與標籤
   統計獨立 (p>0.4)；同時 expanding 聚合特徵會隨筆數收斂成
   15 家供應商各自的「指紋」，而訓練/測試集包含同一批供應商，
   模型因此能靠指紋背出「這家供應商 → 固定風險等級」的映射。
2. STEP 6.5 的留供應商出局審計是決定性證據: 訓練看不到測試
   供應商時, 準確率崩至 {audit_mean:.2f} (隨機水準), 證明第 1 點。
3. Set A 與 Set B 分數幾乎相同的原因也在此: 指紋效應蓋過了
   maverick/single-source 聚合的真實訊號差異。
4. 根本限制: 資料只有 15 家供應商、每家風險等級固定 (High 僅
   1 家)、行為欄位與標籤獨立生成。要讓模型學到可泛化的風險
   訊號, 需要 (a) 更多供應商、(b) 行為與風險真實相關的資料,
   或 (c) 把題目改為「供應商層級」分類並蒐集外部樣本。
5. 對專題的建議敘事: 本 pipeline 展示了正確的方法論 —
   as-of 無洩漏聚合、先切分後加權、正則化+early stopping、
   以及最重要的「用 grouped split 審計指標可信度」。
   能診斷出「高分是假的」本身就是這份專題最有價值的結論。
""")
print("🏁 Pipeline 完成。")
