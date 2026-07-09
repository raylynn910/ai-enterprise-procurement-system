import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

print("=====================================================================")
print("🚀 啟動 AI 驅動企業採購智慧決策系統：高準度優化版 ML Pipeline (v6.0)")
print("=====================================================================")

# =====================================================================
# STEP 1. 資料清洗、多欄位強健型資料轉換與官方 Y 標籤建構
# =====================================================================
print("\n=== STEP 1: 執行自動化清洗與多欄位強健型數值轉換 ===")

file_path = "Data_中英欄位.csv"
if not os.path.exists(file_path):
    file_path = "Data_中英欄位_After_EDA.csv"

df = pd.read_csv(file_path)

# 1.1 移除中文字元與括號，確保欄位錨定
df.columns = [col.split("（")[0].strip() for col in df.columns]

# 【核心型態防禦函數】：確保 100% 數值純淨
def force_numeric_clean(series):
    s_str = series.astype(str).str.strip().str.lower()
    s_str = s_str.replace({
        "yes": "1", "no": "0", 
        "true": "1", "false": "0", 
        "unknown": "0"
    })
    s_str = s_str.str.replace("%", "", regex=False)
    s_num = pd.to_numeric(s_str, errors='coerce')
    
    fallback_value = s_num.median()
    if pd.isna(fallback_value):
        fallback_value = 0.0
    s_num = s_num.fillna(fallback_value)
    
    if s_num.max() > 1.0:
        s_num = s_num / 100.0
    return s_num

# 嚴格校正所有採購與合規特徵
target_numeric_cols = ["On Time Delivery", "Maverick Spend", "Single Source Flag", "Unit Price", "Budget Unit Price"]
for col in target_numeric_cols:
    if col in df.columns:
        df[col] = force_numeric_clean(df[col])

# 1.2 剩餘欄位缺失值填補防線
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=[object, "str"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# 1.3 依官方定義建立 Savings_Category 三元分類基準線
def assign_savings_category(pct):
    if pct > 2.0:
        return 0
    elif pct >= -2.0:
        return 1
    else:
        return 2

df["Savings_Category_True"] = df["Savings Pct"].apply(assign_savings_category)

# 1.4 類別特徵全面 Label Encoding
le_dict = {}
encode_features = ["Supplier ID", "Item Code", "Category", "Department", "PO Quarter", "Supplier Risk", "Supplier Status", "Preferred Supplier"]
for col in encode_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col + "_Encoded"] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# =====================================================================
# STEP 2. 建立時序防禦機制與【高準度特徵衍生工程】
# =====================================================================
print("\n=== STEP 2: 啟動時序防禦機制，執行 8:2 盲測切分 ===")

df = df.sort_values(by=["PO Year", "PO Quarter_Encoded"]).reset_index(drop=True)

# 2.1 計算品項滾動歷史均價 (Item_Avg_Price)
df["Price_Sum"] = df.groupby("Item Code_Encoded")["Unit Price"].transform(lambda x: x.cumsum().shift(1)).fillna(0)
df["Item_Count"] = df.groupby("Item Code_Encoded")["Unit Price"].transform(lambda x: pd.Series(np.arange(len(x)), index=x.index).shift(1)).fillna(0)
df["Item_Avg_Price"] = (df["Price_Sum"] / df["Item_Count"]).replace([np.inf, -np.inf], np.nan).fillna(0)
df = df.drop(columns=["Price_Sum", "Item_Count"])

# 🚀【高準度核心優化特徵】：衍生預算單價與歷史行情的內在價差率（Gap Pct）
# 這能直接賦予模型強大的商業邏輯感知，大幅拉升準確度
df["Budget_vs_Avg_Gap"] = (df["Budget Unit Price"] - df["Item_Avg_Price"]) / (df["Item_Avg_Price"] + 1e-5)

# 2.2 劃分歷史訓練集與未來盲測集
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()
print(f">> 時序切分完成：歷史訓練集 {len(train_df)} 筆，未來測試集 {len(test_df)} 筆")

# =====================================================================
# STEP 3. 核心題目 1：預測 Savings Pct (先迴歸、後分類之雙階優化架構)
# =====================================================================
print("\n=== STEP 3: 開始訓練【核心題目一：Savings Pct 雙階預測模型】 ===")

# 將全新衍生的「Budget_vs_Avg_Gap」納入特徵群
features_t1 = [
    "Category_Encoded", "Department_Encoded", "Quantity", "Budget Unit Price", "Item_Avg_Price", 
    "Budget_vs_Avg_Gap", 
    "Preferred Supplier_Encoded", "Supplier Risk_Encoded", "Supplier Status_Encoded",
    "Maverick Spend", "Single Source Flag"
]

X_train_t1 = train_df[features_t1]
y_train_t1_reg = train_df["Savings Pct"]  

X_test_t1 = test_df[features_t1]
y_test_t1_reg = test_df["Savings Pct"]
y_test_t1_cls = test_df["Savings_Category_True"]  

# 🚀 3.1 第一階迴歸模型優化：進行超參數細緻微調（降 LR、加 Trees、引入特徵隨機抽樣）
reg_model = XGBRegressor(
    n_estimators=300,        # 增加樹量，拉高泛化能力
    learning_rate=0.02,      # 降低步長，細緻逼近最優解
    max_depth=5, 
    subsample=0.8,           # 隨機樣本抽樣防止過擬合
    colsample_bytree=0.8,    # 隨機特徵抽樣降低雜訊影響
    random_state=42
)
reg_model.fit(X_train_t1, y_train_t1_reg)

pred_train_reg = reg_model.predict(X_train_t1)
pred_test_reg = reg_model.predict(X_test_t1)

print(f">> [第一階迴歸評估] MAE: {mean_absolute_error(y_test_t1_reg, pred_test_reg):.4f}%, R2 Score: {r2_score(y_test_t1_reg, pred_test_reg):.4f}")

# 3.2 第二階：建立二次微調特徵矩陣
train_df["Pred_Savings_Pct"] = pred_train_reg
test_df["Pred_Savings_Pct"] = pred_test_reg

# 3.3 針對分類階段進行鄰居自適應 SMOTE 數據防禦
print(">> 執行迴歸轉分類之不平衡特徵微調機制...")
X_train_cls = train_df[features_t1].copy()
X_train_cls["Pred_Value"] = pred_train_reg  
y_train_cls = train_df["Savings_Category_True"]

min_samples = y_train_cls.value_counts().min()
if min_samples > 1:
    safe_k = min(5, min_samples - 1)
    sampler = SMOTE(k_neighbors=safe_k, random_state=42)
else:
    sampler = RandomOverSampler(random_state=42)

X_res, y_res = sampler.fit_resample(X_train_cls, y_train_cls)

# 🚀 3.4 第二階分類器精確收斂微調
cls_refiner = XGBClassifier(
    n_estimators=200, 
    learning_rate=0.03, 
    max_depth=6, 
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
cls_refiner.fit(X_res, y_res)

X_test_cls = test_df[features_t1].copy()
X_test_cls["Pred_Value"] = pred_test_reg
final_cls_preds = cls_refiner.predict(X_test_cls)

print("\n[官方驗證 - Savings Pct 最終三類成本分類報告 (0:節省, 1:接近預算, 2:超支)]:")
print(classification_report(y_test_t1_cls, final_cls_preds, zero_division=0))

# =====================================================================
# STEP 4. 核心題目 2：IT Software 供應商推薦分數引擎
# =====================================================================
print("\n=== STEP 4: 啟動【核心題目二：IT Software 供應商智慧推薦與綜合評分】 ===")

it_train = train_df[train_df["Category"] == "IT Software"].copy()
if len(it_train) == 0:
    it_train = train_df.copy()

supplier_stats = it_train.groupby("Supplier ID").agg({
    "Savings Pct": "mean",
    "On Time Delivery": "mean",
    "Supplier Risk_Encoded": "mean",
    "Supplier ESG Score": "mean",
    "Preferred Supplier_Encoded": "mean",
    "Single Source Flag": "mean"
}).reset_index()

s_min, s_max = supplier_stats["Savings Pct"].min(), supplier_stats["Savings Pct"].max()
supplier_stats["Savings_Score"] = ((supplier_stats["Savings Pct"] - s_min) / (s_max - s_min + 1e-5)) * 100
supplier_stats["Delivery_Score"] = supplier_stats["On Time Delivery"] * 100
supplier_stats["Risk_Score"] = ((supplier_stats["Supplier Risk_Encoded"].max() - supplier_stats["Supplier Risk_Encoded"]) / (supplier_stats["Supplier Risk_Encoded"].max() + 1e-5)) * 100  
supplier_stats["ESG_Score"] = supplier_stats["Supplier ESG Score"]

supplier_stats["Recommendation_Score"] = (
    supplier_stats["Savings_Score"] * 0.30 +
    supplier_stats["Delivery_Score"] * 0.25 +
    supplier_stats["Risk_Score"] * 0.15 +
    supplier_stats["ESG_Score"] * 0.15 +
    supplier_stats["Preferred Supplier_Encoded"] * 15 -
    supplier_stats["Single Source Flag"] * 5
)

top_suppliers = supplier_stats.sort_values(by="Recommendation_Score", ascending=False).head(3)

print("\n⭐ [儀表板決策輔助：IT Software 頂級供應商推薦名單] ⭐")
for idx, row in top_suppliers.iterrows():
    print(f"🏆 推薦排名：{row['Supplier ID']} | 綜合決策得分: {row['Recommendation_Score']:.2f}")
    print(f"   👉 推薦理由：平均採購節省率高達 {row['Savings Pct']:.2f}%, 歷史準時交付率 {row['On Time Delivery']*100:.2f}%, ESG 指標評分 {row['Supplier ESG Score']:.1f}")

print("\n=====================================================================")
print("🏁 Pipeline 安全解鎖，全流程順利執行完畢。")
print("=====================================================================")

# =====================================================================
# 🚀 額外新增：打包匯出模型與編碼器，準備對接到前端 Web App
# =====================================================================
import pickle

print("\n📦 開始打包模型與編碼器元件...")

# 建立一個資料夾來存放模型元件
os.makedirs("models", exist_ok=True)

# 1. 打包第一階迴歸模型與第二階分類模型
with open("models/reg_model.pkl", "wb") as f:
    pickle.dump(reg_model, f)
with open("models/cls_refiner.pkl", "wb") as f:
    pickle.dump(cls_refiner, f)

# 2. 打包所有類別欄位的 LabelEncoder（前端轉換必備）
with open("models/le_dict.pkl", "wb") as f:
    pickle.dump(le_dict, f)

# 3. 打包特徵欄位清單，確保前端輸入的欄位順序與訓練時 100% 一致
with open("models/features_list.pkl", "wb") as f:
    pickle.dump(features_t1, f)

# 4. 打包 STEP 4 計算出來的 IT Software 頂級供應商推薦表，直接當作 Dashboard 的底表
supplier_stats.to_csv("models/it_supplier_recommendations.csv", index=False)

print("💾 所有模型元件已成功導出至 models/ 資料夾！您可以開始對接前端網頁了。")