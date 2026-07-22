import os
import sys
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("=====================================================================")
print("🚀 [模組一] 啟動：Savings Pct 雙階成本預測與分類 Pipeline (v6.0)")
print("=====================================================================")

# =====================================================================
# STEP 1. 資料清洗、多欄位強健型資料轉換與官方 Y 標籤建構
# =====================================================================
print("\n=== STEP 1: 執行自動化清洗與多欄位強健型數值轉換 ===")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Data_中英欄位.csv")
if not os.path.exists(file_path):
    file_path = os.path.join(current_dir, "Data_中英欄位_After_EDA.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ 找不到資料來源檔案，請檢查檔案是否存在。")

df = pd.read_csv(file_path, encoding="utf-8-sig")

# 1.1 移除中文字元與括號，確保欄位錨定
df.columns = [col.split("（")[0].strip() for col in df.columns]

# 【核心型態防禦函數】：確保 100% 數值純淨
def force_numeric_clean(series, is_price=False):
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
    
    if not is_price and s_num.max() > 1.0:
        s_num = s_num / 100.0
    return s_num

# 嚴格校正所有採購與合規特徵
target_numeric_cols = ["On Time Delivery", "Maverick Spend", "Single Source Flag", "Unit Price", "Budget Unit Price"]
for col in target_numeric_cols:
    if col in df.columns:
        is_price_col = ("Price" in col or "Amount" in col)
        df[col] = force_numeric_clean(df[col], is_price=is_price_col)

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
df["Item_Avg_Price"] = df.groupby("Item Code_Encoded")["Unit Price"].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

# 🚀【高準度核心優化特徵】：衍生預算單價與歷史行情的內在價差率（Gap Pct）
df["Budget_vs_Avg_Gap"] = (df["Budget Unit Price"] - df["Item_Avg_Price"]) / (df["Item_Avg_Price"] + 1e-5)

# 2.2 劃分歷史訓練集與未來盲測集
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()
print(f">> 時序切分完成：歷史訓練集 {len(train_df)} 筆，未來測試集 {len(test_df)} 筆")

# =====================================================================
# STEP 3. 核心題目 1 訓練：先迴歸、後分類之雙階優化架構
# =====================================================================
print("\n=== STEP 3: 開始訓練【核心題目一：Savings Pct 雙階預測模型】 ===")

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

# 🚀 3.1 第一階迴歸模型優化
reg_model = XGBRegressor(
    n_estimators=300,        
    learning_rate=0.02,      
    max_depth=5, 
    subsample=0.8,           
    colsample_bytree=0.8,    
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
# STEP 4. 模型與組件打包導出 (與後端對接必備)
# =====================================================================
print("\n📦 開始打包模型與編碼器元件...")
model_dir = os.path.join(os.path.dirname(current_dir), "backend", "models")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "reg_model.pkl"), "wb") as f:
    pickle.dump(reg_model, f)
with open(os.path.join(model_dir, "cls_refiner.pkl"), "wb") as f:
    pickle.dump(cls_refiner, f)
with open(os.path.join(model_dir, "le_dict.pkl"), "wb") as f:
    pickle.dump(le_dict, f)
with open(os.path.join(model_dir, "features_list.pkl"), "wb") as f:
    pickle.dump(features_t1, f)

print(f"💾 [模組一] 所有模型元件 (v6.0) 已成功導出至 {model_dir} 資料夾！")
print("=====================================================================")