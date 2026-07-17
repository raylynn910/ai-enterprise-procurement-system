import os
import numpy as np
import pandas as pd

print("=====================================================================")
print("🚀 [模組二] 啟動：IT Software 供應商智慧推薦與綜合評分引擎 (v6.5 - 企業級強健版)")
print("=====================================================================")

# =====================================================================
# STEP 1. 資料清洗與關鍵數值欄位純淨化
# =====================================================================
file_path = "Data_中英欄位.csv"
if not os.path.exists(file_path):
    file_path = "Data_中英欄位_After_EDA.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ 找不到資料來源檔案，請檢查檔案是否存在。")

df = pd.read_csv(file_path)

# 1.1 移除中文字元與括號，錨定標準英文欄位名稱
df.columns = [col.split("（")[0].strip() for col in df.columns]

# 【修正問題一：逐筆防禦函數】拒絕全體縮放，精準修正單筆百分比登打異常與極端髒資料
def force_numeric_clean_robust(series):
    s_str = series.astype(str).str.strip().str.lower()
    s_str = s_str.replace({
        "yes": "1", "no": "0", 
        "true": "1", "false": "0", 
        "unknown": "0"
    })
    s_str = s_str.str.replace("%", "", regex=False)
    s_num = pd.to_numeric(s_str, errors='coerce')
    
    col_median = s_num.median()
    if pd.isna(col_median) or col_median > 1.0:
        col_median = 0.8  # 給予合理的業務預設值（例如 80% 準交率）
        
    def fix_row_value(val, fallback):
        if pd.isna(val):
            return fallback
        if val > 100.0 or val < 0.0:  # 絕對異常值直接剔除填補
            return fallback
        if val > 1.0:                  # 修正打成 95 的百分比整數（變回 0.95）
            return val / 100.0
        return val                     # 正確的率（0.0 到 1.0 之間）

    return s_num.apply(lambda x: fix_row_value(x, col_median))

# 強制清洗與對齊關鍵業務特徵
if "On Time Delivery" in df.columns:
    df["On Time Delivery"] = force_numeric_clean_robust(df["On Time Delivery"])
if "Single Source Flag" in df.columns:
    df["Single Source Flag"] = force_numeric_clean_robust(df["Single Source Flag"])

# 編碼 Supplier Risk 與 Preferred Supplier 作為權重計算依據
if "Supplier Risk" in df.columns:
    risk_mapping = {"Low": 0, "Medium": 1, "High": 2, "Unknown": 1}
    df["Supplier Risk_Encoded"] = df["Supplier Risk"].map(risk_mapping).fillna(1)
else:
    df["Supplier Risk_Encoded"] = 1

if "Preferred Supplier" in df.columns:
    df["Preferred Supplier_Encoded"] = df["Preferred Supplier"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0)
else:
    df["Preferred Supplier_Encoded"] = 0

# 缺失值常規填補
if "Supplier ESG Score" in df.columns:
    df["Supplier ESG Score"] = df["Supplier ESG Score"].fillna(df["Supplier ESG Score"].median())
else:
    df["Supplier ESG Score"] = 75.0


# =====================================================================
# STEP 2. 全量最新數據排序（修正問題二：移除 8:2 切分，拒絕資訊時空盲區）
# =====================================================================
# 儀表板排行榜需反映當前最新現況（包含新進優質廠商），不進行機器學習的盲測切分
if "PO Quarter" in df.columns:
    df["PO Quarter_Cleaned"] = df["PO Quarter"].astype(str)
    q_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    df["Q_Weight"] = df["PO Quarter_Cleaned"].map(q_map).fillna(1)
    df = df.sort_values(by=["PO Year", "Q_Weight"]).reset_index(drop=True)
    df = df.drop(columns=["Q_Weight"])

# 直接使用全量數據作為排行榜分析底層
full_analysis_df = df.copy()


# =====================================================================
# STEP 3. 篩選 IT Software 類別並改用主鍵聚合（修正問題三、五）
# =====================================================================
print("\n=== STEP 2: 開始聚合計算 IT Software 供應商綜合決策指標 ===")

it_data = full_analysis_df[full_analysis_df["Category"] == "IT Software"].copy()

if len(it_data) == 0:
    print("⚠️ 警告：未偵測到 'IT Software' 類別！自動切換為全品類資料計算以保持推薦引擎正常運作。")
    it_data = full_analysis_df.copy()

# 【修正問題三】：改用唯一的 'Supplier ID' 進行聚合，避免同名同姓的廠商數據混淆失真
# 【修正問題五】：風險指標改用 'max'（一票否決最大風險），單一來源改用 'last'（最新當前狀態）
supplier_stats = it_data.groupby("Supplier ID").agg({
    "Supplier Name": "first",                  # 順便撈出供應商名稱以便前端對照呈現
    "Savings Pct": "mean",                     # 平均省錢效益
    "On Time Delivery": "mean",                # 平均準交表現
    "Supplier Risk_Encoded": "max",            # 🔥 修正五：只要發生過一次高風險，就必須記錄最大風險值
    "Supplier ESG Score": "mean",              # 平均永續評分
    "Preferred Supplier_Encoded": "last",      # 優先供應商以最新的合約資格為準
    "Single Source Flag": "last"               # 🔥 修正五：單一來源狀態以最新一次採購現況為準
}).reset_index()


# =====================================================================
# STEP 4. 權重打分公式收斂（修正問題四：極端值業務截斷處理）
# =====================================================================
# 【修正問題四】：對 Savings Pct 進行業務範圍截斷（Clip），防止省 90% 的學霸摧毀常態分布鑑別度
# 業務合理範圍設定為：超支最多 -10.0%，節省最多 +20.0%
clipped_savings = supplier_stats["Savings Pct"].clip(lower=-10.0, upper=20.0)

s_min, s_max = clipped_savings.min(), clipped_savings.max()
# 有了業務截斷保護後，分母不會被拉大，穩定省下 5% ~ 8% 的優秀廠商就能拿到 75~85 的高鑑別度分數！
supplier_stats["Savings_Score"] = ((clipped_savings - s_min) / (s_max - s_min + 1e-5)) * 100

# 其他基礎評分轉換
supplier_stats["Delivery_Score"] = supplier_stats["On Time Delivery"] * 100
supplier_stats["Risk_Score"] = ((supplier_stats["Supplier Risk_Encoded"].max() - supplier_stats["Supplier Risk_Encoded"]) / (supplier_stats["Supplier Risk_Encoded"].max() + 1e-5)) * 100  
supplier_stats["ESG_Score"] = supplier_stats["Supplier ESG Score"]

# 🚀 複合推薦得分核心計算公式
supplier_stats["Recommendation_Score"] = (
    supplier_stats["Savings_Score"] * 0.30 +
    supplier_stats["Delivery_Score"] * 0.25 +
    supplier_stats["Risk_Score"] * 0.15 +
    supplier_stats["ESG_Score"] * 0.15 +
    supplier_stats["Preferred Supplier_Encoded"] * 15 -
    supplier_stats["Single Source Flag"] * 5
)

# 篩選前三名做為極致推薦名單
top_suppliers = supplier_stats.sort_values(by="Recommendation_Score", ascending=False).head(3)

print("\n⭐ [儀表板決策輔助：頂級供應商智慧推薦名單 (精準安全版)] ⭐")
for index, (idx, row) in enumerate(top_suppliers.iterrows(), 1):
    risk_text = "Low" if row['Supplier Risk_Encoded'] == 0 else ("Medium" if row['Supplier Risk_Encoded'] == 1 else "High")
    print(f"🏆 推薦排名 第 {index} 名：{row['Supplier Name']} ({row['Supplier ID']}) | 綜合決策得分: {row['Recommendation_Score']:.2f}")
    print(f"   👉 推薦理由：平均採購節省率高達 {row['Savings Pct']:.2f}%, 歷史準時交付率 {row['On Time Delivery']*100:.2f}%, ESG 指標評分 {row['Supplier ESG Score']:.1f}, 最大潛在風險: {risk_text}")

# =====================================================================
# STEP 5. 打包導出推薦底表做為前端對接
# =====================================================================
print("\n📦 開始導出推薦決策底表...")
os.makedirs("models", exist_ok=True)

# 儲存全量供應商的聚合得分，供前端 Dashboard 隨時無縫撈取排行
supplier_stats.to_csv("models/it_supplier_recommendations.csv", index=False)

print("💾 [模組二] 安全優化版供應商推薦底表已成功儲存至 models/it_supplier_recommendations.csv！")
print("=====================================================================")