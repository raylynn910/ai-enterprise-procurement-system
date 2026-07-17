from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np

app = FastAPI(title="IT 供應商推薦引擎 API")

# 允許跨網域請求 (CORS)，這樣前端 HTML 才能順利呼叫此 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生產環境建議指定具體網址，開發階段用 "*" 即可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 啟動時：讀取並清洗資料 (與您的 Step 1 & 2 完全相同)
# ==========================================
file_path = "Data_中英欄位.csv"
if not os.path.exists(file_path):
    file_path = "Data_中英欄位_After_EDA.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("❌ 找不到資料來源檔案！")

df = pd.read_csv(file_path)
df.columns = [col.split("（")[0].strip() for col in df.columns]

# 逐筆防禦函數
def force_numeric_clean_robust(series):
    s_str = series.astype(str).str.strip().str.lower()
    s_str = s_str.replace({"yes": "1", "no": "0", "true": "1", "false": "0", "unknown": "0"})
    s_str = s_str.str.replace("%", "", regex=False)
    s_num = pd.to_numeric(s_str, errors='coerce')
    col_median = s_num.median()
    if pd.isna(col_median) or col_median > 1.0:
        col_median = 0.8
    def fix_row_value(val, fallback):
        if pd.isna(val): return fallback
        if val > 100.0 or val < 0.0: return fallback
        if val > 1.0: return val / 100.0
        return val
    return s_num.apply(lambda x: fix_row_value(x, col_median))

# 清洗關鍵欄位
if "On Time Delivery" in df.columns:
    df["On Time Delivery"] = force_numeric_clean_robust(df["On Time Delivery"])
if "Single Source Flag" in df.columns:
    df["Single Source Flag"] = force_numeric_clean_robust(df["Single Source Flag"])

if "Supplier Risk" in df.columns:
    risk_mapping = {"Low": 0, "Medium": 1, "High": 2, "Unknown": 1}
    df["Supplier Risk_Encoded"] = df["Supplier Risk"].map(risk_mapping).fillna(1)
else:
    df["Supplier Risk_Encoded"] = 1

if "Preferred Supplier" in df.columns:
    df["Preferred Supplier_Encoded"] = df["Preferred Supplier"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0)
else:
    df["Preferred Supplier_Encoded"] = 0

if "Supplier ESG Score" in df.columns:
    df["Supplier ESG Score"] = df["Supplier ESG Score"].fillna(df["Supplier ESG Score"].median())
else:
    df["Supplier ESG Score"] = 75.0

# 全量數據排序
if "PO Quarter" in df.columns:
    df["PO Quarter_Cleaned"] = df["PO Quarter"].astype(str)
    q_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    df["Q_Weight"] = df["PO Quarter_Cleaned"].map(q_map).fillna(1)
    df = df.sort_values(by=["PO Year", "Q_Weight"]).reset_index(drop=True)
    df = df.drop(columns=["Q_Weight"])

full_analysis_df = df.copy()

# ==========================================
# API 路由：動態計算推薦排行
# ==========================================
@app.get("/api/recommendations")
def get_recommendations(
    scenario: str = Query("default", description="預設情境: default, cost, urgent, risk"),
    w_savings: float = Query(0.30, description="節省率權重"),
    w_delivery: float = Query(0.25, description="準交率權重"),
    w_risk: float = Query(0.15, description="風險權重"),
    w_esg: float = Query(0.15, description="ESG 權重")
):
    # 1. 根據前端選擇的情境 (Scenario) 自動覆寫權重
    if scenario == "cost":
        w_savings, w_delivery, w_risk, w_esg = 0.50, 0.15, 0.15, 0.10
    elif scenario == "urgent":
        w_savings, w_delivery, w_risk, w_esg = 0.15, 0.50, 0.15, 0.10
    elif scenario == "risk":
        w_savings, w_delivery, w_risk, w_esg = 0.15, 0.15, 0.50, 0.10

    # 2. 篩選與聚合 (與您的 Step 3 相同)
    it_data = full_analysis_df[full_analysis_df["Category"] == "IT Software"].copy()
    if len(it_data) == 0:
        it_data = full_analysis_df.copy()

    supplier_stats = it_data.groupby("Supplier ID").agg({
        "Supplier Name": "first",
        "Savings Pct": "mean",
        "On Time Delivery": "mean",
        "Supplier Risk_Encoded": "max",
        "Supplier ESG Score": "mean",
        "Preferred Supplier_Encoded": "last",
        "Single Source Flag": "last"
    }).reset_index()

    # 3. 業務截斷與評分轉換 (與您的 Step 4 相同)
    clipped_savings = supplier_stats["Savings Pct"].clip(lower=-10.0, upper=20.0)
    s_min, s_max = clipped_savings.min(), clipped_savings.max()
    supplier_stats["Savings_Score"] = ((clipped_savings - s_min) / (s_max - s_min + 1e-5)) * 100
    
    supplier_stats["Delivery_Score"] = supplier_stats["On Time Delivery"] * 100
    supplier_stats["Risk_Score"] = ((supplier_stats["Supplier Risk_Encoded"].max() - supplier_stats["Supplier Risk_Encoded"]) / (supplier_stats["Supplier Risk_Encoded"].max() + 1e-5)) * 100  
    supplier_stats["ESG_Score"] = supplier_stats["Supplier ESG Score"]

    # 4. 套用動態權重計算推薦得分
    supplier_stats["Recommendation_Score"] = (
        supplier_stats["Savings_Score"] * w_savings +
        supplier_stats["Delivery_Score"] * w_delivery +
        supplier_stats["Risk_Score"] * w_risk +
        supplier_stats["ESG_Score"] * w_esg +
        supplier_stats["Preferred Supplier_Encoded"] * 15 -
        supplier_stats["Single Source Flag"] * 5
    )

    # 5. 排序並取前 10 名（或是全部），交給前端渲染
    sorted_stats = supplier_stats.sort_values(by="Recommendation_Score", ascending=False)
    
    # 將 NaN/Inf 轉為相容 JSON 的格式
    sorted_stats = sorted_stats.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return sorted_stats.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)