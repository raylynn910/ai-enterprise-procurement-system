from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import pickle
import pandas as pd

# 資料庫與模型路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'procurement.db')

# 全域載入機器學習模型
try:
    with open(os.path.join(BASE_DIR, "models", "reg_model.pkl"), "rb") as f:
        reg_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "cls_refiner.pkl"), "rb") as f:
        cls_refiner = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "le_dict.pkl"), "rb") as f:
        le_dict = pickle.load(f)
    ML_MODELS_LOADED = True
except Exception as e:
    print(f"Warning: ML models failed to load. Using mock fallback. Error: {e}")
    ML_MODELS_LOADED = False

app = FastAPI(
    title="Smart Procurement API",
    description="API for the AI-driven Enterprise Procurement System",
    version="1.0.0"
)

# 設定 CORS (允許前端跨網域請求)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 在開發階段允許所有來源，上線時應限縮
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 資料庫路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'procurement.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    # 將回傳結果轉為字典形式，方便 FastAPI 轉換為 JSON
    conn.row_factory = sqlite3.Row 
    return conn

@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Procurement API"}

@app.get("/api/procurements")
def get_procurements(
    limit: int = Query(50, description="回傳的資料筆數上限"),
    offset: int = Query(0, description="資料位移量，用於分頁")
):
    """
    取得採購單歷史資料清單 (支援分頁)
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found. Please run import_data.py first.")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 使用參數化查詢以防止 SQL Injection
    cursor.execute(
        "SELECT * FROM procurement_data LIMIT ? OFFSET ?", 
        (limit, offset)
    )
    rows = cursor.fetchall()
    
    # 取得總筆數以利前端計算分頁
    cursor.execute("SELECT COUNT(*) FROM procurement_data")
    total_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "status": "success",
        "data": [dict(row) for row in rows],
        "pagination": {
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
    }

# ----- Mock API 區塊 (供前端開發使用) -----

class SupplierRiskRequest(BaseModel):
    supplier_id: str = Field(..., description="供應商ID", example="SUP-010")
    country: Optional[str] = Field(None, description="供應商所在國家", example="South Korea")
    category: Optional[str] = Field(None, description="採購類別", example="Raw Materials")
    lead_time_days: Optional[int] = Field(None, description="預估交期(天)", example=50)

class SupplierRiskResponse(BaseModel):
    supplier_id: str
    risk_score: float = Field(..., description="風險分數 0-100 (越高越危險)")
    risk_level: str = Field(..., description="風險等級: Low, Medium, High")
    recommendation: str = Field(..., description="AI給予的建議")
    is_mock: bool = Field(True, description="標示此為Mock資料")

@app.post("/api/predict/supplier-risk", response_model=SupplierRiskResponse)
def predict_supplier_risk(request: SupplierRiskRequest = Body(...)):
    """
    [Mock] 預測供應商風險 API
    
    目前模型尚在訓練中，此 API 暫時回傳隨機或固定的 Mock 資料，供前端開發使用。
    """
    # 簡單的 Mock 邏輯：根據 Supplier ID 決定風險高低，或直接給予固定數值
    risk_score = 15.5
    risk_level = "Low"
    recommendation = "此供應商歷史交期穩定，風險極低，建議可簽訂長期合約。"
    
    if "999" in request.supplier_id:
        risk_score = 88.0
        risk_level = "High"
        recommendation = "警告：此地區近期有供應鏈中斷風險，建議尋找備援供應商 (Single Source Flag = Yes 需特別注意)。"
    elif request.lead_time_days and request.lead_time_days > 60:
        risk_score = 65.0
        risk_level = "Medium"
        recommendation = "交期較長，建議提前下單並監控實際到貨日。"

    return SupplierRiskResponse(
        supplier_id=request.supplier_id,
        risk_score=risk_score,
        risk_level=risk_level,
        recommendation=recommendation,
        is_mock=True
    )

@app.get("/api/trends/monthly")
def get_monthly_trends():
    """
    取得歷史採購單的月份趨勢統計 (包含訂單數、總數量等)
    用以串接前端的整體預算消耗或歷史趨勢圖表。
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found.")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 我們改為計算兩個在採購實務上非常有價值的趨勢指標：
    # 1. avg_savings_pct (平均節省率): 追蹤採購議價績效。
    # 2. on_time_delivery_rate (準交率): 追蹤供應商整體交貨品質。
    query = """
        SELECT 
            PO_Year, 
            PO_Month, 
            AVG(CAST(Savings_Pct AS FLOAT)) as avg_savings_pct,
            SUM(CASE WHEN On_Time_Delivery = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as on_time_delivery_rate
        FROM procurement_data 
        GROUP BY PO_Year, PO_Month
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # 將月份轉為數字以利排序
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, 
        "May": 5, "June": 6, "July": 7, "August": 8, 
        "September": 9, "October": 10, "November": 11, "December": 12,
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, 
        "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    
    data = []
    for row in rows:
        r = dict(row)
        month_name = r["PO_Month"]
        r["month_num"] = month_map.get(month_name, 99)
        data.append(r)
        
    # 按年份與月份排序
    data = sorted(data, key=lambda x: (int(x["PO_Year"]), x["month_num"]))
    
    conn.close()
    
    return {
        "status": "success",
        "data": data
    }

# ----- UX 優化與動態補水 API 區塊 -----

@app.get("/api/form-options")
def get_form_options():
    """提供前端下拉選單所需的動態資料 (類別與供應商)"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Category FROM procurement_data WHERE Category IS NOT NULL")
    categories = [r["Category"] for r in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT Supplier_ID, Supplier_Name FROM procurement_data WHERE Supplier_ID IS NOT NULL")
    suppliers = [{"id": r["Supplier_ID"], "name": r["Supplier_Name"]} for r in cursor.fetchall()]
    conn.close()
    
    return {"categories": categories, "suppliers": suppliers}

@app.get("/api/context/category")
def get_category_context(name: str):
    """取得特定採購類別的歷史均價，供前端做錨定參考"""
    # 這裡暫時用長度做一個決定性的 Mock 均價，因為實際 CSV 缺乏直接的 Unit Price 欄位可聚合
    avg_price = 100.0 + (len(name) * 15.5) 
    return {
        "category": name, 
        "historical_avg_price": round(avg_price, 2)
    }

@app.get("/api/context/supplier")
def get_supplier_context(id: str):
    """取得供應商的戰力卡片資料 (風險、ESG等)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Supplier_Name, Supplier_Risk, Supplier_Status, Supplier_Tier, Preferred_Supplier, Supplier_ESG_Score 
        FROM procurement_data 
        WHERE Supplier_ID = ? LIMIT 1
    """, (id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    raise HTTPException(status_code=404, detail="Supplier not found")

class SavingsPredictionRequest(BaseModel):
    category: str
    supplier_id: str
    quantity: int
    budget_price: float

@app.post("/api/predict/savings")
def predict_savings(request: SavingsPredictionRequest = Body(...)):
    """
    整合版預測 API (Data Enrichment + ML Prediction)
    前端只需傳 4 個參數，後端查 DB 補齊 10 個特徵後進行預測。
    """
    # 1. Backend Data Enrichment (自動補水)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Supplier_Risk, Supplier_Status, Preferred_Supplier, Maverick_Spend, Single_Source_Flag, Department
        FROM procurement_data 
        WHERE Supplier_ID = ? LIMIT 1
    """, (request.supplier_id,))
    sup_info = cursor.fetchone()
    conn.close()
    
    if not sup_info:
        raise HTTPException(status_code=404, detail="Supplier not found in DB")
        
    # 2. 準備 Item_Avg_Price (對齊 /api/context/category 的邏輯)
    avg_price_mock = 100.0 + (len(request.category) * 15.5)
    
    # 3. 執行預測
    if ML_MODELS_LOADED:
        try:
            # Label Encoding 轉換 (加入容錯機制)
            def safe_transform(le, val, fallback="Unknown"):
                try:
                    return le.transform([val])[0]
                except ValueError:
                    return le.transform([fallback])[0]

            cat_enc = safe_transform(le_dict["Category"], request.category)
            dept_enc = safe_transform(le_dict["Department"], sup_info["Department"])
            pref_enc = safe_transform(le_dict["Preferred Supplier"], sup_info["Preferred_Supplier"])
            risk_enc = safe_transform(le_dict["Supplier Risk"], sup_info["Supplier_Risk"])
            stat_enc = safe_transform(le_dict["Supplier Status"], sup_info["Supplier_Status"])
            
            mav_val = 1.0 if str(sup_info["Maverick_Spend"]).lower() in ["yes", "1", "true"] else 0.0
            ss_val = 1.0 if str(sup_info["Single_Source_Flag"]).lower() in ["yes", "1", "true"] else 0.0
            
            gap_pct = (request.budget_price - avg_price_mock) / (avg_price_mock + 1e-5)
            
            input_data = pd.DataFrame([{
                "Category_Encoded": cat_enc, "Department_Encoded": dept_enc,
                "Quantity": request.quantity, "Budget Unit Price": request.budget_price,
                "Item_Avg_Price": avg_price_mock, "Budget_vs_Avg_Gap": gap_pct,
                "Preferred Supplier_Encoded": pref_enc, "Supplier Risk_Encoded": risk_enc,
                "Supplier Status_Encoded": stat_enc, "Maverick Spend": mav_val,
                "Single Source Flag": ss_val
            }])
            
            # 第一階迴歸
            pred_savings = float(reg_model.predict(input_data)[0])
            # 第二階分類
            input_data["Pred_Value"] = pred_savings
            pred_class_idx = int(cls_refiner.predict(input_data)[0])
            
            class_mapping = {0: "🟢 預期可節省成本", 1: "🟡 接近預算範圍", 2: "🔴 潛在超出預算風險"}
            
            return {
                "status": "success",
                "pred_savings_pct": round(pred_savings, 2),
                "pred_class_code": pred_class_idx,
                "status_text": class_mapping.get(pred_class_idx, "未知"),
                "enriched_features": {
                    "supplier_risk": sup_info["Supplier_Risk"],
                    "maverick_spend": sup_info["Maverick_Spend"],
                    "preferred_supplier": sup_info["Preferred_Supplier"]
                },
                "is_mock": False
            }
        except Exception as e:
            print("ML Prediction Error:", e)
            # 出錯時回退到 mock

    # --- 以下為 Mock Fallback 邏輯 (模型載入失敗或預測出錯時使用) ---
    mock_savings_pct = 5.0
    if sup_info["Supplier_Risk"] == "High":
        mock_savings_pct = -3.5
    elif request.quantity > 100:
        mock_savings_pct = 12.0
        
    pred_class_code = 0 if mock_savings_pct > 2 else (1 if mock_savings_pct > -2 else 2)
    class_mapping = {0: "🟢 預期可節省成本", 1: "🟡 接近預算範圍", 2: "🔴 潛在超出預算風險"}
    
    return {
        "status": "success",
        "pred_savings_pct": mock_savings_pct,
        "pred_class_code": pred_class_code,
        "status_text": class_mapping[pred_class_code],
        "enriched_features": {
            "supplier_risk": sup_info["Supplier_Risk"],
            "maverick_spend": sup_info["Maverick_Spend"],
            "preferred_supplier": sup_info["Preferred_Supplier"]
        },
        "is_mock": True,
        "message": "Used fallback mock logic."
    }
