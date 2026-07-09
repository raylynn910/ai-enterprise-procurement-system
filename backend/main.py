from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os

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
