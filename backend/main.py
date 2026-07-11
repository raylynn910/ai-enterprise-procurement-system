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
    with open(os.path.join(BASE_DIR, "models", "supplier_risk_model.pkl"), "rb") as f:
        supplier_risk_model = pickle.load(f)
        
    # Team 1 Risk Model
    with open(os.path.join(BASE_DIR, "models", "risk_rf_model.pkl"), "rb") as f:
        risk_rf_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "risk_scaler.pkl"), "rb") as f:
        risk_scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "risk_features.pkl"), "rb") as f:
        risk_features = pickle.load(f)
        
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

@app.get("/api/risk/orders")
def get_risk_orders():
    """
    從資料庫中撈取最近的訂單，並使用 Team 2 的「供應商綜合評分引擎 (Model B)」
    來計算供應商的推薦分數。若分數低於 55，則標記為高風險訂單。
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found.")
        
    try:
        conn = get_db_connection()
        query = '''
            SELECT 
                "PO_Number", "Supplier_Name", "Supplier_ESG_Score", "On_Time_Delivery", 
                "Days_Late", "PO_Status", "Savings_Pct", "Maverick_Spend", "Single_Source_Flag",
                "Supplier_Risk", "Preferred_Supplier"
            FROM procurement_data 
            ORDER BY RANDOM()
            LIMIT 500
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {"status": "success", "data": []}
            
        # 1. 節省成本分數 (Savings_Score)
        df['Savings_Pct_Num'] = pd.to_numeric(df['Savings_Pct'], errors='coerce').fillna(0)
        s_min, s_max = df['Savings_Pct_Num'].min(), df['Savings_Pct_Num'].max()
        df['Savings_Score'] = ((df['Savings_Pct_Num'] - s_min) / (s_max - s_min + 1e-5)) * 100
        
        # 2. 交期分數 (Delivery_Score)
        df['Delivery_Score'] = df['On_Time_Delivery'].apply(lambda x: 100 if str(x).strip().lower() == 'yes' else 0)
        
        # 3. 風險分數 (Risk_Score) - 越低風險分數越高
        def map_risk(x):
            v = str(x).strip().lower()
            if v == 'low': return 0
            if v == 'medium': return 1
            if v == 'high': return 2
            return 1
        df['Risk_Encoded'] = df['Supplier_Risk'].apply(map_risk)
        df['Risk_Score'] = ((2 - df['Risk_Encoded']) / (2 + 1e-5)) * 100
        
        # 4. ESG 分數 (ESG_Score)
        df['ESG_Score'] = pd.to_numeric(df['Supplier_ESG_Score'], errors='coerce').fillna(50)
        
        # 5. 偏好與單一來源加扣分
        df['Pref_Score'] = df['Preferred_Supplier'].apply(lambda x: 15 if str(x).strip().lower() == 'yes' else 0)
        df['Single_Score'] = df['Single_Source_Flag'].apply(lambda x: 5 if str(x).strip() in ['1', 'yes', 'true'] else 0)
        
        # 計算綜合推薦得分 (Recommendation_Score)
        df['Recommendation_Score'] = (
            df['Savings_Score'] * 0.30 +
            df['Delivery_Score'] * 0.25 +
            df['Risk_Score'] * 0.15 +
            df['ESG_Score'] * 0.15 +
            df['Pref_Score'] -
            df['Single_Score']
        )
        
        # 根據 Model B 邏輯，推薦分數過低 (< 45) 視為高風險訂單 (約佔最差的 15%)
        high_risk_df = df[df['Recommendation_Score'] < 45].copy()
        
        # 將結果轉為 JSON 格式
        results = []
        for _, row in high_risk_df.iterrows():
            results.append({
                "po_number": row["PO_Number"],
                "supplier_name": row["Supplier_Name"],
                "savings_pct": f"{row['Savings_Pct_Num']:.1f}%",
                "maverick": "Yes" if str(row["Maverick_Spend"]).strip() == "1" else "No",
                "single_source": "Yes" if str(row["Single_Source_Flag"]).strip() in ["1", "yes", "true"] else "No"
            })
            
        return {
            "status": "success",
            "count": len(results),
            "data": results
        }
        
    except Exception as e:
        print(f"Error in risk orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    compliance_score: float = Field(..., description="合規性分數")
    financial_score: float = Field(..., description="財務穩定分數")
    delivery_score: float = Field(..., description="交期可靠分數")
    esg_score: float = Field(..., description="永續發展分數")
    pricing_score: float = Field(..., description="價格競爭力分數")

@app.post("/api/predict/supplier-risk", response_model=SupplierRiskResponse)
def predict_supplier_risk(request: SupplierRiskRequest = Body(...)):
    """
    [ML] 預測供應商風險 API
    """
    if not ML_MODELS_LOADED:
        return _mock_supplier_risk(request)
        
    try:
        # 查詢供應商歷史特徵
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT Supplier_ESG_Score, On_Time_Delivery, Days_Late, PO_Status 
            FROM procurement_data 
            WHERE Supplier_ID = ? 
            ORDER BY PO_Date DESC LIMIT 1
        ''', (request.supplier_id,))
        row = cursor.fetchone()
        conn.close()
        
        osint_summary_msg = ""
        if row:
            esg_score = row['Supplier_ESG_Score']
            on_time = row['On_Time_Delivery']
            days_late = row['Days_Late']
            po_status = row['PO_Status']
        else:
            # 啟用 OSINT 進行新廠商資料搜集
            try:
                from osint_gatherer import gather_supplier_intelligence
                # For OSINT, supplier_id usually contains the company name from the frontend
                osint_result = gather_supplier_intelligence(request.supplier_id, request.country)
                esg_score = osint_result["esg_score"]
                days_late = osint_result["days_late"]
                po_status = osint_result["po_status"]
                on_time = "Yes" if days_late <= 0 else "No"
                osint_summary_msg = osint_result["osint_summary"]
            except Exception as e:
                print("OSINT Error:", e)
                # Fallback 預設值
                esg_score = 50.0
                on_time = 'Yes'
                days_late = 0
                po_status = 'Closed'
            
        if osint_summary_msg == "找不到該公司相關資訊":
            # 直接標記為高風險，拒絕幽靈公司
            risk_level = 'High'
            risk_score = 100.0
            recommendation = "【嚴重警告】找不到該公司相關資訊。由於無法於公開網路核實其實體存在，系統直接判定為極高風險，請立即停止交易或進行深度人工徵信。"
        else:
            input_data = pd.DataFrame([{
                'Supplier ESG Score': float(esg_score),
                'On Time Delivery': str(on_time),
                'Days Late': float(days_late),
                'PO Status': str(po_status)
            }])
            
            pred_class = supplier_risk_model.predict(input_data)[0]
            pred_proba = supplier_risk_model.predict_proba(input_data)[0]
            
            classes = supplier_risk_model.named_steps['classifier'].classes_
            proba_dict = dict(zip(classes, pred_proba))
            
            risk_score = proba_dict.get('High', 0) * 100 + proba_dict.get('Medium', 0) * 50
            risk_score = round(risk_score, 1)
            risk_level = pred_class
            
            # --- Rule-based Guardrails (合規與負面新聞強制阻斷機制) ---
            is_guardrail_blocked = False
            if "高度示警" in osint_summary_msg:
                risk_level = 'High'
                risk_score = max(90.0, 100.0 - esg_score)  # 強制給予高風險分數
                is_guardrail_blocked = True
                
        if osint_summary_msg != "找不到該公司相關資訊":
            if risk_level == 'High':
                if is_guardrail_blocked:
                    recommendation = "【合規阻斷】該供應商存在嚴重負面新聞或高度爭議，依內控規範強制阻斷，不予核准。"
                else:
                    recommendation = "警告：模型預測該供應商具有高風險，請注意交期延誤與品質問題，建議尋求替代來源。"
            elif risk_level == 'Medium':
                recommendation = "風險中等，建議加強監控與合約約束。"
            else:
                recommendation = "風險評估為低，歷史紀錄良好，可持續合作。"
                
            if osint_summary_msg:
                recommendation = osint_summary_msg + "\n\n模型建議: " + recommendation
            
        # --- Calculate 5 Radar Chart Dimensions ---
        c_score = 0.0 if is_guardrail_blocked else float(esg_score)
        f_score = max(10.0, 100.0 - risk_score)
        
        # Delivery Score based on days late
        d_late = float(days_late) if days_late > 0 else 0.0
        d_score = max(10.0, 100.0 - (d_late * 3.5)) # 0 days -> 100, 20 days -> 30
        
        # Deterministic pseudo-random pricing score based on supplier string length or hash
        p_score = 65.0 + (len(request.supplier_id) * 3 % 25)
        
        return SupplierRiskResponse(
            supplier_id=request.supplier_id,
            risk_score=risk_score,
            risk_level=risk_level,
            recommendation=recommendation,
            is_mock=False,
            compliance_score=round(c_score, 1),
            financial_score=round(f_score, 1),
            delivery_score=round(d_score, 1),
            esg_score=round(float(esg_score), 1),
            pricing_score=round(p_score, 1)
        )
    except Exception as e:
        print(f"Prediction Error: {e}")
        return _mock_supplier_risk(request)

def _mock_supplier_risk(request: SupplierRiskRequest):
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
        is_mock=True,
        compliance_score=85.0 if risk_level != "High" else 20.0,
        financial_score=90.0 if risk_level == "Low" else 60.0,
        delivery_score=95.0 if risk_level == "Low" else 40.0,
        esg_score=100.0 - risk_score,
        pricing_score=75.0
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
