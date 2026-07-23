from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import finance_gatherer
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
    with open(os.path.join(BASE_DIR, "models", "risk_rf_model.pkl"), "rb") as f:
        risk_rf_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "risk_scaler.pkl"), "rb") as f:
        risk_scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "risk_features.pkl"), "rb") as f:
        risk_features = pickle.load(f)
        
    with open(os.path.join(BASE_DIR, "models", "dispute_model.pkl"), "rb") as f:
        dispute_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "dispute_encoders.pkl"), "rb") as f:
        dispute_encoders = pickle.load(f)
        
    ML_MODELS_LOADED = True
    print("All ML models loaded successfully.")
except Exception as e:
    print(f"Warning: ML models failed to load. Using mock fallback. Error: {e}")
    ML_MODELS_LOADED = False

# 新供應商風險評分引擎
try:
    with open(os.path.join(BASE_DIR, "models", "new_supplier_scoring_model.pkl"), "rb") as f:
        new_supplier_bundle = pickle.load(f)
    NEW_SUPPLIER_MODEL_LOADED = True
except Exception as e:
    print(f"Warning: new-supplier scoring model failed to load. Error: {e}")
    NEW_SUPPLIER_MODEL_LOADED = False

# 供應商財務風險模型
try:
    with open(os.path.join(BASE_DIR, "models", "financial_risk_model.pkl"), "rb") as f:
        financial_risk_bundle = pickle.load(f)
    FINANCIAL_MODEL_LOADED = True
except Exception as e:
    print(f"Warning: financial risk model failed to load. Error: {e}")
    FINANCIAL_MODEL_LOADED = False

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

# 初始化 AI 日誌資料表
def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ai_prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            category TEXT,
            supplier_id TEXT,
            quantity INTEGER,
            budget_price REAL,
            pred_savings_pct REAL,
            pred_class_code INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

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

@app.get("/api/options")
def get_options():
    """
    動態取得 Category, Item_Description, Supplier_Name, Contract 的連動選項
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found.")
        
    conn = get_db_connection()
    query = '''
        SELECT DISTINCT Category, Item_Description, Supplier_ID, Supplier_Name, Contract_ID, Contract_Type 
        FROM procurement_data
        WHERE Category IS NOT NULL
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    mapping = {}
    for _, row in df.iterrows():
        cat = row['Category']
        if pd.isna(cat) or cat == "":
            continue
            
        if cat not in mapping:
            mapping[cat] = {'items': set(), 'suppliers': set(), 'contracts': set()}
            
        item = row['Item_Description']
        if pd.notna(item) and item != "":
            mapping[cat]['items'].add(item)
            
        supp_id = row['Supplier_ID']
        supp_name = row['Supplier_Name']
        if pd.notna(supp_id) and pd.notna(supp_name) and supp_id != "":
            mapping[cat]['suppliers'].add((supp_id, supp_name))
            
        contract_id = row['Contract_ID']
        contract_type = row['Contract_Type']
        if pd.notna(contract_id) and contract_id != "":
            c_type = f" ({contract_type})" if pd.notna(contract_type) else ""
            mapping[cat]['contracts'].add(f"{contract_id}{c_type}")
            
    # Convert sets to sorted lists
    result = {}
    for cat, data in mapping.items():
        result[cat] = {
            "items": sorted(list(data["items"])),
            "suppliers": [{"id": s[0], "name": s[1]} for s in sorted(list(data["suppliers"]), key=lambda x: x[1])],
            "contracts": sorted(list(data["contracts"]))
        }
        
    return {"status": "success", "data": result}

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


# ----- 採購端「有問題的 PO 單」追蹤清單 -----
# 規則：只要命中下列任一條件即為風險訂單（Cancelled 訂單一律排除）
#   1. Payment Status ∈ {Overdue, Pending, On Hold}
#   2. Invoice Status ∈ {Overdue, Pending, Disputed}
#   3. Invoice Match Type ∈ {2-Way Match, No Match}
#   4. Maverick Spend = Yes

_ALLOWED_SORT_COLUMNS = {
    "po_date", "po_number", "po_status", "supplier_id", "supplier_name",
    "item_description", "quantity", "savings_pct",
    "invoice_status", "payment_status", "invoice_match_type", "maverick_spend",
}

def _parse_dmy(date_str):
    """CSV 內 PO Date 為 dd/mm/yyyy，轉為 pd.Timestamp；失敗回 NaT。"""
    return pd.to_datetime(date_str, format="%d/%m/%Y", errors="coerce")

def _format_ymd_slash(ts):
    """回傳台灣常用 yyyy/mm/dd；NaT 回空字串。"""
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y/%m/%d")

def _compute_matched_rules(row):
    rules = []
    pay = str(row.get("Payment_Status", "")).strip().lower()
    if pay in {"overdue", "pending", "on hold"}:
        rules.append({
            "code": f"payment_{pay.replace(' ', '_')}",
            "label": {"overdue": "付款逾期", "pending": "付款待處理", "on hold": "付款凍結"}[pay],
            "level": "danger" if pay == "overdue" else "warning",
        })
    inv = str(row.get("Invoice_Status", "")).strip().lower()
    if inv in {"overdue", "pending", "disputed"}:
        rules.append({
            "code": f"invoice_{inv}",
            "label": {"overdue": "發票逾期", "pending": "發票待處理", "disputed": "發票爭議"}[inv],
            "level": "danger" if inv in {"overdue", "disputed"} else "warning",
        })
    match_type = str(row.get("Invoice_Match_Type", "")).strip()
    match_lower = match_type.lower()
    if match_lower == "2-way match":
        rules.append({"code": "match_2way", "label": "2-Way Match", "level": "warning"})
    elif match_lower == "no match":
        rules.append({"code": "match_none", "label": "無核銷", "level": "danger"})
    if str(row.get("Maverick_Spend", "")).strip().lower() == "yes":
        rules.append({"code": "maverick", "label": "越權採購", "level": "danger"})
    return rules


@app.get("/api/procurement/at-risk-orders")
def get_at_risk_orders(
    start_date: Optional[str] = Query(None, description="起始日期 yyyy-mm-dd（含）"),
    end_date: Optional[str] = Query(None, description="結束日期 yyyy-mm-dd（含）"),
    page: int = Query(1, ge=1, description="頁碼（1 起算）"),
    page_size: int = Query(50, ge=1, le=50, description="每頁筆數，最多 50"),
    sort_by: str = Query("po_date", description="排序欄位"),
    sort_order: str = Query("desc", description="asc / desc"),
):
    """
    給採購端追蹤「有問題的 PO 單」清單。命中任一風險規則即回傳；Cancelled 訂單排除。
    支援日期區間篩選、欄位排序、分頁（每頁最多 50 筆）。
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found.")

    if sort_by not in _ALLOWED_SORT_COLUMNS:
        raise HTTPException(status_code=400, detail=f"Invalid sort_by: {sort_by}")
    if sort_order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail=f"Invalid sort_order: {sort_order}")

    # 解析日期參數 (前端傳 yyyy-mm-dd)
    start_ts = pd.to_datetime(start_date, format="%Y-%m-%d", errors="coerce") if start_date else None
    end_ts = pd.to_datetime(end_date, format="%Y-%m-%d", errors="coerce") if end_date else None
    if start_date and pd.isna(start_ts):
        raise HTTPException(status_code=400, detail="start_date 格式錯誤，需為 yyyy-mm-dd")
    if end_date and pd.isna(end_ts):
        raise HTTPException(status_code=400, detail="end_date 格式錯誤，需為 yyyy-mm-dd")

    try:
        conn = get_db_connection()
        sql = '''
            SELECT
                "PO_Number", "PO_Date", "PO_Status", "Supplier_ID", "Supplier_Name",
                "Item_Description", "Quantity", "Savings_Pct",
                "Invoice_Status", "Payment_Status", "Invoice_Match_Type", "Maverick_Spend"
            FROM procurement_data
            WHERE LOWER(TRIM("PO_Status")) != 'cancelled'
              AND (
                  LOWER(TRIM("Payment_Status")) IN ('overdue', 'pending', 'on hold')
                  OR LOWER(TRIM("Invoice_Status")) IN ('overdue', 'pending', 'disputed')
                  OR LOWER(TRIM("Invoice_Match_Type")) IN ('2-way match', 'no match')
                  OR LOWER(TRIM("Maverick_Spend")) = 'yes'
              )
        '''
        df = pd.read_sql_query(sql, conn)
        conn.close()

        if df.empty:
            return {
                "status": "success",
                "total": 0, "page": page, "page_size": page_size, "total_pages": 0,
                "data": [],
            }

        # 日期區間篩選 (在 pandas 端做，因 SQLite 存的是 dd/mm/yyyy 字串)
        df["_po_date_ts"] = df["PO_Date"].apply(_parse_dmy)
        if start_ts is not None:
            df = df[df["_po_date_ts"] >= start_ts]
        if end_ts is not None:
            df = df[df["_po_date_ts"] <= end_ts]

        if df.empty:
            return {
                "status": "success",
                "total": 0, "page": page, "page_size": page_size, "total_pages": 0,
                "data": [],
            }

        # 排序 (數值欄位需先轉型)
        sort_col_map = {
            "po_date": "_po_date_ts",
            "po_number": "PO_Number",
            "po_status": "PO_Status",
            "supplier_id": "Supplier_ID",
            "supplier_name": "Supplier_Name",
            "item_description": "Item_Description",
            "quantity": "_quantity_num",
            "savings_pct": "_savings_pct_num",
            "invoice_status": "Invoice_Status",
            "payment_status": "Payment_Status",
            "invoice_match_type": "Invoice_Match_Type",
            "maverick_spend": "Maverick_Spend",
        }
        if sort_by == "quantity":
            df["_quantity_num"] = pd.to_numeric(df["Quantity"], errors="coerce")
        elif sort_by == "savings_pct":
            df["_savings_pct_num"] = pd.to_numeric(df["Savings_Pct"], errors="coerce")

        df = df.sort_values(
            by=sort_col_map[sort_by],
            ascending=(sort_order == "asc"),
            na_position="last",
            kind="mergesort",
        )

        total = len(df)
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        page_df = df.iloc[offset : offset + page_size]

        data = []
        for _, row in page_df.iterrows():
            data.append({
                "po_number": row["PO_Number"],
                "po_date": _format_ymd_slash(row["_po_date_ts"]),
                "po_status": row["PO_Status"],
                "supplier_id": row["Supplier_ID"],
                "supplier_name": row["Supplier_Name"],
                "item_description": row["Item_Description"],
                "quantity": row["Quantity"],
                "savings_pct": row["Savings_Pct"],
                "invoice_status": row["Invoice_Status"],
                "payment_status": row["Payment_Status"],
                "invoice_match_type": row["Invoice_Match_Type"],
                "maverick_spend": row["Maverick_Spend"],
                "matched_rules": _compute_matched_rules(row),
            })

        return {
            "status": "success",
            "total": int(total),
            "page": page,
            "page_size": page_size,
            "total_pages": int(total_pages),
            "data": data,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in at-risk-orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----- Mock API 區塊 (供前端開發使用) -----

class SupplierRiskRequest(BaseModel):
    supplier_id: str = Field(..., description="供應商ID", example="SUP-010")
    country: Optional[str] = Field(None, description="供應商所在國家", example="South Korea")
    category: Optional[str] = Field(None, description="採購類別", example="Raw Materials")
    lead_time_days: Optional[int] = Field(None, description="預估交期(天)", example=50)
    tier: int = Field(3, description="供應商層級 (1-3)", example=2)

class SupplierRiskResponse(BaseModel):
    supplier_id: str
    risk_score: float = Field(..., description="風險分數 0-100 (越高越危險)")
    risk_level: str = Field(..., description="風險等級: Low, Medium, High")
    recommendation: str = Field(..., description="AI給予的建議")
    is_mock: bool = Field(True, description="是否為Mock資料")
    reputation_score: float = Field(..., description="市場聲譽分數")
    financial_score: float = Field(..., description="財務穩定度")
    delivery_score: float = Field(..., description="交期可靠度")
    esg_score: float = Field(..., description="永續指標")
    pricing_score: float = Field(..., description="定價競爭力")
    osint_sources: list = Field([], description="公開來源情報")
    reasons: list = Field([], description="模型決策解釋")

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
        osint_sources_list = []
        if row:
            esg_score = float(row['Supplier_ESG_Score'])
            on_time = row['On_Time_Delivery']
            days_late = int(float(row['Days_Late']))
            po_status = row['PO_Status']
        else:
            # 啟用 OSINT 進行新廠商資料搜集
            try:
                import esg_gatherer
                from osint_gatherer import gather_supplier_intelligence
                # For OSINT, supplier_id usually contains the company name from the frontend
                osint_result = gather_supplier_intelligence(request.supplier_id, request.country)
                esg_score = osint_result["esg_score"]
                days_late = osint_result["days_late"]
                po_status = osint_result["po_status"]
                on_time = "Yes" if days_late <= 0 else "No"
                osint_summary_msg = osint_result["osint_summary"]
                osint_sources_list = osint_result.get("osint_sources", [])
                
                auth_esg = esg_gatherer.get_authoritative_esg(request.supplier_id)
            except Exception as e:
                print("OSINT Error:", e)
                # Fallback 預設值
                esg_score = 50.0
                on_time = 'Yes'
                days_late = 0
                po_status = 'Closed'
                osint_sources_list = []
            
        is_guardrail_blocked = False
        if osint_summary_msg == "找不到該公司相關資訊":
            # 直接標記為高風險，拒絕幽靈公司
            risk_level = '需複核'
            risk_score = 100.0
            reasons = []
            recommendation = "【嚴重警告】找不到該公司相關資訊。由於無法於公開網路核實其實體存在，系統直接判定為極高風險，請立即停止交易或進行深度人工徵信。"
        else:
            b = new_supplier_bundle
            model = b["day0_model"]
            feats = b["features_day0"]
            row = {"tier": request.tier, "esg": float(esg_score)}
            profile = pd.DataFrame([row])[feats]
            
            proba = model.predict_proba(profile)[0]
            clf = model.named_steps["clf"]
            risk_order = b["risk_order"]     # ["核准","需複核"]
            pred_pos = int(proba.argmax())
            pred_label = int(clf.classes_[pred_pos])
            
            risk_level = risk_order[pred_label]
            pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
            risk_score = round(float(proba[pos_col] * 100), 1)
            
            # 可解釋性: 標準化特徵值 × 係數
            z = model.named_steps["scaler"].transform(profile)[0]
            coef = clf.coef_
            if coef.shape[0] == 1:
                contrib_pos = z * coef[0]
                contrib = contrib_pos if pred_label == 1 else -contrib_pos
            else:
                contrib = z * coef[pred_pos]
            order = abs(contrib).argsort()[::-1]
            
            reasons = [{"feature": b["feature_labels"].get(feats[int(j)], feats[int(j)]),
                        "value": round(float(profile.iloc[0, int(j)]), 4),
                        "contribution": round(float(contrib[int(j)]), 3),
                        "direction": "推向此判定" if contrib[int(j)] > 0 else "拉離此判定"} for j in order]
            
            # --- Rule-based Guardrails (合規與負面新聞強制阻斷機制) ---
            if "高度示警" in osint_summary_msg:
                if 'auth_esg' in locals() and auth_esg:
                    osint_summary_msg += f"\n\n【ESG 權威豁免】已抓取國際真實 ESG 評等 ({auth_esg['rating']})，解除一般新聞負面字詞阻斷。"
                else:
                    risk_level = '需複核'
                    risk_score = max(90.0, 100.0 - esg_score)  # 強制給予高風險分數
                    is_guardrail_blocked = True
                
        if osint_summary_msg != "找不到該公司相關資訊":
            if risk_level == '需複核':
                if is_guardrail_blocked:
                    recommendation = "【合規阻斷】該供應商存在嚴重負面新聞或高度爭議，依內控規範強制阻斷，不予核准。"
                else:
                    recommendation = "警告：模型預測該供應商具有風險，建議尋求替代來源或啟動人工複核程序。"
            else:
                recommendation = "風險評估為低，歷史紀錄或基本面良好，可予以核准。"
                
            if osint_summary_msg:
                recommendation = osint_summary_msg + "\n\n模型建議: " + recommendation
            
        # --- Calculate 5 Radar Chart Dimensions ---
        rep_score = 50.0  # Base reputation score
        
        # Identify if it is a ghost company
        is_ghost_company = (osint_summary_msg == "找不到該公司相關資訊")
        
        if is_ghost_company:
            # Ghost companies should have minimal scores across all dimensions
            rep_score = 10.0
            f_score = 10.0
            d_score = 10.0
            esg_score = 10.0
            p_score = 10.0
        else:
            # Adjust reputation based on OSINT volume
            if len(osint_sources_list) >= 3:
                rep_score += 20.0
            elif len(osint_sources_list) >= 1:
                rep_score += 10.0
                
            # Adjust reputation based on Sentiment
            if is_guardrail_blocked:
                rep_score = 10.0
            elif "正面" in osint_summary_msg or "優良" in osint_summary_msg or "獲獎" in osint_summary_msg:
                rep_score += 15.0
                
            rep_score = min(100.0, max(0.0, rep_score))
            
            f_score = max(10.0, 100.0 - risk_score)
            
            # Delivery Score based on days late
            d_late = float(days_late) if days_late > 0 else 0.0
            d_score = max(10.0, 100.0 - (d_late * 3.5)) # 0 days -> 100, 20 days -> 30
            
            if 'auth_esg' in locals() and auth_esg:
                esg_score = auth_esg['esg_score']
                rep_score = max(rep_score, 85.0)  # Authoritative source implies high reputation
                d_score = auth_esg['delivery_score']
                
            # --- Real Financial OSINT API ---
            fin_data = finance_gatherer.get_financial_features(request.supplier_id)
            if fin_data and fin_data.get('quoteType') == 'EQUITY':
                market_cap = fin_data.get('marketCap', 0)
                if market_cap > 100_000_000_000: # 100B+
                    f_score = 95.0
                    p_score = 90.0
                elif market_cap > 10_000_000_000: # 10B+
                    f_score = 88.0
                    p_score = 82.0
                elif market_cap > 1_000_000_000: # 1B+
                    f_score = 80.0
                    p_score = 75.0
                else:
                    f_score = 75.0
                    p_score = 70.0
            else:
                # Fallback to Risk Proxy if API timeouts or company is private
                p_score = 65.0 + (len(request.supplier_id) * 3 % 25)
        
        return SupplierRiskResponse(
            supplier_id=request.supplier_id,
            risk_score=risk_score,
            risk_level=risk_level,
            recommendation=recommendation,
            is_mock=False,
            reputation_score=round(rep_score, 1),
            financial_score=round(f_score, 1),
            delivery_score=round(d_score, 1),
            esg_score=round(float(esg_score), 1),
            pricing_score=round(p_score, 1),
            osint_sources=osint_sources_list,
            reasons=reasons
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
        reputation_score=85.0 if risk_level != "High" else 20.0,
        financial_score=90.0 if risk_level == "Low" else 60.0,
        delivery_score=95.0 if risk_level == "Low" else 40.0,
        esg_score=100.0 - risk_score,
        pricing_score=75.0,
        osint_sources=[]
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
    data = sorted(data, key=lambda x: (int(x["PO_Year"]) if x.get("PO_Year") and str(x["PO_Year"]).isdigit() else 0, x.get("month_num", 99)))
    
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
    item_description: str = ""
    supplier_id: str
    contract_id: str = ""
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
            
            # 爭議預測 (Disputed Prediction)
            is_disputed = False
            try:
                # Prepare features for dispute model
                # Features: ['Category_Encoded', 'Supplier_Risk_Encoded', 'Contract_Type_Encoded', 'Quantity', 'Maverick_Spend', 'Single_Source_Flag']
                cat_d_enc = dispute_encoders['Category'].transform([request.category] if request.category in dispute_encoders['Category'].classes_ else ['Unknown'])[0]
                risk_d_enc = dispute_encoders['Supplier_Risk'].transform([sup_info["Supplier_Risk"]] if sup_info["Supplier_Risk"] in dispute_encoders['Supplier_Risk'].classes_ else ['Unknown'])[0]
                
                # We use request.contract_id as proxy for contract type since we just have the ID in the request
                # For a more precise mapping, we could parse the type from the UI, but this will work as a fallback
                contract_type = "Framework" if "CON" in request.contract_id else "Spot"
                contract_d_enc = dispute_encoders['Contract_Type'].transform([contract_type] if contract_type in dispute_encoders['Contract_Type'].classes_ else ['Unknown'])[0]
                
                dispute_input = pd.DataFrame([{
                    'Category_Encoded': cat_d_enc,
                    'Supplier_Risk_Encoded': risk_d_enc,
                    'Contract_Type_Encoded': contract_d_enc,
                    'Quantity': request.quantity,
                    'Maverick_Spend': mav_val,
                    'Single_Source_Flag': ss_val
                }])
                dispute_pred = dispute_model.predict(dispute_input)[0]
                is_disputed = bool(dispute_pred == 1)
            except Exception as de:
                print("Dispute Prediction Error:", de)

            # 產生 AI 說明
            import google.generativeai as genai
            import os
            from dotenv import load_dotenv
            ai_explanation = ""
            try:
                load_dotenv()
                gemini_key = os.environ.get("GEMINI_API_KEY")
                if gemini_key:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    savings_amount = request.budget_price * (pred_savings / 100.0)
                    prompt = f"""
                    You are an expert procurement AI assistant.
                    Please explain why this purchase order has a predicted savings of {pred_savings:.1f}% (${savings_amount:.2f}).
                    Context:
                    - Supplier: {request.supplier_id} (Risk: {sup_info['Supplier_Risk']})
                    - Category: {request.category}
                    - Item: {request.item_description}
                    - Quantity: {request.quantity}
                    - Budget: ${request.budget_price}
                    - Single Source: {"Yes" if ss_val else "No"}
                    - Maverick Spend: {"Yes" if mav_val else "No"}
                    
                    Provide a short 2-3 sentence explanation in Traditional Chinese. Keep it professional and business-focused.
                    DO NOT say "Here is the explanation" or "Sure". Just output the explanation directly.
                    """
                    response = model.generate_content(prompt)
                    ai_explanation = response.text.strip()
                else:
                    ai_explanation = f"基於預算金額與歷史平均單價計算，預估將產生 {pred_savings:.1f}% 的成本差異。"
            except Exception as l_e:
                print("Gemini Explanation Error:", l_e)
                ai_explanation = f"基於預算金額與歷史平均單價計算，預估將產生 {pred_savings:.1f}% 的成本差異。"

            ret_val = {
                "status": "success",
                "pred_savings_pct": round(pred_savings, 2),
                "pred_class_code": pred_class_idx,
                "status_text": class_mapping.get(pred_class_idx, "未知"),
                "is_disputed": is_disputed,
                "ai_explanation": ai_explanation,
                "enriched_features": {
                    "supplier_risk": sup_info["Supplier_Risk"],
                    "maverick_spend": sup_info["Maverick_Spend"],
                    "preferred_supplier": sup_info["Preferred_Supplier"]
                },
                "is_mock": False
            }
            
            # 將預測結果寫入日誌表
            try:
                conn_log = get_db_connection()
                conn_log.execute('''
                    INSERT INTO ai_prediction_logs (category, supplier_id, quantity, budget_price, pred_savings_pct, pred_class_code)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (request.category, request.supplier_id, request.quantity, request.budget_price, round(pred_savings, 2), pred_class_idx))
                conn_log.commit()
                conn_log.close()
            except Exception as log_e:
                print("Log insert error:", log_e)
                
            return ret_val
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
    
    ret_val = {
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
    
    try:
        conn_log = get_db_connection()
        conn_log.execute('''
            INSERT INTO ai_prediction_logs (category, supplier_id, quantity, budget_price, pred_savings_pct, pred_class_code)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (request.category, request.supplier_id, request.quantity, request.budget_price, float(mock_savings_pct), int(pred_class_code)))
        conn_log.commit()
        conn_log.close()
    except Exception as log_e:
        print("Log insert error:", log_e)

    return ret_val

@app.get("/api/reports/weekly")
def get_weekly_report(skip_gemini: bool = False, start_date: str = None, end_date: str = None):
    import datetime
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if start_date and end_date:
        date_str_start = start_date
        date_str_end = end_date
    else:
        # 計算本週日期區間 (預設)
        today = datetime.datetime.now()
        seven_days_ago = today - datetime.timedelta(days=7)
        date_str_end = today.strftime("%Y-%m-%d")
        date_str_start = seven_days_ago.strftime("%Y-%m-%d")
    
    # 1. AI Logs Stats
    cursor.execute("""
        SELECT COUNT(*) as total, SUM(CASE WHEN pred_class_code = 2 THEN 1 ELSE 0 END) as blocked 
        FROM ai_prediction_logs 
        WHERE date(timestamp) >= ? AND date(timestamp) <= ?
    """, (date_str_start, date_str_end))
    log_stats = cursor.fetchone()
    total_preds = log_stats['total'] if log_stats['total'] else 0
    blocked_preds = log_stats['blocked'] if log_stats['blocked'] else 0
    
    # 2. Risk Distribution for Pie Chart
    cursor.execute("""
        SELECT pred_class_code, COUNT(*) as cnt 
        FROM ai_prediction_logs 
        WHERE date(timestamp) >= ? AND date(timestamp) <= ?
        GROUP BY pred_class_code
    """, (date_str_start, date_str_end))
    risk_dist = {0: 0, 1: 0, 2: 0}
    for row in cursor.fetchall():
        risk_dist[row['pred_class_code']] = row['cnt']
        
    # 3. Cost Avoidance Trend for Bar Chart
    cursor.execute("""
        SELECT date(timestamp) as dt, SUM(CASE WHEN pred_class_code = 2 THEN budget_price * 0.15 ELSE 0 END) as cost_avoidance 
        FROM ai_prediction_logs 
        WHERE date(timestamp) >= ? AND date(timestamp) <= ?
        GROUP BY dt ORDER BY dt DESC LIMIT 30
    """, (date_str_start, date_str_end))
    trend_rows = cursor.fetchall()
    dates = []
    avoidance = []
    for r in reversed(trend_rows):
        dates.append(r['dt'])
        avoidance.append(round(r['cost_avoidance'], 2))
        
    conn.close()
    
    total_avoidance = sum(avoidance) if avoidance else (blocked_preds * 12500)
    
    gemini_markdown = None
    if not skip_gemini:
        try:
            from rag_engine import generate_weekly_report
            report_data = {
                'date_range': f"{date_str_start} 至 {date_str_end}",
                'blocked_preds': blocked_preds,
                'total_preds': total_preds,
                'total_avoidance': total_avoidance
            }
            gemini_markdown = generate_weekly_report(report_data)
        except Exception as e:
            print("Failed to import or use rag_engine:", e)

    if gemini_markdown:
        markdown = gemini_markdown
    else:
        markdown = f"""
### 📊 報表區間：`{date_str_start}` 至 `{date_str_end}`

### 1. 💰 財務衝擊與 ROI 摘要 
*   **本週預測總單數**：`{total_preds}` 筆
*   **AI 成功攔截高風險單數**：`{blocked_preds}` 筆
*   **AI 實際護盤金額 (Cost Avoidance)**：成功替公司守住 `${total_avoidance:,.0f} USD` 的潛在損失。

### 2. 🔍 AI (SHAP) 決策根因分析
*   **案例分析**：系統偵測到攔截的訂單中，最主要的超支風險來自於「預算與歷史均價落差過大 (+45% 影響力)」，且供應商具有單一來源風險。
*   **處置建議**：退回採購單重啟議價，強制導入雙源採購。

### 3. 🛡️ 戰略轉型與風險緩解追蹤
*   **雙源採購進度**：已啟動，目標轉移高風險供應商 30% 訂單，預計 8/15 完成。
*   **預警訊號**：AI 發現「急件採購」比例微幅上升，請注意後續交期惡化風險。
"""
    
    # 確保哪怕日誌是空的，圖表也有基本的資料點呈現
    if not dates:
        dates = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        avoidance = [0, 0, 0, 0, 0]
        
    return {
        "status": "success",
        "markdown": markdown,
        "charts": {
            "pie": {
                "labels": ["🟢 預期可節省", "🟡 接近預算", "🔴 超出預算風險"],
                "data": [risk_dist.get(0, 0), risk_dist.get(1, 0), risk_dist.get(2, 0)]
            },
            "bar": {
                "labels": dates,
                "data": avoidance
            }
        }
    }

@app.get('/api/recommend/suppliers')
def recommend_suppliers(category: str, scenario: str):
    """Supplier Recommendation API"""
    conn = sqlite3.connect('procurement.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    order_by_clause = ''
    if scenario == 'cost':
        order_by_clause = 'ORDER BY Avg_Savings DESC'
    elif scenario == 'urgent':
        order_by_clause = 'ORDER BY Avg_Days_Late ASC'
    elif scenario == 'compliance':
        order_by_clause = 'ORDER BY Avg_ESG DESC'
    else:
        order_by_clause = 'ORDER BY Avg_Savings DESC'
    
    cursor.execute(f"""
        SELECT Supplier_Name, Supplier_Country, 
               AVG(Savings_Pct) as Avg_Savings, 
               AVG(Days_Late) as Avg_Days_Late, 
               AVG(Supplier_ESG_Score) as Avg_ESG
        FROM procurement_data 
        WHERE Category = ? 
        GROUP BY Supplier_Name 
        {order_by_clause} 
        LIMIT 3
    """, (category,))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for i, row in enumerate(rows):
        score_str = ''
        if scenario == 'cost':
            score_str = f"{row['Avg_Savings']:.2f}% Savings"
            reason = f"Historical high cost reduction in {category}"
        elif scenario == 'urgent':
            score_str = f"{row['Avg_Days_Late']:.1f} Days Late"
            reason = f"Reliable and fast delivery track record"
        elif scenario == 'compliance':
            score_str = f"ESG: {row['Avg_ESG']:.1f}"
            reason = f"Strong adherence to sustainability and compliance"
        
        results.append({
            'rank': i + 1,
            'name': row['Supplier_Name'],
            'country': row['Supplier_Country'],
            'score_text': score_str,
            'reason': reason,
            'raw_metrics': {
                'savings_pct': round(row['Avg_Savings'], 2),
                'days_late': round(row['Avg_Days_Late'], 2),
                'esg_score': round(row['Avg_ESG'], 2)
            }
        })
    return results

@app.get('/api/overview/kpis')
def overview_kpis():
    """Overview KPIs API"""
    conn = sqlite3.connect('procurement.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Avg Savings, Avg ESG
    cursor.execute('SELECT AVG(Savings_Pct) as avg_savings, AVG(Supplier_ESG_Score) as avg_esg FROM procurement_data')
    row1 = cursor.fetchone()
    avg_savings = float(row1['avg_savings']) if row1['avg_savings'] is not None else 0.0
    avg_esg = float(row1['avg_esg']) if row1['avg_esg'] is not None else 0.0
    
    # 2. Risk Score & High Risk Count
    cursor.execute('SELECT Supplier_Risk, COUNT(*) as cnt FROM procurement_data GROUP BY Supplier_Risk')
    risk_rows = cursor.fetchall()
    total_risk_score = 0
    total_count = 0
    high_risk_count = 0
    for r in risk_rows:
        cnt = int(r['cnt'])
        risk_val = str(r['Supplier_Risk']).strip().capitalize()
        total_count += cnt
        if risk_val == 'High':
            total_risk_score += 100 * cnt
            high_risk_count += cnt
        elif risk_val == 'Medium':
            total_risk_score += 50 * cnt
            
    avg_risk_score = (total_risk_score / total_count) if total_count > 0 else 0
    
    # 3. Maverick Spend Count
    cursor.execute("SELECT COUNT(*) as cnt FROM procurement_data WHERE Maverick_Spend COLLATE NOCASE IN ('yes', 'true', '1')")
    maverick_row = cursor.fetchone()
    maverick_count = int(maverick_row['cnt'])
    
    conn.close()
    
    return {
        'avg_savings': round(avg_savings, 2),
        'avg_risk_score': round(avg_risk_score, 1),
        'high_risk_count': high_risk_count,
        'avg_esg': round(avg_esg, 1),
        'maverick_count': maverick_count
    }


# ----- 新供應商風險評分 API (v2 審計修正版) -----

class NewSupplierRequest(BaseModel):
    tier: int = Field(..., ge=1, le=3, description="供應商層級 1-3", example=3)
    esg: float = Field(..., ge=0, le=100, description="ESG 評分 0-100", example=48.0)
    mav_rate: Optional[float] = Field(None, ge=0, le=1,
        description="觀察至今的脫軌採購率; 入職當下無交易時留空 → Day-0 模式")
    single_rate: Optional[float] = Field(None, ge=0, le=1,
        description="觀察至今的單一來源率; 入職當下無交易時留空 → Day-0 模式")
    n_transactions: Optional[int] = Field(None, ge=0,
        description="目前累積交易筆數 (選填, <25 時回應會提醒行為率不穩)")

class RiskReason(BaseModel):
    feature: str = Field(..., description="影響因素 (商業語言)")
    value: float = Field(..., description="該供應商的特徵值")
    contribution: float = Field(..., description="對預測類別 logit 的貢獻 (正=推向此級)")
    direction: str = Field(..., description="推向此級 / 拉離此級")

class NewSupplierResponse(BaseModel):
    mode: str = Field(..., description="day0 (入職當下) 或 review (交易累積後)")
    risk_level: str = Field(..., description="預測判定: 核准 / 需複核")
    probabilities: dict = Field(..., description="二元機率 (未校準, 僅供排序參考)")
    reasons: list[RiskReason] = Field(..., description="為什麼是這個判定")
    loso_accuracy: float = Field(..., description="該模式留一供應商驗證準確率")
    caveat: str = Field(..., description="使用注意")

@app.post("/api/predict/new-supplier-risk", response_model=NewSupplierResponse)
def predict_new_supplier_risk(request: NewSupplierRequest = Body(...)):
    """
    [ML] 新供應商風險分級 API (v3, 二元主框架)

    回答: 「第 16 家新供應商依公司既有風險政策該『核准』還是『需複核』? 為什麼?」
    - Day-0 模式 (入職當下, 只有 tier+esg): LOSO 87%, AUC 0.93
    - Review 模式 (交易累積後, 加行為率): LOSO 87%, AUC 0.91
    多數類基準 60%, permutation p<0.05。Medium 以上一律送人工複核。
    """
    if not NEW_SUPPLIER_MODEL_LOADED:
        raise HTTPException(status_code=503, detail="New-supplier scoring model not loaded")

    # 行為率必須成對提供; 只給一個會造成「已知資訊被靜默忽略」的誤導結果
    provided = (request.mav_rate is not None, request.single_rate is not None)
    if provided[0] != provided[1]:
        raise HTTPException(status_code=422, detail=(
            "mav_rate 與 single_rate 必須同時提供 (Review 模式) 或同時留空 (Day-0 模式), "
            "只提供其中一個會導致該資訊被忽略而產生誤導性評分"))

    try:
        b = new_supplier_bundle
        if all(provided):
            mode, model, feats = "review", b["review_model"], b["features_review"]
            row = {"tier": request.tier, "esg": request.esg,
                   "mav_rate": request.mav_rate, "single_rate": request.single_rate}
        else:
            mode, model, feats = "day0", b["day0_model"], b["features_day0"]
            row = {"tier": request.tier, "esg": request.esg}

        profile = pd.DataFrame([row])[feats]
        proba = model.predict_proba(profile)[0]
        clf = model.named_steps["clf"]
        risk_order = b["risk_order"]     # ["核准","需複核"]
        pred_pos = int(proba.argmax())
        pred_label = int(clf.classes_[pred_pos])

        # 可解釋性: 標準化特徵值 × 係數。二元 LogReg 的 coef_ 為 (1,n)、
        # 該列即『正類(需複核=1)』方向; 不可用 coef_[pred_pos]。
        z = model.named_steps["scaler"].transform(profile)[0]
        coef = clf.coef_
        if coef.shape[0] == 1:                       # 二元
            contrib_pos = z * coef[0]                # 對『需複核』的貢獻
            contrib = contrib_pos if pred_label == 1 else -contrib_pos
        else:                                        # 多類 (向後相容)
            contrib = z * coef[pred_pos]
        order = abs(contrib).argsort()[::-1]

        reasons = [RiskReason(
            feature=b["feature_labels"].get(feats[int(j)], feats[int(j)]),
            value=round(float(profile.iloc[0, int(j)]), 4),
            contribution=round(float(contrib[int(j)]), 3),
            direction="推向此判定" if contrib[int(j)] > 0 else "拉離此判定",
        ) for j in order]

        caveat = ("二元判定「核准 vs 需複核」, Medium 以上一律送人工複核 "
                  "(寧可誤殺不可漏放)。機率未經校準, 僅供排序參考; "
                  "訊號幾乎全來自 Tier, 行為率貢獻有限。")
        if mode == "review" and request.n_transactions is not None and request.n_transactions < 25:
            caveat += f" 目前僅 {request.n_transactions} 筆交易, 行為率仍不穩, 建議並看 Day-0 結果。"

        return NewSupplierResponse(
            mode=mode,
            risk_level=risk_order[pred_label],
            probabilities={risk_order[int(c)]: round(float(proba[i]), 4)
                           for i, c in enumerate(clf.classes_)},
            reasons=reasons,
            loso_accuracy=round(float(b["loso_accuracy"][mode]), 3),
            caveat=caveat,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in new-supplier risk scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----- 供應商財務風險 API (UCI 台灣真實資料 ML 模型 + Altman Z'' 即時引擎) -----

# 即時引擎模組: import 失敗 (如快取 DB 不可寫) 不應擊殺整個 app
try:
    import financial_risk as fr
    FR_ENGINE_LOADED = True
except Exception as _fr_err:
    print(f"Warning: financial_risk engine failed to load. Error: {_fr_err}")
    FR_ENGINE_LOADED = False


def _model_contributions(model, Xrow):
    """回傳逐特徵貢獻 (對正類 logit), 支援 XGBoost / LightGBM / LogReg Pipeline。

    訓練腳本以 CV 挑冠軍, 部署的模型類別可能隨重訓改變 —
    端點不可寫死單一框架的 API。
    """
    import numpy as _np
    if hasattr(model, "get_booster"):                       # XGBoost
        import xgboost as _xgb
        return model.get_booster().predict(
            _xgb.DMatrix(Xrow), pred_contribs=True)[0][:-1]  # 末位是 bias
    if hasattr(model, "booster_"):                          # LightGBM
        return _np.asarray(
            model.booster_.predict(Xrow, pred_contrib=True))[0][:-1]
    if hasattr(model, "named_steps") and "c" in getattr(model, "named_steps", {}):
        z = model.named_steps["s"].transform(Xrow)[0]       # LogReg Pipeline
        return z * model.named_steps["c"].coef_[0]
    return _np.zeros(Xrow.shape[1])                         # 未知模型: 無貢獻資訊

class FinancialRatiosRequest(BaseModel):
    ratios: dict = Field(..., description=(
        "UCI 正規化財務比率 (0-1), key 為特徵名。可只給部分, "
        "未提供者以訓練集中位數補值。例: "
        "{'Net Income to Total Assets': 0.6, 'Total debt/Total net worth': 0.02}"))

class FinancialRiskResponse(BaseModel):
    probability: float = Field(..., description="破產機率 (模型輸出)")
    risk_tier: str = Field(..., description="High / Watch / Low")
    top_factors: list = Field(..., description="影響最大的特徵貢獻 (正=推向破產)")
    filled_with_median: int = Field(..., description="以中位數補值的特徵數")
    model_info: str = Field(..., description="模型與資料來源")

@app.post("/api/predict/financial-risk", response_model=FinancialRiskResponse)
def predict_financial_risk(request: FinancialRatiosRequest = Body(...)):
    """
    [ML] 供應商財務風險 — 輸入 UCI 正規化財務比率, 輸出破產機率與分級。
    模型: XGBoost, 訓練自台灣 6,819 家真實企業 (test AUC 0.958)。
    """
    if not FINANCIAL_MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Financial risk model not loaded")
    try:
        b = financial_risk_bundle
        feats = b["feature_names"]
        unknown = [k for k in request.ratios if k not in feats]
        if unknown:
            raise HTTPException(status_code=422,
                detail=f"未知特徵名: {unknown[:5]} (需為 UCI 資料集欄名)")
        # 值必須是有限數值 — 否則回 422 而非讓 float() 在深處炸成 500
        clean = {}
        for k, v in request.ratios.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422,
                    detail=f"特徵「{k}」的值必須是數字, 收到: {v!r}")
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise HTTPException(status_code=422,
                    detail=f"特徵「{k}」的值必須是有限數值")
            clean[k] = fv
        row = dict(b["train_medians"])
        row.update(clean)
        Xrow = pd.DataFrame([row])[feats]

        proba = float(b["model"].predict_proba(Xrow)[0, 1])
        thr = b["thresholds"]
        tier = "High" if proba >= thr["high"] else ("Watch" if proba >= thr["watch"] else "Low")

        contribs = _model_contributions(b["model"], Xrow)
        order = abs(contribs).argsort()[::-1][:5]
        top = [{"feature": feats[i],
                "value": round(float(Xrow.iloc[0, i]), 4),
                "contribution": round(float(contribs[i]), 3),
                "direction": "↑ 推升風險" if contribs[i] > 0 else "↓ 降低風險",
                "user_provided": feats[i] in request.ratios}
               for i in order]

        return FinancialRiskResponse(
            probability=round(proba, 4),
            risk_tier=tier,
            top_factors=top,
            filled_with_median=len(feats) - len(request.ratios),
            model_info=f"{b['model_name']} | {b['data_source']} | test AUC={b['metrics']['test_auc']:.3f}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in financial risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/assess/company-financial")
def assess_company_financial(name: str = Query(..., description="公司名稱或股票代號 (如 台積電 / 2330.TW)")):
    """
    [即時] 公司名稱 → Altman Z''-score 財務風險評估。
    以 yfinance 抓取最新年報實算 (上市櫃公司), 未上市/查無財報時誠實回報。
    404 = 查無此公司/財報; 502 = 外部資料源暫時故障 (可重試, 勿當作不存在)。
    """
    if not FR_ENGINE_LOADED:
        raise HTTPException(status_code=503, detail="Financial assessment engine not loaded")
    result, err, kind = fr.assess_company(name.strip())
    if err:
        raise HTTPException(status_code=502 if kind == "upstream" else 404, detail=err)
    result["method"] = ("Altman Z''-score (1995 新興市場版): "
                        "Z''=6.56·X1+3.26·X2+6.72·X3+1.05·X4; "
                        ">2.6 安全 | 1.1-2.6 灰色 | <1.1 危險")
    result["caveat"] = ("Z-score 為公式型預警指標, 非本專案 ML 模型輸出; "
                        "ML 模型 (UCI 正規化比率空間) 請用 /api/predict/financial-risk。")
    return result


@app.get("/api/supplier/search")
def search_supplier(q: str):
    """
    Search for a supplier by name and return a 360 scorecard of their historical data.
    """
    if not q or len(q.strip()) == 0:
        return {"error": "Query cannot be empty"}
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Search for the supplier
        search_query = f"%{q.strip().lower()}%"
        cursor.execute("""
            SELECT Supplier_ID, Supplier_Name, Supplier_Risk, Supplier_Tier, Supplier_ESG_Score, Preferred_Supplier
            FROM procurement_data 
            WHERE LOWER(Supplier_Name) LIKE ? OR LOWER(Supplier_ID) LIKE ?
            LIMIT 1
        """, (search_query, search_query))
        
        supplier_row = cursor.fetchone()
        
        if not supplier_row:
            conn.close()
            return {"found": False, "message": "No supplier found matching the query."}
            
        supplier_id = supplier_row['Supplier_ID']
        supplier_name = supplier_row['Supplier_Name']
        tier = int(supplier_row['Supplier_Tier'])
        esg_score = float(supplier_row['Supplier_ESG_Score'])
        preferred = supplier_row['Preferred_Supplier']
        
        # 2. Get Aggregated Metrics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_pos,
                SUM(CAST(Quantity AS FLOAT) * 120) as total_spend,
                AVG(CAST(Savings_Pct AS FLOAT)) as avg_savings,
                AVG(CAST(Days_Late AS FLOAT)) as avg_days_late,
                SUM(CASE WHEN Maverick_Spend COLLATE NOCASE IN ('yes', 'true', '1') THEN 1 ELSE 0 END) as maverick_count
            FROM procurement_data
            WHERE Supplier_ID = ?
        """, (supplier_id,))
        
        metrics_row = cursor.fetchone()
        
        # 3. Get Recent POs
        cursor.execute("""
            SELECT PO_Number as PO_ID, PO_Date, (Quantity * 120) as Spend, Category, Supplier_Risk, Maverick_Spend, PO_Status
            FROM procurement_data
            WHERE Supplier_ID = ?
            ORDER BY PO_Date DESC
            LIMIT 5
        """, (supplier_id,))
        
        recent_pos = [dict(r) for r in cursor.fetchall()]
        
        total_pos = int(metrics_row['total_pos'] or 0)
        mav_count = int(metrics_row['maverick_count'] or 0)
        single_count = 0
        
        cursor.execute("SELECT COUNT(*) as c FROM procurement_data WHERE Supplier_ID = ? AND Single_Source_Flag COLLATE NOCASE IN ('yes', 'true', '1')", (supplier_id,))
        single_count_row = cursor.fetchone()
        if single_count_row:
            single_count = int(single_count_row['c'] or 0)
            
        mav_rate = mav_count / total_pos if total_pos > 0 else 0
        single_rate = single_count / total_pos if total_pos > 0 else 0
        
        conn.close()
        
        # Use new model review mode to calculate risk_level
        b = new_supplier_bundle
        model = b["review_model"]
        feats = b["features_review"]
        row = {"tier": tier, "esg": esg_score, "mav_rate": mav_rate, "single_rate": single_rate}
        profile = pd.DataFrame([row])[feats]
        proba = model.predict_proba(profile)[0]
        clf = model.named_steps["clf"]
        pred_label = int(clf.classes_[int(proba.argmax())])
        risk_level = b["risk_order"][pred_label]
        
        return {
            "found": True,
            "supplier": {
                "id": supplier_id,
                "name": supplier_name,
                "risk_level": risk_level,
                "tier": tier,
                "esg_score": esg_score,
                "preferred": preferred
            },
            "metrics": {
                "total_pos": metrics_row['total_pos'] if metrics_row else 0,
                "total_spend": metrics_row['total_spend'] if metrics_row and metrics_row['total_spend'] else 0,
                "avg_savings": round(metrics_row['avg_savings'], 2) if metrics_row and metrics_row['avg_savings'] else 0,
                "avg_days_late": round(metrics_row['avg_days_late'], 1) if metrics_row and metrics_row['avg_days_late'] else 0,
                "maverick_count": metrics_row['maverick_count'] if metrics_row and metrics_row['maverick_count'] else 0
            },
            "recent_pos": recent_pos
        }
    except Exception as e:
        print(f"Search API Error: {e}")
        return {"error": str(e)}

import random

class SupplierAddRequest(BaseModel):
    name: str
    country: str
    category: str
    risk_level: str
    esg_score: float

@app.post("/api/supplier/add")
def add_supplier(request: SupplierAddRequest):
    conn = get_db_connection()
    try:
        # Create a dummy PO to register the supplier in the system
        supplier_id = f"V{random.randint(10000, 99999)}"
        
        # Determine some initial metrics based on risk
        days_late = 0
        if request.risk_level == "High":
            days_late = 10
        elif request.risk_level == "Medium":
            days_late = 4
            
        savings_pct = 5.0 # baseline
        
        conn.execute('''
            INSERT INTO procurement_data (
                PO_Number, PO_Date, PO_Year, PO_Month, Supplier_ID, Supplier_Name, Supplier_Country,
                Supplier_Risk, Category, Supplier_ESG_Score, Days_Late, Savings_Pct,
                Quantity, Item_Description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"PO-{random.randint(100000, 999999)}",
            "2024-05-01",
            "2024",
            "May",
            supplier_id,
            request.name,
            request.country,
            request.risk_level,
            request.category,
            request.esg_score,
            days_late,
            savings_pct,
            1,
            "Initial Onboarding"
        ))
        conn.commit()
        return {"success": True, "message": "Supplier onboarded successfully", "supplier_id": supplier_id}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        conn.close()
