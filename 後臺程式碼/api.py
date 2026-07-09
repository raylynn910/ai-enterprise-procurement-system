import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 建立 API 應用程式
app = FastAPI(title="供應商智慧採購 AI 預測服務")

# 🚀 關鍵防禦：允許跨來源資源共用 (CORS)
# 這樣組員的 HTML 網頁才能從瀏覽器發送請求給你的 Python 後端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許任何網頁連線
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 載入訓練好的模型與組件
with open("models/reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)
with open("models/cls_refiner.pkl", "rb") as f:
    cls_refiner = pickle.load(f)
with open("models/le_dict.pkl", "rb") as f:
    le_dict = pickle.load(f)

# 2. 定義前端網頁要傳過來的資料格式（必須跟前端欄位對齊）
class PurchaseOrder(BaseModel):
    category: str
    department: str
    quantity: int
    budget_price: float
    item_avg_price: float
    pref_supplier: str
    supplier_risk: str
    supplier_status: str
    maverick_spend: bool
    single_source: bool

# 3. 建立預測接口 (API Endpoint)
@app.post("/predict")
def predict_purchase_order(order: PurchaseOrder):
    # 類別特徵文字轉數字 (Label Encoding)
    cat_enc = le_dict["Category"].transform([order.category])[0]
    dept_enc = le_dict["Department"].transform([order.department])[0]
    pref_enc = le_dict["Preferred Supplier"].transform([order.pref_supplier])[0]
    risk_enc = le_dict["Supplier Risk"].transform([order.supplier_risk])[0]
    stat_enc = le_dict["Supplier Status"].transform([order.supplier_status])[0]
    
    mav_val = 1.0 if order.maverick_spend else 0.0
    ss_val = 1.0 if order.single_source else 0.0
    
    # 計算 v6.0 黃金特徵 Gap Pct (預算行情價差率)
    gap_pct = (order.budget_price - order.item_avg_price) / (order.item_avg_price + 1e-5)
    
    # 嚴格按照模型訓練時的欄位順序組裝成 DataFrame
    input_data = pd.DataFrame([{
        "Category_Encoded": cat_enc, "Department_Encoded": dept_enc,
        "Quantity": order.quantity, "Budget Unit Price": order.budget_price,
        "Item_Avg_Price": order.item_avg_price, "Budget_vs_Avg_Gap": gap_pct,
        "Preferred Supplier_Encoded": pref_enc, "Supplier Risk_Encoded": risk_enc,
        "Supplier Status_Encoded": stat_enc, "Maverick Spend": mav_val,
        "Single Source Flag": ss_val
    }])
    
    # 第一階迴歸預測
    pred_savings = float(reg_model.predict(input_data)[0])
    
    # 第二階分類預測
    input_data["Pred_Value"] = pred_savings
    pred_class_idx = int(cls_refiner.predict(input_data)[0])
    
    class_mapping = {0: "🟢 預期可節省成本", 1: "🟡 接近預算範圍", 2: "🔴 潛在超出預算風險"}
    
    # 回傳給前端 HTML 的 JSON 結果
    return {
        "pred_savings_pct": round(pred_savings, 2),
        "pred_class_code": pred_class_idx,
        "status_text": class_mapping.get(pred_class_idx, "未知")
    }