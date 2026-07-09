# ai_service.py
import pickle
import pandas as pd
import streamlit as st

@st.cache_resource
def load_ml_models():
    """使用 Streamlit 快取機制，讓網頁只載入一次模型，大幅優化網頁速度"""
    with open("models/reg_model.pkl", "rb") as f:
        reg_model = pickle.load(f)
    with open("models/cls_refiner.pkl", "rb") as f:
        cls_refiner = pickle.load(f)
    with open("models/le_dict.pkl", "rb") as f:
        le_dict = pickle.load(f)
    return reg_model, cls_refiner, le_dict

def predict_new_order(category, department, quantity, budget_price, item_avg_price, 
                      pref_supplier, supplier_risk, supplier_status, maverick_spend, single_source):
    """
    接收前端 Streamlit 的輸入值，進行 LabelEncoding 與 v6.0 特徵工程，並回傳預測結果
    """
    # 載入模型組件
    reg_model, cls_refiner, le_dict = load_ml_models()
    
    # 1. 類別特徵文字轉數字 (Label Encoding 防禦機制)
    cat_enc = le_dict["Category"].transform([category])[0]
    dept_enc = le_dict["Department"].transform([department])[0]
    pref_enc = le_dict["Preferred Supplier"].transform([pref_supplier])[0]
    risk_enc = le_dict["Supplier Risk"].transform([supplier_risk])[0]
    stat_enc = le_dict["Supplier Status"].transform([supplier_status])[0]
    
    mav_val = 1.0 if maverick_spend else 0.0
    ss_val = 1.0 if single_source else 0.0
    
    # 2. 自動計算 v6.0 核心黃金優化特徵：Gap Pct (預算行情價差率)
    gap_pct = (budget_price - item_avg_price) / (item_avg_price + 1e-5)
    
    # 3. 嚴格按照模型訓練時的欄位順序重組 DataFrame
    input_data = pd.DataFrame([{
        "Category_Encoded": cat_enc,
        "Department_Encoded": dept_enc,
        "Quantity": quantity,
        "Budget Unit Price": budget_price,
        "Item_Avg_Price": item_avg_price,
        "Budget_vs_Avg_Gap": gap_pct,
        "Preferred Supplier_Encoded": pref_enc,
        "Supplier Risk_Encoded": risk_enc,
        "Supplier Status_Encoded": stat_enc,
        "Maverick Spend": mav_val,
        "Single Source Flag": ss_val
    }])
    
    # 4. 第一階迴歸預測連續型的 Savings Pct
    pred_savings = reg_model.predict(input_data)[0]
    
    # 5. 第二階分類器：將預測值疊加，進行二次細緻邊界劃分
    input_data["Pred_Value"] = pred_savings
    pred_class_idx = cls_refiner.predict(input_data)[0]
    
    # 分類映射
    class_mapping = {0: "🟢 預期可節省成本", 1: "🟡 接近預算範圍", 2: "🔴 潛在超出預算風險"}
    final_status = class_mapping.get(pred_class_idx, "未知")
    
    return round(pred_savings, 2), pred_class_idx, final_status