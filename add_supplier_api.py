append_code = '''
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
        
        conn.execute(\'\'\'
            INSERT INTO procurement_data (
                PO_Number, PO_Date, Supplier_ID, Supplier_Name, Supplier_Country,
                Supplier_Risk, Category, Supplier_ESG_Score, Days_Late, Savings_Pct,
                Quantity, Item_Description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        \'\'\', (
            f"PO-{random.randint(100000, 999999)}",
            "2024-05-01",
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
'''

with open('backend/main.py', 'a', encoding='utf-8') as f:
    f.write(append_code)
