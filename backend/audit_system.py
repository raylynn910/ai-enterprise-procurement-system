import sqlite3
import os
import requests
import json
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'procurement.db')
API_URL = "http://127.0.0.1:8000"

def audit_database():
    print("=== Database Audit ===")
    if not os.path.exists(DB_PATH):
        print(f"[FAIL] Database file not found at {DB_PATH}")
        return False
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='procurement_data';")
        if not cursor.fetchone():
            print("[FAIL] procurement_data table not found")
            return False
            
        # Check row count
        cursor.execute("SELECT COUNT(*) FROM procurement_data")
        count = cursor.fetchone()[0]
        print(f"[PASS] Table found. Total rows: {count}")
        
        # Check schema
        cursor.execute("PRAGMA table_info(procurement_data);")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"[PASS] Found {len(columns)} columns.")
        
        conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] Database audit failed: {e}")
        return False

def test_api_endpoint(method, endpoint, payload=None):
    url = f"{API_URL}{endpoint}"
    print(f"Testing {method} {endpoint}...")
    try:
        if method == "GET":
            res = requests.get(url, timeout=10)
        else:
            res = requests.post(url, json=payload, timeout=10)
            
        if res.status_code == 200:
            print(f"  [PASS] Status 200")
            return True, res.json()
        else:
            print(f"  [FAIL] Status {res.status_code}: {res.text}")
            return False, None
    except Exception as e:
        print(f"  [FAIL] Connection error: {e}")
        return False, None

def audit_apis():
    print("\n=== API Audit ===")
    endpoints = [
        ("GET", "/api/procurements", None),
        ("GET", "/api/risk/orders", None),
        ("GET", "/api/trends/monthly", None),
        ("GET", "/api/form-options", None),
    ]
    for method, ep, payload in endpoints:
        test_api_endpoint(method, ep, payload)
        
    # GET /api/context/category
    test_api_endpoint("GET", "/api/context/category?name=IT Software", None)
    
    # GET /api/context/supplier
    test_api_endpoint("GET", "/api/context/supplier?id=SUP-001", None)
    
    # POST /api/predict/supplier-risk
    payload_risk = {
        "supplier_id": "SUP-001",
        "supplier_name": "CloudTech Solutions",
        "category": "IT Software",
        "country": "USA",
        "days_late": 0,
        "esg_score": 80
    }
    test_api_endpoint("POST", "/api/predict/supplier-risk", payload_risk)
    
    # POST /api/predict/savings
    payload_savings = {
        "category": "IT Software",
        "supplier_id": "SUP-001",
        "quantity": 100,
        "budget_price": 500,
        "contract_type": "Annual Framework"
    }
    test_api_endpoint("POST", "/api/predict/savings", payload_savings)

if __name__ == "__main__":
    audit_database()
    audit_apis()
