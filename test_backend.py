import requests

def test_predict():
    print("Testing /api/predict/supplier-risk...")
    payload = {
        "supplier_id": "SUP-NEW-01",
        "country": "Taiwan",
        "category": "Hardware",
        "tier": 2
    }
    try:
        r = requests.post("http://localhost:8000/api/predict/supplier-risk", json=payload)
        print("Status:", r.status_code)
        print(r.json())
    except Exception as e:
        print("Error:", e)

def test_search():
    print("Testing /api/supplier/search...")
    try:
        r = requests.get("http://localhost:8000/api/supplier/search?q=SUP-001")
        print("Status:", r.status_code)
        print(r.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    import threading
    import time
    import subprocess
    import os

    os.chdir("backend")
    print("Starting backend...")
    proc = subprocess.Popen([r".venv\Scripts\uvicorn.exe", "main:app", "--port", "8000"])
    time.sleep(3)
    try:
        test_predict()
        print("-" * 40)
        test_search()
    finally:
        print("Shutting down backend...")
        proc.terminate()
