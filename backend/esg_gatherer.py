import re

# 模擬真實的權威 ESG 評級資料庫 (例如 MSCI, Sustainalytics 的公開評等)
# 數值代表映射到雷達圖的滿分 100 評分
ESG_AUTHORITY_DB = {
    "APPLE": {"esg_score": 92.0, "rating": "MSCI: AA", "compliance": 95.0, "delivery_score": 90.0},
    "GOOGLE": {"esg_score": 90.0, "rating": "MSCI: AA", "compliance": 90.0, "delivery_score": 92.0},
    "ALPHABET": {"esg_score": 90.0, "rating": "MSCI: AA", "compliance": 90.0, "delivery_score": 92.0},
    "MICROSOFT": {"esg_score": 95.0, "rating": "MSCI: AAA", "compliance": 98.0, "delivery_score": 95.0},
    "INTEL": {"esg_score": 88.0, "rating": "MSCI: A", "compliance": 90.0, "delivery_score": 88.0},
    "BMW": {"esg_score": 85.0, "rating": "MSCI: A", "compliance": 90.0, "delivery_score": 90.0},
    "TSMC": {"esg_score": 95.0, "rating": "MSCI: AAA", "compliance": 98.0, "delivery_score": 98.0},
    "TAIWAN SEMICONDUCTOR": {"esg_score": 95.0, "rating": "MSCI: AAA", "compliance": 98.0, "delivery_score": 98.0},
    "NVIDIA": {"esg_score": 85.0, "rating": "MSCI: A", "compliance": 92.0, "delivery_score": 90.0},
    "AMAZON": {"esg_score": 75.0, "rating": "MSCI: BBB", "compliance": 85.0, "delivery_score": 95.0},
    "TESLA": {"esg_score": 70.0, "rating": "MSCI: BBB", "compliance": 80.0, "delivery_score": 80.0},
}

def get_authoritative_esg(supplier_name: str) -> dict:
    """
    查詢該企業是否具備國際權威 ESG 評分。
    如果存在，回傳真實分數與合規底分。如果不存在，回傳 None。
    """
    if not supplier_name:
        return None
        
    name_upper = supplier_name.strip().upper()
    
    # 簡單的別名/包含比對 (例如 "Apple Inc." -> "APPLE")
    for key, data in ESG_AUTHORITY_DB.items():
        if key in name_upper:
            return data
            
    return None
