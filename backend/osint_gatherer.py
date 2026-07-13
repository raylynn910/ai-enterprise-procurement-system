import wikipedia
from duckduckgo_search import DDGS
import re

def gather_supplier_intelligence(supplier_name: str, country: str = None) -> dict:
    """
    Search OSINT sources for supplier information and derive estimated features.
    Returns:
        dict: {
            "esg_score": float,
            "days_late": int,
            "po_status": str,
            "osint_summary": str
        }
    """
    summary_text = ""
    found_info = False

    is_chinese = bool(re.search(r'[\u4e00-\u9fff]', supplier_name))
    
    # 1. Try Wikipedia
    try:
        if is_chinese:
            wikipedia.set_lang("zh")
        else:
            wikipedia.set_lang("en")
            
        search_results = wikipedia.search(supplier_name)
        if search_results:
            # 確保標題或內文有包含公司名稱，避免維基百科的模糊推薦
            wiki_summary = wikipedia.summary(search_results[0], sentences=2, auto_suggest=False)
            if supplier_name.lower() in search_results[0].lower() or supplier_name.lower() in wiki_summary.lower():
                summary_text += wiki_summary + " "
                found_info = True
    except Exception as e:
        pass

    # 2. Try Tavily API (Adverse Media Search)
    osint_sources_list = []
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key="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")
        
        # Craft a targeted search query for news and controversies
        if is_chinese:
            query = f'"{supplier_name}" AND (新聞 OR 爭議 OR 食安 OR 違法 OR 永續)'
        else:
            query = f'"{supplier_name}" AND (news OR controversy OR lawsuit OR sustainability)'
            
        response = tavily.search(query=query, search_depth="basic", max_results=5)
        for r in response.get("results", []):
            body = r.get("content", "")
            title = r.get("title", "")
            
            # 防呆：確保該公司名字真的出現在搜尋結果中，否則可能只是查到後面的 general keyword
            if supplier_name.lower() in body.lower() or supplier_name.lower() in title.lower():
                summary_text += body + " "
                osint_sources_list.append({
                    "title": title or "News Source",
                    "url": r.get("url", "#"),
                    "snippet": body[:150] + "..." if len(body) > 150 else body
                })
                found_info = True
    except Exception as e:
        print("Tavily Search Error:", e)

    if not found_info or not summary_text.strip():
        return {
            "esg_score": 30.0, # High risk for ghost companies
            "days_late": 15,
            "po_status": "Pending",
            "osint_summary": "找不到該公司相關資訊",
            "osint_sources": []
        }

    # 3. Simple Sentiment / Keyword Analysis
    text_lower = summary_text.lower()
    
    negative_keywords = ["lawsuit", "scandal", "fraud", "penalty", "fine", "violation", "bankruptcy", "delay", "court", "investigation", "sued", "breach", "controversy", 
                         "訴訟", "裁罰", "違反", "詐欺", "醜聞", "食安", "下架", "毒", "延遲", "違法", "黑心", "罰單", "風波", "調查", "停工"]
    positive_keywords = ["award", "sustainability", "green", "leader", "innovation", "reliable", "partner", "top", "excellence", "growth", "global",
                         "獲獎", "永續", "領先", "優良", "認證", "ESG", "綠能", "創新", "卓越"]
    
    neg_count = sum(1 for word in negative_keywords if word in text_lower)
    pos_count = sum(1 for word in positive_keywords if word in text_lower)
    
    # Calculate ESG score (base 70 to be more forgiving)
    esg_score = 70.0 + (pos_count * 10) - (neg_count * 5)
    esg_score = max(0.0, min(100.0, esg_score))
    
    # Estimate days late based on negative/positive
    if neg_count > pos_count:
        days_late = 2 + (neg_count * 1)
    elif pos_count > 0:
        days_late = -2 # Early delivery
    else:
        days_late = 2
        
    # Generate human readable summary
    if neg_count >= 5 and neg_count > pos_count + 2:
        osint_summary = f"【OSINT 高度示警】於公開情報網發現大量負面關鍵字 (如訴訟、裁罰或爭議)。片段擷取：'...{summary_text[:120]}...'"
    elif pos_count > 0 and pos_count >= neg_count:
        osint_summary = f"【OSINT 評估良好】於公開情報網發現正面評價 (如永續、獲獎或領先)。片段擷取：'...{summary_text[:120]}...'"
    else:
        osint_summary = f"【OSINT 中性結果】已於公開網路核實公司實體，未發現重大爭議。片段擷取：'...{summary_text[:120]}...'"
        
    return {
        "esg_score": esg_score,
        "days_late": days_late,
        "po_status": "Completed", # Default
        "osint_summary": osint_summary,
        "osint_sources": osint_sources_list
    }

if __name__ == "__main__":
    print("Testing Apple...")
    print(gather_supplier_intelligence("Apple Inc."))
    print("\nTesting Unknown Ghost Company...")
    print(gather_supplier_intelligence("SomeGhostCompany999"))
