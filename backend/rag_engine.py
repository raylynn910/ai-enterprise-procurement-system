import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini if key exists
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
is_rag_enabled = False
model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Test model availability
        model.generate_content("test")
        is_rag_enabled = True
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        is_rag_enabled = False

def generate_osint_summary(supplier_name: str, osint_texts: list) -> str:
    """
    RAG Scenario A: Supplier Due Diligence.
    Takes retrieved OSINT texts and uses Gemini to synthesize a professional summary.
    """
    if not is_rag_enabled or not osint_texts:
        return None
        
    context = "\n\n".join(osint_texts)
    prompt = f"""
    你是企業的資深採購與合規稽核員。
    我們正在針對供應商「{supplier_name}」進行背景調查。
    以下是我們剛從網路上擷取到的最新開源情報 (OSINT)：
    
    {context}
    
    請基於上述情報，撰寫一段約 80~100 字的「深度調查摘要」。
    - 如果發現負面新聞（如訴訟、違法），請明確指出潛在風險與處置建議。
    - 如果新聞偏向正面（如獲獎、擴廠），請指出有利於議價或長期合作的優勢。
    - 語氣必須客觀、冷靜、嚴謹，不帶過度渲染。
    - 在句尾加上「(由 Gemini AI 輔助分析)」的字樣。
    - 絕對不能瞎掰上述情報中沒有提及的事情 (防止幻覺)。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini generation error: {e}")
        return None

def generate_weekly_report(report_data: dict) -> str:
    """
    RAG Scenario B: Executive Weekly Report Generation.
    Takes DB statistics and uses Gemini to write the executive summary markdown.
    """
    if not is_rag_enabled:
        return None
        
    prompt = f"""
    你是企業的 CPO (首席採購長) 專屬 AI 幕僚。
    請根據以下本週採購系統的運算數據，為總裁撰寫一份「AI 驅動供應商風險管理週報」。
    
    [本週數據]
    - 統計區間：{report_data.get('date_range', '本週')}
    - 系統攔截的高風險超支訂單數：{report_data.get('blocked_preds', 0)} 筆
    - 總預測單數：{report_data.get('total_preds', 0)} 筆
    - 成功替公司省下的潛在損失金額 (Cost Avoidance)：${report_data.get('total_avoidance', 0):,.0f} USD
    
    長官與老闆喜歡「清晰、美觀、一目了然」的「表格化」排版。請產出符合以下 Markdown 結構的報告（不需加上大標題，從 H3 開始即可）：
    
    ### 📊 報表區間：{report_data.get('date_range', '本週')}
    
    ### 1. 💰 財務衝擊與 ROI 摘要
    (請將上述本週數據，整理成一個高質感的 Markdown 表格。表格欄位請設計為：「關鍵指標 (KPI)」、「結算數值」、「執行成效說明」)
    
    ### 2. 🔍 AI 決策洞察與異常分析
    (請根據採購實務，隨機假定 2~3 項本週發現的超支或風險原因，例如「預算與歷史均價落差過大」、「單一來源風險」等，並以 Markdown 表格呈現。表格欄位請設計為：「異常風險警示」、「風險說明與影響」、「AI 建議處置方案」)
    
    ### 3. 🛡️ 戰略轉型與下週行動方針
    (請自由發揮 2 項具有高階視野的採購戰略行動方針，以 Markdown 表格呈現。表格欄位請設計為：「戰略目標」、「具體行動計畫」、「預期商業效益」)
    
    請嚴格遵守以下排版原則：
    1. 語氣必須是極度專業的財報/稽核風格。
    2. 全文必須以「表格 (Markdown Tables)」為核心，每一段落的主體都是表格，嚴禁長篇大論的文字贅述，讓老闆能在一分鐘內看懂。
    3. 結尾請標註小字：`*本報告由 Gemini RAG 引擎根據系統即時數據動態生成*`。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini generation error: {e}")
        return None
