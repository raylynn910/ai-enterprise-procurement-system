# -*- coding: utf-8 -*-
"""即時「公司名稱 → 財務風險」評估引擎 (Altman Z''-score)。

為什麼用 Altman Z'' 而不是 ML 模型直接吃即時財報:
UCI 訓練資料的 95 個比率已被原提供者 min-max 正規化 (原始
min/max 未公開), 即時抓到的原始財報數字無法映射進該特徵空間。
Altman Z''-score (1995, 新興市場版) 是學術與實務通用的破產預警
公式, 直接以原始財報計算, 完全可解釋:

    Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4
    X1 = 營運資金/總資產      X2 = 保留盈餘/總資產
    X3 = EBIT/總資產          X4 = 帳面淨值/總負債
    區間: Z'' > 2.6 安全 | 1.1–2.6 灰色 | < 1.1 危險

資料來源: yfinance 年報 (上市櫃公司)。查無財報時回報原因,
不編造分數。快取沿用 finance_cache 模式 (7 天)。
"""

import json
import logging
import re
import sqlite3
import time

import requests
import yfinance as yf

from finance_gatherer import _get_ticker_from_name, CACHE_DB_PATH

logger = logging.getLogger(__name__)
CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60
REGISTRY_EXPIRY_SECONDS = 30 * 24 * 60 * 60

# 官方公司名冊: 上市 (TWSE, 中文欄位) → .TW; 上櫃 (TPEx, 英文欄位) → .TWO
REGISTRY_SOURCES = [
    ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW",
     ("公司代號", "公司簡稱", "公司名稱")),
    ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO",
     ("SecuritiesCompanyCode", "CompanyAbbreviation", "CompanyName")),
]

_registry_cache = {"data": None, "ts": 0.0}


def _load_tw_registry():
    """載入台灣上市/上櫃公司名冊 (記憶體快取 30 天)。

    回傳 list[(代號, 簡稱, 全名, ticker後綴)]。
    Yahoo 搜尋 API 對中文查詢會回 400, 因此中文名稱一律走
    官方名冊匹配, 也保證對應到台股本尊 (不會抓到海外存託憑證)。
    """
    if _registry_cache["data"] and time.time() - _registry_cache["ts"] < REGISTRY_EXPIRY_SECONDS:
        return _registry_cache["data"]
    rows = []
    for url, suffix, (k_code, k_short, k_full) in REGISTRY_SOURCES:
        try:
            resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            for r in resp.json():
                code = str(r.get(k_code, "")).strip()
                if code:
                    rows.append((code, str(r.get(k_short, "")).strip(),
                                 str(r.get(k_full, "")).strip(), suffix))
        except Exception as e:
            logger.warning(f"registry fetch failed ({url}): {e}")
    if rows:
        _registry_cache.update(data=rows, ts=time.time())
    return rows


_CJK = re.compile(r"[一-鿿]")


def _yahoo_fallback(q: str):
    """英文名稱的 Yahoo 搜尋 fallback — 必須通過名稱驗證才採用。

    防幽靈公司: Yahoo 會把「不存在幽靈公司xyz」剝成 xyz 亂配一檔
    美股, 因此要求查詢字串必須出現在配對結果的公司名稱中,
    否則寧可回報找不到 (與階段九 OSINT 防呆同一原則)。
    """
    try:
        res = requests.get(
            f"https://query2.finance.yahoo.com/v1/finance/search?q={q}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=3.0)
        res.raise_for_status()
        quotes = [x for x in res.json().get("quotes", [])
                  if x.get("quoteType") == "EQUITY"]
        # 台股掛牌優先
        quotes.sort(key=lambda x: 0 if str(x.get("symbol", "")).endswith((".TW", ".TWO")) else 1)
        ql = q.lower()
        for x in quotes:
            names = f"{x.get('shortname','')} {x.get('longname','')}".lower()
            if ql in names:
                return x.get("symbol")
    except Exception as e:
        logger.warning(f"yahoo fallback failed for {q}: {e}")
    return None


def _resolve_ticker(query: str):
    """公司名稱/代號 → yfinance ticker。官方名冊優先, Yahoo 驗證後墊底。"""
    q = query.strip()
    if "." in q:                        # 已是完整代號 (2330.TW)
        return q
    registry = _load_tw_registry()
    if re.fullmatch(r"\d{4,6}[A-Z]?", q):   # 純數字股號 → 查上市/上櫃後綴
        for code, _, _, suffix in registry:
            if code == q:
                return q + suffix
        return q + ".TW"                # 名冊拿不到時預設上市
    for code, short, full, suffix in registry:      # 1) 簡稱完全相符
        if q == short:
            return code + suffix
    for code, short, full, suffix in registry:      # 2) 簡稱/全名包含
        if q in short or q in full:
            return code + suffix
    if _CJK.search(q):
        return None                     # 中文查不到名冊 = 非台股上市櫃, 不亂猜
    return _yahoo_fallback(q)           # 英文外國公司 (含名稱驗證)

ZONES = [
    (2.6, "Safe", "安全區: 財務體質穩健, 短期倒閉風險低"),
    (1.1, "Grey", "灰色地帶: 體質偏弱, 建議加強付款條件與監控"),
    (float("-inf"), "Distress", "危險區: 具財務危機特徵, 建議人工盡職調查後再往來"),
]


def _init_cache():
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS altman_cache (
            company_name TEXT PRIMARY KEY,
            data_json TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    conn.close()


_init_cache()


def _first_available(df, labels):
    """從財報 DataFrame 依序找第一個存在且非空的科目 (最新一期)。"""
    for label in labels:
        if label in df.index:
            series = df.loc[label].dropna()
            if len(series):
                return float(series.iloc[0])
    return None


def _compute_altman(ticker_symbol: str):
    tk = yf.Ticker(ticker_symbol)
    bs = tk.balance_sheet
    ist = tk.income_stmt
    if bs is None or bs.empty or ist is None or ist.empty:
        return None, "查無財報資料 (可能未上市或資料源無涵蓋)"

    total_assets = _first_available(bs, ["Total Assets"])
    current_assets = _first_available(bs, ["Current Assets"])
    current_liab = _first_available(bs, ["Current Liabilities"])
    total_liab = _first_available(
        bs, ["Total Liabilities Net Minority Interest", "Total Liab"])
    retained = _first_available(bs, ["Retained Earnings"])
    equity = _first_available(
        bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    ebit = _first_available(ist, ["EBIT", "Operating Income", "Pretax Income"])

    missing = [n for n, v in [
        ("總資產", total_assets), ("流動資產", current_assets),
        ("流動負債", current_liab), ("總負債", total_liab),
        ("保留盈餘", retained), ("股東權益", equity), ("EBIT", ebit),
    ] if v is None]
    if missing or not total_assets or not total_liab:
        return None, f"財報科目不完整, 缺: {', '.join(missing) or '總資產/總負債為零'}"

    x1 = (current_assets - current_liab) / total_assets
    x2 = retained / total_assets
    x3 = ebit / total_assets
    x4 = equity / total_liab
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    for cut, zone, desc in ZONES:
        if z > cut or zone == "Distress":
            zone_name, zone_desc = zone, desc
            break

    return {
        "ticker": ticker_symbol,
        "z_score": round(z, 2),
        "zone": zone_name,
        "zone_description": zone_desc,
        "components": {
            "X1 營運資金/總資產": round(x1, 4),
            "X2 保留盈餘/總資產": round(x2, 4),
            "X3 EBIT/總資產": round(x3, 4),
            "X4 淨值/總負債": round(x4, 4),
        },
        "statement_date": str(bs.columns[0].date()) if len(bs.columns) else None,
    }, None


def assess_company(company_name: str):
    """主入口: 公司名稱或股票代號 → Altman Z'' 評估。

    回傳 (result_dict, error_message); 成功時 error 為 None。
    """
    # 1. 快取
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        row = conn.execute(
            "SELECT data_json, timestamp FROM altman_cache WHERE company_name = ?",
            (company_name,)).fetchone()
        conn.close()
        if row and time.time() - row[1] < CACHE_EXPIRY_SECONDS:
            return json.loads(row[0]), None
    except Exception as e:
        logger.warning(f"altman cache read error: {e}")

    # 2. 解析代號: 官方名冊優先 (支援中文簡稱/全名/股號), Yahoo 搜尋墊後
    symbol = _resolve_ticker(company_name)
    if not symbol:
        return None, f"找不到「{company_name}」對應的股票代號 (未上市公司無公開財報)"

    try:
        result, err = _compute_altman(symbol)
    except Exception as e:
        logger.warning(f"altman compute error for {symbol}: {e}")
        return None, f"財報抓取失敗: {e}"
    if err:
        return None, err

    result["query"] = company_name
    # 3. 寫快取
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO altman_cache VALUES (?, ?, ?)",
            (company_name, json.dumps(result, ensure_ascii=False), time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"altman cache write error: {e}")
    return result, None
