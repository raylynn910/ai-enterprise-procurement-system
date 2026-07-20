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
不編造分數; 名稱解析失敗一律拒絕, 絕不猜測 (防幽靈公司原則)。

assess_company() 回傳 (result, err_msg, err_kind):
    err_kind = "not_found" (查無此公司/無財報) 或
               "upstream"  (外部資料源暫時性故障, 應重試)
"""

import json
import logging
import os
import re
import sqlite3
import time

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# 路徑以本檔案位置錨定 — 不依賴啟動時的工作目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DB_PATH = os.path.join(BASE_DIR, "procurement.db")

CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60
REGISTRY_EXPIRY_SECONDS = 30 * 24 * 60 * 60
REGISTRY_PARTIAL_EXPIRY_SECONDS = 10 * 60   # 名冊只抓到一部分時, 10 分鐘後重試
REGISTRY_FAIL_BACKOFF_SECONDS = 60          # 全部失敗時的退避, 避免每個請求都卡 timeout

# 官方公司名冊: 上市 (TWSE, 中文欄位) → .TW; 上櫃 (TPEx, 英文欄位) → .TWO
REGISTRY_SOURCES = [
    ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW",
     ("公司代號", "公司簡稱", "公司名稱")),
    ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO",
     ("SecuritiesCompanyCode", "CompanyAbbreviation", "CompanyName")),
]

ZONES = [
    (2.6, "Safe", "安全區: 財務體質穩健, 短期倒閉風險低"),
    (1.1, "Grey", "灰色地帶: 體質偏弱, 建議加強付款條件與監控"),
    (float("-inf"), "Distress", "危險區: 具財務危機特徵, 建議人工盡職調查後再往來"),
]

_registry_cache = {"data": None, "ts": 0.0, "complete": False, "fail_ts": 0.0}
_CJK = re.compile(r"[一-鿿]")
_TICKER_SHAPE = re.compile(r"^[A-Za-z0-9-]{1,10}\.[A-Za-z]{1,4}$")  # 2330.TW / BRK-B.DE


def _init_cache():
    try:
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
    except Exception as e:
        # import 時的快取初始化失敗不應擊殺整個 app; 之後讀寫各自再容錯
        logger.warning(f"altman cache init failed (non-fatal): {e}")


_init_cache()


def _load_tw_registry():
    """載入台灣上市/上櫃公司名冊。

    快取策略: 兩來源皆成功 → 30 天; 只成功一部分 → 10 分鐘
    (避免把殘缺名冊鎖一個月, 造成上櫃公司整月解析失敗);
    全部失敗 → 60 秒退避, 期間不再重試 (避免每請求都卡 timeout)。
    """
    now = time.time()
    if _registry_cache["data"]:
        ttl = REGISTRY_EXPIRY_SECONDS if _registry_cache["complete"] else REGISTRY_PARTIAL_EXPIRY_SECONDS
        if now - _registry_cache["ts"] < ttl:
            return _registry_cache["data"]
    if now - _registry_cache["fail_ts"] < REGISTRY_FAIL_BACKOFF_SECONDS:
        return _registry_cache["data"] or []

    rows, ok_sources = [], 0
    for url, suffix, (k_code, k_short, k_full) in REGISTRY_SOURCES:
        try:
            resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            for r in resp.json():
                code = str(r.get(k_code, "")).strip()
                if code:
                    rows.append((code, str(r.get(k_short, "")).strip(),
                                 str(r.get(k_full, "")).strip(), suffix))
            ok_sources += 1
        except Exception as e:
            logger.warning(f"registry fetch failed ({url}): {e}")

    if rows:
        _registry_cache.update(data=rows, ts=now,
                               complete=(ok_sources == len(REGISTRY_SOURCES)))
    else:
        _registry_cache["fail_ts"] = now
    return rows


def _yahoo_fallback(q: str):
    """英文名稱的 Yahoo 搜尋 fallback — 名稱驗證後才採用。

    防幽靈公司: Yahoo 會把「不存在幽靈公司xyz」剝成 xyz 亂配一檔
    美股。要求查詢字串以「完整單詞」出現在配對結果的公司名稱中
    (bare substring 會讓 'ON' 匹配到任何含 on 的名字), 否則寧可
    回報找不到。
    """
    try:
        res = requests.get(
            f"https://query2.finance.yahoo.com/v1/finance/search?q={q}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=3.0)
        res.raise_for_status()
        quotes = [x for x in res.json().get("quotes", [])
                  if x.get("quoteType") == "EQUITY"]
        quotes.sort(key=lambda x: 0 if str(x.get("symbol", "")).endswith((".TW", ".TWO")) else 1)
        word = re.compile(rf"\b{re.escape(q.lower())}\b")
        for x in quotes:
            names = f"{x.get('shortname', '')} {x.get('longname', '')}".lower()
            if word.search(names):
                return x.get("symbol")
    except Exception as e:
        logger.warning(f"yahoo fallback failed for {q}: {e}")
    return None


def _resolve_ticker(query: str):
    """公司名稱/代號 → (ticker | None, err_msg | None)。

    誤配比配不到更危險 (會回報別家公司的財務風險), 因此:
    - 模糊查詢命中多家 → 拒絕並要求更精確, 絕不取第一筆
    - 中文查不到官方名冊 → 直接回報, 不落入 Yahoo 亂猜
    """
    q = query.strip()
    if _TICKER_SHAPE.fullmatch(q):          # 已是完整代號 (2330.TW)
        return q, None

    registry = _load_tw_registry()
    if re.fullmatch(r"\d{4,6}[A-Z]?", q):   # 純數字股號
        for code, _, _, suffix in registry:
            if code == q:
                return q + suffix, None
        if registry and _registry_cache["complete"]:
            return None, f"股號 {q} 不在上市/上櫃名冊中"
        # 名冊缺漏 (如上櫃來源暫時抓不到) → 先猜上市, 呼叫端會再以 .TWO 重試
        return q + ".TW", None

    for code, short, full, suffix in registry:      # 1) 簡稱完全相符
        if q == short or q == full:
            return code + suffix, None
    hits = [(code, short, suffix) for code, short, full, suffix in registry
            if q in short or q in full]             # 2) 部分相符 → 必須唯一
    if len(hits) == 1:
        return hits[0][0] + hits[0][2], None
    if len(hits) > 1:
        examples = "、".join(f"{s}({c})" for c, s, _ in hits[:5])
        return None, (f"「{q}」過於籠統, 匹配到 {len(hits)} 家公司 (如: {examples}...), "
                      f"請輸入完整公司簡稱或股號")
    if _CJK.search(q):
        hint = ("" if _registry_cache["complete"]
                else " (註: 上櫃名冊暫時無法取得, 若為上櫃公司請改用股號查詢)")
        return None, f"「{q}」不在台股上市/上櫃名冊中 (未上市公司無公開財報){hint}"
    sym = _yahoo_fallback(q)
    if sym:
        return sym, None
    return None, f"找不到「{q}」對應的股票代號 (未上市公司無公開財報)"


def _extract_period(df, labels_map):
    """從財報 DataFrame 取「同一期」的所有科目, 不跨年度混用。

    依欄位由新到舊逐期嘗試: 該期所有必要科目皆非空才採用。
    回傳 (values_dict, period_date) 或 (None, None)。
    """
    for col in df.columns:
        vals = {}
        for name, labels in labels_map.items():
            v = None
            for label in labels:
                if label in df.index:
                    cell = df.loc[label, col]
                    if cell is not None and cell == cell:  # 非 NaN
                        v = float(cell)
                        break
            if v is None:
                break
            vals[name] = v
        else:
            return vals, col
    return None, None


def _compute_altman(ticker_symbol: str):
    tk = yf.Ticker(ticker_symbol)
    bs = tk.balance_sheet
    ist = tk.income_stmt
    if bs is None or bs.empty or ist is None or ist.empty:
        return None, "查無財報資料 (可能未上市或資料源無涵蓋)", "not_found"

    bs_vals, bs_date = _extract_period(bs, {
        "total_assets": ["Total Assets"],
        "current_assets": ["Current Assets"],
        "current_liab": ["Current Liabilities"],
        "total_liab": ["Total Liabilities Net Minority Interest", "Total Liab"],
        "retained": ["Retained Earnings"],
        "equity": ["Stockholders Equity", "Common Stock Equity",
                   "Total Equity Gross Minority Interest"],
    })
    if bs_vals is None:
        return None, "資產負債表科目不完整, 無法計算 Z-score", "not_found"

    ist_vals, ist_date = _extract_period(ist, {
        "ebit": ["EBIT", "Operating Income", "Pretax Income"],
    })
    if ist_vals is None:
        return None, "損益表缺少 EBIT/營業利益, 無法計算 Z-score", "not_found"

    if not bs_vals["total_assets"] or not bs_vals["total_liab"]:
        return None, "總資產/總負債為零, 無法計算比率", "not_found"

    x1 = (bs_vals["current_assets"] - bs_vals["current_liab"]) / bs_vals["total_assets"]
    x2 = bs_vals["retained"] / bs_vals["total_assets"]
    x3 = ist_vals["ebit"] / bs_vals["total_assets"]
    x4 = bs_vals["equity"] / bs_vals["total_liab"]
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    for cut, zone, desc in ZONES:
        if z > cut or zone == "Distress":
            zone_name, zone_desc = zone, desc
            break

    result = {
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
        "statement_date": str(bs_date.date()) if hasattr(bs_date, "date") else str(bs_date),
    }
    if hasattr(bs_date, "year") and hasattr(ist_date, "year") and bs_date.year != ist_date.year:
        result["period_note"] = (f"注意: 資產負債表 ({bs_date.date()}) 與損益表 "
                                 f"({ist_date.date()}) 取自不同年度期別")
    return result, None, None


def assess_company(company_name: str):
    """主入口: 公司名稱或股票代號 → Altman Z'' 評估。

    回傳 (result, err_msg, err_kind); 成功時後兩者為 None。
    err_kind: "not_found" → 404 | "upstream" → 502 (暫時性, 可重試)
    """
    # 1. 快取
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        row = conn.execute(
            "SELECT data_json, timestamp FROM altman_cache WHERE company_name = ?",
            (company_name,)).fetchone()
        conn.close()
        if row and time.time() - row[1] < CACHE_EXPIRY_SECONDS:
            return json.loads(row[0]), None, None
    except Exception as e:
        logger.warning(f"altman cache read error: {e}")

    # 2. 解析代號 (誤配即拒絕)
    symbol, err = _resolve_ticker(company_name)
    if not symbol:
        return None, err, "not_found"

    try:
        result, err, kind = _compute_altman(symbol)
        # 純數字股號在名冊缺漏時先猜了 .TW; 查無財報就再試上櫃 .TWO
        if err and kind == "not_found" and symbol.endswith(".TW") \
                and re.fullmatch(r"\d{4,6}[A-Z]?", company_name.strip()):
            result, err, kind = _compute_altman(company_name.strip() + ".TWO")
    except Exception as e:
        logger.warning(f"altman compute error for {symbol}: {e}")
        return None, "外部財報資料源暫時無法使用, 請稍後重試", "upstream"
    if err:
        return None, err, kind

    result["query"] = company_name
    # 3. 寫快取 (只快取成功結果)
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO altman_cache VALUES (?, ?, ?)",
            (company_name, json.dumps(result, ensure_ascii=False), time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"altman cache write error: {e}")
    return result, None, None
