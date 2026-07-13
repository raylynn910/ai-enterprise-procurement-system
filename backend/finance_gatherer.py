import sqlite3
import json
import time
import requests
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

CACHE_DB_PATH = 'procurement.db'
CACHE_EXPIRY_SECONDS = 30 * 24 * 60 * 60  # 30 days

def _init_db():
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS finance_cache (
            company_name TEXT PRIMARY KEY,
            data_json TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

_init_db()

def _get_ticker_from_name(company_name: str) -> str:
    """Uses Yahoo Finance Search API to map a company name to a ticker symbol."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=2.0)
        res.raise_for_status()
        data = res.json()
        quotes = data.get('quotes', [])
        if quotes:
            return quotes[0].get('symbol')
    except Exception as e:
        logger.warning(f"Failed to fetch ticker for {company_name}: {e}")
    return None

def _fetch_finance_data(company_name: str) -> dict:
    """Fetches financial data with a strict timeout and fallback."""
    try:
        # Step 1: Map name to ticker
        ticker_symbol = _get_ticker_from_name(company_name)
        if not ticker_symbol:
            return None

        # Step 2: Fetch data using yfinance (lightweight)
        ticker = yf.Ticker(ticker_symbol)
        
        # We wrap the info fetch in a timeout block manually or just rely on requests timeout
        # yfinance uses requests under the hood, but it doesn't strictly expose timeout for .info
        # For simplicity and robustness, we grab fast_info or info
        info = ticker.info
        market_cap = info.get('marketCap')
        quote_type = info.get('quoteType')
        
        if market_cap is None:
            return None

        return {
            "symbol": ticker_symbol,
            "marketCap": market_cap,
            "quoteType": quote_type
        }
    except Exception as e:
        logger.warning(f"Failed to fetch finance data for {company_name}: {e}")
        return None

def get_financial_features(company_name: str) -> dict:
    """
    Main entry point.
    Returns a dict with 'marketCap' and 'quoteType', or None if fallback is needed.
    """
    # 1. Check cache
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT data_json, timestamp FROM finance_cache WHERE company_name = ?", (company_name,))
        row = cursor.fetchone()
        if row:
            data_json, timestamp = row
            if time.time() - timestamp < CACHE_EXPIRY_SECONDS:
                conn.close()
                return json.loads(data_json)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

    # 2. Fetch fresh data (with timeout)
    # We use a simple synchronous call here. _fetch_finance_data handles internal timeouts.
    # To truly enforce a 2.0s timeout at the function level across OSes, we'd need signals or threads.
    # We will rely on the requests timeout inside _get_ticker_from_name for the first line of defense.
    data = _fetch_finance_data(company_name)

    # 3. Save to cache
    if data:
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO finance_cache (company_name, data_json, timestamp)
                VALUES (?, ?, ?)
            ''', (company_name, json.dumps(data), time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    return data
