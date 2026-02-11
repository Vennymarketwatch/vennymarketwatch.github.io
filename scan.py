import json
import time
from datetime import datetime, timezone
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

# =========================
# Template rules
# =========================
DROP_LOOKBACK = 500
MIN_DROP_PCT = 50.0
MA_LEN = 250
BASE_BARS = 80
MAX_BASE_HEIGHT_PCT = 22.0
NEAR_HIGH_PCT = 10.0

# Filters
MIN_MARKET_CAP = 1_000_000_000
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 20_000_000
EXCLUDE_REIT = True

# Universe sources
SPTM_HOLDINGS_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-sptm.xlsx"
IWV_HOLDINGS_CSV = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?dataType=fund&fileName=IWV_holdings&fileType=csv"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def _safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except:
        return None

def normalize(sym):
    return str(sym).strip().upper().replace(".", "-")

def fetch_sptm_symbols():
    r = session.get(SPTM_HOLDINGS_XLSX)
    xls = pd.ExcelFile(BytesIO(r.content))
    sheet = xls.sheet_names[0]

    raw = pd.read_excel(BytesIO(r.content), sheet_name=sheet, header=None)

    header_row = None
    for i in range(50):
        row = raw.iloc[i].astype(str).str.lower().tolist()
        if any("ticker" in v or "symbol" in v for v in row):
            header_row = i
            break

    df = pd.read_excel(BytesIO(r.content), sheet_name=sheet, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    sym_col = [c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower()][0]

    syms = []
    for s in df[sym_col].dropna():
        sym = normalize(s)
        if EXCLUDE_REIT and "REIT" in sym:
            continue
        syms.append(sym)

    return sorted(set(syms))

def fetch_iwv_symbols():
    r = session.get(IWV_HOLDINGS_CSV)
    text = r.text
    start = [i for i,l in enumerate(text.splitlines()) if l.startswith("Ticker")][0]
    df = pd.read_csv(StringIO("\n".join(text.splitlines()[start:])))

    syms = []
    for s in df["Ticker"].dropna():
        sym = normalize(s)
        syms.append(sym)

    return sorted(set(syms))

def get_universe():
    sptm = fetch_sptm_symbols()
    iwv = fetch_iwv_symbols()
    return sorted(set(sptm) | set(iwv))

def get_market_cap(t):
    try:
        return yf.Ticker(t).fast_info["marketCap"]
    except:
        return None

def scan_shape(ticker):
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if df.empty:
        return None

    close = df["Close"].iloc[-1]
    vol = df["Volume"].iloc[-1]

    if close < MIN_PRICE or close * vol < MIN_DOLLAR_VOL:
        return None

    mc = get_market_cap(ticker)
    if mc is None or mc < MIN_MARKET_CAP:
        return None

    df["MA250"] = df["Close"].rolling(250).mean()
    if close <= df["MA250"].iloc[-1]:
        return None

    window = df.tail(DROP_LOOKBACK)
    hi = window["High"].max()
    lo = window["Low"].min()
    drop = (hi - lo) / hi * 100

    if drop < MIN_DROP_PCT:
        return None

    base = df.tail(BASE_BARS)
    bh = base["High"].max()
    bl = base["Low"].min()

    if (bh - bl) / close * 100 > MAX_BASE_HEIGHT_PCT:
        return None

    early = base["Low"].iloc[:40].min()
    late = base["Low"].iloc[40:].min()

    if late <= early:
        return None

    if close < bh * (1 - NEAR_HIGH_PCT/100):
        return None

    return {
        "ticker": ticker,
        "price": _safe_float(close),
        "drop_pct": _safe_float(drop),
        "base_high": _safe_float(bh),
        "market_cap": _safe_float(mc)
    }

def main():
    universe = get_universe()
    print("Universe:", len(universe))

    results = []
    for t in tqdm(universe):
        r = scan_shape(t)
        if r:
            results.append(r)

    payload = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "count": len(results),
        "results": results
    }

    with open("picks.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Scan complete. Found:", len(results))

if __name__ == "__main__":
    main()
