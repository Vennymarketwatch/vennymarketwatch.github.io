#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
US Stock Pattern Scanner (EOD, cached)

Your updated model (as requested):
- Keep everything unchanged unless explicitly mentioned.

Changes applied:
5) Refined to:
   In last 500 bars:
   - drop from highest close to lowest close >= 30%
   - AND there exists at least one day where price is simultaneously below EMA20, EMA60, EMA120, EMA250
6) Removed (no more "trough early in window")
7) Changed to: last 100 trading days cannot break the 500-day window low
9) Changed to: current price must be above EMA60, EMA120, EMA250
10) Changed to: within last 150 trading days, must have at least one cross up above EMA250
11) Removed (no base range/sideways constraint)
12) Changed to: distance to EMA250 < 30% (instead of 15%)
13) Removed
14) Removed
15) Changed to: current price must be BELOW the recent 90-day high (no 90% threshold)

Universe:
- SPTM + IWV holdings (proxy for S&P 1500 + Russell 3000-ish)
- Clean tickers and exclude obvious non-common instruments by heuristics

Filters (unchanged):
- price > $5
- last day $ volume > $20M
- market cap > $1B (best effort via yfinance, cached)

Outputs:
- picks.json (payload with results)

Cache:
- data_cache/<TICKER>.pkl  (history)
- data_cache/meta_<TICKER>.json (market cap)
"""

import os
import re
import io
import json
import time
import contextlib
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm


# -----------------------
# Config
# -----------------------
HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Model parameters (updated per your request)
WINDOW_BARS = 500
MIN_DROP_PCT = 40.0                 # from highest close to lowest close in window
RECENT_NO_BREAK_LOW_BARS = 100      # last 100 bars cannot break window low
EMA_CROSS_LOOKBACK = 150            # last 150 bars must have a cross up above EMA250
EMA_DISTANCE_MAX_PCT = 30.0         # |close - EMA250| / EMA250 < 30%
RECENT_HIGH_LOOKBACK = 120           # close must be below recent 90-day high

# EMA lengths
EMA20 = 20
EMA60 = 60
EMA120 = 120
EMA250 = 250

# Liquidity / size filters (unchanged)
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 20_000_000        # last close * last volume
MIN_MKTCAP = 1_000_000_000         # 1B

# Data fetch (unchanged)
HIST_PERIOD = "3y"
HIST_INTERVAL = "1d"
REQUEST_TIMEOUT = 30

# Universe sources (unchanged)
SPTM_HOLDINGS_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-sptm.xlsx"
IWV_HOLDINGS_URL  = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"


# -----------------------
# Utils
# -----------------------
def safe_float(x) -> Optional[float]:
    try:
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return None
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        return None

def normalize_ticker_for_yahoo(t: str) -> str:
    t = str(t).strip().upper()
    t = t.replace(".", "-")  # BRK.B -> BRK-B
    return t

def is_plausible_us_common_ticker(t: str) -> bool:
    t = t.strip().upper()
    if not t:
        return False
    if len(t) > 10:
        return False
    if any(ch in t for ch in [" ", "/", "+", "&", "(", ")", ",", ":", "$"]):
        return False
    if t in {"-", "N/A", "NA", "NULL"}:
        return False
    return bool(re.match(r"^[A-Z0-9]{1,6}(-[A-Z0-9]{1,3})?$", t))

def exclude_special_instruments(t: str) -> bool:
    """
    Best-effort exclusions: preferred/warrants/rights/units.
    (You asked to exclude ETF/Preferred/Warrants/Rights/Units — ETFs should not be in holdings universe, but we keep this heuristic.)
    """
    tt = t.upper()

    # Preferred / warrants / rights / units
    if re.search(r"-P[A-Z]?$", tt):
        return True
    if re.search(r"-W(S)?$", tt):
        return True
    if re.search(r"-R$", tt):
        return True
    if re.search(r"-U$", tt):
        return True

    return False

def _suppress_yf_noise():
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())


# -----------------------
# Universe
# -----------------------
def fetch_holdings_tickers_sptm() -> List[str]:
    r = requests.get(SPTM_HOLDINGS_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))

    tickers: List[str] = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df is None or df.empty:
            continue

        col_name = None
        for c in df.columns:
            cl = str(c).strip().lower()
            if "ticker" in cl or "symbol" in cl:
                col_name = c
                break
        if col_name is None:
            continue

        vals = df[col_name].dropna().astype(str).tolist()
        tickers.extend(vals)

        if len(tickers) > 500:
            break

    return tickers

def fetch_holdings_tickers_iwv() -> List[str]:
    r = requests.get(IWV_HOLDINGS_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    text = r.text

    df = None
    sym_col = None
    for skip in range(0, 25):
        try:
            tmp = pd.read_csv(io.StringIO(text), skiprows=skip)
            if tmp is None or tmp.empty:
                continue
            for c in tmp.columns:
                cl = str(c).strip().lower()
                if cl in ("ticker", "symbol"):
                    df = tmp
                    sym_col = c
                    break
            if df is not None:
                break
        except Exception:
            continue

    if df is None or sym_col is None:
        return []
    return df[sym_col].dropna().astype(str).tolist()

def get_universe() -> List[str]:
    raw: List[str] = []

    try:
        raw.extend(fetch_holdings_tickers_sptm())
    except Exception as e:
        print("WARN: SPTM holdings fetch failed:", e)

    try:
        raw.extend(fetch_holdings_tickers_iwv())
    except Exception as e:
        print("WARN: IWV holdings fetch failed:", e)

    cleaned: List[str] = []
    seen = set()

    for t in raw:
        t = normalize_ticker_for_yahoo(t)
        if not is_plausible_us_common_ticker(t):
            continue
        if exclude_special_instruments(t):
            continue
        if t in seen:
            continue
        seen.add(t)
        cleaned.append(t)

    return cleaned


# -----------------------
# Cache + Data
# -----------------------
def cache_path_price(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.pkl")

def load_cached_history(ticker: str) -> Optional[pd.DataFrame]:
    p = cache_path_price(ticker)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_pickle(p)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        return None
    return None

def save_cached_history(ticker: str, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(cache_path_price(ticker))
    except Exception:
        pass

def fetch_history_cached(ticker: str) -> Optional[pd.DataFrame]:
    df = load_cached_history(ticker)
    if df is not None:
        return df

    with _suppress_yf_noise()[0], _suppress_yf_noise()[1]:
        try:
            df = yf.download(
                ticker,
                period=HIST_PERIOD,
                interval=HIST_INTERVAL,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception:
            return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        return None

    save_cached_history(ticker, df)
    return df

def get_marketcap_best_effort(ticker: str) -> Optional[float]:
    meta_path = os.path.join(CACHE_DIR, f"meta_{ticker}.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            mc = meta.get("marketCap")
            if isinstance(mc, (int, float)) and mc > 0:
                return float(mc)
        except Exception:
            pass

    mc = None
    with _suppress_yf_noise()[0], _suppress_yf_noise()[1]:
        try:
            tk = yf.Ticker(ticker)
            fi = getattr(tk, "fast_info", None)
            if fi and isinstance(fi, dict):
                mc = fi.get("market_cap") or fi.get("marketCap")
            if not mc:
                info = tk.info
                mc = info.get("marketCap")
        except Exception:
            mc = None

    if mc and isinstance(mc, (int, float)) and mc > 0:
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"marketCap": float(mc), "updated": time.time()}, f)
        except Exception:
            pass
        return float(mc)

    return None


# -----------------------
# Pattern logic (UPDATED)
# -----------------------
def scan_shape(ticker: str) -> Optional[Dict]:
    df = fetch_history_cached(ticker)
    if df is None or df.empty:
        return None

    close_last = safe_float(df["Close"].iloc[-1])
    vol_last = safe_float(df["Volume"].iloc[-1])
    if close_last is None or vol_last is None:
        return None

    # Basic filters (unchanged)
    if close_last < MIN_PRICE:
        return None
    dollar_vol = close_last * vol_last
    if dollar_vol < MIN_DOLLAR_VOL:
        return None

    # Need enough bars
    need = max(WINDOW_BARS, EMA250, RECENT_NO_BREAK_LOW_BARS, EMA_CROSS_LOOKBACK, RECENT_HIGH_LOOKBACK) + 30
    if len(df) < need:
        return None

    # Market cap (unchanged; best effort)
    mc = get_marketcap_best_effort(ticker)
    if mc is not None and mc < MIN_MKTCAP:
        return None

    # Compute EMAs on full df
    ema20 = df["Close"].ewm(span=EMA20, adjust=False).mean()
    ema60 = df["Close"].ewm(span=EMA60, adjust=False).mean()
    ema120 = df["Close"].ewm(span=EMA120, adjust=False).mean()
    ema250 = df["Close"].ewm(span=EMA250, adjust=False).mean()

    ema60_last = safe_float(ema60.iloc[-1])
    ema120_last = safe_float(ema120.iloc[-1])
    ema250_last = safe_float(ema250.iloc[-1])
    if ema60_last is None or ema120_last is None or ema250_last is None or ema250_last <= 0:
        return None

    # -------------------------
    # 5) 500-day window: drop >= 30% from highest close to lowest close
    #    AND there exists a day where price simultaneously below EMA20/60/120/250
    # -------------------------
    window = df.tail(WINDOW_BARS).copy()

    w_high = safe_float(window["Close"].max())
    w_low = safe_float(window["Close"].min())
    if w_high is None or w_low is None or w_high <= 0:
        return None

    drop_pct = (w_high - w_low) / w_high * 100.0
    if drop_pct < MIN_DROP_PCT:
        return None

    # Condition: close simultaneously below all EMAs at least once in window
    ema20_w = ema20.tail(WINDOW_BARS)
    ema60_w = ema60.tail(WINDOW_BARS)
    ema120_w = ema120.tail(WINDOW_BARS)
    ema250_w = ema250.tail(WINDOW_BARS)

    bear = (
    (window["Close"] < ema20_w) &
    (window["Close"] < ema60_w) &
    (window["Close"] < ema120_w) &
    (window["Close"] < ema250_w)
)

# NEW: 在500天窗口内，至少出现过连续5个交易日同时低于所有EMA
    max_consecutive_bear = safe_float(bear.rolling(7).sum().max())
    if max_consecutive_bear is None or max_consecutive_bear < 7:
        return None

    # -------------------------
    # 7) Last 100 bars cannot break the window low
    # -------------------------
    recent100_min = safe_float(df.tail(RECENT_NO_BREAK_LOW_BARS)["Low"].min())
    if recent100_min is None:
        return None
    if recent100_min < w_low:
        return None

    # -------------------------
    # 9) Current price above EMA60/120/250
    # -------------------------
    if close_last <= ema60_last or close_last <= ema120_last or close_last <= ema250_last:
        return None

    # -------------------------
    # 10) Last 150 bars must have a cross up above EMA250
    # -------------------------
    recent = df.tail(EMA_CROSS_LOOKBACK)
    ema250_r = ema250.tail(EMA_CROSS_LOOKBACK)

    cross_up = (
        (recent["Close"].shift(1) < ema250_r.shift(1)) &
        (recent["Close"] > ema250_r)
    )
    if not bool(cross_up.any()):
        return None

    # -------------------------
    # 12) Distance to EMA250 < 30%
    # -------------------------
    dist_pct = abs(close_last - ema250_last) / ema250_last * 100.0
    if dist_pct > EMA_DISTANCE_MAX_PCT:
        return None

    # -------------------------
    # 15) Current price BELOW recent 90-day high (no 90% threshold)
    # -------------------------
    high120 = safe_float(df.tail(RECENT_HIGH_LOOKBACK)["Close"].max())
    if high120 is None or high120 <= 0:
        return None
    if close_last >= high120:
        return None

    # If passed all filters, return result
    return {
        "ticker": ticker,
        "close": round(close_last, 2),
        "high120": round(high120, 2),
        "drop500_pct": round(drop_pct, 2),
        "ema60": round(ema60_last, 2),
        "ema120": round(ema120_last, 2),
        "ema250": round(ema250_last, 2),
        "ema250_dist_pct": round(dist_pct, 2),
        "dollar_vol": int(dollar_vol),
        "marketCap": int(mc) if mc else None,
    }


# -----------------------
# Main
# -----------------------
def main():
    # best-effort raise open-file limit
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 8192), hard), hard))
    except Exception:
        pass

    universe = get_universe()
    print(f"Universe (clean): {len(universe)}")

    results: List[Dict] = []
    failed = 0

    for t in tqdm(universe, desc="Scanning (cached)", unit="stk"):
        try:
            r = scan_shape(t)
            if r:
                results.append(r)
        except Exception:
            failed += 1
            continue

    # Sort: closest to high90 first, then dollar volume
    def sort_key(x: Dict) -> tuple:
        high120 = x.get("high90") or 0
        close = x.get("close") or 0
        ratio = (close / high90) if high120 else 0
        return (ratio, x.get("dollar_vol", 0))

    results.sort(key=sort_key, reverse=True)

    payload = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "universe_size": len(universe),
        "found": len(results),
        "failed": failed,
        "params": {
            "WINDOW_BARS": WINDOW_BARS,
            "MIN_DROP_PCT": MIN_DROP_PCT,
            "RECENT_NO_BREAK_LOW_BARS": RECENT_NO_BREAK_LOW_BARS,
            "EMA_CROSS_LOOKBACK": EMA_CROSS_LOOKBACK,
            "EMA_DISTANCE_MAX_PCT": EMA_DISTANCE_MAX_PCT,
            "RECENT_HIGH_LOOKBACK": RECENT_HIGH_LOOKBACK,
            "MIN_PRICE": MIN_PRICE,
            "MIN_DOLLAR_VOL": MIN_DOLLAR_VOL,
            "MIN_MKTCAP": MIN_MKTCAP,
            "EMA": [EMA20, EMA60, EMA120, EMA250],
        },
        "results": results,
    }

    with open("picks.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nScan complete. Found:", len(results))
    print("Wrote: picks.json")
    print("Cache folder:", CACHE_DIR)


if __name__ == "__main__":
    main()
