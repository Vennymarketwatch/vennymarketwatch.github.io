#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
US Stock Pattern Scanner (EOD, cached)

Your model (latest):
1) Last 500 bars max drawdown >= 50% (time-ordered peak -> trough)
2) Trough must occur in first 60% of that 500-bar window (avoid trend stocks)
3) Recent 200d must NOT break the old trough (allow 2% wiggle): recent_min >= window_min * 0.98
4) Recovered from trough by at least +30%
5) Price is above MA250 (structure recovery)
6) Base (last 80 bars) is sideways: range <= 30% of median close
7) Base has higher lows: 2nd half min low > 1st half min low
8) MA20 rising: MA20 today > MA20 5 bars ago
9) Near breakout: close within 10% of the recent 50-day high (close >= high_50d * 0.90)
10) Break above MA250 must have happened within last 120 bars (breakout-then-base model)
11) Base should be near MA250 (avoid trend continuation): |median(base) - MA250| / MA250 <= 15%

Universe:
- SPTM + IWV holdings (proxy for S&P 1500 + Russell 3000-ish)
- Clean tickers and exclude obvious non-common instruments by heuristics
- Extra cleaning to drop weird tokens

Filters:
- US exchanges by holdings (implicit)
- price > $5
- last day $ volume > $20M
- market cap > $1B (best effort via yfinance, cached)

Outputs:
- picks.json

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

# Model parameters
DROP_LOOKBACK = 500
MIN_MAX_DRAWDOWN_PCT = 50.0
TROUGH_MUST_BE_EARLY_FRAC = 0.60

RECOVERY_MIN_FROM_TROUGH = 1.30

MA_LONG = 250
MA_SHORT = 20
MA_SHORT_SLOPE_LAG = 5  # compare to 5 bars ago

BASE_BARS = 80
MAX_BASE_RANGE_PCT = 30.0

NEAR_HIGH_LOOKBACK = 50
NEAR_HIGH_PCT = 0.90

CROSS_LOOKBACK = 120

BASE_NEAR_MA250_MAX_DIST_PCT = 15.0

# Liquidity / size filters
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 20_000_000        # last close * last volume
MIN_MKTCAP = 1_000_000_000         # 1B

# Data fetch
HIST_PERIOD = "3y"
HIST_INTERVAL = "1d"
REQUEST_TIMEOUT = 30

# Universe sources
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
    Best-effort exclusions: preferred/warrants/rights/units + leveraged names
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

    # Some holdings files put weird words, reject obvious non-tickers already by regex.

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

        # usually enough; avoid parsing too many sheets
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

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        return None

    save_cached_history(ticker, df)
    return df

def get_marketcap_best_effort(ticker: str) -> Optional[float]:
    meta_path = os.path.join(CACHE_DIR, f"meta_{ticker}.json")

    # cache
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
# Pattern logic
# -----------------------
def max_drawdown_pct(close: pd.Series) -> Tuple[float, int, int]:
    """
    Time-ordered max drawdown:
    dd = 1 - close / cummax(close)
    returns: (max_dd_pct, peak_idx, trough_idx)
    """
    close = close.dropna()
    if len(close) < 50:
        return 0.0, -1, -1

    roll_max = close.cummax()
    dd = 1.0 - (close / roll_max)
    trough_i = int(np.argmax(dd.values))
    max_dd = float(dd.iloc[trough_i] * 100.0)

    peak_i = int(np.argmax(close.iloc[: trough_i + 1].values)) if trough_i >= 0 else -1
    return max_dd, peak_i, trough_i

def scan_shape(ticker: str) -> Optional[Dict]:
    df = fetch_history_cached(ticker)
    if df is None or df.empty:
        return None

    close_last = safe_float(df["Close"].iloc[-1])
    vol_last = safe_float(df["Volume"].iloc[-1])
    if close_last is None or vol_last is None:
        return None

    # Basic filters
    if close_last < MIN_PRICE:
        return None
    dollar_vol = close_last * vol_last
    if dollar_vol < MIN_DOLLAR_VOL:
        return None

    # Need enough bars
    need = max(DROP_LOOKBACK, MA_LONG, BASE_BARS, NEAR_HIGH_LOOKBACK, CROSS_LOOKBACK) + 30
    if len(df) < need:
        return None

    # Market cap (best effort)
    mc = get_marketcap_best_effort(ticker)
    if mc is not None and mc < MIN_MKTCAP:
        return None

    # 1) Big drop: max drawdown in last DROP_LOOKBACK (time-ordered)
    window = df.tail(DROP_LOOKBACK).copy()
    max_dd, peak_i, trough_i = max_drawdown_pct(window["Close"])
    if max_dd < MIN_MAX_DRAWDOWN_PCT:
        return None

    # --- 深跌后修复过滤（防趋势股） ---
    # trough must be early in window
    if trough_i > int(len(window) * TROUGH_MUST_BE_EARLY_FRAC):
        return None

    # recent 200d must not undercut old trough (allow 2% wiggle)
    recent_min = safe_float(df.tail(200)["Low"].min())
    window_min = safe_float(window["Low"].min())
    if recent_min is None or window_min is None:
        return None
    if recent_min < window_min * 0.98:
        return None

    trough_close = safe_float(window["Close"].iloc[trough_i])
    if trough_close is None or trough_close <= 0:
        return None

    # 2) Recovery from trough
    if close_last < trough_close * RECOVERY_MIN_FROM_TROUGH:
        return None
    # --- 必须经历过深熊：价格曾跌破 MA250 - 20% ---
    ma250_window = window["Close"].rolling(MA_LONG).mean()
    bear_zone = window["Close"] < ma250_window * 0.8
    if not bear_zone.any():
        return None

    # 3) Above MA250
    ma250 = df["Close"].rolling(MA_LONG).mean()
    ma250_last = safe_float(ma250.iloc[-1])
    if ma250_last is None:
        return None
    if close_last <= ma250_last:
        return None

    # 10) Cross above MA250 must occur within last CROSS_LOOKBACK
    recent = df.tail(CROSS_LOOKBACK).copy()
    ma250_recent = ma250.tail(CROSS_LOOKBACK)

    cross_up = ((recent["Close"].shift(1) < ma250_recent.shift(1)) &
                (recent["Close"] > ma250_recent)).any()
    if not bool(cross_up):
        return None

    # 4) Base tightness over last BASE_BARS
    base = df.tail(BASE_BARS).copy()
    base_high = safe_float(base["High"].max())
    base_low = safe_float(base["Low"].min())
    base_mid = safe_float(base["Close"].median())
    if base_high is None or base_low is None or base_mid is None or base_mid <= 0:
        return None

    base_range_pct = (base_high - base_low) / base_mid * 100.0
    if base_range_pct > MAX_BASE_RANGE_PCT:
        return None

    # 11) base near MA250 (avoid trend continuation far above MA250)
    distance_pct = abs(base_mid - ma250_last) / ma250_last * 100.0
    if distance_pct > BASE_NEAR_MA250_MAX_DIST_PCT:
        return None

    # 5) Higher lows within base
    first_half_min = safe_float(base["Low"].iloc[: BASE_BARS // 2].min())
    second_half_min = safe_float(base["Low"].iloc[BASE_BARS // 2 :].min())
    if first_half_min is None or second_half_min is None:
        return None
    if second_half_min <= first_half_min:
        return None

    # 6) MA20 rising
    ma20 = df["Close"].rolling(MA_SHORT).mean()
    ma20_last = safe_float(ma20.iloc[-1])
    ma20_prev = safe_float(ma20.iloc[-(MA_SHORT_SLOPE_LAG + 1)])  # 5 bars ago
    if ma20_last is None or ma20_prev is None:
        return None
    if ma20_last <= ma20_prev:
        return None

    # 7) Near recent 50-day high
    prior_high = safe_float(df.tail(NEAR_HIGH_LOOKBACK)["Close"].max())
    if prior_high is None or prior_high <= 0:
        return None
    if close_last < prior_high * NEAR_HIGH_PCT:
        return None

    return {
        "ticker": ticker,
        "close": round(close_last, 2),
        "high_50d": round(prior_high, 2),
        "ma250": round(ma250_last, 2),
        "ma20": round(ma20_last, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "base_range_pct": round(base_range_pct, 2),
        "base_ma250_dist_pct": round(distance_pct, 2),
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

    # Optional pre-filter pass: keep only tickers that have cached history or can be fetched quickly
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

    # Sort: closest to 50d high first, then dollar volume
    def sort_key(x: Dict) -> Tuple[float, float]:
        prior = x.get("high_50d") or 0
        close = x.get("close") or 0
        ratio = (close / prior) if prior else 0
        return (ratio, x.get("dollar_vol", 0))

    results.sort(key=sort_key, reverse=True)

    payload = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "universe_size": len(universe),
        "found": len(results),
        "failed": failed,
        "params": {
            "DROP_LOOKBACK": DROP_LOOKBACK,
            "MIN_MAX_DRAWDOWN_PCT": MIN_MAX_DRAWDOWN_PCT,
            "TROUGH_MUST_BE_EARLY_FRAC": TROUGH_MUST_BE_EARLY_FRAC,
            "RECOVERY_MIN_FROM_TROUGH": RECOVERY_MIN_FROM_TROUGH,
            "MA_LONG": MA_LONG,
            "MA_SHORT": MA_SHORT,
            "MA_SHORT_SLOPE_LAG": MA_SHORT_SLOPE_LAG,
            "BASE_BARS": BASE_BARS,
            "MAX_BASE_RANGE_PCT": MAX_BASE_RANGE_PCT,
            "NEAR_HIGH_LOOKBACK": NEAR_HIGH_LOOKBACK,
            "NEAR_HIGH_PCT": NEAR_HIGH_PCT,
            "CROSS_LOOKBACK": CROSS_LOOKBACK,
            "BASE_NEAR_MA250_MAX_DIST_PCT": BASE_NEAR_MA250_MAX_DIST_PCT,
            "MIN_PRICE": MIN_PRICE,
            "MIN_DOLLAR_VOL": MIN_DOLLAR_VOL,
            "MIN_MKTCAP": MIN_MKTCAP,
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
