import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# =========================
# Settings (match your template)
# =========================
DROP_LOOKBACK = 500          # lookback window for major high/low
MIN_DROP_PCT = 50.0          # at least 50% drop
MA_LEN = 250                 # MA250
BASE_BARS = 80               # last 80 days base / consolidation
MAX_BASE_HEIGHT_PCT = 22.0   # define "sideways": (base_high-base_low)/close <= 22%
NEAR_HIGH_PCT = 10.0         # within 10% of base high => close >= 0.9 * base_high

# =========================
# Universe (edit this list)
# =========================
TICKERS = [
    "AAPL", "NVDA", "MSFT", "TSLA", "META", "AMZN",
    "AMD", "AES", "GOOGL", "SMCI", "AVGO", "NFLX"
]

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def scan_stock(ticker: str):
    """
    Returns dict if the stock matches template, else None.
    Data source: yfinance daily.
    """
    # Need enough bars: 500 lookback + MA250 + base window
    # We'll fetch ~3y to be safe.
    df = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None

    # Standardize columns (yfinance sometimes returns multiindex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Drop rows with missing close
    df = df.dropna(subset=["Close", "High", "Low"])
    if len(df) < max(DROP_LOOKBACK, MA_LEN, BASE_BARS) + 5:
        return None

    # Indicators
    df["MA250"] = df["Close"].rolling(MA_LEN).mean()

    # Latest values
    close_last = df["Close"].iloc[-1]
    ma_last = df["MA250"].iloc[-1]
    if np.isnan(ma_last) or np.isnan(close_last):
        return None

    # ===== 1) 500-day drawdown >= 50% (use rolling highest/lowest over last 500)
    window = df.tail(DROP_LOOKBACK)
    hi500 = window["High"].max()
    lo500 = window["Low"].min()
    if hi500 <= 0:
        return None

    drop_pct = (hi500 - lo500) / hi500 * 100.0
    big_drop = drop_pct >= MIN_DROP_PCT

    # ===== 2) Above MA250 now (break condition simplified to current above)
    above_ma = close_last > ma_last

    # ===== 3) Base in last 80 bars: sideways range constraint
    base = df.tail(BASE_BARS)
    base_high = base["High"].max()
    base_low = base["Low"].min()
    if close_last <= 0:
        return None

    base_height_pct = (base_high - base_low) / close_last * 100.0
    sideways = base_height_pct <= MAX_BASE_HEIGHT_PCT

    # ===== 4) Higher lows within the base: last half low > first half low
    half = BASE_BARS // 2
    early_low = base["Low"].iloc[:half].min()
    late_low = base["Low"].iloc[half:].min()
    higher_lows = late_low > early_low

    # ===== 5) Within 10% of base high: close >= 90% of base_high
    near_high = close_last >= base_high * (1.0 - NEAR_HIGH_PCT / 100.0)

    if big_drop and above_ma and sideways and higher_lows and near_high:
        return {
            "ticker": ticker,
            "price": _safe_float(close_last),
            "ma250": _safe_float(ma_last),
            "drop_pct_500d": _safe_float(drop_pct),
            "base_high_80d": _safe_float(base_high),
            "base_low_80d": _safe_float(base_low),
            "base_height_pct": _safe_float(base_height_pct),
            "early_low_80d": _safe_float(early_low),
            "late_low_80d": _safe_float(late_low),
        }

    return None

def main():
    results = []

    for t in tqdm(TICKERS, desc="Scanning"):
        try:
            r = scan_stock(t)
            if r:
                results.append(r)
        except Exception as e:
            # Skip broken tickers without killing whole scan
            print(f"[WARN] {t} failed: {e}")

    # Sort: closest to base high first (more "ready")
    def readiness(item):
        # bigger is closer to high => (price/base_high)
        try:
            return item["price"] / item["base_high_80d"]
        except Exception:
            return 0

    results.sort(key=readiness, reverse=True)

    payload = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "rules": {
            "drop_lookback": DROP_LOOKBACK,
            "min_drop_pct": MIN_DROP_PCT,
            "ma_len": MA_LEN,
            "base_bars": BASE_BARS,
            "max_base_height_pct": MAX_BASE_HEIGHT_PCT,
            "near_high_pct": NEAR_HIGH_PCT
        },
        "count": len(results),
        "results": results
    }

    with open("picks.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nScan complete. Found: {len(results)}")
    print("Wrote: picks.json")

if __name__ == "__main__":
    main()
