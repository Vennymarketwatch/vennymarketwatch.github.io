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
# Your template rules
# =========================
DROP_LOOKBACK = 500
MIN_DROP_PCT = 50.0
MA_LEN = 250
BASE_BARS = 80
MAX_BASE_HEIGHT_PCT = 22.0
NEAR_HIGH_PCT = 10.0

# =========================
# Your filters (speed + quality)
# =========================
MIN_MARKET_CAP = 1_000_000_000   # $1B
MIN_PRICE = 5.0                 # > $5
MIN_DOLLAR_VOL = 20_000_000      # last day Close*Volume > $20M
EXCLUDE_REIT = True

# =========================
# Universe sources: S&P1500 + Russell3000 via ETF holdings
# =========================
SPTM_HOLDINGS_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-sptm.xlsx"  #  [oai_citation:2‡State Street Global Advisors](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-portfolio-sp-1500-composite-stock-market-etf-sptm)
IWV_HOLDINGS_CSV = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?dataType=fund&fileName=IWV_holdings&fileType=csv"  #  [oai_citation:3‡BlackRock](https://www.ishares.com/us/products/239714/ishares-russell-3000-etf)

# =========================
# Practical controls
# =========================
BATCH_QUOTES = 200        # download last-6-days quotes in batches (fast prefilter)
SLEEP_EVERY = 10          # sleep every N tickers during per-ticker marketcap calls
SLEEP_SEC = 1.0
MAX_TICKERS = None        # set e.g. 800 to test; None = full

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def _safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def _clean_symbol(sym: str) -> str:
    sym = str(sym).strip().upper()
    return sym

def _is_excluded_symbol(sym: str) -> bool:
    """
    Exclude obvious non-common-stock suffixes typically used in US tickers.
    (Not perfect, but helps filter preferred/warrants/units/rights etc.)
    """
    s = sym
    if not s or "^" in s or "/" in s:
        return True

    # Common patterns:
    # -W (warrants) sometimes like ABCW or ABC+?; many warrants end with W
    # -U (units), -R (rights) appear as suffixes in holdings sometimes as "XXXXU" or "XXXXR"
    # Preferred often like "ABC-P" or "ABC.PR" but holdings may normalize.
    bad_suffixes = ("W", "U", "R")
    if len(s) >= 2 and s[-1] in bad_suffixes:
        return True

    # Dots often indicate preferred/classes; keep class A/B like BRK.B (yfinance uses BRK-B)
    # We'll normalize '.' -> '-' later; don't exclude just for dot.
    return False

def normalize_for_yf(sym: str) -> str:
    # Yahoo Finance uses '-' for class shares (BRK-B) instead of BRK.B
    return sym.replace(".", "-")

def fetch_sptm_symbols():
    r = session.get(SPTM_HOLDINGS_XLSX, timeout=60)
    r.raise_for_status()
    xls = pd.ExcelFile(BytesIO(r.content))
    # Usually the first sheet contains holdings
    df = xls.parse(xls.sheet_names[0])

    # Try to find a symbol column
    sym_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"ticker", "symbol", "holding ticker", "holding"}:
            sym_col = c
            break
    if sym_col is None:
        # fallback: find any column containing "Ticker" or "Symbol"
        for c in df.columns:
            name = str(c).lower()
            if "ticker" in name or "symbol" in name:
                sym_col = c
                break
    if sym_col is None:
        raise ValueError("SPTM holdings file: cannot find ticker/symbol column.")

    # Optional sector column
    sector_col = None
    for c in df.columns:
        if "sector" in str(c).lower():
            sector_col = c
            break

    syms = []
    for _, row in df.iterrows():
        sym = row.get(sym_col, None)
        if pd.isna(sym):
            continue
        sym = _clean_symbol(sym)
        if _is_excluded_symbol(sym):
            continue

        if EXCLUDE_REIT and sector_col is not None:
            sec = str(row.get(sector_col, "")).strip().lower()
            if "real estate" in sec:
                continue

        syms.append(normalize_for_yf(sym))
    return sorted(set(syms))

def fetch_iwv_symbols():
    r = session.get(IWV_HOLDINGS_CSV, timeout=60)
    r.raise_for_status()
    # iShares holdings CSV includes some header lines before the table; pandas can handle with comment-like junk poorly
    text = r.text
    # Find the first line that starts with "Ticker"
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Ticker,"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("IWV holdings CSV: cannot find table header.")

    csv_text = "\n".join(lines[start_idx:])
    df = pd.read_csv(StringIO(csv_text))

    # Basic cleanup
    df = df.dropna(subset=["Ticker"])
    # Optional sector column
    sector_col = "Sector" if "Sector" in df.columns else None
    asset_col = "Asset Class" if "Asset Class" in df.columns else None

    syms = []
    for _, row in df.iterrows():
        if asset_col and str(row.get(asset_col, "")).strip().lower() not in {"equity", "common stock", "stock"}:
            # skip futures/cash/other
            continue

        sym = _clean_symbol(row["Ticker"])
        if _is_excluded_symbol(sym):
            continue

        if EXCLUDE_REIT and sector_col:
            sec = str(row.get(sector_col, "")).strip().lower()
            if "real estate" in sec:
                continue

        syms.append(normalize_for_yf(sym))
    return sorted(set(syms))

def get_universe():
    sptm = fetch_sptm_symbols()
    iwv = fetch_iwv_symbols()
    uni = sorted(set(sptm) | set(iwv))
    if MAX_TICKERS:
        uni = uni[:MAX_TICKERS]
    return uni

def batch_prefilter_price_dvol(tickers):
    """
    Fast prefilter using only last ~6 days data in batches:
    Keep those with last Close > $5 and last Close*Volume > $20M.
    """
    keep = []
    for i in range(0, len(tickers), BATCH_QUOTES):
        batch = tickers[i:i + BATCH_QUOTES]
        try:
            df = yf.download(batch, period="6d", interval="1d", group_by="ticker", auto_adjust=False, progress=False)
        except Exception:
            continue

        # Single ticker returns normal columns
        if isinstance(df.columns, pd.MultiIndex):
            for t in batch:
                if (t, "Close") not in df.columns or (t, "Volume") not in df.columns:
                    continue
                close = df[(t, "Close")].dropna()
                vol = df[(t, "Volume")].dropna()
                if close.empty or vol.empty:
                    continue
                c = float(close.iloc[-1])
                v = float(vol.iloc[-1])
                if c >= MIN_PRICE and c * v >= MIN_DOLLAR_VOL:
                    keep.append(t)
        else:
            # Rare case: yfinance returns flat columns if only 1 ticker
            close = df["Close"].dropna()
            vol = df["Volume"].dropna()
            if not close.empty and not vol.empty:
                c = float(close.iloc[-1])
                v = float(vol.iloc[-1])
                if c >= MIN_PRICE and c * v >= MIN_DOLLAR_VOL:
                    keep.extend(batch)

    return sorted(set(keep))

def get_market_cap_fast(ticker):
    """
    market cap via fast_info; fallback to info (slower).
    """
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and "marketCap" in fi and fi["marketCap"]:
            return float(fi["marketCap"])
    except Exception:
        pass

    try:
        t = yf.Ticker(ticker)
        info = t.get_info()
        mc = info.get("marketCap")
        return float(mc) if mc else None
    except Exception:
        return None

def scan_shape(ticker):
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    if len(df) < max(DROP_LOOKBACK, MA_LEN, BASE_BARS) + 5:
        return None

    close_last = float(df["Close"].iloc[-1])
    vol_last = float(df["Volume"].iloc[-1])
    dollar_vol = close_last * vol_last

    df["MA250"] = df["Close"].rolling(MA_LEN).mean()
    ma_last = df["MA250"].iloc[-1]
    if np.isnan(ma_last):
        return None

    # 1) 500d drawdown >= 50%
    window = df.tail(DROP_LOOKBACK)
    hi500 = float(window["High"].max())
    lo500 = float(window["Low"].min())
    if hi500 <= 0:
        return None
    drop_pct = (hi500 - lo500) / hi500 * 100.0
    big_drop = drop_pct >= MIN_DROP_PCT

    # 2) Above MA250 now
    above_ma = close_last > float(ma_last)

    # 3) 80d base sideways
    base = df.tail(BASE_BARS)
    base_high = float(base["High"].max())
    base_low = float(base["Low"].min())
    base_height_pct = (base_high - base_low) / close_last * 100.0
    sideways = base_height_pct <= MAX_BASE_HEIGHT_PCT

    # 4) Higher lows within base (second half low > first half low)
    half = BASE_BARS // 2
    early_low = float(base["Low"].iloc[:half].min())
    late_low = float(base["Low"].iloc[half:].min())
    higher_lows = late_low > early_low

    # 5) within 10% of base high
    near_high = close_last >= base_high * (1.0 - NEAR_HIGH_PCT / 100.0)

    if big_drop and above_ma and sideways and higher_lows and near_high:
        return {
            "ticker": ticker,
            "price": _safe_float(close_last),
            "dollar_vol": _safe_float(dollar_vol),
            "ma250": _safe_float(ma_last),
            "drop_pct_500d": _safe_float(drop_pct),
            "base_high_80d": _safe_float(base_high),
            "base_low_80d": _safe_float(base_low),
            "base_height_pct": _safe_float(base_height_pct),
        }

    return None

def main():
    universe = get_universe()
    print(f"Universe size (SPTM + IWV, deduped): {len(universe)}")

    # Stage 1: cheap prefilter
    pre = batch_prefilter_price_dvol(universe)
    print(f"After price>${MIN_PRICE} & $vol>${MIN_DOLLAR_VOL/1e6:.0f}M: {len(pre)}")

    # Stage 2: market cap filter (per ticker)
    cap_ok = []
    for i, t in enumerate(tqdm(pre, desc="Market cap filter")):
        mc = get_market_cap_fast(t)
        if mc is not None and mc >= MIN_MARKET_CAP:
            cap_ok.append(t)

        if SLEEP_EVERY and (i + 1) % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SEC)

    print(f"After market cap>${MIN_MARKET_CAP/1e9:.0f}B: {len(cap_ok)}")

    # Stage 3: expensive shape scan
    results = []
    for t in tqdm(cap_ok, desc="Shape scan"):
        try:
            r = scan_shape(t)
            if r:
                # attach market cap (so your website shows it later if you want)
                r["market_cap"] = _safe_float(get_market_cap_fast(t))
                results.append(r)
        except Exception:
            pass

    # Sort: closest to base high first
    def readiness(item):
        try:
            return (item["price"] / item["base_high_80d"])
        except Exception:
            return 0

    results.sort(key=readiness, reverse=True)

    payload = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "universe": "SPTM (S&P1500 proxy) + IWV (Russell3000 proxy)",
        "filters": {
            "min_market_cap": MIN_MARKET_CAP,
            "min_price": MIN_PRICE,
            "min_dollar_vol": MIN_DOLLAR_VOL,
            "exclude_reit": EXCLUDE_REIT,
            "exclude_etf_preferred_warrants_rights_units": True
        },
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
