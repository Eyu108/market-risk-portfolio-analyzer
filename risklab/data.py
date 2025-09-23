# risklab/data.py — robust loaders: Stooq (no key) → Alpha Vantage (API key) → Yahoo

from __future__ import annotations

import io
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Optional Stooq via pandas-datareader is flaky behind some networks/proxies,
# so we also hit Stooq's CSV endpoint directly with a user-agent.
_STQ_BASES = ("https://stooq.com", "https://stooq.pl")
_STQ_SUFFIXES = (".US", ".PL", ".DE", ".UK", ".FR", ".JP")

def _sanitize_index(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, errors="coerce")
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    return idx

def _series_from_df(df: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype="float64", name=name)
    if isinstance(df, pd.Series):
        s = df.rename(name)
    else:
        # prefer Adj Close -> Close -> adjusted_close (Alpha)
        cols = [c for c in df.columns]
        lower = {str(c).lower(): c for c in cols}
        pick = None
        for key in ("adj close", "adjclose", "close", "adjusted_close"):
            if key in lower:
                pick = lower[key]; break
        if pick is None:
            # Stooq CSV capitalizes "Close"
            if "Close" in df.columns: pick = "Close"
        if pick is None:
            return pd.Series(dtype="float64", name=name)
        s = df[pick].rename(name)
    s.index = _sanitize_index(s.index)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return pd.to_numeric(s, errors="coerce").dropna()

# ---------------------------- STQOOQ (no key) ----------------------------

def _stooq_aliases(t: str, map_tsx_to_us: bool) -> list[str]:
    t0 = t.upper()
    specials = {
        "^DXY": ["DXY", "USDIDX", "USDOLLAR"],
        "CL=F": ["CL", "WTI", "OIL"],
        "^GSPC": ["SPX", "^SPX", "SP500"],
        "^TNX": ["US10Y", "TNX"],
    }
    if t0 in specials: return specials[t0]
    if t0.endswith(".TO") and map_tsx_to_us:
        base = t0.split(".")[0]
        return [base, base + ".US"]
    if t0.endswith(_STQ_SUFFIXES): return [t0]
    if re.fullmatch(r"[A-Z0-9.\-]+", t0) and not any(ch in t0 for ch in ["^","="]):
        return [t0, t0 + ".US"]
    return [t0]

def _stooq_csv(symbol: str, interval: str) -> pd.DataFrame:
    # interval: 1d/1wk/1mo -> i: d/w/m (stooq)
    i = {"1d":"d", "1wk":"w", "1mo":"m"}.get(interval, "d")
    headers = {"User-Agent": "Mozilla/5.0"}
    last_exc = None
    for base in _STQ_BASES:
        url = f"{base}/q/d/l/?s={symbol.lower()}&i={i}"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.ok and r.text and "Date,Open,High,Low,Close" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                return df
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc
    return pd.DataFrame()

def _stooq_fetch_one(t: str, start=None, end=None, interval="1d", map_tsx_to_us=True) -> pd.Series:
    for cand in _stooq_aliases(t, map_tsx_to_us):
        try:
            df = _stooq_csv(cand, interval)
            if not df.empty:
                s = _series_from_df(df, t)
                if start is not None: s = s[s.index >= pd.to_datetime(start)]
                if end is not None:   s = s[s.index <= pd.to_datetime(end)]
                if not s.empty: return s
        except Exception:
            continue
    return pd.Series(dtype="float64", name=t)

# ------------------------- ALPHA VANTAGE (key) --------------------------

def _get_alpha_key(explicit_key: str | None) -> str | None:
    if explicit_key: return explicit_key
    # try Streamlit secrets when running under Streamlit
    try:
        import streamlit as st  # type: ignore
        k = st.secrets.get("ALPHAVANTAGE_API_KEY", None)
        if k: return k
    except Exception:
        pass
    return os.getenv("ALPHAVANTAGE_API_KEY")

def _alpha_function(interval: str) -> str:
    return {
        "1d": "TIME_SERIES_DAILY_ADJUSTED",
        "1wk": "TIME_SERIES_WEEKLY_ADJUSTED",
        "1mo": "TIME_SERIES_MONTHLY_ADJUSTED",
    }.get(interval, "TIME_SERIES_DAILY_ADJUSTED")

def _alpha_fetch_one(t: str, start=None, end=None, interval="1d", api_key: str | None = None) -> pd.Series:
    key = _get_alpha_key(api_key)
    if not key:
        return pd.Series(dtype="float64", name=t)
    func = _alpha_function(interval)
    params = {
        "function": func,
        "symbol": t,
        "apikey": key,
        "outputsize": "full",
        "datatype": "csv",
    }
    try:
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        if not r.ok or "Error Message" in r.text or "Invalid API call" in r.text:
            return pd.Series(dtype="float64", name=t)
        df = pd.read_csv(io.StringIO(r.text))
        # csv columns: timestamp, open, high, low, close, adjusted_close, volume, ...
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        df.index.name = "Date"
        s = _series_from_df(df, t)  # picks adjusted_close
        if start is not None: s = s[s.index >= pd.to_datetime(start)]
        if end is not None:   s = s[s.index <= pd.to_datetime(end)]
        return s
    except Exception:
        return pd.Series(dtype="float64", name=t)

# ------------------------------ YAHOO -----------------------------------

def _yahoo_download(t: str, start=None, end=None, interval="1d", timeout=60) -> pd.Series:
    df = yf.download(
        t, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, group_by="column",
        threads=False, timeout=timeout
    )
    return _series_from_df(df, t)

def _yahoo_history(t: str, start=None, end=None, interval="1d") -> pd.Series:
    df = yf.Ticker(t).history(start=start, end=end, interval=interval, auto_adjust=True, actions=False)
    return _series_from_df(df, t)

# ------------------------------- PUBLIC --------------------------------

def get_prices(
    tickers: list[str],
    start=None,
    end=None,
    interval: str = "1d",
    retries: int = 2,
    pause: float = 1.0,
    timeout: int = 60,
    provider: str = "Auto (Stooq→Alpha→Yahoo)",  # "Stooq (no API key)" | "Alpha Vantage (API key)" | "Yahoo (yfinance)" | "CSV only"
    map_tsx_to_us: bool = True,
    alpha_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch prices for each ticker with selected provider strategy.
    """
    out: list[pd.Series] = []

    for t in dict.fromkeys(tickers):
        s = pd.Series(dtype="float64", name=t)

        if provider.startswith("Stooq"):
            s = _stooq_fetch_one(t, start, end, interval, map_tsx_to_us)

        elif provider.startswith("Alpha"):
            s = _alpha_fetch_one(t, start, end, interval, alpha_key)

        elif provider.startswith("Yahoo"):
            last_exc = None
            for _ in range(retries + 1):
                try:
                    s = _yahoo_download(t, start, end, interval, timeout)
                    if not s.empty: break
                except Exception as e:
                    last_exc = e
                time.sleep(pause)
            if s.empty:
                try:
                    s = _yahoo_history(t, start, end, interval)
                except Exception:
                    pass

        else:  # Auto
            # 1) Stooq direct CSV (no key)
            s = _stooq_fetch_one(t, start, end, interval, map_tsx_to_us)
            # 2) Alpha Vantage (needs key)
            if s.empty:
                s = _alpha_fetch_one(t, start, end, interval, alpha_key)
                if s.empty and interval != "1d":
                    # try daily if weekly/monthly missing
                    s = _alpha_fetch_one(t, start, end, "1d", alpha_key)
            # 3) Yahoo last resort
            if s.empty:
                for _ in range(retries + 1):
                    try:
                        s = _yahoo_download(t, start, end, interval, timeout)
                        if not s.empty: break
                    except Exception:
                        pass
                    time.sleep(pause)
                if s.empty:
                    try:
                        s = _yahoo_history(t, start, end, interval)
                    except Exception:
                        pass

        if s.empty:
            print(f"[WARN] Failed to fetch {t} via provider='{provider}'")
        out.append(s)

    if not out: return pd.DataFrame()

    prices = pd.concat(out, axis=1)
    prices.index = _sanitize_index(prices.index)
    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    return prices.dropna(how="all")

def pct_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if prices is None or prices.empty: return pd.DataFrame()
    prices = prices.sort_index()
    rets = np.log(prices / prices.shift(1)) if method.lower() == "log" else prices.pct_change()
    return rets.dropna(how="all")
