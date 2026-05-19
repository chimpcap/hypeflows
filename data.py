"""
Data fetchers for the HYPE flows dashboard.

Three sources feed the dashboard:
  1. HYPE price       -> Hyperliquid public API (live, hourly)
  2. PURR holdings    -> data/purr_holdings.csv (hand-maintained from 8-Ks for now)
  3. BHYP/THYP flows  -> data/etf_daily.csv     (hand-maintained from issuer pages / SoSoValue for now)

Replace the CSV-backed loaders with live fetchers as we wire them up.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------- HYPE price (live) -----------------------------------------------

def fetch_hype_price_hourly(lookback_days: int = 30) -> pd.DataFrame:
    """Hourly HYPE/USD candles from Hyperliquid (perp, deepest book).

    Returns columns: ts (UTC), open, high, low, close, volume.
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_days * 24 * 60 * 60 * 1000
    body = {
        "type": "candleSnapshot",
        "req": {"coin": "HYPE", "interval": "1h", "startTime": start_ms, "endTime": end_ms},
    }
    r = requests.post(HL_INFO_URL, json=body, timeout=15)
    r.raise_for_status()
    candles = r.json()
    if not candles:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(candles)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    return df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)


def latest_hype_price(price_df: pd.DataFrame) -> float | None:
    if price_df.empty:
        return None
    return float(price_df["close"].iloc[-1])


def daily_hype_close(price_df: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly candles to daily close (UTC). Columns: date, hype_close."""
    if price_df.empty:
        return pd.DataFrame(columns=["date", "hype_close"])
    daily = (
        price_df.set_index("ts")["close"]
        .resample("1D")
        .last()
        .dropna()
        .reset_index()
        .rename(columns={"ts": "date", "close": "hype_close"})
    )
    daily["date"] = daily["date"].dt.date
    return daily


# ---------- ETF live quote + intraday volume (Yahoo Finance) ----------------

def fetch_etf_quote(ticker: str) -> dict:
    """Latest snapshot for an ETF ticker. Returns a dict with at minimum:
      price, day_volume_shares, day_volume_usd, previous_close, market_state, as_of (UTC).
    Uses Yahoo's public chart endpoint — no auth, no key.
    """
    url = YF_CHART_URL.format(ticker=ticker)
    r = requests.get(url, params={"interval": "1m", "range": "1d"}, headers=YF_HEADERS, timeout=10)
    r.raise_for_status()
    payload = r.json()["chart"]["result"][0]
    meta = payload["meta"]
    price = float(meta["regularMarketPrice"])
    day_shares = float(meta.get("regularMarketVolume") or 0)
    return {
        "ticker": ticker,
        "price": price,
        "day_volume_shares": day_shares,
        "day_volume_usd": day_shares * price,
        "previous_close": float(meta.get("previousClose") or 0),
        "day_high": float(meta.get("regularMarketDayHigh") or 0),
        "day_low": float(meta.get("regularMarketDayLow") or 0),
        "market_state": meta.get("currentTradingPeriod", {}),
        "as_of": pd.Timestamp.now(tz="UTC"),
        "exchange_tz": meta.get("exchangeTimezoneName", "America/New_York"),
        "regular_market_time": pd.to_datetime(meta.get("regularMarketTime"), unit="s", utc=True)
        if meta.get("regularMarketTime")
        else None,
    }


def fetch_etf_intraday(ticker: str) -> pd.DataFrame:
    """One-minute intraday bars for the current trading day.

    Returns columns: ts (UTC), open, high, low, close, volume, dollar_volume, cum_volume_shares,
    cum_volume_usd.
    """
    url = YF_CHART_URL.format(ticker=ticker)
    r = requests.get(url, params={"interval": "1m", "range": "1d"}, headers=YF_HEADERS, timeout=10)
    r.raise_for_status()
    payload = r.json()["chart"]["result"][0]
    ts = payload.get("timestamp") or []
    if not ts:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume", "dollar_volume",
                     "cum_volume_shares", "cum_volume_usd"]
        )
    q = payload["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, unit="s", utc=True),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    ).dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0)
    df["dollar_volume"] = df["close"] * df["volume"]
    df["cum_volume_shares"] = df["volume"].cumsum()
    df["cum_volume_usd"] = df["dollar_volume"].cumsum()
    return df.reset_index(drop=True)


def estimated_inflow_usd(quote: dict, ratio: float) -> float:
    """Apply a flat ratio to today's $ volume to estimate net creation $."""
    return float(quote["day_volume_usd"]) * float(ratio)


def estimated_inflow_hype(quote: dict, ratio: float, hype_price: float) -> float:
    """Convert estimated USD inflow into HYPE token units."""
    if not hype_price:
        return 0.0
    return estimated_inflow_usd(quote, ratio) / float(hype_price)


# ---------- PURR treasury (CSV until SEC/on-chain wired) --------------------

def load_purr_holdings() -> pd.DataFrame:
    """Disclosed PURR HYPE holdings.

    Returns: date, hype_tokens_held, tokens_added (vs previous row), source.
    """
    df = pd.read_csv(DATA_DIR / "purr_holdings.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["tokens_added"] = df["hype_tokens_held"].diff()
    return df


# ---------- BHYP / THYP ETF flows (CSV until issuer pages wired) ------------

def load_etf_daily() -> pd.DataFrame:
    """Daily ETF state.

    Returns: date, fund, net_flow_usd, aum_usd, source.
    `net_flow_usd` may be NaN on rows where only AUM is known.
    """
    df = pd.read_csv(DATA_DIR / "etf_daily.csv", parse_dates=["date"])
    return df.sort_values(["date", "fund"]).reset_index(drop=True)


# ---------- Combined HYPE absorbed -----------------------------------------

def unified_hype_absorbed(
    purr: pd.DataFrame, etf: pd.DataFrame, daily_price: pd.DataFrame
) -> pd.DataFrame:
    """Combine all three sources into a daily HYPE-units-absorbed series.

    Treasury adds are already in tokens. ETF flows are in USD and get converted
    using the day's HYPE close (last available close as a fallback).

    Returns: date, source, hype_added.
    """
    rows: list[dict] = []

    if not purr.empty:
        for _, r in purr.iterrows():
            if pd.notna(r.get("tokens_added")) and r["tokens_added"] != 0:
                rows.append(
                    {"date": r["date"].date(), "source": "PURR", "hype_added": float(r["tokens_added"])}
                )

    if not etf.empty and not daily_price.empty:
        price_lookup = dict(zip(daily_price["date"], daily_price["hype_close"]))
        last_known = daily_price["hype_close"].iloc[-1] if len(daily_price) else None
        for _, r in etf.iterrows():
            if pd.isna(r.get("net_flow_usd")):
                continue
            d = r["date"].date()
            px = price_lookup.get(d, last_known)
            if px is None or px == 0:
                continue
            rows.append(
                {"date": d, "source": r["fund"], "hype_added": float(r["net_flow_usd"]) / float(px)}
            )

    return pd.DataFrame(rows).sort_values(["date", "source"]).reset_index(drop=True)
