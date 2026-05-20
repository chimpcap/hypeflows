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
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YF_HEADERS = {"User-Agent": "Mozilla/5.0"}
FARSIDE_URL = "https://farside.co.uk/hyp/"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


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


def fetch_etf_daily_history(ticker: str, range_str: str = "1mo") -> pd.DataFrame:
    """Daily bars since the ETF started trading.

    Returns: date, close, volume_shares, volume_usd.
    """
    r = requests.get(
        YF_CHART_URL.format(ticker=ticker),
        params={"interval": "1d", "range": range_str},
        headers=YF_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()["chart"]["result"][0]
    ts = payload.get("timestamp") or []
    if not ts:
        return pd.DataFrame(columns=["date", "close", "volume_shares", "volume_usd"])
    q = payload["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).date,
            "close": q.get("close"),
            "volume_shares": q.get("volume"),
        }
    ).dropna(subset=["close"])
    df["volume_shares"] = df["volume_shares"].fillna(0)
    df["volume_usd"] = df["close"] * df["volume_shares"]
    return df.reset_index(drop=True)


def fetch_etf_intraday_history(ticker: str, range_str: str = "10d") -> pd.DataFrame:
    """15-minute bars across the past `range_str` (Yahoo supports up to ~60d at 15m).

    Returns: ts, date, close, volume, dollar_volume.
    """
    r = requests.get(
        YF_CHART_URL.format(ticker=ticker),
        params={"interval": "15m", "range": range_str},
        headers=YF_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()["chart"]["result"][0]
    ts = payload.get("timestamp") or []
    if not ts:
        return pd.DataFrame(columns=["ts", "date", "close", "volume", "dollar_volume"])
    q = payload["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, unit="s", utc=True),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    ).dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0)
    df["dollar_volume"] = df["close"] * df["volume"]
    df["date"] = df["ts"].dt.tz_convert("America/New_York").dt.date
    return df.reset_index(drop=True)


def compute_implied_ratios(
    etf_csv: pd.DataFrame, daily_by_ticker: dict[str, pd.DataFrame]
) -> dict:
    """Implied inflow ratio per ETF: cumulative known inflow / cumulative volume since launch.

    Uses latest known AUM as the proxy for cumulative net inflows (valid only when
    price moves are modest, which is the case for these brand-new funds).
    Returns: {ticker: {"ratio": float|None, "inflow_usd": float, "volume_usd": float,
                       "through_date": date, "note": str}, "combined": {...}}.
    """
    out: dict = {}
    total_inflow = 0.0
    total_volume = 0.0
    have_any = False
    for ticker, daily in daily_by_ticker.items():
        rows = etf_csv[etf_csv["fund"] == ticker].sort_values("date")
        if rows.empty or daily.empty:
            out[ticker] = {"ratio": None, "inflow_usd": 0.0, "volume_usd": 0.0,
                           "through_date": None, "note": "no data"}
            continue
        aum_rows = rows.dropna(subset=["aum_usd"])
        inflow_rows = rows.dropna(subset=["net_flow_usd"])
        if not aum_rows.empty:
            latest_aum = float(aum_rows["aum_usd"].iloc[-1])
            through_date = aum_rows["date"].iloc[-1].date()
            note = "cumulative AUM / volume since launch"
        elif not inflow_rows.empty:
            latest_aum = float(inflow_rows["net_flow_usd"].sum())
            through_date = inflow_rows["date"].iloc[-1].date()
            note = "sum of known daily inflows / volume on those days"
        else:
            out[ticker] = {"ratio": None, "inflow_usd": 0.0, "volume_usd": 0.0,
                           "through_date": None, "note": "no inflow data"}
            continue
        vol_total = float(daily[daily["date"] <= through_date]["volume_usd"].sum())
        ratio = (latest_aum / vol_total) if vol_total > 0 else None
        out[ticker] = {"ratio": ratio, "inflow_usd": latest_aum, "volume_usd": vol_total,
                       "through_date": through_date, "note": note}
        if ratio is not None:
            total_inflow += latest_aum
            total_volume += vol_total
            have_any = True
    out["combined"] = {
        "ratio": (total_inflow / total_volume) if (have_any and total_volume > 0) else None,
        "inflow_usd": total_inflow,
        "volume_usd": total_volume,
    }
    return out


FARSIDE_CACHE_PATH = DATA_DIR / "farside_cache.csv"


def _parse_farside_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    if len(tables) < 2:
        raise RuntimeError("Farside table layout changed")
    raw = tables[1].copy()
    raw.columns = ["date_str", "BHYP", "THYP", "total"]
    date_pat = r"^\d{1,2}\s+\w{3}\s+\d{4}$"
    body = raw[raw["date_str"].astype(str).str.match(date_pat, na=False)].copy()
    if body.empty:
        raise RuntimeError("no date rows found in Farside table")
    body["date"] = pd.to_datetime(body["date_str"], format="%d %b %Y").dt.date

    def parse(v: object) -> float | None:
        s = str(v).strip()
        if s in ("-", "", "nan", "NaN"):
            return None
        s = s.replace(",", "")
        try:
            return float(s) * 1e6
        except ValueError:
            return None

    long = body.melt(
        id_vars=["date"], value_vars=["BHYP", "THYP"], var_name="ticker", value_name="raw"
    )
    long["actual_inflow_usd"] = long["raw"].map(parse)
    return long[["date", "ticker", "actual_inflow_usd"]].sort_values(["date", "ticker"]).reset_index(drop=True)


def fetch_farside_flows() -> tuple[pd.DataFrame, str]:
    """Daily net flows for HYPE ETFs from farside.co.uk/hyp/ (USD).

    Tries live Farside first. If that fails (e.g. cloud IP is blocked), falls
    back to data/farside_cache.csv. Always returns a (df, source) tuple so the
    UI can show provenance — source is "live" or "cache: <reason>".

    df columns: date, ticker, actual_inflow_usd (NaN if no flow reported).
    """
    live_err: str | None = None
    try:
        r = requests.get(FARSIDE_URL, headers=BROWSER_HEADERS, timeout=15)
        if r.status_code != 200:
            live_err = f"HTTP {r.status_code} ({len(r.text)} bytes)"
        elif len(r.text) < 1000:
            live_err = f"suspiciously small response ({len(r.text)} bytes)"
        else:
            df = _parse_farside_html(r.text)
            try:
                df.to_csv(FARSIDE_CACHE_PATH, index=False)
            except Exception:
                pass
            return df, "live"
    except Exception as e:
        live_err = f"{type(e).__name__}: {e}"

    if FARSIDE_CACHE_PATH.exists():
        df = pd.read_csv(FARSIDE_CACHE_PATH, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df, f"cache (live failed: {live_err})"
    raise RuntimeError(f"Farside live fetch failed and no cache file: {live_err}")


def build_actuals_table(
    farside: pd.DataFrame,
    daily_by_ticker: dict[str, pd.DataFrame],
    slider_ratio: float,
) -> pd.DataFrame:
    """Join Farside actual flows with Yahoo daily volume + slider-based estimate.

    Returns: date, ticker, actual_inflow_usd, day_volume_usd, actual_ratio,
             estimated_at_slider_usd, diff_usd.
    """
    if farside.empty:
        return pd.DataFrame()
    rows = []
    daily_lookup = {
        t: {row["date"]: row["volume_usd"] for _, row in df.iterrows()}
        for t, df in daily_by_ticker.items()
    }
    for _, r in farside.iterrows():
        t = r["ticker"]
        vol = daily_lookup.get(t, {}).get(r["date"])
        actual = r["actual_inflow_usd"]
        est = float(vol) * float(slider_ratio) if vol else None
        ratio = (float(actual) / float(vol)) if (vol and actual is not None) else None
        rows.append(
            {
                "date": r["date"],
                "ticker": t,
                "actual_inflow_usd": actual,
                "day_volume_usd": float(vol) if vol else None,
                "actual_ratio": ratio,
                "estimated_at_slider_usd": est,
                "diff_usd": (float(actual) - est) if (actual is not None and est is not None) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def market_clock(now_utc: pd.Timestamp | None = None) -> dict:
    """US equity market clock (NYSE/Nasdaq core session).

    Honest scope: handles weekends and the 09:30–16:00 ET window. Does NOT know
    about US holidays — if you want bullet-proof state on holidays, swap to
    pandas_market_calendars later.
    """
    from zoneinfo import ZoneInfo
    from datetime import datetime, time, timedelta

    et = ZoneInfo("America/New_York")
    now = (now_utc or pd.Timestamp.now(tz="UTC")).tz_convert(et).to_pydatetime()
    open_t, close_t = time(9, 30), time(16, 0)

    def next_open_after(d: datetime) -> datetime:
        cursor = datetime.combine(d.date(), open_t, tzinfo=et)
        if d.time() >= open_t:
            cursor = cursor + timedelta(days=1)
        while cursor.weekday() >= 5:  # 5=Sat, 6=Sun
            cursor = cursor + timedelta(days=1)
        return cursor

    weekday = now.weekday() < 5
    if weekday and open_t <= now.time() < close_t:
        close_dt = datetime.combine(now.date(), close_t, tzinfo=et)
        return {
            "state": "open",
            "message": "Market is OPEN",
            "until": close_dt,
            "seconds_until": int((close_dt - now).total_seconds()),
            "label_until": "closes",
        }
    nxt = next_open_after(now)
    return {
        "state": "premarket" if weekday and now.time() < open_t else "closed",
        "message": "Market is CLOSED",
        "until": nxt,
        "seconds_until": int((nxt - now).total_seconds()),
        "label_until": "opens",
    }


def fmt_countdown(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def historical_cumulative_inflows(
    intraday_by_ticker: dict[str, pd.DataFrame],
    etf_csv: pd.DataFrame,
    daily_by_ticker: dict[str, pd.DataFrame],
    fallback_ratio: float,
) -> pd.DataFrame:
    """Per-bar cumulative inflow estimate across the launch-to-now window.

    For each (ticker, date):
      ratio_for_day = known_day_inflow / known_day_volume   if disclosed in etf_csv
                    = fallback_ratio                        otherwise
    Per 15-min bar: estimated_inflow = bar_dollar_volume * ratio_for_day.
    Cumulated per ticker.

    Returns: ts, ticker, bar_inflow_usd, cum_inflow_usd.
    """
    frames = []
    for ticker, bars in intraday_by_ticker.items():
        if bars.empty:
            continue
        daily = daily_by_ticker.get(ticker, pd.DataFrame())
        known = etf_csv[(etf_csv["fund"] == ticker) & etf_csv["net_flow_usd"].notna()]
        day_ratios: dict = {}
        if not known.empty and not daily.empty:
            daily_lookup = {row["date"]: row["volume_usd"] for _, row in daily.iterrows()}
            for _, r in known.iterrows():
                d = r["date"].date()
                vol = daily_lookup.get(d)
                if vol and vol > 0:
                    day_ratios[d] = float(r["net_flow_usd"]) / float(vol)
        df = bars.copy()
        df["ratio"] = df["date"].map(day_ratios).fillna(fallback_ratio)
        df["bar_inflow_usd"] = df["dollar_volume"] * df["ratio"]
        df["cum_inflow_usd"] = df["bar_inflow_usd"].cumsum()
        df["ticker"] = ticker
        frames.append(df[["ts", "ticker", "bar_inflow_usd", "cum_inflow_usd"]])
    if not frames:
        return pd.DataFrame(columns=["ts", "ticker", "bar_inflow_usd", "cum_inflow_usd"])
    return pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)


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
