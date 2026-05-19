"""HYPE Flows Dashboard — volume-based ETF inflow estimator.

Estimated daily ETF inflow = today's USD volume * net-inflow ratio (slider, 25%-35%).
The live tiles auto-refresh every ~15s without any page interaction.

Run:
    python -m streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as src

PRICE_TTL = 3600
VOLUME_TTL = 10
FRAGMENT_REFRESH = "15s"
ETF_TICKERS = ("BHYP", "THYP")


# ---------- cached fetchers ----------

@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def get_hype_price(days: int) -> pd.DataFrame:
    return src.fetch_hype_price_hourly(lookback_days=days)


@st.cache_data(ttl=VOLUME_TTL, show_spinner=False)
def get_etf_quote(ticker: str) -> dict:
    return src.fetch_etf_quote(ticker)


@st.cache_data(ttl=VOLUME_TTL, show_spinner=False)
def get_etf_intraday(ticker: str) -> pd.DataFrame:
    return src.fetch_etf_intraday(ticker)


@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def get_purr() -> pd.DataFrame:
    return src.load_purr_holdings()


# ---------- page ----------

st.set_page_config(page_title="HYPE Flows", page_icon="📈", layout="wide")

st.title("HYPE Flows Dashboard")
st.caption(
    "Estimated **HYPE ETF inflows** for **BHYP** and **THYP**, "
    "computed as today's USD trading volume × your inflow ratio. "
    "Tiles refresh automatically — no reload needed."
)

ratio_pct = st.slider(
    "Estimated net-inflow ratio (% of $ volume that represents new creations)",
    min_value=25, max_value=35, value=30, step=1,
    help="Rule-of-thumb proxy: of all the dollars that traded in the ETF today, "
         "this share is assumed to be net new creations (inflow). Tune as we learn.",
)
ratio = ratio_pct / 100.0

st.divider()


# ---------- live fragment (auto-refresh) ----------

@st.fragment(run_every=FRAGMENT_REFRESH)
def live_block(ratio: float) -> None:
    quotes = {}
    intraday = {}
    errors = []
    for tkr in ETF_TICKERS:
        try:
            quotes[tkr] = get_etf_quote(tkr)
            intraday[tkr] = get_etf_intraday(tkr)
        except Exception as e:
            errors.append(f"{tkr}: {e}")
    if errors:
        st.error("Yahoo fetch issue → " + " | ".join(errors))
    if not quotes:
        return

    try:
        price_df = get_hype_price(30)
        hype_px = src.latest_hype_price(price_df)
    except Exception:
        hype_px = None

    total_usd_inflow = sum(src.estimated_inflow_usd(q, ratio) for q in quotes.values())
    total_hype_inflow = (total_usd_inflow / hype_px) if hype_px else 0.0
    total_usd_volume = sum(q["day_volume_usd"] for q in quotes.values())

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("HYPE spot", f"${hype_px:,.2f}" if hype_px else "—")
    h2.metric("Combined $ volume", f"${total_usd_volume/1e6:,.2f}M")
    h3.metric(
        f"Est. inflow @ {int(round(ratio*100))}%",
        f"${total_usd_inflow/1e6:,.2f}M",
    )
    h4.metric(
        "Est. HYPE absorbed",
        f"{total_hype_inflow/1e3:,.1f}k" if hype_px else "—",
        help=f"= ${total_usd_inflow/1e6:,.2f}M ÷ ${hype_px:,.2f}" if hype_px else None,
    )

    st.caption(
        f"As of {datetime.now(timezone.utc):%H:%M:%S UTC}  ·  "
        f"BHYP last quote {quotes['BHYP']['regular_market_time']:%H:%M UTC}  ·  "
        f"THYP last quote {quotes['THYP']['regular_market_time']:%H:%M UTC}"
    )

    st.write("")

    cols = st.columns(2)
    for col, tkr in zip(cols, ETF_TICKERS):
        q = quotes[tkr]
        est_usd = src.estimated_inflow_usd(q, ratio)
        est_hype = (est_usd / hype_px) if hype_px else 0.0
        change = (q["price"] - q["previous_close"]) / q["previous_close"] * 100 if q["previous_close"] else 0
        with col:
            st.subheader(tkr)
            t1, t2 = st.columns(2)
            t1.metric("Price", f"${q['price']:,.2f}", f"{change:+.2f}% vs prev close")
            t2.metric("Volume (shares)", f"{q['day_volume_shares']:,.0f}")
            t3, t4 = st.columns(2)
            t3.metric("$ volume today", f"${q['day_volume_usd']/1e6:,.2f}M")
            t4.metric(
                f"Est. inflow @ {int(round(ratio*100))}%",
                f"${est_usd/1e6:,.2f}M",
                help=f"≈ {est_hype:,.0f} HYPE absorbed" if hype_px else None,
            )

    st.write("")

    chart_frames = []
    for tkr, df in intraday.items():
        if df.empty:
            continue
        sub = df[["ts", "cum_volume_usd"]].copy()
        sub["ticker"] = tkr
        sub["est_inflow_usd"] = sub["cum_volume_usd"] * ratio
        chart_frames.append(sub)
    if chart_frames:
        combined = pd.concat(chart_frames, ignore_index=True)
        fig = px.line(
            combined,
            x="ts",
            y="est_inflow_usd",
            color="ticker",
            labels={"est_inflow_usd": f"Est. cumulative inflow today (USD @ {int(round(ratio*100))}%)", "ts": ""},
        )
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        fig.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Intraday bars not yet available for today.")


live_block(ratio)

st.divider()

# ---------- secondary context ----------

with st.expander("Context: HYPE price (30d) and PURR treasury"):
    try:
        price_df = get_hype_price(30)
    except Exception as e:
        price_df = pd.DataFrame()
        st.warning(f"HYPE price fetch failed: {e}")

    purr_df = get_purr()
    purr_latest = purr_df["hype_tokens_held"].iloc[-1] if not purr_df.empty else None
    st.metric(
        "PURR HYPE held",
        f"{purr_latest/1e6:.2f}M HYPE" if purr_latest else "—",
        help="Latest disclosed balance from 8-K filings.",
    )

    if not price_df.empty:
        fig_px = go.Figure(
            data=[
                go.Candlestick(
                    x=price_df["ts"],
                    open=price_df["open"],
                    high=price_df["high"],
                    low=price_df["low"],
                    close=price_df["close"],
                    name="HYPE",
                )
            ]
        )
        fig_px.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            yaxis_title="USD",
        )
        st.plotly_chart(fig_px, use_container_width=True)

with st.sidebar:
    st.header("Source status")
    st.markdown(
        f"- ✅ **BHYP / THYP volume** — Yahoo Finance (refreshes every {FRAGMENT_REFRESH})\n"
        "- ✅ **HYPE spot** — Hyperliquid public API (hourly)\n"
        "- 🟡 **PURR treasury** — manual CSV from 8-K filings\n"
    )
    st.divider()
    st.subheader("How the estimate works")
    st.markdown(
        "1. Pull today's traded share volume × last price for each ETF (Yahoo).\n"
        "2. Multiply by your **inflow ratio** to estimate net creations in USD.\n"
        "3. Divide by HYPE spot to get HYPE tokens absorbed.\n\n"
        "The 25–35% range is a rough proxy — for brand-new ETFs the primary-market "
        "share of total activity is high, so this is defensible for now."
    )
