"""
Multi-Index Live Scalper Dashboard
Upstox WebSocket V3 feed + Black-Scholes Greeks
----------------------------------------
SETUP:
  1. pip install streamlit upstox-python-sdk websockets protobuf scipy numpy
  2. Download the proto file:
       curl -O https://assets.upstox.com/feed/market-data-feed/v3/MarketDataFeed.proto
  3. Compile it (requires protoc):
       protoc --python_out=. MarketDataFeed.proto
     -> This generates MarketDataFeedV3_pb2.py in your working dir
  4. Add your token to .streamlit/secrets.toml:
       [upstox]
       access_token = "your_token_here"
  5. Run:
       streamlit run scalper_dashboard.py
"""

import asyncio
import json
import ssl
import threading
import time
from datetime import datetime

import numpy as np
import streamlit as st
from scipy.stats import norm

import upstox_client
from upstox_client.rest import ApiException

# Attempt to import compiled protobuf — graceful fallback for dev/testing
try:
    import MarketDataFeedV3_pb2 as pb
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Index Scalper",
    layout="wide",
    page_icon="⚡",
)

# ─────────────────────────────────────────────
# 2. SECRETS
# ─────────────────────────────────────────────
try:
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("❌ Missing `access_token` in `.streamlit/secrets.toml`")
    st.stop()

# ─────────────────────────────────────────────
# 3. INDEX METADATA
# ─────────────────────────────────────────────
INDEX_CONFIG = {
    "NIFTY 50": {
        "instrument_key": "NSE_INDEX|Nifty 50",
        "lot_size": 75,
        "strike_step": 50,
    },
    "BANK NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "lot_size": 30,
        "strike_step": 100,
    },
    "FINNIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Fin Service",
        "lot_size": 65,
        "strike_step": 50,
    },
    "MIDCAP NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Midcap Select",
        "lot_size": 120,
        "strike_step": 25,
    },
}

ALL_INSTRUMENT_KEYS = [v["instrument_key"] for v in INDEX_CONFIG.values()]

# ─────────────────────────────────────────────
# 4. SHARED STATE (WebSocket thread → Streamlit)
# ─────────────────────────────────────────────
# Thread-safe dict: instrument_key → {"ltp": float, "close": float, "ts": datetime}
if "price_feed" not in st.session_state:
    st.session_state.price_feed = {}

if "ws_status" not in st.session_state:
    st.session_state.ws_status = "disconnected"   # "connecting" | "live" | "error" | "disconnected"

if "ws_thread_started" not in st.session_state:
    st.session_state.ws_thread_started = False

if "ws_error_msg" not in st.session_state:
    st.session_state.ws_error_msg = ""

if "tick_count" not in st.session_state:
    st.session_state.tick_count = 0

# ─────────────────────────────────────────────
# 5. WEBSOCKET CLIENT
# ─────────────────────────────────────────────

def _get_authorized_ws_url(token: str) -> str:
    """Fetch the redirect URI for WebSocket V3 from Upstox."""
    config = upstox_client.Configuration()
    config.access_token = token
    api = upstox_client.WebsocketApi(upstox_client.ApiClient(config))
    res = api.get_market_data_feed_authorize_v3("2.0")
    return res.data.authorized_redirect_uri


async def _stream_market_data(
    token: str,
    instrument_keys: list,
    price_feed: dict,
    status_holder: list,   # mutable list so thread can write status
):
    """
    Connects to Upstox WebSocket V3, subscribes to instrument keys,
    and continuously writes LTP updates into price_feed.
    """
    status_holder[0] = "connecting"

    try:
        url = _get_authorized_ws_url(token)
    except Exception as e:
        status_holder[0] = f"error:Failed to get WS URL — {e}"
        return

    ssl_ctx = ssl.create_default_context()

    # Subscription payload — binary format required by V3
    sub_payload = json.dumps({
        "guid": "scalper-multi-index-001",
        "method": "sub",
        "data": {
            "mode": "ltpc",           # lowest latency: LTP + close price only
            "instrumentKeys": instrument_keys,
        },
    }).encode("utf-8")

    try:
        async with websockets.connect(
            url,
            ssl=ssl_ctx,
            ping_interval=20,        # keep-alive ping every 20s
            ping_timeout=10,
        ) as ws:
            await ws.send(sub_payload)
            status_holder[0] = "live"

            async for raw_bytes in ws:
                # Skip text frames (market_info status messages come as text/JSON)
                if isinstance(raw_bytes, str):
                    continue

                if not PROTO_AVAILABLE:
                    # Fallback: can't decode without proto — just mark live
                    continue

                try:
                    feed_response = pb.FeedResponse()
                    feed_response.ParseFromString(raw_bytes)

                    for ikey, feed_data in feed_response.feeds.items():
                        if feed_data.HasField("ltpc"):
                            ltpc = feed_data.ltpc
                            price_feed[ikey] = {
                                "ltp": ltpc.ltp,
                                "close": ltpc.cp,
                                "change_pct": round(
                                    ((ltpc.ltp - ltpc.cp) / ltpc.cp) * 100, 2
                                ) if ltpc.cp else 0.0,
                                "ts": datetime.now(),
                            }

                except Exception:
                    # Malformed frame — skip silently
                    continue

    except websockets.exceptions.ConnectionClosedError as e:
        status_holder[0] = f"error:Connection closed — {e}"
    except Exception as e:
        status_holder[0] = f"error:{e}"


def _ws_thread_target(token: str, instrument_keys: list, price_feed: dict, status_holder: list):
    """Entry point for the background thread."""
    asyncio.run(_stream_market_data(token, instrument_keys, price_feed, status_holder))


def start_websocket_feed():
    """Launch the WebSocket feed in a daemon thread if not already running."""
    if st.session_state.ws_thread_started:
        return

    status_holder = ["connecting"]          # mutable so async coroutine can write into it
    price_feed = st.session_state.price_feed

    t = threading.Thread(
        target=_ws_thread_target,
        args=(TOKEN, ALL_INSTRUMENT_KEYS, price_feed, status_holder),
        daemon=True,
        name="upstox-ws-feed",
    )
    t.start()

    # Poll briefly so the UI can show "connecting"
    def _update_status():
        while True:
            raw = status_holder[0]
            if raw.startswith("error:"):
                st.session_state.ws_status = "error"
                st.session_state.ws_error_msg = raw[6:]
                break
            else:
                st.session_state.ws_status = raw
            time.sleep(0.3)

    threading.Thread(target=_update_status, daemon=True).start()
    st.session_state.ws_thread_started = True


# ─────────────────────────────────────────────
# 6. BLACK-SCHOLES GREEKS ENGINE
# ─────────────────────────────────────────────

def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """
    Returns delta, gamma, vega, theta (per day), rho for a European option.
    S     = spot price
    K     = strike price
    T     = time to expiry in years
    r     = risk-free rate (e.g. 0.07 for 7%)
    sigma = implied volatility (e.g. 0.15 for 15%)
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0, "iv": sigma}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100   # per 1% move in IV

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "vega":  round(vega, 2),
        "theta": round(theta_annual / 365, 2),   # per calendar day
        "rho":   round(rho, 4),
        "iv":    sigma,
    }


# ─────────────────────────────────────────────
# 7. FORMATTING HELPERS
# ─────────────────────────────────────────────

def fmt_inr(value: float) -> str:
    """Format a rupee value in Indian notation (L = lakhs, Cr = crores)."""
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"


def fmt_spot(value: float) -> str:
    return f"₹{value:,.2f}"


def delta_color(delta: float, option_type: str) -> str:
    """Return a Streamlit metric delta string with sign for display."""
    return f"+{delta}" if delta >= 0 else str(delta)


# ─────────────────────────────────────────────
# 8. SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Scalper Controls")
    st.divider()

    selected_index = st.selectbox("Index", list(INDEX_CONFIG.keys()), index=0)
    conf = INDEX_CONFIG[selected_index]

    st.divider()
    st.markdown("### Strategy")

    run_live = st.toggle("🚀 Start Live Feed", value=False)
    lots = st.number_input("Lots", min_value=1, max_value=500, value=1, step=1)

    strike_mode = st.selectbox(
        "Strike Selection",
        ["ATM", "1-Strike ITM", "2-Strike ITM"],
    )
    strike_offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]

    st.divider()
    st.markdown("### Greeks Parameters")

    iv_pct = st.slider("Implied Volatility (%)", min_value=5, max_value=80, value=15, step=1)
    iv = iv_pct / 100.0

    dte_days = st.slider("Days to Expiry", min_value=0, max_value=30, value=4, step=1)
    dte = dte_days / 365.0

    risk_free = st.slider("Risk-Free Rate (%)", min_value=4, max_value=12, value=7, step=1)
    r = risk_free / 100.0

    st.divider()
    refresh_ms = st.select_slider(
        "UI Refresh Interval",
        options=[500, 1000, 2000, 3000, 5000],
        value=1000,
        format_func=lambda x: f"{x}ms",
    )

    # WebSocket status indicator
    st.divider()
    ws_status = st.session_state.ws_status
    if ws_status == "live":
        st.success("🟢 WebSocket: Live")
    elif ws_status == "connecting":
        st.info("🟡 WebSocket: Connecting…")
    elif ws_status == "error":
        st.error(f"🔴 WS Error: {st.session_state.ws_error_msg}")
    else:
        st.caption("⚪ WebSocket: Not started")

    if not PROTO_AVAILABLE:
        st.warning("⚠️ `MarketDataFeedV3_pb2` not found. Compile the proto file to decode live data.")
    if not WS_AVAILABLE:
        st.warning("⚠️ `websockets` package not installed. Run `pip install websockets`.")


# ─────────────────────────────────────────────
# 9. MAIN DASHBOARD
# ─────────────────────────────────────────────

st.markdown(f"# ⚡ {selected_index} Scalper")

# Start WebSocket on user request
if run_live and not st.session_state.ws_thread_started:
    start_websocket_feed()

# Pull latest data for selected index
ikey = conf["instrument_key"]
feed_entry = st.session_state.price_feed.get(ikey)

# Determine spot price
if feed_entry and run_live:
    spot = feed_entry["ltp"]
    change_pct = feed_entry.get("change_pct", 0.0)
    close_price = feed_entry.get("close", spot)
    data_age = (datetime.now() - feed_entry["ts"]).total_seconds()
    data_source = "WebSocket V3"
else:
    # Offline / paused state — show empty placeholders
    spot = None
    change_pct = 0.0
    close_price = None
    data_age = None
    data_source = "—"


# ── Status bar ──────────────────────────────
col_stat1, col_stat2, col_stat3 = st.columns([2, 2, 3])
with col_stat1:
    if run_live and st.session_state.ws_status == "live":
        st.caption(f"🟢 Source: {data_source}")
    elif run_live:
        st.caption(f"🟡 {st.session_state.ws_status.capitalize()}…")
    else:
        st.caption("⚪ Dashboard paused — toggle Live Feed to start")

with col_stat2:
    if data_age is not None:
        age_str = f"{data_age:.1f}s ago" if data_age > 1 else "< 1s ago"
        st.caption(f"🕐 Last tick: {age_str}")

with col_stat3:
    if feed_entry:
        st.caption(f"📡 Instrument: `{ikey}`")

st.divider()

# ── Top metrics ─────────────────────────────
m1, m2, m3, m4 = st.columns(4)

step = conf["strike_step"]

if spot:
    atm = int(round(spot / step) * step)
    exposure = spot * lots * conf["lot_size"]
    ce_strike = atm - (strike_offset * step)
    pe_strike = atm + (strike_offset * step)

    m1.metric(
        label=f"📈 {selected_index} Spot",
        value=fmt_spot(spot),
        delta=f"{change_pct:+.2f}% vs prev close",
    )
    m2.metric(label="🎯 ATM Strike", value=f"{atm:,}")
    m3.metric(
        label="💰 Total Exposure",
        value=fmt_inr(exposure),
        delta=f"{lots} lot × {conf['lot_size']}",
    )
    m4.metric(
        label="📅 DTE",
        value=f"{dte_days}d",
        delta=f"IV: {iv_pct}%",
    )
else:
    m1.metric(label=f"📈 {selected_index} Spot", value="—")
    m2.metric(label="🎯 ATM Strike", value="—")
    m3.metric(label="💰 Total Exposure", value="—")
    m4.metric(label="📅 DTE", value=f"{dte_days}d", delta=f"IV: {iv_pct}%")
    atm = ce_strike = pe_strike = None

st.divider()

# ── Greeks panels ────────────────────────────
if spot and atm:
    g_ce = black_scholes_greeks(spot, ce_strike, dte, r, iv, "call")
    g_pe = black_scholes_greeks(spot, pe_strike, dte, r, iv, "put")

    col_ce, col_pe = st.columns(2)

    with col_ce:
        st.success(f"### 🟢 {ce_strike:,} CE — Call")
        c1, c2, c3 = st.columns(3)
        c1.metric("Delta", g_ce["delta"], delta=delta_color(g_ce["delta"], "call"))
        c2.metric("Theta/day", g_ce["theta"])
        c3.metric("Gamma", g_ce["gamma"])
        c4, c5, c6 = st.columns(3)
        c4.metric("Vega", g_ce["vega"])
        c5.metric("Rho", g_ce["rho"])
        c6.metric("IV", f"{iv_pct}%")

    with col_pe:
        st.error(f"### 🔴 {pe_strike:,} PE — Put")
        p1, p2, p3 = st.columns(3)
        p1.metric("Delta", g_pe["delta"], delta=delta_color(g_pe["delta"], "put"))
        p2.metric("Theta/day", g_pe["theta"])
        p3.metric("Gamma", g_pe["gamma"])
        p4, p5, p6 = st.columns(3)
        p4.metric("Vega", g_pe["vega"])
        p5.metric("Rho", g_pe["rho"])
        p6.metric("IV", f"{iv_pct}%")

    st.divider()

    # ── Net position summary ─────────────────
    st.markdown("### 📊 Net Position Summary")
    net_delta = round(g_ce["delta"] + g_pe["delta"], 4)
    net_theta = round(g_ce["theta"] + g_pe["theta"], 2)
    net_gamma = round(g_ce["gamma"] + g_pe["gamma"], 6)
    net_vega  = round(g_ce["vega"] + g_pe["vega"], 2)

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Net Delta", net_delta, delta="neutral ✓" if abs(net_delta) < 0.05 else "directional ⚠️")
    n2.metric("Net Theta/day", net_theta)
    n3.metric("Net Gamma", net_gamma)
    n4.metric("Net Vega", net_vega)

    # Theta P&L projection
    st.divider()
    st.markdown("### 🗓️ Theta Decay Projection")
    theta_table_data = []
    for day in range(1, min(dte_days + 1, 8)):
        remaining_dte = max((dte_days - day) / 365, 0.0001)
        g_ce_proj = black_scholes_greeks(spot, ce_strike, remaining_dte, r, iv, "call")
        g_pe_proj = black_scholes_greeks(spot, pe_strike, remaining_dte, r, iv, "put")
        daily_theta = round((g_ce_proj["theta"] + g_pe_proj["theta"]) * lots * conf["lot_size"], 2)
        theta_table_data.append({
            "Day": f"Day +{day}",
            "DTE Remaining": dte_days - day,
            "CE Theta": g_ce_proj["theta"],
            "PE Theta": g_pe_proj["theta"],
            f"Net Theta × {lots}L (₹)": daily_theta,
        })

    import pandas as pd
    if theta_table_data:
        df = pd.DataFrame(theta_table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("⏸️ Dashboard paused. Enable **Live Feed** in the sidebar to begin streaming.")

# ─────────────────────────────────────────────
# 10. ALL-INDEX SUMMARY (when live)
# ─────────────────────────────────────────────
if run_live and len(st.session_state.price_feed) > 0:
    st.divider()
    st.markdown("### 🌐 All Index Prices")
    rows = []
    for idx_name, idx_conf in INDEX_CONFIG.items():
        entry = st.session_state.price_feed.get(idx_conf["instrument_key"])
        if entry:
            rows.append({
                "Index": idx_name,
                "LTP": f"₹{entry['ltp']:,.2f}",
                "Prev Close": f"₹{entry['close']:,.2f}" if entry.get("close") else "—",
                "Change %": f"{entry.get('change_pct', 0):+.2f}%",
                "Last Update": entry["ts"].strftime("%H:%M:%S"),
            })
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# 11. AUTO-REFRESH
# ─────────────────────────────────────────────
if run_live:
    time.sleep(refresh_ms / 1000)
    st.rerun()
