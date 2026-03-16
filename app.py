"""
Multi-Index Live Scalper Dashboard
Upstox MarketDataStreamer (SDK built-in) — no proto compile needed
------------------------------------------------------------------
SETUP:
  1. pip install -r requirements.txt
  2. Add your token to .streamlit/secrets.toml:
       [upstox]
       access_token = "your_token_here"
  3. streamlit run scalper_dashboard.py
"""

import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
import upstox_client

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
# 4. SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
if "price_feed" not in st.session_state:
    st.session_state.price_feed = {}

if "ws_status" not in st.session_state:
    st.session_state.ws_status = "disconnected"

if "ws_error_msg" not in st.session_state:
    st.session_state.ws_error_msg = ""

if "ws_thread_started" not in st.session_state:
    st.session_state.ws_thread_started = False

# ─────────────────────────────────────────────
# 5. WEBSOCKET FEED  (upstox_client.MarketDataStreamer)
# ─────────────────────────────────────────────

def _stream_market_data(
    token: str,
    instrument_keys: list,
    price_feed: dict,
    status_holder: list,
):
    """
    Uses upstox_client.MarketDataStreamer — the SDK handles proto
    decoding internally so no protoc compile step is needed.
    The on_message callback receives an already-decoded Python dict.
    """
    status_holder[0] = "connecting"

    def on_open(ws):
        status_holder[0] = "live"

    def on_message(ws, message):
        try:
            feeds = message.get("feeds", {})
            for ikey, data in feeds.items():
                ltpc = data.get("ltpc", {})
                ltp  = ltpc.get("ltp")
                cp   = ltpc.get("cp")
                if ltp is None:
                    continue
                ltp = float(ltp)
                cp  = float(cp) if cp else ltp
                price_feed[ikey] = {
                    "ltp":        ltp,
                    "close":      cp,
                    "change_pct": round(((ltp - cp) / cp) * 100, 2) if cp else 0.0,
                    "ts":         datetime.now(),
                }
        except Exception:
            pass

    def on_error(ws, error):
        status_holder[0] = f"error:{error}"

    def on_close(ws, *args):
        status_holder[0] = "disconnected"

    try:
        api_client = upstox_client.ApiClient(
            upstox_client.Configuration(access_token=token)
        )
        streamer = upstox_client.MarketDataStreamer(
            api_client,
            instrument_keys,
            "ltpc",   # lowest latency: LTP + close price only
        )
        streamer.on("open",    on_open)
        streamer.on("message", on_message)
        streamer.on("error",   on_error)
        streamer.on("close",   on_close)
        streamer.connect()    # blocking — runs inside daemon thread

    except Exception as e:
        status_holder[0] = f"error:{e}"


def start_websocket_feed():
    """Launch MarketDataStreamer in a background daemon thread."""
    if st.session_state.ws_thread_started:
        return

    # Mutable list so the async thread can write status back
    status_holder = ["connecting"]

    def _run():
        _stream_market_data(
            TOKEN,
            ALL_INSTRUMENT_KEYS,
            st.session_state.price_feed,
            status_holder,
        )

    def _watch_status():
        """Mirror status_holder into session_state for Streamlit to read."""
        while True:
            raw = status_holder[0]
            if raw.startswith("error:"):
                st.session_state.ws_status    = "error"
                st.session_state.ws_error_msg = raw[6:]
                break
            st.session_state.ws_status = raw
            time.sleep(0.3)

    threading.Thread(target=_run,          daemon=True, name="upstox-streamer").start()
    threading.Thread(target=_watch_status, daemon=True, name="status-watcher").start()
    st.session_state.ws_thread_started = True


# ─────────────────────────────────────────────
# 6. BLACK-SCHOLES GREEKS ENGINE
# ─────────────────────────────────────────────

def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict:
    """
    Full first-order Greeks for a European option.
    S     = spot price
    K     = strike price
    T     = time to expiry in years
    r     = risk-free rate  (e.g. 0.07 for 7%)
    sigma = implied volatility (e.g. 0.15 for 15%)
    Returns: delta, gamma, vega, theta (per calendar day), rho
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta        = norm.cdf(d1)
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta        = norm.cdf(d1) - 1
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100   # per 1% move in IV

    return {
        "delta": round(delta,             4),
        "gamma": round(gamma,             6),
        "vega":  round(vega,              2),
        "theta": round(theta_annual / 365, 2),
        "rho":   round(rho,               4),
    }


# ─────────────────────────────────────────────
# 7. FORMATTING HELPERS
# ─────────────────────────────────────────────

def fmt_inr(value: float) -> str:
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    return f"₹{value:,.0f}"


def sign_str(v: float) -> str:
    return f"+{v}" if v >= 0 else str(v)


# ─────────────────────────────────────────────
# 8. SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Scalper Controls")
    st.divider()

    selected_index = st.selectbox("Index", list(INDEX_CONFIG.keys()))
    conf = INDEX_CONFIG[selected_index]

    st.divider()
    st.markdown("### Strategy")

    run_live = st.toggle("🚀 Start Live Feed", value=False)
    lots     = st.number_input("Lots", min_value=1, max_value=500, value=1, step=1)

    strike_mode   = st.selectbox("Strike Selection", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    strike_offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]

    st.divider()
    st.markdown("### Greeks Parameters")

    iv_pct    = st.slider("Implied Volatility (%)", min_value=5,  max_value=80, value=15, step=1)
    iv        = iv_pct / 100.0
    dte_days  = st.slider("Days to Expiry",          min_value=0,  max_value=30, value=4,  step=1)
    dte       = dte_days / 365.0
    risk_free = st.slider("Risk-Free Rate (%)",       min_value=4,  max_value=12, value=7,  step=1)
    r         = risk_free / 100.0

    st.divider()
    refresh_ms = st.select_slider(
        "UI Refresh Interval",
        options=[500, 1000, 2000, 3000, 5000],
        value=1000,
        format_func=lambda x: f"{x}ms",
    )

    # ── WebSocket status ──────────────────────
    st.divider()
    ws_status = st.session_state.ws_status
    if ws_status == "live":
        st.success("🟢 WebSocket: Live")
    elif ws_status == "connecting":
        st.info("🟡 WebSocket: Connecting…")
    elif ws_status == "error":
        st.error(f"🔴 Error: {st.session_state.ws_error_msg}")
    else:
        st.caption("⚪ WebSocket: Not started")

    # Reconnect button after failure
    if ws_status in ("error", "disconnected") and st.session_state.ws_thread_started:
        if st.button("🔄 Reconnect"):
            st.session_state.ws_thread_started = False
            st.session_state.ws_status = "disconnected"
            st.rerun()


# ─────────────────────────────────────────────
# 9. START FEED ON TOGGLE
# ─────────────────────────────────────────────

if run_live and not st.session_state.ws_thread_started:
    start_websocket_feed()


# ─────────────────────────────────────────────
# 10. PULL LATEST DATA FOR SELECTED INDEX
# ─────────────────────────────────────────────

ikey       = conf["instrument_key"]
feed_entry = st.session_state.price_feed.get(ikey) if run_live else None

if feed_entry:
    spot        = feed_entry["ltp"]
    change_pct  = feed_entry.get("change_pct", 0.0)
    close_price = feed_entry.get("close", spot)
    data_age    = (datetime.now() - feed_entry["ts"]).total_seconds()
else:
    spot = change_pct = close_price = data_age = None


# ─────────────────────────────────────────────
# 11. PAGE HEADER
# ─────────────────────────────────────────────

st.markdown(f"# ⚡ {selected_index} Scalper")

h1, h2, h3 = st.columns([2, 2, 3])
with h1:
    if run_live and ws_status == "live":
        st.caption("🟢 Source: WebSocket V3 — SDK Streamer")
    elif run_live:
        st.caption(f"🟡 {ws_status.capitalize()}…")
    else:
        st.caption("⚪ Paused — enable Live Feed in sidebar")
with h2:
    if data_age is not None:
        st.caption(f"🕐 Last tick: {'< 1s' if data_age < 1 else f'{data_age:.1f}s'} ago")
with h3:
    st.caption(f"📡 `{ikey}`")

st.divider()


# ─────────────────────────────────────────────
# 12. TOP METRICS ROW
# ─────────────────────────────────────────────

step = conf["strike_step"]

if spot:
    atm       = int(round(spot / step) * step)
    exposure  = spot * lots * conf["lot_size"]
    ce_strike = atm - strike_offset * step
    pe_strike = atm + strike_offset * step
else:
    atm = ce_strike = pe_strike = exposure = None

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    f"📈 {selected_index} Spot",
    f"₹{spot:,.2f}" if spot else "—",
    f"{change_pct:+.2f}%" if change_pct is not None else None,
)
m2.metric("🎯 ATM Strike",     f"{atm:,}"         if atm      else "—")
m3.metric("💰 Total Exposure",  fmt_inr(exposure) if exposure  else "—",
          f"{lots} lot × {conf['lot_size']}" if exposure else None)
m4.metric("📅 DTE", f"{dte_days}d", f"IV {iv_pct}%")

st.divider()


# ─────────────────────────────────────────────
# 13. GREEKS PANELS
# ─────────────────────────────────────────────

if spot and atm:
    g_ce = black_scholes_greeks(spot, ce_strike, dte, r, iv, "call")
    g_pe = black_scholes_greeks(spot, pe_strike, dte, r, iv, "put")

    col_ce, col_pe = st.columns(2)

    with col_ce:
        st.success(f"### 🟢 {ce_strike:,} CE — Call")
        a, b, c_ = st.columns(3)
        a.metric("Delta",     g_ce["delta"], sign_str(g_ce["delta"]))
        b.metric("Theta/day", g_ce["theta"])
        c_.metric("Gamma",    g_ce["gamma"])
        d_, e_, f_ = st.columns(3)
        d_.metric("Vega", g_ce["vega"])
        e_.metric("Rho",  g_ce["rho"])
        f_.metric("IV",   f"{iv_pct}%")

    with col_pe:
        st.error(f"### 🔴 {pe_strike:,} PE — Put")
        a, b, c_ = st.columns(3)
        a.metric("Delta",     g_pe["delta"], sign_str(g_pe["delta"]))
        b.metric("Theta/day", g_pe["theta"])
        c_.metric("Gamma",    g_pe["gamma"])
        d_, e_, f_ = st.columns(3)
        d_.metric("Vega", g_pe["vega"])
        e_.metric("Rho",  g_pe["rho"])
        f_.metric("IV",   f"{iv_pct}%")

    st.divider()

    # ── Net position summary ──────────────────
    st.markdown("### 📊 Net Position Summary")

    net_delta = round(g_ce["delta"] + g_pe["delta"], 4)
    net_theta = round(g_ce["theta"] + g_pe["theta"], 2)
    net_gamma = round(g_ce["gamma"] + g_pe["gamma"], 6)
    net_vega  = round(g_ce["vega"]  + g_pe["vega"],  2)

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Net Delta", net_delta,
              "✅ Neutral" if abs(net_delta) < 0.05 else "⚠️ Directional")
    n2.metric("Net Theta/day", net_theta)
    n3.metric("Net Gamma",     net_gamma)
    n4.metric("Net Vega",      net_vega)

    st.divider()

    # ── Theta decay table ─────────────────────
    st.markdown("### 🗓️ Theta Decay Projection")

    if dte_days > 0:
        rows = []
        for day in range(1, min(dte_days + 1, 8)):
            rem = max((dte_days - day) / 365, 1e-6)
            gc  = black_scholes_greeks(spot, ce_strike, rem, r, iv, "call")
            gp  = black_scholes_greeks(spot, pe_strike, rem, r, iv, "put")
            combined = round((gc["theta"] + gp["theta"]) * lots * conf["lot_size"], 2)
            rows.append({
                "Day":               f"Day +{day}",
                "DTE Remaining":     dte_days - day,
                "CE Theta":          gc["theta"],
                "PE Theta":          gp["theta"],
                f"Net P&L ₹ × {lots}L": combined,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Set DTE > 0 to see the theta decay table.")

else:
    st.info("⏸️ Enable **Live Feed** in the sidebar to start streaming.")


# ─────────────────────────────────────────────
# 14. ALL-INDEX OVERVIEW
# ─────────────────────────────────────────────

if run_live and st.session_state.price_feed:
    st.divider()
    st.markdown("### 🌐 All Index Prices")

    rows = []
    for idx_name, idx_conf in INDEX_CONFIG.items():
        entry = st.session_state.price_feed.get(idx_conf["instrument_key"])
        if entry:
            rows.append({
                "Index":       idx_name,
                "LTP":         f"₹{entry['ltp']:,.2f}",
                "Prev Close":  f"₹{entry['close']:,.2f}",
                "Change %":    f"{entry['change_pct']:+.2f}%",
                "Last Update": entry["ts"].strftime("%H:%M:%S"),
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("⏳ Waiting for first tick…")


# ─────────────────────────────────────────────
# 15. AUTO-REFRESH LOOP
# ─────────────────────────────────────────────

if run_live:
    time.sleep(refresh_ms / 1000)
    st.rerun()
