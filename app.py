import threading
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
import upstox_client

# ==============================================================
# 1. PAGE CONFIG
# ==============================================================
st.set_page_config(page_title="Multi-Index Scalper", layout="wide", page_icon="⚡")

# ==============================================================
# 2. SECRETS & CONFIG
# ==============================================================
try:
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing `access_token` in `.streamlit/secrets.toml`")
    st.stop()

INDEX_CONFIG = {
    "NIFTY 50": {"instrument_key": "NSE_INDEX|Nifty 50", "lot_size": 75, "strike_step": 50},
    "BANK NIFTY": {"instrument_key": "NSE_INDEX|Nifty Bank", "lot_size": 30, "strike_step": 100},
    "FINNIFTY": {"instrument_key": "NSE_INDEX|Nifty Fin Service", "lot_size": 65, "strike_step": 50},
    "MIDCAP NIFTY": {"instrument_key": "NSE_INDEX|Nifty Midcap Select", "lot_size": 120, "strike_step": 25},
}
ALL_INSTRUMENT_KEYS = [v["instrument_key"] for v in INDEX_CONFIG.values()]

# Shared state for background thread
_price_feed = {}
_ws_status = ["disconnected"]
_ws_started = [False]

# ==============================================================
# 3. WEBSOCKET FEED
# ==============================================================
def _stream_market_data(token, instrument_keys):
    _ws_status[0] = "connecting"
    configuration = upstox_client.Configuration()
    configuration.access_token = token
    api_client = upstox_client.ApiClient(configuration)
    streamer = upstox_client.MarketDataStreamerV3(api_client)
    streamer.auto_reconnect(True, 5, 3)

    def on_open():
        streamer.subscribe(instrument_keys, "ltpc")
        _ws_status[0] = "live"

    def on_message(message):
        for ikey, data in message.get("feeds", {}).items():
            ltpc = data.get("ltpc", {})
            ltp, cp = ltpc.get("ltp"), ltpc.get("cp")
            if ltp:
                cp = float(cp) if cp else float(ltp)
                _price_feed[ikey] = {
                    "ltp": float(ltp),
                    "change_pct": round(((float(ltp) - cp) / cp) * 100, 2),
                    "ts": datetime.now()
                }

    streamer.on("open", on_open)
    streamer.on("message", on_message)
    streamer.connect()

def start_feed():
    if not _ws_started[0]:
        _ws_started[0] = True
        threading.Thread(target=_stream_market_data, args=(TOKEN, ALL_INSTRUMENT_KEYS), daemon=True).start()

# ==============================================================
# 4. GREEKS ENGINE (Black-Scholes)
# ==============================================================
def greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0:
        return dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf1 = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return {
        "delta": round(delta, 3),
        "gamma": round(pdf1 / (S * sigma * np.sqrt(T)), 6),
        "theta": round(theta / 365, 2),
        "vega": round(S * pdf1 * np.sqrt(T) / 100, 2)
    }

# ==============================================================
# 5. SIDEBAR & UI
# ==============================================================
with st.sidebar:
    st.header("Scalper Controls")
    selected_index = st.selectbox("Index", list(INDEX_CONFIG.keys()))
    run_live = st.toggle("Start Live Feed", value=False)
    strike_mode = st.selectbox("Strike Selection", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    iv = st.slider("IV (%)", 5, 80, 15) / 100
    dte = st.slider("Days to Expiry", 0, 30, 4) / 365
    r = st.slider("Risk-Free Rate (%)", 4, 12, 7) / 100

dashboard_placeholder = st.empty()

if run_live:
    start_feed()
    while True:
        with dashboard_placeholder.container():
            conf = INDEX_CONFIG[selected_index]
            data = _price_feed.get(conf["instrument_key"])
            
            if data:
                ltp = data["ltp"]
                st.metric(f"{selected_index} Spot", f"₹{ltp:,.2f}", f"{data['change_pct']}%")
                
                # Calculate Strikes
                step = conf["strike_step"]
                offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
                atm = round(ltp / step) * step
                ce_s, pe_s = atm - (offset * step), atm + (offset * step)

                col1, col2 = st.columns(2)
                for col, strike, opt in zip([col1, col2], [ce_s, pe_s], ["call", "put"]):
                    with col:
                        st.subheader(f"{opt.upper()} {strike}")
                        g = greeks(ltp, strike, dte, r, iv, opt)
                        st.json(g)
            else:
                st.info("Waiting for data...")
        time.sleep(0.5)
