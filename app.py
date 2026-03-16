import threading
import time
from datetime import datetime
import numpy as np
import streamlit as st
from scipy.stats import norm
import upstox_client

# 1. PAGE SETUP
st.set_page_config(page_title="Multi-Index Scalper", layout="wide", page_icon="⚡")

# 2. CONFIG & SECRETS
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
ALL_KEYS = [v["instrument_key"] for v in INDEX_CONFIG.values()]

# Global state using a dictionary for thread-safety
if "price_feed" not in st.session_state:
    st.session_state.price_feed = {}
if "ws_started" not in st.session_state:
    st.session_state.ws_started = False

# 3. GREEKS ENGINE
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
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

# 4. WEBSOCKET HANDLER
def run_v3_streamer(token, keys):
    conf = upstox_client.Configuration()
    conf.access_token = token
    api_client = upstox_client.ApiClient(conf)
    streamer = upstox_client.MarketDataStreamerV3(api_client)
    
    def on_message(message):
        feeds = message.get("feeds", {})
        for ikey, val in feeds.items():
            ltpc = val.get("ltpc", {})
            if ltpc.get("ltp"):
                st.session_state.price_feed[ikey] = {
                    "ltp": float(ltpc["ltp"]),
                    "cp": float(ltpc.get("cp", ltpc["ltp"])),
                    "ts": datetime.now()
                }

    streamer.on("open", lambda: streamer.subscribe(keys, "ltpc"))
    streamer.on("message", on_message)
    streamer.connect()

# 5. UI LAYOUT
with st.sidebar:
    st.title("⚡ Scalper Pro")
    selected_index = st.selectbox("Select Index", list(INDEX_CONFIG.keys()))
    run_live = st.toggle("Go Live", value=False)
    
    st.divider()
    strike_mode = st.selectbox("Strike", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    iv = st.slider("IV %", 5, 80, 18) / 100
    dte = st.slider("DTE (Days)", 0, 30, 5) / 365
    r_rate = st.slider("Risk Free %", 5, 12, 7) / 100

# Placeholder for dynamic content
main_container = st.empty()

if run_live:
    if not st.session_state.ws_started:
        thread = threading.Thread(target=run_v3_streamer, args=(TOKEN, ALL_KEYS), daemon=True)
        thread.start()
        st.session_state.ws_started = True

    while True:
        with main_container.container():
            conf = INDEX_CONFIG[selected_index]
            data = st.session_state.price_feed.get(conf["instrument_key"])
            
            if not data:
                st.warning("Connecting to Upstox Feed...")
            else:
                ltp = data["ltp"]
                change = round(((ltp - data["cp"]) / data["cp"]) * 100, 2)
                
                # Strike Calculation
                step = conf["strike_step"]
                offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
                atm = round(ltp / step) * step
                ce_strike, pe_strike = atm - (offset * step), atm + (offset * step)

                # Header Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric(selected_index, f"₹{ltp:,.2f}", f"{change}%")
                m2.metric("Time", data["ts"].strftime("%H:%M:%S"))
                m3.metric("ATM Strike", atm)

                # Greeks Cards
                c1, c2 = st.columns(2)
                for col, strike, opt in [(c1, ce_strike, "call"), (c2, pe_strike, "put")]:
                    with col:
                        st.subheader(f"{opt.upper()} {strike}")
                        g = calculate_greeks(ltp, strike, dte, r_rate, iv, opt)
                        st.json(g)
        
        time.sleep(0.5)
else:
    main_container.info("Toggle 'Go Live' to start streaming market data.")
