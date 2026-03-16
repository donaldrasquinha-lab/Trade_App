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
    "BANK NIFTY": {"instrument_key": "NSE_INDEX|Nifty Bank", "lot_size": 15, "strike_step": 100},
    "FINNIFTY": {"instrument_key": "NSE_INDEX|Nifty Fin Service", "lot_size": 40, "strike_step": 50},
}
ALL_KEYS = [v["instrument_key"] for v in INDEX_CONFIG.values()]

# Global thread-safe storage for market data
if "LIVE_FEED" not in globals():
    globals()["LIVE_FEED"] = {}

# 3. GREEKS ENGINE (Black-Scholes)
def get_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf1 = norm.pdf(d1)
    if option_type == "call":
        delta, theta = norm.cdf(d1), (-(S * pdf1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta, theta = norm.cdf(d1) - 1, (-(S * pdf1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return {
        "delta": round(delta, 3), "gamma": round(pdf1 / (S * sigma * np.sqrt(T)), 6),
        "theta": round(theta / 365, 2), "vega": round(S * pdf1 * np.sqrt(T) / 100, 2)
    }

# 4. UPSTOX V3 STREAMER
def run_v3_streamer(token, keys):
    api_config = upstox_client.Configuration()
    api_config.access_token = token
    streamer = upstox_client.MarketDataStreamerV3(upstox_client.ApiClient(api_config))
    
    def on_message(message):
        # V3 Payload: feeds -> {key} -> ltpc -> {ltp, cp}
        feeds = message.get("feeds", {})
        for ikey, val in feeds.items():
            data = val.get("ltpc") or val.get("ff", {}).get("marketFF", {}).get("ltpc")
            if data and "ltp" in data:
                globals()["LIVE_FEED"][ikey] = {
                    "ltp": float(data["ltp"]),
                    "cp": float(data.get("cp", data["ltp"])),
                    "ts": datetime.now()
                }

    streamer.on("open", lambda: streamer.subscribe(keys, "ltpc"))
    streamer.on("message", on_message)
    streamer.connect()

# 5. UI COMPONENTS
with st.sidebar:
    st.title("⚡ Scalper Pro")
    selected_index = st.selectbox("Index", list(INDEX_CONFIG.keys()))
    run_live = st.toggle("Start Feed", value=False)
    st.divider()
    strike_mode = st.selectbox("Strike", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    iv = st.slider("IV %", 5, 80, 18) / 100
    dte = st.slider("DTE", 0, 30, 5) / 365
    r_rate = st.slider("RF Rate %", 5, 12, 7) / 100

dashboard = st.empty()

# 6. EXECUTION LOOP
if run_live:
    if "thread_started" not in st.session_state:
        threading.Thread(target=run_v3_streamer, args=(TOKEN, ALL_KEYS), daemon=True).start()
        st.session_state.thread_started = True

    while True:
        with dashboard.container():
            conf = INDEX_CONFIG[selected_index]
            data = globals()["LIVE_FEED"].get(conf["instrument_key"])
            
            if not data:
                st.info("🔄 Connecting to Upstox Market Feed...")
            else:
                ltp = data["ltp"]
                chg = round(((ltp - data["cp"]) / data["cp"]) * 100, 2)
                
                # Strike Calculation
                step, offset = conf["strike_step"], {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
                atm = round(ltp / step) * step
                ce_s, pe_s = atm - (offset * step), atm + (offset * step)

                # Display Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric(selected_index, f"₹{ltp:,.2f}", f"{chg}%")
                c2.metric("Update Time", data["ts"].strftime("%H:%M:%S"))
                c3.metric("ATM", atm)

                # Option Cards
                col_ce, col_pe = st.columns(2)
                for col, strike, opt in [(col_ce, ce_s, "call"), (col_pe, pe_s, "put")]:
                    with col:
                        st.subheader(f"{opt.upper()} {strike}")
                        st.write(get_greeks(ltp, strike, dte, r_rate, iv, opt))
        
        time.sleep(0.4) # Control UI refresh rate
else:
    dashboard.warning("Feed is stopped. Toggle 'Start Feed' in the sidebar.")
