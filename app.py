import streamlit as st

import streamlit as st  # MUST BE AT THE TOP
import threading
import time
from datetime import datetime
import numpy as np
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

# Global data store to bridge the Background Thread and Streamlit UI
if "LIVE_DATA" not in globals():
    globals()["LIVE_DATA"] = {}

# 3. GREEKS ENGINE
def get_greeks(S, K, T, r, sigma, option_type="call"):
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
        "Delta": round(delta, 3),
        "Gamma": round(pdf1 / (S * sigma * np.sqrt(T)), 6),
        "Theta": round(theta / 365, 2),
        "Vega": round(S * pdf1 * np.sqrt(T) / 100, 2)
    }

# 4. UPSTOX V3 STREAMER (Background Thread)
def run_upstox_thread(token, keys):
    api_config = upstox_client.Configuration()
    api_config.access_token = token
    api_client = upstox_client.ApiClient(api_config)
    streamer = upstox_client.MarketDataStreamerV3(api_client)
    
    def on_message(message):
        feeds = message.get("feeds", {})
        for ikey, val in feeds.items():
            # Extract LTP from V3 nested structure
            data = val.get("ltpc") or val.get("ff", {}).get("marketFF", {}).get("ltpc")
            if data and "ltp" in data:
                globals()["LIVE_DATA"][ikey] = {
                    "ltp": float(data["ltp"]),
                    "cp": float(data.get("cp", data["ltp"])),
                    "ts": datetime.now()
                }

    streamer.on("open", lambda: streamer.subscribe(keys, "ltpc"))
    streamer.on("message", on_message)
    streamer.connect()

# 5. SIDEBAR CONTROLS
with st.sidebar:
    st.title("⚡ Scalper Pro")
    selected_name = st.selectbox("Select Index", list(INDEX_CONFIG.keys()))
    run_live = st.toggle("Start Live Feed", value=False)
    st.divider()
    strike_mode = st.selectbox("Strike Mode", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    iv = st.slider("IV %", 5, 80, 18) / 100
    dte = st.slider("Days to Expiry", 0, 30, 5) / 365
    r_rate = st.slider("Risk Free Rate %", 5, 12, 7) / 100

# 6. MAIN DISPLAY AREA
main_ui = st.empty()

if run_live:
    # Start the thread once
    if "ws_active" not in st.session_state:
        threading.Thread(target=run_upstox_thread, args=(TOKEN, ALL_KEYS), daemon=True).start()
        st.session_state.ws_active = True

    # Refresh Loop
    while True:
        with main_ui.container():
            conf = INDEX_CONFIG[selected_name]
            live_val = globals()["LIVE_DATA"].get(conf["instrument_key"])
            
            if not live_val:
                st.info("⌛ Requesting data from Upstox (Market must be open)...")
            else:
                ltp = live_val["ltp"]
                pct = round(((ltp - live_val["cp"]) / live_val["cp"]) * 100, 2)
                
                # Logic
                step = conf["strike_step"]
                offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
                atm = round(ltp / step) * step
                ce_s, pe_s = atm - (offset * step), atm + (offset * step)

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(selected_name, f"₹{ltp:,.2f}", f"{pct}%")
                col2.metric("Strike Selection", f"{strike_mode} ({atm})")
                col3.metric("Last Update", live_val["ts"].strftime("%H:%M:%S"))

                st.divider()

                # Greeks
                left, right = st.columns(2)
                with left:
                    st.subheader(f"CALL {ce_s}")
                    st.table(pd.DataFrame([get_greeks(ltp, ce_s, dte, r_rate, iv, "call")]))
                with right:
                    st.subheader(f"PUT {pe_s}")
                    st.table(pd.DataFrame([get_greeks(ltp, pe_s, dte, r_rate, iv, "put")]))
        
        time.sleep(0.5)
else:
    main_ui.warning("Feed Stopped. Enable 'Start Live Feed' in the sidebar.")
