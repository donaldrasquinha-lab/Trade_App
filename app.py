import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
from upstox_client.rest import ApiException
import time

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="Multi-Index Live Scalper", layout="wide", page_icon="📈")

try:
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing 'access_token' in .streamlit/secrets.toml")
    st.stop()

# --- 2. INDEX METADATA (Lot sizes and strike steps) ---
# Note: Lot sizes are based on current 2024/2025 NSE revisions
INDEX_CONFIG = {
    "NIFTY 50": {
        "key": "NSE_INDEX|Nifty 50", 
        "lot_size": 75, 
        "strike_step": 50
    },
    "BANK NIFTY": {
        "key": "NSE_INDEX|Nifty Bank", 
        "lot_size": 30, 
        "strike_step": 100
    },
    "FINNIFTY": {
        "key": "NSE_INDEX|Nifty Fin Service", 
        "lot_size": 65, 
        "strike_step": 50
    },
    "MIDCAP NIFTY": {
        "key": "NSE_INDEX|Nifty Midcap 100", 
        "lot_size": 120, 
        "strike_step": 25
    }
}

# --- 3. GREEKS ENGINE ---
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: return {"delta": 0, "theta": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    return {"delta": round(delta, 2), "theta": round(theta / 365, 2)}

# --- 4. UPSTOX DATA FETCH ---
def get_live_data(instrument_key):
    config = upstox_client.Configuration()
    config.access_token = TOKEN
    api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
    try:
        # OHLC endpoint is more stable for Indices
        res = api.get_market_quote_ohlc(instrument_key, "1d", '2.0')
        if res.status == "success" and instrument_key in res.data:
            return res.data[instrument_key].last_price
        return None
    except Exception:
        return None

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🎯 Index Selection")
    selected_index_name = st.selectbox("Select Index", list(INDEX_CONFIG.keys()))
    conf = INDEX_CONFIG[selected_index_name]
    
    st.divider()
    st.header("🛡️ Strategy")
    run_live = st.toggle("🚀 Start Dashboard", value=False)
    lots = st.number_input("Lots", min_value=1, value=1)
    strike_mode = st.selectbox("Strike Choice", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
    
    st.subheader("Greeks Settings")
    iv = st.slider("Implied Volatility (IV)", 0.05, 0.50, 0.15)
    dte = st.slider("Days to Expiry", 0, 7, 4) / 365

# --- 6. DASHBOARD MAIN LOOP ---
st.title(f"⚡ {selected_index_name} Scalper")
placeholder = st.empty()

if run_live:
    while True:
        spot = get_live_data(conf["key"])
        
        if spot:
            # Dynamic ATM calculation based on index's strike step
            step = conf["strike_step"]
            atm = int(round(spot / step) * step)
            
            # CE is ITM below spot; PE is ITM above spot
            ce_strike = atm - (offset * step)
            pe_strike = atm + (offset * step)
            
            with placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{selected_index_name} SPOT", f"₹{spot}")
                m2.metric("TOTAL EXPOSURE", f"₹{spot * lots * conf['lot_size']:,.0f}")
                m3.metric("CURRENT ATM", atm)
                
                st.divider()
                c1, c2 = st.columns(2)
                
                with c1:
                    st.success(f"🟢 CALL: {ce_strike} CE")
                    g_ce = calculate_greeks(spot, ce_strike, dte, 0.07, iv, "call")
                    st.write(f"**Delta:** `{g_ce['delta']}` | **Theta:** `{g_ce['theta']}`")
                
                with c2:
                    st.error(f"🔴 PUT: {pe_strike} PE")
                    g_pe = calculate_greeks(spot, pe_strike, dte, 0.07, iv, "put")
                    st.write(f"**Delta:** `{g_pe['delta']}` | **Theta:** `{g_pe['theta']}`")
        
        else:
            st.warning(f"Unable to fetch data for {selected_index_name}. Ensure you have index permissions.")
            
        time.sleep(1.5)
else:
    st.info("Dashboard Paused. Start live updates from the sidebar.")
