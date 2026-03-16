import streamlit as st
st.write("Available Secret Keys:", list(st.secrets.keys()))
if "upstox" in st.secrets:
    st.write("Upstox Keys:", list(st.secrets["upstox"].keys()))



import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
import time

# --- 1. SECRETS LOADING ---
try:
    # We only need the token for live data
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing 'access_token' in .streamlit/secrets.toml")
    st.stop()

# --- 2. OPTION GREEKS ENGINE ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0: return {"delta": 0, "gamma": 0, "theta": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "call":
        delta = norm.cdf(d1)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return {"delta": round(delta, 2), "gamma": round(gamma, 4), "theta": round(theta / 365, 2)}

# --- 3. LIVE DATA FETCHING ---
def get_live_spot(token):
    config = upstox_client.Configuration()
    config.access_token = token
    api_instance = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
    try:
        instrument_key = "NSE_INDEX|Nifty 50"
        res = api_instance.ltp(instrument_key, 'v2')
        return res.data[instrument_key].last_price
    except Exception as e:
        return None

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Nifty Live Scalper", layout="wide")

# Sidebar for Strategy
with st.sidebar:
    st.header("🛡️ Strategy & Risk")
    lots = st.number_input("Lots", min_value=1, value=1)
    sl_pts = st.number_input("Stop Loss (Points)", value=15.0)
    strike_mode = st.selectbox("Strike Choice", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
    st.divider()
    st.success("API Connected via Token")

# --- 5. MAIN DASHBOARD ---
st.title("🚀 Nifty Live Scalper Dashboard")
placeholder = st.empty()

# Persistent Live Loop
while True:
    spot = get_live_spot(TOKEN)
    
    if spot:
        atm = round(spot / 50) * 50
        ce_strike = atm - (offset * 50)
        pe_strike = atm + (offset * 50)
        
        with placeholder.container():
            # Real-time Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("NIFTY 50 SPOT", f"₹{spot}")
            m2.metric("NET EXPOSURE", f"₹{spot * lots * 50:,.0f}")
            m3.metric("ATM STRIKE", atm)
            
            st.divider()
            
            # Trading Recommendations
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"🟢 CALL: {ce_strike} CE")
                g = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                st.info(f"**Greeks:** Delta: {g['delta']} | Theta: {g['theta']}")
                st.caption(f"Exit {sl_pts} pts below entry price")

            with c2:
                st.error(f"🔴 PUT: {pe_strike} PE")
                g = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                st.info(f"**Greeks:** Delta: {g['delta']} | Theta: {g['theta']}")
                st.caption(f"Exit {sl_pts} pts below entry price")
    else:
        st.error("Error fetching spot price. Check if your Token is expired.")
        break
        
    time.sleep(3)
