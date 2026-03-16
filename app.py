import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
import time

# --- 1. SECRETS LOADING ---
try:
    # Ensure your .streamlit/secrets.toml has: [upstox] access_token = "..."
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing 'access_token' in .streamlit/secrets.toml")
    st.stop()

# --- 2. OPTION GREEKS ENGINE ---
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: 
        return {"delta": 0, "gamma": 0, "theta": 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return {"delta": round(delta, 2), "gamma": round(gamma, 4), "theta": round(theta / 365, 2)}

# --- 3. UPSTOX API SETUP ---
def get_api_instance(token):
    config = upstox_client.Configuration()
    config.access_token = token
    # Initialize the specific Market Quote API
    return upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Nifty Live Scalper", layout="wide")

with st.sidebar:
    st.header("🛡️ Strategy & Risk")
    lots = st.number_input("Lots", min_value=1, value=1)
    lot_size = st.number_input("Lot Size (Nifty)", value=25) 
    sl_pts = st.number_input("Stop Loss (Points)", value=15.0)
    strike_mode = st.selectbox("Strike Choice", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
    st.divider()
    st.info("Refreshing every 2 seconds...")

st.title("🚀 Nifty Live Scalper Dashboard")
placeholder = st.empty()

# Initialize API once outside the loop
api_instance = get_api_instance(TOKEN)

# --- 5. PERSISTENT LIVE LOOP ---
while True:
    try:
        # Correct Instrument Key for Nifty 50 Index
        instrument_key = "NSE_INDEX|Nifty 50"
        
        # FIX: Use .ltp() instead of .get_ltp() for MarketQuoteApi
        # The version string must be 'v2' or '2.0' depending on SDK version
        api_response = api_instance.ltp(instrument_key, 'v2')
        
        # Access data using dictionary key
        quote_data = api_response.data[instrument_key]
        spot = quote_data.last_price
        
        if spot:
            # Calculate Strikes
            atm = int(round(spot / 50) * 50)
            ce_strike = atm - (offset * 50)
            pe_strike = atm + (offset * 50)
            
            with placeholder.container():
                # Top Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("NIFTY 50 SPOT", f"₹{spot}")
                m2.metric("NET EXPOSURE", f"₹{spot * lots * lot_size:,.0f}")
                m3.metric("ATM STRIKE", atm)
                
                st.divider()
                
                # Trading Recommendations
                c1, c2 = st.columns(2)
                
                with c1:
                    st.success(f"🟢 CALL: {ce_strike} CE")
                    # Assuming 4 days to expiry, 7% risk-free rate, 15% IV
                    g = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                    st.write(f"**Delta:** `{g['delta']}` | **Theta:** `{g['theta']}`")
                    st.caption(f"Target Entry: CMP | SL: {sl_pts} pts")

                with c2:
                    st.error(f"🔴 PUT: {pe_strike} PE")
                    g = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                    st.write(f"**Delta:** `{g['delta']}` | **Theta:** `{g['theta']}`")
                    st.caption(f"Target Entry: CMP | SL: {sl_pts} pts")
                    
    except Exception as e:
        st.error(f"Error: {e}")
        # If it's a 'NoneType' error, the market might be closed or symbol wrong
        st.warning("Check if the market is open or your token is still valid.")
        break
        
    time.sleep(2)
