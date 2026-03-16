import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
from upstox_client.rest import ApiException
import time

# --- 1. SECRETS LOADING ---
try:
    # TOML structure must be: [upstox] \n access_token = "..."
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
        delta = norm.cdf(d1) - 1 # Corrected Put Delta (Negative)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return {"delta": round(delta, 2), "gamma": round(gamma, 4), "theta": round(theta / 365, 2)}

# --- 3. UPSTOX API SETUP ---
def get_api_instance(token):
    config = upstox_client.Configuration()
    config.access_token = token
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
        # EXACT CASE SENSITIVE KEY
        instrument_key = "NSE_INDEX|Nifty 50" 
        
        # CORRECT SDK METHOD: .ltp(symbol, api_version)
        api_response = api_instance.ltp(instrument_key, '2.0')
        
        # Accessing data safely
        if instrument_key in api_response.data:
            spot = api_response.data[instrument_key].last_price
            
            # Calculate Strikes
            atm = int(round(spot / 50) * 50)
            ce_strike = atm - (offset * 50)
            pe_strike = atm + (offset * 50)
            
            with placeholder.container():
                # Real-time Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("NIFTY 50 SPOT", f"₹{spot}")
                m2.metric("NET EXPOSURE", f"₹{spot * lots * lot_size:,.0f}")
                m3.metric("ATM STRIKE", atm)
                
                st.divider()
                
                # Trading Recommendations
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"🟢 CALL: {ce_strike} CE")
                    # Assuming 4 days to expiry, 7% rate, 15% IV
                    g = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                    st.write(f"**Delta:** `{g['delta']}` | **Theta:** `{g['theta']}`")
                    st.caption(f"Strategy: Buy at CMP | SL: {sl_pts} pts")

                with c2:
                    st.error(f"🔴 PUT: {pe_strike} PE")
                    g = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                    st.write(f"**Delta:** `{g['delta']}` | **Theta:** `{g['theta']}`")
                    st.caption(f"Strategy: Buy at CMP | SL: {sl_pts} pts")
        else:
            st.error(f"Instrument '{instrument_key}' not found. Check if market is closed.")
            
    except ApiException as e:
        st.error(f"Upstox API Error: {e.body if hasattr(e, 'body') else e}")
        break
    except Exception as e:
        st.error(f"System Error: {e}")
        break
        
    time.sleep(2)
