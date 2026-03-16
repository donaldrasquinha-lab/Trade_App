import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
from upstox_client.rest import ApiException
import time

# --- 1. SECRETS LOADING ---
try:
    # Ensure .streamlit/secrets.toml has: [upstox] \n access_token = "..."
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing 'access_token' in .streamlit/secrets.toml. Check your TOML formatting.")
    st.stop()

# --- 2. OPTION GREEKS ENGINE ---
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: return {"delta": 0, "gamma": 0, "theta": 0}
    
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
    api_client = upstox_client.ApiClient(config)
    return upstox_client.MarketQuoteApi(api_client)

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

# Initialize API Instance
api_instance = get_api_instance(TOKEN)

# --- 5. PERSISTENT LIVE LOOP ---
while True:
    try:
        # EXACT KEY: NSE_INDEX|Nifty 50 (Pipe is correct for Nifty)
        instrument_key = "NSE_INDEX|Nifty 50" 
        
        # VERSION: '2.0' (String format required for v2 SDK)
        api_response = api_instance.ltp(instrument_key, '2.0')
        
        if api_response.status == 'success' and instrument_key in api_response.data:
            spot = api_response.data[instrument_key].last_price
            
            # Strike Logic
            atm = int(round(spot / 50) * 50)
            ce_strike = atm - (offset * 50)
            pe_strike = atm + (offset * 50)
            
            with placeholder.container():
                # Dashboard Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("NIFTY 50 SPOT", f"₹{spot}")
                m2.metric("NET EXPOSURE", f"₹{spot * lots * lot_size:,.0f}")
                m3.metric("ATM STRIKE", atm)
                
                st.divider()
                
                # Trade Signals
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"🟢 CALL: {ce_strike} CE")
                    g = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                    st.write(f"**Delta:** `{g['delta']}` | **Theta:** `{g['theta']}`")
                    st.caption(f"Entry: CMP | SL: {sl_pts} pts")

                with c2:
                    st.error(f"🔴 PUT: {pe_strike} PE")
                    g = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                    st.write(f"**Delta:** `{g['delta']}` | **Theta:** `{g['theta']}`")
                    st.caption(f"Entry: CMP | SL: {sl_pts} pts")
        else:
            st.error(f"Error: Could not find '{instrument_key}'. API Response: {api_response.status}")
            
    except ApiException as e:
        # Parsing error body to see the specific UDAPI code
        st.error(f"Upstox API Error: {e.body}")
        break
    except Exception as e:
        st.error(f"System Error: {e}")
        break
        
    time.sleep(2)
