import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
from upstox_client.rest import ApiException
import time

# --- 1. SECRETS LOADING ---
try:
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing 'access_token' in .streamlit/secrets.toml")
    st.stop()

# --- 2. OPTION GREEKS ENGINE ---
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    S: Spot Price, K: Strike Price, T: Time to Expiry (Years), 
    r: Risk-free rate, sigma: Volatility (IV)
    """
    if T <= 0 or sigma <= 0: 
        return {"delta": 0, "gamma": 0, "theta": 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
        # Call Theta formula
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1 # Put Delta is always negative
        # Put Theta formula
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    # Return Theta as daily decay
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
    lot_size = st.number_input("Lot Size (Nifty)", value=25) # Nifty changed to 25
    sl_pts = st.number_input("Stop Loss (Points)", value=15.0)
    strike_mode = st.selectbox("Strike Choice", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
    st.divider()
    st.info("🔄 Refresh Rate: 1 second")

st.title("🚀 Nifty Live Scalper Dashboard")
placeholder = st.empty()

# Initialize API once
api_instance = get_api_instance(TOKEN)

# --- 5. PERSISTENT LIVE LOOP ---
while True:
    try:
        instrument_key = "NSE_INDEX|Nifty 50"
        # Fetching LTP (Last Traded Price)
        api_response = api_instance.get_ltp(instrument_key, 'v2')
        spot = api_response.data[instrument_key].last_price
        
        if spot:
            # Nifty Strike logic (steps of 50)
            atm = int(round(spot / 50) * 50)
            ce_strike = atm - (offset * 50)
            pe_strike = atm + (offset * 50)
            
            with placeholder.container():
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("NIFTY 50 SPOT", f"₹{spot}")
                m2.metric("NET EXPOSURE", f"₹{spot * lots * lot_size:,.0f}")
                m3.metric("ATM STRIKE", atm)
                
                st.divider()
                
                # Trading Panels
                c1, c2 = st.columns(2)
                
                # CALL Side (Using generic IV of 15% and 4 days to expiry)
                with c1:
                    st.success(f"🟢 CALL OPTION: {ce_strike} CE")
                    g_ce = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                    st.write(f"**Delta:** `{g_ce['delta']}` | **Theta:** `{g_ce['theta']}`")
                    st.caption(f"Strategy: Long {lots} Lot(s) | SL: {sl_pts} pts")

                # PUT Side
                with c2:
                    st.error(f"🔴 PUT OPTION: {pe_strike} PE")
                    g_pe = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                    st.write(f"**Delta:** `{g_pe['delta']}` | **Theta:** `{g_pe['theta']}`")
                    st.caption(f"Strategy: Long {lots} Lot(s) | SL: {sl_pts} pts")
        
    except ApiException as e:
        st.error(f"Upstox API Error: {e}")
        break
    except Exception as e:
        st.error(f"System Error: {e}")
        break
        
    time.sleep(1) # Refresh interval
