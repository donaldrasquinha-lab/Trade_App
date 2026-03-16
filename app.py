import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
import requests
import time

# --- 1. CONFIGURATION (REPLACE THESE) ---
API_KEY = "YOUR_API_KEY_HERE"
API_SECRET = "YOUR_API_SECRET_HERE"
REDIRECT_URI = "http://localhost:8501"

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

# --- 3. UPSTOX AUTHENTICATION LOGIC ---
def get_access_token(auth_code):
    url = "https://api.upstox.com"
    payload = {
        'code': auth_code, 
        'client_id': API_KEY, 
        'client_secret': API_SECRET,
        'redirect_uri': REDIRECT_URI, 
        'grant_type': 'authorization_code'
    }
    headers = {'accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=payload, headers=headers)
    return response.json().get('access_token')

def get_live_spot(token):
    config = upstox_client.Configuration()
    config.access_token = token
    api_instance = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
    try:
        # Correct Instrument Key for Nifty 50 Spot
        instrument_key = "NSE_INDEX|Nifty 50"
        res = api_instance.ltp(instrument_key, 'v2')
        return res.data[instrument_key].last_price
    except Exception as e:
        return None

# --- 4. UI SETUP ---
st.set_page_config(page_title="Pro Nifty Scalper", layout="wide")

# Sidebar: Auth & Risk Management
with st.sidebar:
    st.title("🔑 Upstox Login")
    
    # Check for 'code' in the browser URL
    query_params = st.query_params
    auth_code = query_params.get("code")
    
    if 'token' not in st.session_state:
        if not auth_code:
            # FIXED URL STRUCTURE: Added missing slashes and question marks
            login_url = f"https://api.upstox.com{API_KEY}&redirect_uri={REDIRECT_URI}"
            st.link_button("Authorize Upstox", login_url, type="primary")
            st.warning("Please click above to login.")
            st.stop()
        else:
            with st.spinner("Generating Token..."):
                token = get_access_token(auth_code)
                if token:
                    st.session_state.token = token
                    st.success("Authenticated!")
                    st.rerun() # Refresh to clean URL parameters
                else:
                    st.error("Authentication Failed. Check API Keys.")
                    st.stop()
    
    if st.button("Logout / Reset"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

    st.divider()
    st.header("🛡️ Risk & Strategy")
    lots = st.number_input("Lots", min_value=1, value=1)
    sl_val = st.number_input("Stop Loss (Points)", value=15.0)
    mode = st.selectbox("Strike Mode", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[mode]

# --- 5. MAIN DASHBOARD ---
st.title("🚀 Nifty Live Scalper")

# Placeholder for real-time updates
placeholder = st.empty()

if 'token' in st.session_state:
    while True:
        spot = get_live_spot(st.session_state.token)
        
        if spot:
            atm = round(spot / 50) * 50
            ce_strike = atm - (offset * 50)
            pe_strike = atm + (offset * 50)
            
            with placeholder.container():
                # Top Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("NIFTY 50 SPOT", f"₹{spot}")
                col_m2.metric("NET EXPOSURE", f"₹{spot * lots * 50:,.0f}")
                col_m3.metric("ATM STRIKE", atm)
                
                st.divider()
                
                # Trading Cards
                c1, c2 = st.columns(2)
                
                with c1:
                    st.success(f"🟢 CALL OPTION: {ce_strike} CE")
                    g = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                    st.info(f"**Greeks:** Δ: {g['delta']} | Γ: {g['gamma']} | Θ: {g['theta']}")
                    st.caption(f"Suggested Exit: ₹{sl_val} pts Stop Loss")

                with c2:
                    st.error(f"🔴 PUT OPTION: {pe_strike} PE")
                    g = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                    st.info(f"**Greeks:** Δ: {g['delta']} | Γ: {g['gamma']} | Θ: {g['theta']}")
                    st.caption(f"Suggested Exit: ₹{sl_val} pts Stop Loss")
        else:
            st.error("Unable to fetch live data. Ensure market is open or check API status.")
            break
        
        time.sleep(3) # Refresh speed (Upstox rate limits apply)
