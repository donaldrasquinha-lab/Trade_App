import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
import requests
import time

# --- 1. CONFIGURATION (Replace with your actual keys) ---
API_KEY = "YOUR_UPSTOX_API_KEY"
API_SECRET = "YOUR_UPSTOX_API_SECRET"
REDIRECT_URI = "http://localhost:8501"  # Matches Upstox App Config

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
        'code': auth_code, 'client_id': API_KEY, 'client_secret': API_SECRET,
        'redirect_uri': REDIRECT_URI, 'grant_type': 'authorization_code'
    }
    headers = {'accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=payload, headers=headers)
    return response.json().get('access_token')

def get_live_full_data(token, instrument_key):
    config = upstox_client.Configuration()
    config.access_token = token
    api_instance = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
    try:
        res = api_instance.ltp(instrument_key, 'v2')
        return res.data[instrument_key].last_price
    except Exception: return None

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Pro Nifty Scalper", layout="wide")

# Sidebar: Auth & Risk
with st.sidebar:
    st.title("🔑 Upstox Login")
    auth_code = st.query_params.get("code")
    
    if 'token' not in st.session_state:
        if not auth_code:
            login_url = f"https://api.upstox.com{API_KEY}&redirect_uri={REDIRECT_URI}"
            st.link_button("Authorize Upstox", login_url)
            st.stop()
        else:
            st.session_state.token = get_access_token(auth_code)
            st.success("Authenticated!")

    st.divider()
    st.header("🛡️ Risk Management")
    lots = st.number_input("Lots", min_value=1, value=1)
    sl_type = st.radio("SL Type", ["Points", "Percentage"])
    sl_val = st.number_input("SL Value", value=10.0)
    
    st.header("🎯 Strike Selection")
    mode = st.selectbox("Strike", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[mode]

# --- 5. LIVE DASHBOARD ---
st.title("🚀 Pro Nifty Scalper Dashboard")

# Refresh every 5 seconds
placeholder = st.empty()

while True:
    spot = get_live_full_data(st.session_state.token, "NSE_INDEX|Nifty 50")
    
    if spot:
        atm = round(spot / 50) * 50
        ce_strike = atm - (offset * 50)
        pe_strike = atm + (offset * 50)
        
        with placeholder.container():
            m1, m2 = st.columns(2)
            m1.metric("NIFTY 50 SPOT", f"₹{spot}")
            m2.info(f"Net Exposure: ₹{spot * lots * 50:,.0f}")
            
            c1, c2 = st.columns(2)
            # Call Option Column
            with c1:
                st.success(f"🟢 BUY NIFTY {ce_strike} CE")
                g = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                st.code(f"Delta: {g['delta']} | Gamma: {g['gamma']} | Theta: {g['theta']}")
                st.caption(f"Suggested SL: ₹{sl_val} {sl_type} below entry")

            # Put Option Column
            with c2:
                st.error(f"🔴 BUY NIFTY {pe_strike} PE")
                g = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                st.code(f"Delta: {g['delta']} | Gamma: {g['gamma']} | Theta: {g['theta']}")
                st.caption(f"Suggested SL: ₹{sl_val} {sl_type} below entry")
    
    time.sleep(5)
