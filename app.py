import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
import requests
import time

# --- CONFIGURATION FROM SECRETS ---
try:
    API_KEY = st.secrets["upstox"]["api_key"]
    API_SECRET = st.secrets["upstox"]["api_secret"]
    REDIRECT_URI = st.secrets["upstox"]["redirect_uri"]
except KeyError:
    st.error("Missing secrets! Add [upstox] section to your .streamlit/secrets.toml file.")
    st.stop()

# --- OPTION GREEKS ENGINE (Black-Scholes) ---
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

# --- UPSTOX AUTH & DATA FETCHING ---
def get_access_token(auth_code):
    url = "https://api.upstox.com"
    payload = {
        'code': auth_code, 'client_id': API_KEY, 'client_secret': API_SECRET,
        'redirect_uri': REDIRECT_URI, 'grant_type': 'authorization_code'
    }
    headers = {'accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=payload, headers=headers)
    return response.json().get('access_token')

def get_live_spot(token):
    config = upstox_client.Configuration()
    config.access_token = token
    api_instance = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
    try:
        instrument_key = "NSE_INDEX|Nifty 50"
        res = api_instance.ltp(instrument_key, 'v2')
        return res.data[instrument_key].last_price
    except Exception: return None

# --- UI SETUP ---
st.set_page_config(page_title="Nifty Pro Scalper", layout="wide")

with st.sidebar:
    st.title("🔑 Upstox Login")
    auth_code = st.query_params.get("code")
    
    if 'token' not in st.session_state:
        if not auth_code:
            login_url = f"https://api.upstox.com{API_KEY}&redirect_uri={REDIRECT_URI}"
            st.link_button("Authorize Upstox", login_url, type="primary")
            st.stop()
        else:
            token = get_access_token(auth_code)
            if token:
                st.session_state.token = token
                st.rerun()
            else:
                st.error("Auth Failed. Check your API Keys/Redirect URI.")
                st.stop()

    if st.button("Reset Session"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

    st.divider()
    st.header("🛡️ Risk & Strategy")
    lots = st.number_input("Lots", min_value=1, value=1)
    sl_pts = st.number_input("Stop Loss (Points)", value=15.0)
    strike_mode = st.selectbox("Strike Choice", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]

# --- MAIN DASHBOARD ---
st.title("🚀 Nifty Live Scalper Dashboard")
placeholder = st.empty()

if 'token' in st.session_state:
    while True:
        spot = get_live_spot(st.session_state.token)
        if spot:
            atm = round(spot / 50) * 50
            ce_strike = atm - (offset * 50)
            pe_strike = atm + (offset * 50)
            
            with placeholder.container():
                # Summary Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("NIFTY SPOT", f"₹{spot}")
                m2.metric("NET EXPOSURE", f"₹{spot * lots * 50:,.0f}")
                m3.metric("ATM STRIKE", atm)
                
                st.divider()
                
                # Option Suggestions
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"🟢 CALL: {ce_strike} CE")
                    g = calculate_greeks(spot, ce_strike, 4/365, 0.07, 0.15, "call")
                    st.info(f"**Greeks:** Δ: {g['delta']} | Θ: {g['theta']}")
                    st.caption(f"Suggested SL: {sl_pts} pts below entry")
                
                with c2:
                    st.error(f"🔴 PUT: {pe_strike} PE")
                    g = calculate_greeks(spot, pe_strike, 4/365, 0.07, 0.15, "put")
                    st.info(f"**Greeks:** Δ: {g['delta']} | Θ: {g['theta']}")
                    st.caption(f"Suggested SL: {sl_pts} pts below entry")
        else:
            st.warning("Waiting for live data... ensure market is open.")
            
        time.sleep(3)
