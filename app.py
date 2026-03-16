import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
from upstox_client.rest import ApiException
import time

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="Nifty Live Scalper", layout="wide", page_icon="🚀")

try:
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing 'access_token' in .streamlit/secrets.toml")
    st.stop()

# --- 2. GREEKS ENGINE (Black-Scholes) ---
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

# --- 3. UPSTOX API HANDLER ---
def get_market_data():
    configuration = upstox_client.Configuration()
    configuration.access_token = TOKEN
    api_instance = upstox_client.MarketQuoteApi(upstox_client.ApiClient(configuration))
    
    try:
        # Use get_market_quote_ohlc as it is more stable for Indices
        # The key must be exactly "NSE_INDEX|Nifty 50"
        api_response = api_instance.get_market_quote_ohlc("NSE_INDEX|Nifty 50", "1d", '2.0')
        
        if api_response.status == "success" and "NSE_INDEX|Nifty 50" in api_response.data:
            return api_response.data["NSE_INDEX|Nifty 50"].last_price
        return None
    except ApiException as e:
        st.sidebar.error(f"API Error: {e.reason}")
        return None

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🛡️ Strategy & Risk")
    run_live = st.toggle("🚀 Start Live Dashboard", value=False)
    st.divider()
    lots = st.number_input("Lots", min_value=1, value=1)
    lot_size = 25  # Nifty current lot size
    strike_mode = st.selectbox("Strike Choice", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]
    
    st.subheader("Greeks Input")
    iv = st.slider("Implied Volatility (IV)", 0.05, 0.50, 0.15)
    dte = st.slider("Days to Expiry", 0, 7, 4) / 365

# --- 5. LIVE DASHBOARD LOOP ---
st.title("⚡ Nifty Live Scalper")
placeholder = st.empty()

if run_live:
    while True:
        spot = get_market_data()
        
        if spot:
            # Rounding to nearest 50 for Nifty ATM
            atm = int(round(spot / 50) * 50)
            ce_strike = atm - (offset * 50)
            pe_strike = atm + (offset * 50)
            
            with placeholder.container():
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("NIFTY 50 SPOT", f"₹{spot}")
                m2.metric("EXPOSURE", f"₹{spot * lots * lot_size:,.0f}")
                m3.metric("ATM STRIKE", atm)
                
                st.divider()
                
                # Option Cards
                c1, c2 = st.columns(2)
                
                # Call Section
                with c1:
                    st.success(f"🟢 CALL: {ce_strike} CE")
                    g_ce = calculate_greeks(spot, ce_strike, dte, 0.07, iv, "call")
                    st.write(f"**Delta:** `{g_ce['delta']}` | **Theta:** `{g_ce['theta']}`")
                
                # Put Section
                with c2:
                    st.error(f"🔴 PUT: {pe_strike} PE")
                    g_pe = calculate_greeks(spot, pe_strike, dte, 0.07, iv, "put")
                    st.write(f"**Delta:** `{g_pe['delta']}` | **Theta:** `{g_pe['theta']}`")
        
        else:
            st.warning("Waiting for data from Upstox... Check console or sidebar for errors.")
        
        time.sleep(1)  # Refresh rate
else:
    st.info("Dashboard is currently paused. Toggle 'Start Live Dashboard' in the sidebar to begin.")

