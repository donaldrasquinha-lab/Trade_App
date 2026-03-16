import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import upstox_client
import time

# --- 1. SECRETS ---
TOKEN = st.secrets["upstox"]["access_token"]

# --- 2. GREEKS ENGINE (Vectorized for speed) ---
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

# --- 3. UI ---
st.set_page_config(page_title="Nifty Scalper", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    run_live = st.toggle("🚀 Go Live", value=False)
    lots = st.number_input("Lots", 1, 100, 1)
    iv = st.slider("Implied Volatility (IV)", 0.05, 0.50, 0.15)
    dte = st.slider("Days to Expiry", 0, 7, 4) / 365

st.title("⚡ Nifty Live Scalper")
placeholder = st.empty()

# --- 4. API & LOOP ---
def get_data():
    config = upstox_client.Configuration()
    config.access_token = TOKEN
    api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
    # Note: Use 'NSE_INDEX|Nifty 50' for Nifty Spot
    res = api.get_ltp_full("NSE_INDEX|Nifty 50", '2.0')
    # Accessing data based on Upstox V2 Response Structure
    return res.data["NSE_INDEX|Nifty 50"].last_price

if run_live:
    while True:
        try:
            spot = get_data()
            atm = int(round(spot / 50) * 50)
            
            with placeholder.container():
                cols = st.columns(3)
                cols[0].metric("SPOT", f"₹{spot}")
                cols[1].metric("ATM", atm)
                cols[2].metric("EXP.", f"₹{spot * lots * 25:,.0f}")

                c1, c2 = st.columns(2)
                ce_g = calculate_greeks(spot, atm, dte, 0.07, iv, "call")
                pe_g = calculate_greeks(spot, atm, dte, 0.07, iv, "put")
                
                c1.info(f"**CALL {atm}**  \nDelta: `{ce_g['delta']}`  \nTheta: `{ce_g['theta']}`")
                c2.error(f"**PUT {atm}**  \nDelta: `{pe_g['delta']}`  \nTheta: `{pe_g['theta']}`")
                
            time.sleep(1) # Frequency limit for Upstox API
        except Exception as e:
            st.error(f"Connection Error: {e}")
            time.sleep(5)
else:
    st.warning("Dashboard Paused. Toggle 'Go Live' in the sidebar to start.")
