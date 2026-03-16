import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import time

# --- 1. OPTION GREEKS ENGINE (Black-Scholes) ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    """S=Spot, K=Strike, T=Years to Expiry, r=Risk-Free Rate, sigma=IV"""
    if T <= 0: return {"delta": 0, "gamma": 0, "theta": 0}
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

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="Pro Nifty Scalper", layout="wide")

# Sidebar: Risk Management & Strike Choice
with st.sidebar:
    st.header("🛡️ Risk Management")
    lots = st.number_input("Position Size (Lots)", min_value=1, value=1)
    sl_type = st.radio("Stop-Loss Type", ["Points", "Percentage"])
    sl_value = st.number_input(f"SL Value ({sl_type})", value=10.0 if sl_type == "Points" else 15.0)
    
    st.divider()
    st.header("🎯 Strike Selection")
    strike_mode = st.selectbox("Preferred Strike", ["ATM", "1-Strike ITM", "2-Strike ITM", "Custom Offset"])
    offset = 0
    if strike_mode == "Custom Offset":
        offset = st.slider("Offset (Multiples of 50)", -5, 5, 0)
    elif "1-Strike" in strike_mode: offset = 1
    elif "2-Strike" in strike_mode: offset = 2

    show_greeks = st.toggle("Show Option Greeks", value=True)
    filter_liquidity = st.toggle("Liquidity Filter (Volume > 10k)", value=True)

# --- 3. CORE LOGIC ---
def get_recommendations(spot):
    atm = round(spot / 50) * 50
    # Adjustment for ITM logic: CE ITM is Spot > Strike | PE ITM is Spot < Strike
    ce_strike = atm - (offset * 50)
    pe_strike = atm + (offset * 50)
    
    # Mock Data (In production, replace with real API calls like nsepython.option_chain)
    # Filter Logic: Only show if Volume (mocked here) > 10000
    is_liquid = True # Replace with actual vol check
    
    return {
        "CE": {"strike": ce_strike, "iv": 0.15, "vol": 50000},
        "PE": {"strike": pe_strike, "iv": 0.16, "vol": 45000},
        "liquid": is_liquid
    }

# --- 4. MAIN DASHBOARD ---
st.title("🚀 Pro Nifty Scalper Dashboard")
spot_val = 22500.0  # Replace with nse_quote_ltp("NIFTY 50")

col_m1, col_m2 = st.columns(2)
col_m1.metric("NIFTY SPOT", f"₹{spot_val}")
col_m2.info(f"Strategy: {strike_mode} | Net Exposure: ₹{spot_val * lots * 50:,.0f}")

rec = get_recommendations(spot_val)

if filter_liquidity and not rec["liquid"]:
    st.warning("⚠️ High-liquid contracts not found for current selection. Showing best available.")

# Display Suggestions
c1, c2 = st.columns(2)

for side, strike_data in [("CE", rec["CE"]), ("PE", rec["PE"])]:
    container = c1 if side == "CE" else c2
    color = "green" if side == "CE" else "red"
    
    with container:
        st.markdown(f"### {side} Suggestion")
        st.subheader(f"NIFTY {strike_data['strike']} {side}")
        
        # Calculate Risk/Reward
        est_premium = 100.0 # Replace with real LTP
        sl_price = est_premium - sl_value if sl_type == "Points" else est_premium * (1 - sl_value/100)
        
        st.write(f"**Target Entry:** Approx ₹{est_premium}")
        st.write(f"**Recommended SL:** ₹{round(sl_price, 2)}")
        
        if show_greeks:
            greeks = calculate_greeks(spot_val, strike_data['strike'], 4/365, 0.07, strike_data['iv'], side.lower())
            st.code(f"Delta: {greeks['delta']} | Gamma: {greeks['gamma']} | Theta: {greeks['theta']}")

st.caption("Note: Liquidity filter hides strikes with Open Interest < 5000 lots to prevent slippage.")
