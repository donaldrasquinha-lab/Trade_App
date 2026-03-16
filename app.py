"""
Multi-Index Live Scalper Dashboard
Upstox REST Market Quote API -- works on Streamlit Cloud
---------------------------------------------------------
SETUP:
  1. pip install -r requirements.txt
  2. Create .streamlit/secrets.toml:
       [upstox]
       access_token = "your_token_here"
  3. streamlit run scalper_dashboard.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
import upstox_client

# ==============================================================
# 1. PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Multi-Index Scalper",
    layout="wide",
    page_icon="⚡",
)

# ==============================================================
# 2. SECRETS
# ==============================================================
try:
    TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    st.error("Missing `access_token` in `.streamlit/secrets.toml`")
    st.stop()

# ==============================================================
# 3. INDEX METADATA
# ==============================================================
INDEX_CONFIG = {
    "NIFTY 50": {
        "instrument_key": "NSE_INDEX|Nifty 50",
        "lot_size": 75,
        "strike_step": 50,
    },
    "BANK NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "lot_size": 30,
        "strike_step": 100,
    },
    "FINNIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Fin Service",
        "lot_size": 65,
        "strike_step": 50,
    },
    "MIDCAP NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Midcap Select",
        "lot_size": 120,
        "strike_step": 25,
    },
}

ALL_INSTRUMENT_KEYS = [v["instrument_key"] for v in INDEX_CONFIG.values()]

# ==============================================================
# 4. SESSION STATE INIT
# ==============================================================
if "token_ok"    not in st.session_state: st.session_state.token_ok    = None
if "token_msg"   not in st.session_state: st.session_state.token_msg   = ""
if "live_feed"   not in st.session_state: st.session_state.live_feed   = False
if "last_prices" not in st.session_state: st.session_state.last_prices = {}
if "show_debug"  not in st.session_state: st.session_state.show_debug  = True

# ==============================================================
# 5. PRICE FETCH  (with full debug output)
# ==============================================================
def fetch_all_prices(token, debug=False):
    """
    Fetches LTP for all indices via REST.
    Returns (prices_dict, error_string_or_None).
    If debug=True writes raw response into st.sidebar.
    """
    try:
        conf = upstox_client.Configuration()
        conf.access_token = token
        api  = upstox_client.MarketQuoteApi(upstox_client.ApiClient(conf))

        keys_str = ",".join(ALL_INSTRUMENT_KEYS)
        res = api.get_market_quote_ohlc(keys_str, "1d", "2.0")

        if debug:
            st.sidebar.markdown("**Raw API response:**")
            st.sidebar.write(f"status: `{res.status}`")
            st.sidebar.write(f"data type: `{type(res.data)}`")
            if res.data:
                st.sidebar.write(f"data keys: `{list(res.data.keys())}`")
                # Show full first item so we can see field names
                first_key = list(res.data.keys())[0]
                first_val = res.data[first_key]
                st.sidebar.write(f"first key: `{first_key}`")
                st.sidebar.write(f"first value: `{first_val}`")
                # Try common field names
                for attr in ["last_price", "ltp", "close_price", "ohlc",
                             "last_traded_price", "avg_price"]:
                    val = getattr(first_val, attr, "NOT FOUND")
                    st.sidebar.write(f"  .{attr} = `{val}`")
            else:
                st.sidebar.write("data is None or empty")

        prices = {}
        if res.status == "success" and res.data:
            for ikey, quote in res.data.items():
                # Try all possible LTP field names
                ltp = (getattr(quote, "last_price",        None) or
                       getattr(quote, "ltp",               None) or
                       getattr(quote, "last_traded_price", None))

                # Try all possible close field names
                ohlc  = getattr(quote, "ohlc", None)
                close = None
                if ohlc:
                    close = (getattr(ohlc, "close", None) or
                             getattr(ohlc, "close_price", None))
                if close is None:
                    close = (getattr(quote, "close_price", None) or
                             getattr(quote, "prev_close",  None))

                if ltp is None:
                    continue

                ltp   = float(ltp)
                close = float(close) if close else ltp
                change_pct = round(((ltp - close) / close) * 100, 2) if close else 0.0

                prices[ikey] = {
                    "ltp":        ltp,
                    "close":      close,
                    "change_pct": change_pct,
                    "ts":         datetime.now(),
                }

        return prices, None

    except Exception as e:
        return {}, str(e)


def validate_token(token):
    try:
        conf = upstox_client.Configuration()
        conf.access_token = token
        api  = upstox_client.MarketQuoteApi(upstox_client.ApiClient(conf))
        api.get_market_quote_ohlc(ALL_INSTRUMENT_KEYS[0], "1d", "2.0")
        return True, "Token valid"
    except Exception as e:
        return False, str(e)

# ==============================================================
# 6. GREEKS ENGINE
# ==============================================================
def greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0:
        return dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    pdf1 = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = pdf1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf1 * np.sqrt(T) / 100
    return dict(
        delta = round(delta,       4),
        gamma = round(gamma,       6),
        vega  = round(vega,        2),
        theta = round(theta / 365, 2),
        rho   = round(rho,         4),
    )

# ==============================================================
# 7. HELPERS
# ==============================================================
def fmt_inr(v):
    if v >= 1e7: return f"Rs.{v/1e7:.2f} Cr"
    if v >= 1e5: return f"Rs.{v/1e5:.2f} L"
    return f"Rs.{v:,.0f}"

def sign_str(v):
    return f"+{v}" if v >= 0 else str(v)

# ==============================================================
# 8. SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## Scalper Controls")
    st.divider()

    selected_index = st.selectbox("Index", list(INDEX_CONFIG.keys()))
    conf = INDEX_CONFIG[selected_index]

    st.divider()
    st.markdown("### Strategy")
    run_live      = st.toggle("Start Live Feed", key="live_feed")
    lots          = st.number_input("Lots", min_value=1, max_value=500, value=1, step=1)
    strike_mode   = st.selectbox("Strike Selection", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    strike_offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]

    st.divider()
    st.markdown("### Greeks Parameters")
    iv_pct    = st.slider("Implied Volatility (%)", 5,  80, 15, step=1)
    iv        = iv_pct / 100.0
    dte_days  = st.slider("Days to Expiry",          0,  30,  4, step=1)
    dte       = dte_days / 365.0
    risk_free = st.slider("Risk-Free Rate (%)",       4,  12,  7, step=1)
    r         = risk_free / 100.0

    st.divider()
    refresh_ms = st.select_slider(
        "Refresh Interval",
        options=[1000, 2000, 3000, 5000],
        value=2000,
        format_func=lambda x: f"{x}ms",
    )

    st.divider()
    st.markdown("### Connection")

    if st.button("Check Token"):
        with st.spinner("Validating..."):
            ok, msg = validate_token(TOKEN)
            st.session_state.token_ok  = ok
            st.session_state.token_msg = msg

    if st.session_state.token_ok is True:
        st.success("Token valid")
    elif st.session_state.token_ok is False:
        st.error(f"Token error: {st.session_state.token_msg}")
    else:
        st.caption("Press Check Token to validate")

    st.divider()
    st.caption("Market hours: 9:15 AM - 3:30 PM IST, Mon-Fri")

    # Debug toggle -- leave ON until prices are confirmed working
    st.divider()
    show_debug = st.toggle("Show Debug Info", value=True, key="show_debug")

# ==============================================================
# 9. FETCH PRICES
# ==============================================================
fetch_error = None

if run_live:
    prices, fetch_error = fetch_all_prices(TOKEN, debug=show_debug)
    if prices:
        st.session_state.last_prices = prices
else:
    prices = {}

# Show fetch error prominently if present
if fetch_error:
    st.sidebar.error(f"Fetch error: {fetch_error}")

# Use last known prices if current fetch empty
all_prices = st.session_state.last_prices
ikey       = conf["instrument_key"]
feed_entry = all_prices.get(ikey)

if feed_entry:
    spot       = feed_entry["ltp"]
    change_pct = feed_entry["change_pct"]
    data_age   = (datetime.now() - feed_entry["ts"]).total_seconds()
else:
    spot = change_pct = data_age = None

# ==============================================================
# 10. PAGE HEADER
# ==============================================================
st.markdown(f"# {selected_index} Scalper")

h1, h2, h3 = st.columns([2, 2, 3])
with h1:
    if run_live and fetch_error:
        st.caption(f"🔴 API error")
    elif run_live and spot:
        st.caption("🟢 REST feed live")
    elif run_live:
        st.caption("🟡 Fetching... (market may be closed)")
    else:
        st.caption("⚪ Paused -- toggle Live Feed to start")
with h2:
    if data_age is not None:
        st.caption(f"Last update: {'< 1s' if data_age < 1 else f'{data_age:.0f}s'} ago")
with h3:
    st.caption(f"Instrument: `{ikey}`")

st.divider()

# ==============================================================
# 11. TOP METRICS
# ==============================================================
step = conf["strike_step"]

if spot:
    atm       = int(round(spot / step) * step)
    exposure  = spot * lots * conf["lot_size"]
    ce_strike = atm - strike_offset * step
    pe_strike = atm + strike_offset * step
else:
    atm = ce_strike = pe_strike = exposure = None

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    f"{selected_index} Spot",
    f"Rs.{spot:,.2f}" if spot else "--",
    f"{change_pct:+.2f}%" if change_pct is not None else None,
)
m2.metric("ATM Strike",     f"{atm:,}"        if atm     else "--")
m3.metric("Total Exposure", fmt_inr(exposure) if exposure else "--",
          f"{lots} lot x {conf['lot_size']}"  if exposure else None)
m4.metric("DTE", f"{dte_days}d", f"IV {iv_pct}%")

st.divider()

# ==============================================================
# 12. GREEKS PANELS
# ==============================================================
if spot and atm:
    g_ce = greeks(spot, ce_strike, dte, r, iv, "call")
    g_pe = greeks(spot, pe_strike, dte, r, iv, "put")

    col_ce, col_pe = st.columns(2)

    with col_ce:
        st.success(f"### {ce_strike:,} CE -- Call")
        a, b, c_ = st.columns(3)
        a.metric("Delta",     g_ce["delta"], sign_str(g_ce["delta"]))
        b.metric("Theta/day", g_ce["theta"])
        c_.metric("Gamma",    g_ce["gamma"])
        d_, e_, f_ = st.columns(3)
        d_.metric("Vega", g_ce["vega"])
        e_.metric("Rho",  g_ce["rho"])
        f_.metric("IV",   f"{iv_pct}%")

    with col_pe:
        st.error(f"### {pe_strike:,} PE -- Put")
        a, b, c_ = st.columns(3)
        a.metric("Delta",     g_pe["delta"], sign_str(g_pe["delta"]))
        b.metric("Theta/day", g_pe["theta"])
        c_.metric("Gamma",    g_pe["gamma"])
        d_, e_, f_ = st.columns(3)
        d_.metric("Vega", g_pe["vega"])
        e_.metric("Rho",  g_pe["rho"])
        f_.metric("IV",   f"{iv_pct}%")

    st.divider()

    # Net position summary
    st.markdown("### Net Position Summary")
    net_delta = round(g_ce["delta"] + g_pe["delta"], 4)
    net_theta = round(g_ce["theta"] + g_pe["theta"], 2)
    net_gamma = round(g_ce["gamma"] + g_pe["gamma"], 6)
    net_vega  = round(g_ce["vega"]  + g_pe["vega"],  2)

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Net Delta", net_delta,
              "Neutral" if abs(net_delta) < 0.05 else "Directional")
    n2.metric("Net Theta/day", net_theta)
    n3.metric("Net Gamma",     net_gamma)
    n4.metric("Net Vega",      net_vega)

    st.divider()

    # Theta decay projection
    st.markdown("### Theta Decay Projection")
    if dte_days > 0:
        rows = []
        for day in range(1, min(dte_days + 1, 8)):
            rem = max((dte_days - day) / 365, 1e-6)
            gc  = greeks(spot, ce_strike, rem, r, iv, "call")
            gp  = greeks(spot, pe_strike, rem, r, iv, "put")
            combined = round((gc["theta"] + gp["theta"]) * lots * conf["lot_size"], 2)
            rows.append({
                "Day":                    f"Day +{day}",
                "DTE Remaining":          dte_days - day,
                "CE Theta":               gc["theta"],
                "PE Theta":               gp["theta"],
                f"Net P&L Rs. x {lots}L": combined,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Set DTE > 0 to see the theta decay table.")

else:
    if not run_live:
        st.info("Toggle **Start Live Feed** in the sidebar to begin.")
    elif fetch_error:
        st.error(f"API error: {fetch_error}")
    else:
        st.info("Fetching prices... Market must be open (9:15 AM - 3:30 PM IST).")

# ==============================================================
# 13. ALL-INDEX OVERVIEW
# ==============================================================
if run_live and all_prices:
    st.divider()
    st.markdown("### All Index Prices")
    rows = []
    for idx_name, idx_conf in INDEX_CONFIG.items():
        entry = all_prices.get(idx_conf["instrument_key"])
        if entry:
            rows.append({
                "Index":       idx_name,
                "LTP":         f"Rs.{entry['ltp']:,.2f}",
                "Prev Close":  f"Rs.{entry['close']:,.2f}",
                "Change %":    f"{entry['change_pct']:+.2f}%",
                "Last Update": entry["ts"].strftime("%H:%M:%S"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No data yet. Market must be open.")

# ==============================================================
# 14. AUTO-REFRESH
# ==============================================================
if run_live:
    time.sleep(refresh_ms / 1000)
    st.rerun()
