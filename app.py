"""
Multi-Index Live Scalper Dashboard
Upstox REST Market Quote API -- works on Streamlit Cloud
---------------------------------------------------------
Auto features:
  - IV pulled live from India VIX
  - DTE calculated from current date to weekly expiry
  - Refresh rate adapts to market session (fast open/close, slow midday)

SETUP:
  1. pip install -r requirements.txt
  2. Create .streamlit/secrets.toml:
       [upstox]
       access_token = "your_token_here"
  3. streamlit run scalper_dashboard.py
"""

import time
from datetime import datetime, date, timedelta, timezone

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
        "response_key":   "NSE_INDEX:Nifty 50",
        "lot_size":       75,
        "strike_step":    50,
        "expiry_weekday": 3,    # Thursday
    },
    "BANK NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "response_key":   "NSE_INDEX:Nifty Bank",
        "lot_size":       30,
        "strike_step":    100,
        "expiry_weekday": 2,    # Wednesday
    },
    "FINNIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Fin Service",
        "response_key":   "NSE_INDEX:Nifty Fin Service",
        "lot_size":       65,
        "strike_step":    50,
        "expiry_weekday": 2,    # Tuesday (index 1) -- adjust if needed
    },
    "MIDCAP NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Midcap Select",
        "response_key":   "NSE_INDEX:Nifty Midcap Select",
        "lot_size":       120,
        "strike_step":    25,
        "expiry_weekday": 0,    # Monday
    },
}

ALL_INSTRUMENT_KEYS = [v["instrument_key"] for v in INDEX_CONFIG.values()]

# India VIX -- fetched alongside index prices
VIX_INSTRUMENT_KEY = "NSE_INDEX|India VIX"
VIX_RESPONSE_KEY   = "NSE_INDEX:India VIX"

# ==============================================================
# 4. SESSION STATE INIT
# ==============================================================
if "token_ok"    not in st.session_state: st.session_state.token_ok    = None
if "token_msg"   not in st.session_state: st.session_state.token_msg   = ""
if "live_feed"   not in st.session_state: st.session_state.live_feed   = False
if "last_prices" not in st.session_state: st.session_state.last_prices = {}
if "last_vix"    not in st.session_state: st.session_state.last_vix    = None

# ==============================================================
# 5. AUTO DTE
# ==============================================================
def get_dte(expiry_weekday):
    """
    Returns (dte_days, expiry_date) for the next weekly expiry.
    Rolls to the following week if today IS the expiry day and
    market has closed (after 15:30 IST).
    """
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    today   = now_ist.date()

    days_ahead = expiry_weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0:
        # Today is expiry -- roll after 15:30 IST
        if now_ist.hour >= 15 and now_ist.minute >= 30:
            days_ahead = 7

    expiry_date = today + timedelta(days=days_ahead)
    dte_days    = (expiry_date - today).days
    return dte_days, expiry_date


# ==============================================================
# 6. AUTO REFRESH RATE
# ==============================================================
def get_refresh_ms():
    """
    Fast (1s) near open/close, normal (2s) midday, slow (3s) pre-market.
    Based on IST time.
    """
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    h, m    = now_ist.hour, now_ist.minute
    t       = h * 60 + m    # minutes since midnight IST

    if 555 <= t <= 570:     # 9:15 - 9:30 IST  (opening volatility)
        return 1000
    elif 870 <= t <= 930:   # 14:30 - 15:30 IST (closing volatility)
        return 1000
    elif 555 <= t <= 930:   # market hours midday
        return 2000
    else:
        return 3000         # outside market hours


# ==============================================================
# 7. PRICE FETCH  (index prices + VIX in one call)
# ==============================================================
def fetch_all_prices(token):
    try:
        conf = upstox_client.Configuration()
        conf.access_token = token
        api  = upstox_client.MarketQuoteApi(upstox_client.ApiClient(conf))

        # Fetch all indices + VIX in a single API call
        all_keys = ALL_INSTRUMENT_KEYS + [VIX_INSTRUMENT_KEY]
        keys_str = ",".join(all_keys)
        res = api.get_market_quote_ohlc(keys_str, "1d", "2.0")

        prices = {}
        vix    = None

        if res.status == "success" and res.data:
            for response_key, quote in res.data.items():
                if isinstance(quote, dict):
                    ltp   = quote.get("last_price")
                    ohlc  = quote.get("ohlc", {})
                    close = ohlc.get("close") if isinstance(ohlc, dict) else None
                else:
                    ltp   = getattr(quote, "last_price", None)
                    ohlc  = getattr(quote, "ohlc", None)
                    close = (ohlc.get("close") if isinstance(ohlc, dict)
                             else getattr(ohlc, "close", None))

                if ltp is None:
                    continue

                ltp        = float(ltp)
                close      = float(close) if close else ltp
                change_pct = round(((ltp - close) / close) * 100, 2) if close else 0.0

                entry = {
                    "ltp":        ltp,
                    "close":      close,
                    "change_pct": change_pct,
                    "ts":         datetime.now(),
                }

                if response_key == VIX_RESPONSE_KEY:
                    vix = ltp    # VIX is quoted as a percentage e.g. 13.5
                else:
                    prices[response_key] = entry

        return prices, vix, None

    except Exception as e:
        return {}, None, str(e)


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
# 8. GREEKS ENGINE
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
# 9. HELPERS
# ==============================================================
def fmt_inr(v):
    if v >= 1e7: return f"Rs.{v/1e7:.2f} Cr"
    if v >= 1e5: return f"Rs.{v/1e5:.2f} L"
    return f"Rs.{v:,.0f}"

def sign_str(v):
    return f"+{v}" if v >= 0 else str(v)

# ==============================================================
# 10. COMPUTE AUTO VALUES
# ==============================================================
conf_selected = None   # set after sidebar

# Auto DTE (computed before sidebar so we can show it)
# We'll recompute after index selection; placeholder here
_now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
_refresh  = get_refresh_ms()

# ==============================================================
# 11. SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## Scalper Controls")
    st.divider()

    selected_index = st.selectbox("Index", list(INDEX_CONFIG.keys()))
    conf = INDEX_CONFIG[selected_index]

    # Auto DTE for selected index
    auto_dte, expiry_date = get_dte(conf["expiry_weekday"])

    st.divider()
    st.markdown("### Strategy")
    run_live      = st.toggle("Start Live Feed", key="live_feed")
    lots          = st.number_input("Lots", min_value=1, max_value=500, value=1, step=1)
    strike_mode   = st.selectbox("Strike Selection", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    strike_offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]

    st.divider()
    st.markdown("### Greeks Parameters")

    # IV -- show live VIX value, allow manual override
    vix_display = (f"India VIX: {st.session_state.last_vix:.2f}%"
                   if st.session_state.last_vix else "India VIX: fetching...")
    st.caption(f"Auto IV from {vix_display}")

    auto_iv_pct = round(st.session_state.last_vix, 1) if st.session_state.last_vix else 15.0
    iv_pct = st.slider(
        "Implied Volatility (%)",
        min_value=5, max_value=80,
        value=int(auto_iv_pct),
        step=1,
        help="Auto-set from India VIX. Override manually if needed.",
    )
    iv = iv_pct / 100.0

    # DTE -- show auto value, allow manual override
    st.caption(f"Auto DTE: {auto_dte}d to expiry ({expiry_date.strftime('%d %b')})")
    dte_days = st.slider(
        "Days to Expiry",
        min_value=0, max_value=30,
        value=auto_dte,
        step=1,
        help="Auto-set from weekly expiry calendar. Override if needed.",
    )
    dte = dte_days / 365.0

    risk_free = st.slider("Risk-Free Rate (%)", 4, 12, 7, step=1)
    r         = risk_free / 100.0

    st.divider()

    # Auto refresh display
    st.caption(f"Auto refresh: {_refresh}ms "
               f"({'fast' if _refresh == 1000 else 'normal' if _refresh == 2000 else 'slow'})")

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
    st.caption("Market: 9:15 AM - 3:30 PM IST, Mon-Fri")

# ==============================================================
# 12. FETCH PRICES
# ==============================================================
fetch_error = None

if run_live:
    prices, vix, fetch_error = fetch_all_prices(TOKEN)
    if prices:
        st.session_state.last_prices = prices
    if vix is not None:
        st.session_state.last_vix = vix

# Use last known prices if current fetch empty
all_prices = st.session_state.last_prices
rkey       = conf["response_key"]
feed_entry = all_prices.get(rkey)

if feed_entry:
    spot       = feed_entry["ltp"]
    change_pct = feed_entry["change_pct"]
    data_age   = (datetime.now() - feed_entry["ts"]).total_seconds()
else:
    spot = change_pct = data_age = None

# ==============================================================
# 13. PAGE HEADER
# ==============================================================
st.markdown(f"# {selected_index} Scalper")

h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
with h1:
    if run_live and fetch_error:
        st.caption(f"🔴 {fetch_error[:40]}")
    elif run_live and spot:
        st.caption("🟢 Live")
    elif run_live:
        st.caption("🟡 Fetching...")
    else:
        st.caption("⚪ Paused")
with h2:
    if data_age is not None:
        st.caption(f"Updated: {'< 1s' if data_age < 1 else f'{data_age:.0f}s'} ago")
with h3:
    if st.session_state.last_vix:
        st.caption(f"VIX: {st.session_state.last_vix:.2f}%  |  IV: {iv_pct}%")
with h4:
    st.caption(f"Expiry: {expiry_date.strftime('%d %b')} ({dte_days}d)")

st.divider()

# ==============================================================
# 14. TOP METRICS
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
m4.metric("DTE / Expiry",   f"{dte_days}d",   expiry_date.strftime("%d %b %Y"))

st.divider()

# ==============================================================
# 15. GREEKS PANELS
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
# 16. ALL-INDEX OVERVIEW
# ==============================================================
if run_live and all_prices:
    st.divider()
    st.markdown("### All Index Prices")
    rows = []
    for idx_name, idx_conf in INDEX_CONFIG.items():
        entry = all_prices.get(idx_conf["response_key"])
        if entry:
            dte_i, exp_i = get_dte(idx_conf["expiry_weekday"])
            rows.append({
                "Index":       idx_name,
                "LTP":         f"Rs.{entry['ltp']:,.2f}",
                "Change %":    f"{entry['change_pct']:+.2f}%",
                "Expiry":      exp_i.strftime("%d %b"),
                "DTE":         dte_i,
                "Last Update": entry["ts"].strftime("%H:%M:%S"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No data yet. Market must be open.")

# ==============================================================
# 17. AUTO-REFRESH  (rate adapts to market session)
# ==============================================================
if run_live:
    time.sleep(_refresh / 1000)
    st.rerun()
