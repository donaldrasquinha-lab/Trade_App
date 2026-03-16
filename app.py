"""
Multi-Index Live Scalper Dashboard
With Signal Engine: VWAP, EMA9/21, Momentum, Trade Recommendation
------------------------------------------------------------------
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
        "expiry_weekday": 3,
    },
    "BANK NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Bank",
        "response_key":   "NSE_INDEX:Nifty Bank",
        "lot_size":       30,
        "strike_step":    100,
        "expiry_weekday": 2,
    },
    "FINNIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Fin Service",
        "response_key":   "NSE_INDEX:Nifty Fin Service",
        "lot_size":       65,
        "strike_step":    50,
        "expiry_weekday": 1,
    },
    "MIDCAP NIFTY": {
        "instrument_key": "NSE_INDEX|Nifty Midcap Select",
        "response_key":   "NSE_INDEX:Nifty Midcap Select",
        "lot_size":       120,
        "strike_step":    25,
        "expiry_weekday": 0,
    },
}

ALL_INSTRUMENT_KEYS = [v["instrument_key"] for v in INDEX_CONFIG.values()]
VIX_INSTRUMENT_KEY  = "NSE_INDEX|India VIX"
VIX_RESPONSE_KEY    = "NSE_INDEX:India VIX"

# ==============================================================
# 4. SESSION STATE
# ==============================================================
defaults = {
    "token_ok":    None,
    "token_msg":   "",
    "live_feed":   False,
    "last_prices": {},
    "last_vix":    None,
    "last_chain":  {},
    "chain_ts":    None,
    "candle_ts":   None,
    "candle_df":   None,   # pd.DataFrame of 1m candles
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================
# 5. HELPERS
# ==============================================================
def get_dte(expiry_weekday):
    now_ist    = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    today      = now_ist.date()
    days_ahead = expiry_weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0 and now_ist.hour >= 15 and now_ist.minute >= 30:
        days_ahead = 7
    expiry_date = today + timedelta(days=days_ahead)
    return (expiry_date - today).days, expiry_date

def get_refresh_ms():
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    t = now_ist.hour * 60 + now_ist.minute
    if 555 <= t <= 570 or 870 <= t <= 930:
        return 1000
    elif 555 <= t <= 930:
        return 2000
    return 3000

def fmt_inr(v):
    if v >= 1e7: return f"Rs.{v/1e7:.2f} Cr"
    if v >= 1e5: return f"Rs.{v/1e5:.2f} L"
    return f"Rs.{v:,.0f}"

def sign_str(v):
    return f"+{v}" if v >= 0 else str(v)

def is_high_volatility_window():
    """Returns True during 9:15-10:15 AM and 2:45-3:30 PM IST."""
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    t = now_ist.hour * 60 + now_ist.minute
    return (555 <= t <= 615) or (885 <= t <= 930)

# ==============================================================
# 6. API: SPOT PRICES + VIX
# ==============================================================
def fetch_all_prices(token):
    try:
        conf = upstox_client.Configuration()
        conf.access_token = token
        api  = upstox_client.MarketQuoteApi(upstox_client.ApiClient(conf))
        keys_str = ",".join(ALL_INSTRUMENT_KEYS + [VIX_INSTRUMENT_KEY])
        res = api.get_market_quote_ohlc(keys_str, "1d", "2.0")
        prices = {}
        vix    = None
        if res.status == "success" and res.data:
            for rkey, quote in res.data.items():
                q     = quote if isinstance(quote, dict) else (quote.to_dict() if hasattr(quote, "to_dict") else {})
                ltp   = q.get("last_price") or getattr(quote, "last_price", None)
                ohlc  = q.get("ohlc", {}) or {}
                close = ohlc.get("close") if isinstance(ohlc, dict) else getattr(ohlc, "close", None)
                if ltp is None:
                    continue
                ltp   = float(ltp)
                close = float(close) if close else ltp
                entry = {
                    "ltp":        ltp,
                    "close":      close,
                    "change_pct": round(((ltp - close) / close) * 100, 2) if close else 0.0,
                    "ts":         datetime.now(),
                }
                if rkey == VIX_RESPONSE_KEY:
                    vix = ltp
                else:
                    prices[rkey] = entry
        return prices, vix, None
    except Exception as e:
        return {}, None, str(e)

# ==============================================================
# 7. API: INTRADAY CANDLES (1-minute, V3)
#    Candle format: [timestamp, open, high, low, close, volume, oi]
#    Fetched every 60s -- one full candle per minute.
# ==============================================================
def fetch_candles(token, instrument_key, interval_min=1):
    try:
        conf = upstox_client.Configuration()
        conf.access_token = token
        api  = upstox_client.HistoryV3Api(upstox_client.ApiClient(conf))
        res  = api.get_intra_day_candle_data(instrument_key, "minutes", str(interval_min))

        if res.status != "success" or not res.data or not res.data.candles:
            return None, "No candle data"

        rows = []
        for c in res.data.candles:
            rows.append({
                "ts":     pd.to_datetime(c[0]),
                "open":   float(c[1]),
                "high":   float(c[2]),
                "low":    float(c[3]),
                "close":  float(c[4]),
                "volume": float(c[5]),
            })

        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        return df, None

    except Exception as e:
        return None, str(e)

# ==============================================================
# 8. API: OPTION CHAIN
# ==============================================================
def fetch_option_chain(token, instrument_key, expiry_date_str, atm, step, n=3):
    try:
        conf = upstox_client.Configuration()
        conf.access_token = token
        api  = upstox_client.OptionsApi(upstox_client.ApiClient(conf))
        res  = api.get_put_call_option_chain(instrument_key, expiry_date_str)
        chain  = {}
        wanted = set(atm + i * step for i in range(-n, n + 1))
        if res.status == "success" and res.data:
            for item in res.data:
                strike = (item.strike_price if hasattr(item, "strike_price")
                          else item.get("strike_price") if isinstance(item, dict) else None)
                if not strike or strike not in wanted:
                    continue

                def ex(opt):
                    if opt is None: return {}
                    if isinstance(opt, dict):
                        md = opt.get("market_data", {}) or {}
                        og = opt.get("option_greeks", {}) or {}
                    else:
                        md = getattr(opt, "market_data", None) or {}
                        og = getattr(opt, "option_greeks", None) or {}
                        if hasattr(md, "to_dict"): md = md.to_dict()
                        if hasattr(og, "to_dict"): og = og.to_dict()
                    return {
                        "ltp":   float(md.get("ltp", 0) or 0),
                        "oi":    int(md.get("oi",  0) or 0),
                        "iv":    float(og.get("iv",    0) or 0),
                        "delta": float(og.get("delta", 0) or 0),
                        "theta": float(og.get("theta", 0) or 0),
                        "gamma": float(og.get("gamma", 0) or 0),
                        "vega":  float(og.get("vega",  0) or 0),
                    }

                ce = ex(item.call_options if hasattr(item, "call_options") else item.get("call_options"))
                pe = ex(item.put_options  if hasattr(item, "put_options")  else item.get("put_options"))
                chain[int(strike)] = {
                    "ce_ltp": ce.get("ltp", 0), "pe_ltp": pe.get("ltp", 0),
                    "ce_iv":  ce.get("iv",  0), "pe_iv":  pe.get("iv",  0),
                    "ce_delta":ce.get("delta",0),"pe_delta":pe.get("delta",0),
                    "ce_theta":ce.get("theta",0),"pe_theta":pe.get("theta",0),
                    "ce_gamma":ce.get("gamma",0),"pe_gamma":pe.get("gamma",0),
                    "ce_vega": ce.get("vega", 0),"pe_vega": pe.get("vega",0),
                    "ce_oi":  ce.get("oi",   0), "pe_oi":  pe.get("oi",  0),
                }
        return chain, None
    except Exception as e:
        return {}, str(e)

# ==============================================================
# 9. INDICATORS
# ==============================================================
def compute_indicators(df):
    """
    Computes EMA9, EMA21, VWAP, RSI14 on a candle DataFrame.
    Returns the updated df + latest values dict.
    """
    if df is None or len(df) < 5:
        return df, {}

    df = df.copy()

    # EMA
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["tpv"]  = df["tp"] * df["volume"]
    df["vwap"] = df["tpv"].cumsum() / df["volume"].cumsum()

    # RSI 14
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Latest values
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    return df, {
        "close":    last["close"],
        "ema9":     round(last["ema9"], 2),
        "ema21":    round(last["ema21"], 2),
        "vwap":     round(last["vwap"], 2),
        "rsi":      round(last["rsi"], 1),
        "volume":   int(last["volume"]),
        "ema9_prev":  round(prev["ema9"], 2),
        "ema21_prev": round(prev["ema21"], 2),
        "prev_close": prev["close"],
    }

# ==============================================================
# 10. SIGNAL ENGINE
#     Scores each condition +1 (bullish) or -1 (bearish).
#     Returns recommendation: BUY CE / BUY PE / WAIT / AVOID
# ==============================================================
def generate_signal(ind, spot, vix, high_vol_window):
    """
    Returns a signal dict with score, direction, confidence,
    recommendation, target, stop-loss, and reason list.
    """
    if not ind or spot is None:
        return None

    score   = 0
    reasons = []

    close = ind["close"]
    ema9  = ind["ema9"]
    ema21 = ind["ema21"]
    vwap  = ind["vwap"]
    rsi   = ind["rsi"]

    # ── Trend: EMA9 vs EMA21 ──────────────────
    if ema9 > ema21:
        score += 1
        reasons.append(("bull", "EMA9 > EMA21 (uptrend)"))
    else:
        score -= 1
        reasons.append(("bear", "EMA9 < EMA21 (downtrend)"))

    # ── EMA crossover (last candle) ───────────
    if ind["ema9_prev"] <= ind["ema21_prev"] and ema9 > ema21:
        score += 2
        reasons.append(("bull", "EMA9 just crossed ABOVE EMA21 (strong signal)"))
    elif ind["ema9_prev"] >= ind["ema21_prev"] and ema9 < ema21:
        score -= 2
        reasons.append(("bear", "EMA9 just crossed BELOW EMA21 (strong signal)"))

    # ── Price vs VWAP ─────────────────────────
    if close > vwap:
        score += 1
        reasons.append(("bull", f"Price ({close:.0f}) above VWAP ({vwap:.0f})"))
    else:
        score -= 1
        reasons.append(("bear", f"Price ({close:.0f}) below VWAP ({vwap:.0f})"))

    # ── Price vs EMA9 (momentum) ──────────────
    if close > ema9:
        score += 1
        reasons.append(("bull", f"Price above EMA9 ({ema9:.0f})"))
    else:
        score -= 1
        reasons.append(("bear", f"Price below EMA9 ({ema9:.0f})"))

    # ── RSI ───────────────────────────────────
    if 55 <= rsi <= 75:
        score += 1
        reasons.append(("bull", f"RSI {rsi} in bullish zone (55–75)"))
    elif 25 <= rsi <= 45:
        score -= 1
        reasons.append(("bear", f"RSI {rsi} in bearish zone (25–45)"))
    elif rsi > 80:
        score -= 1
        reasons.append(("warn", f"RSI {rsi} overbought — avoid CE"))
    elif rsi < 20:
        score += 1
        reasons.append(("warn", f"RSI {rsi} oversold — possible bounce"))

    # ── High volatility window bonus ─────────
    if high_vol_window:
        reasons.append(("info", "Inside high-volatility window (9:15–10:15 AM)"))
    else:
        score = int(score * 0.7)   # reduce confidence outside window
        reasons.append(("warn", "Outside prime scalping window — reduce size"))

    # ── VIX check ─────────────────────────────
    if vix:
        if vix > 20:
            reasons.append(("warn", f"VIX {vix:.1f} elevated — widen stop-loss"))
        elif vix < 12:
            reasons.append(("warn", f"VIX {vix:.1f} very low — momentum may be weak"))

    # ── Direction & recommendation ────────────
    abs_score = abs(score)

    if score >= 3:
        direction      = "CE"
        recommendation = "BUY CE"
        confidence     = "High" if abs_score >= 4 else "Medium"
        emoji          = "🟢"
    elif score <= -3:
        direction      = "PE"
        recommendation = "BUY PE"
        confidence     = "High" if abs_score >= 4 else "Medium"
        emoji          = "🔴"
    elif abs_score <= 1:
        direction      = "NEUTRAL"
        recommendation = "WAIT"
        confidence     = "Low"
        emoji          = "🟡"
    else:
        direction      = "CE" if score > 0 else "PE"
        recommendation = "WEAK SIGNAL — AVOID"
        confidence     = "Low"
        emoji          = "⚪"

    # ── Target & stop-loss (10–20 pt scalp) ──
    strike_move = 15   # mid-range of 10–20 pts
    if direction == "CE":
        target    = round(close + strike_move, 0)
        stop_loss = round(close - (strike_move * 0.5), 0)
    elif direction == "PE":
        target    = round(close - strike_move, 0)
        stop_loss = round(close + (strike_move * 0.5), 0)
    else:
        target = stop_loss = None

    return {
        "score":          score,
        "direction":      direction,
        "recommendation": recommendation,
        "confidence":     confidence,
        "emoji":          emoji,
        "target":         target,
        "stop_loss":      stop_loss,
        "reasons":        reasons,
    }

# ==============================================================
# 11. BS GREEKS FALLBACK
# ==============================================================
def bs_greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0:
        return dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    pdf1 = norm.pdf(d1)
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    theta = (-(S * pdf1 * sigma) / (2 * np.sqrt(T))
             + (r * K * np.exp(-r * T) * (norm.cdf(-d2) if option_type == "put" else -norm.cdf(d2))))
    return dict(
        delta = round(delta,       4),
        gamma = round(pdf1 / (S * sigma * np.sqrt(T)), 6),
        vega  = round(S * pdf1 * np.sqrt(T) / 100, 2),
        theta = round(theta / 365, 2),
    )

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
# 12. REFRESH RATE
# ==============================================================
_refresh = get_refresh_ms()
_now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
_high_vol = is_high_volatility_window()

# ==============================================================
# 13. SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## Scalper Controls")
    st.divider()

    selected_index = st.selectbox("Index", list(INDEX_CONFIG.keys()))
    conf = INDEX_CONFIG[selected_index]
    auto_dte, expiry_date = get_dte(conf["expiry_weekday"])

    st.divider()
    st.markdown("### Strategy")
    run_live      = st.toggle("Start Live Feed", key="live_feed")
    lots          = st.number_input("Lots", min_value=1, max_value=500, value=1, step=1)
    strike_mode   = st.selectbox("Strike Selection", ["ATM", "1-Strike ITM", "2-Strike ITM"])
    strike_offset = {"ATM": 0, "1-Strike ITM": 1, "2-Strike ITM": 2}[strike_mode]

    candle_interval = st.selectbox("Candle Interval", ["1", "3", "5"], index=0,
                                    format_func=lambda x: f"{x} min")

    st.divider()
    st.markdown("### Greeks Parameters")
    vix_val = st.session_state.last_vix
    st.caption(f"Auto IV from India VIX: {vix_val:.2f}%" if vix_val else "Auto IV from India VIX: fetching...")
    auto_iv = round(vix_val, 1) if vix_val else 15.0
    iv_pct  = st.slider("Implied Volatility (%)", 5, 80, int(auto_iv), step=1)
    iv      = iv_pct / 100.0
    st.caption(f"Auto DTE: {auto_dte}d → {expiry_date.strftime('%d %b')}")
    dte_days = st.slider("Days to Expiry", 0, 30, auto_dte, step=1)
    dte      = dte_days / 365.0
    risk_free = st.slider("Risk-Free Rate (%)", 4, 12, 7, step=1)
    r         = risk_free / 100.0

    st.divider()
    st.caption(f"Refresh: {_refresh}ms  |  Window: {'🔥 Hot' if _high_vol else '❄️ Calm'}")

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
# 14. FETCH SPOT PRICES
# ==============================================================
if run_live:
    prices, vix, _ = fetch_all_prices(TOKEN)
    if prices:
        st.session_state.last_prices = prices
    if vix:
        st.session_state.last_vix = vix

all_prices = st.session_state.last_prices
feed_entry = all_prices.get(conf["response_key"])
spot       = feed_entry["ltp"]       if feed_entry else None
change_pct = feed_entry["change_pct"] if feed_entry else None
data_age   = (datetime.now() - feed_entry["ts"]).total_seconds() if feed_entry else None

# ==============================================================
# 15. FETCH CANDLES  (every 60s)
# ==============================================================
candle_df  = st.session_state.candle_df
indicators = {}
candle_err = None

if run_live and spot:
    candle_age = (
        (datetime.now() - st.session_state.candle_ts).total_seconds()
        if st.session_state.candle_ts else 999
    )
    if candle_age >= 60:
        new_df, candle_err = fetch_candles(TOKEN, conf["instrument_key"], int(candle_interval))
        if new_df is not None and len(new_df) >= 5:
            st.session_state.candle_df = new_df
            st.session_state.candle_ts = datetime.now()
            candle_df = new_df

    if candle_df is not None:
        candle_df, indicators = compute_indicators(candle_df)

# ==============================================================
# 16. FETCH OPTION CHAIN  (every 5s)
# ==============================================================
step  = conf["strike_step"]
atm   = int(round(spot / step) * step) if spot else None
chain = st.session_state.last_chain

if run_live and spot and atm:
    chain_age = (
        (datetime.now() - st.session_state.chain_ts).total_seconds()
        if st.session_state.chain_ts else 999
    )
    if chain_age >= 5:
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        new_chain, _ = fetch_option_chain(TOKEN, conf["instrument_key"],
                                           expiry_str, atm, step, n=3)
        if new_chain:
            st.session_state.last_chain = new_chain
            st.session_state.chain_ts   = datetime.now()
            chain = new_chain

ce_strike = atm - strike_offset * step if atm else None
pe_strike = atm + strike_offset * step if atm else None
ce_data   = chain.get(ce_strike, {}) if chain and ce_strike else {}
pe_data   = chain.get(pe_strike, {}) if chain and pe_strike else {}
ce_ltp    = ce_data.get("ce_ltp", 0)
pe_ltp    = pe_data.get("pe_ltp", 0)

if ce_data and ce_data.get("ce_delta"):
    g_ce = {"delta": round(ce_data["ce_delta"], 4), "theta": round(ce_data["ce_theta"], 2),
            "gamma": round(ce_data["ce_gamma"], 6), "vega":  round(ce_data["ce_vega"],  2),
            "iv":    round(ce_data["ce_iv"], 2)}
    g_pe = {"delta": round(pe_data["pe_delta"], 4), "theta": round(pe_data["pe_theta"], 2),
            "gamma": round(pe_data["pe_gamma"], 6), "vega":  round(pe_data["pe_vega"],  2),
            "iv":    round(pe_data["pe_iv"], 2)}
    greeks_source = "Exchange"
elif spot and atm:
    g_ce = bs_greeks(spot, ce_strike, dte, r, iv, "call"); g_ce["iv"] = iv_pct
    g_pe = bs_greeks(spot, pe_strike, dte, r, iv, "put");  g_pe["iv"] = iv_pct
    greeks_source = "BS Model"
else:
    g_ce = g_pe = None
    greeks_source = "--"

# ==============================================================
# 17. GENERATE SIGNAL
# ==============================================================
signal = generate_signal(indicators, spot, st.session_state.last_vix, _high_vol)

# ==============================================================
# 18. PAGE HEADER
# ==============================================================
st.markdown(f"# {selected_index} Scalper")

h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
with h1:
    if run_live and spot:
        st.caption(f"🟢 Live  |  {greeks_source} Greeks")
    elif run_live:
        st.caption("🟡 Fetching...")
    else:
        st.caption("⚪ Paused")
with h2:
    if data_age is not None:
        st.caption(f"Spot: {'< 1s' if data_age < 1 else f'{data_age:.0f}s'} ago")
with h3:
    if st.session_state.last_vix:
        st.caption(f"VIX: {st.session_state.last_vix:.2f}%")
with h4:
    st.caption(f"Expiry: {expiry_date.strftime('%d %b')} ({dte_days}d)")

st.divider()

# ==============================================================
# 19. BEST TO BUY NOW  (prominent recommendation card)
# ==============================================================
if signal and spot and atm:
    direction = signal["direction"]
    rec       = signal["recommendation"]
    conf_str  = signal["confidence"]
    score     = signal["score"]

    # Pick the recommended option details
    if direction == "CE":
        rec_strike  = ce_strike
        rec_ltp     = ce_ltp
        rec_margin  = ce_ltp * conf["lot_size"] * lots if ce_ltp else 0
        rec_delta   = g_ce["delta"] if g_ce else "--"
        rec_theta   = g_ce["theta"] if g_ce else "--"
        rec_iv      = g_ce["iv"]    if g_ce else "--"
        bg_color    = "#0d3320"
        border_col  = "#00c853"
        label_col   = "#00e676"
        opt_type    = "CALL"
    elif direction == "PE":
        rec_strike  = pe_strike
        rec_ltp     = pe_ltp
        rec_margin  = pe_ltp * conf["lot_size"] * lots if pe_ltp else 0
        rec_delta   = g_pe["delta"] if g_pe else "--"
        rec_theta   = g_pe["theta"] if g_pe else "--"
        rec_iv      = g_pe["iv"]    if g_pe else "--"
        bg_color    = "#3d0a0a"
        border_col  = "#f44336"
        label_col   = "#ff5252"
        opt_type    = "PUT"
    else:
        rec_strike = rec_ltp = rec_margin = rec_delta = rec_theta = rec_iv = None
        bg_color   = "#2a2a1a"
        border_col = "#ffc107"
        label_col  = "#ffd740"
        opt_type   = "WAIT"

    # Confidence bar (filled dots)
    conf_dots = {"High": "●●●●●", "Medium": "●●●○○", "Low": "●●○○○"}
    dots      = conf_dots.get(conf_str, "●○○○○")

    # Score bar
    max_score  = 6
    score_pct  = min(abs(score) / max_score * 100, 100)
    score_fill = int(score_pct / 10)
    score_bar  = "█" * score_fill + "░" * (10 - score_fill)

    # Window badge
    window_badge = (
        '<span style="background:#ff6d00;color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">🔥 HOT WINDOW</span>'
        if _high_vol else
        '<span style="background:#37474f;color:#ccc;padding:2px 8px;border-radius:4px;font-size:12px;">❄️ CALM WINDOW</span>'
    )

    # Target / SL
    tgt = f"{signal['target']:,.0f}" if signal["target"] else "--"
    sl  = f"{signal['stop_loss']:,.0f}" if signal["stop_loss"] else "--"

    # Reason pills
    icons = {"bull": "🟢", "bear": "🔴", "warn": "🟡", "info": "🔵"}
    reason_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;font-size:13px;">'
        f'<span>{icons.get(k,"•")}</span><span style="color:#ddd;">{t}</span></div>'
        for k, t in signal["reasons"]
    )

    if direction in ("CE", "PE") and rec_ltp:
        card_html = f"""
<div style="
    background:{bg_color};
    border:2px solid {border_col};
    border-radius:12px;
    padding:20px 24px;
    margin-bottom:16px;
">
    <!-- Header row -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
        <div>
            <div style="font-size:13px;color:#aaa;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;">
                Best Option to Buy Now
            </div>
            <div style="font-size:28px;font-weight:700;color:{label_col};margin-top:2px;">
                {rec_strike:,} {opt_type} &nbsp;
                <span style="font-size:16px;background:{border_col};color:white;
                    padding:3px 10px;border-radius:6px;vertical-align:middle;">
                    {rec}
                </span>
            </div>
        </div>
        <div style="text-align:right;">
            {window_badge}
            <div style="margin-top:8px;font-size:13px;color:#aaa;">
                Confidence &nbsp;<span style="color:{label_col};font-size:15px;letter-spacing:2px;">{dots}</span>
                &nbsp; <b style="color:{label_col};">{conf_str}</b>
            </div>
            <div style="font-size:12px;color:#888;margin-top:4px;">
                Signal score &nbsp; <span style="font-family:monospace;color:{label_col};">{score_bar}</span>
                &nbsp; {score:+d}/6
            </div>
        </div>
    </div>

    <!-- Key numbers row -->
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px;">
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;">Premium</div>
            <div style="font-size:18px;font-weight:600;color:white;">Rs.{rec_ltp:,.2f}</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;">Buyer Margin</div>
            <div style="font-size:18px;font-weight:600;color:white;">{fmt_inr(rec_margin)}</div>
            <div style="font-size:10px;color:#888;">{rec_ltp:.1f} x {conf["lot_size"]} x {lots}L</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;">Entry Spot</div>
            <div style="font-size:18px;font-weight:600;color:white;">{spot:,.0f}</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;">Target</div>
            <div style="font-size:18px;font-weight:600;color:#00e676;">{tgt}</div>
            <div style="font-size:10px;color:#888;">+15 pts on spot</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;">Stop-Loss</div>
            <div style="font-size:18px;font-weight:600;color:#ff5252;">{sl}</div>
            <div style="font-size:10px;color:#888;">-7.5 pts on spot</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;">
            <div style="font-size:11px;color:#aaa;text-transform:uppercase;">R:R Ratio</div>
            <div style="font-size:18px;font-weight:600;color:white;">2 : 1</div>
            <div style="font-size:10px;color:#888;">15 tgt / 7.5 sl</div>
        </div>
    </div>

    <!-- Greeks mini row -->
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px;">
        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:8px 10px;font-size:13px;">
            <span style="color:#888;">Delta</span>
            <span style="color:white;font-weight:600;float:right;">{rec_delta}</span>
        </div>
        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:8px 10px;font-size:13px;">
            <span style="color:#888;">Theta/day</span>
            <span style="color:#ff5252;font-weight:600;float:right;">{rec_theta}</span>
        </div>
        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:8px 10px;font-size:13px;">
            <span style="color:#888;">IV</span>
            <span style="color:white;font-weight:600;float:right;">{rec_iv:.1f}%</span>
        </div>
    </div>

    <!-- Signal reasons -->
    <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:12px;">
        <div style="font-size:11px;color:#888;text-transform:uppercase;margin-bottom:6px;">
            Why this signal
        </div>
        {reason_html}
    </div>
</div>
"""
    else:
        # WAIT / AVOID card
        card_html = f"""
<div style="
    background:{bg_color};
    border:2px solid {border_col};
    border-radius:12px;
    padding:20px 24px;
    margin-bottom:16px;
">
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
            <div style="font-size:13px;color:#aaa;text-transform:uppercase;letter-spacing:0.08em;">
                Best Option to Buy Now
            </div>
            <div style="font-size:26px;font-weight:700;color:{label_col};margin-top:4px;">
                {signal["emoji"]} {rec}
            </div>
            <div style="font-size:13px;color:#aaa;margin-top:6px;">
                Score {score:+d}/6 — indicators are mixed, no clear edge right now
            </div>
        </div>
        <div>{window_badge}</div>
    </div>
    <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:12px;margin-top:12px;">
        {reason_html}
    </div>
</div>
"""

    st.markdown(card_html, unsafe_allow_html=True)
    st.divider()

elif run_live and not indicators:
    st.markdown("""
<div style="background:#1a1a2e;border:1px solid #444;border-radius:10px;padding:16px 20px;margin-bottom:16px;">
    <div style="font-size:13px;color:#aaa;text-transform:uppercase;letter-spacing:0.08em;">Best Option to Buy Now</div>
    <div style="font-size:18px;color:#ffd740;margin-top:6px;">⏳ Waiting for candle data...</div>
    <div style="font-size:13px;color:#888;margin-top:4px;">Candles are fetched every 60s. Signal will appear shortly after the first fetch.</div>
</div>
""", unsafe_allow_html=True)
    st.divider()

# ==============================================================
# 20. TOP METRICS
# ==============================================================
if spot and atm:
    exposure = spot * lots * conf["lot_size"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{selected_index} Spot",
              f"Rs.{spot:,.2f}", f"{change_pct:+.2f}%" if change_pct else None)
    m2.metric("ATM Strike",      f"{atm:,}")
    m3.metric("Notional Exposure", fmt_inr(exposure),
              f"{lots} lot x {conf['lot_size']}")
    m4.metric("DTE", f"{dte_days}d", expiry_date.strftime("%d %b %Y"))
    st.divider()

# ==============================================================
# 21. INDICATOR SUMMARY BAR
# ==============================================================
if indicators:
    st.markdown("### Technical Indicators")
    i1, i2, i3, i4, i5 = st.columns(5)
    close = indicators["close"]
    ema9  = indicators["ema9"]
    ema21 = indicators["ema21"]
    vwap  = indicators["vwap"]
    rsi   = indicators["rsi"]

    i1.metric("EMA 9",  f"{ema9:,.2f}",
              "▲ above price" if ema9 > close else "▼ below price")
    i2.metric("EMA 21", f"{ema21:,.2f}",
              "Bullish" if ema9 > ema21 else "Bearish")
    i3.metric("VWAP",   f"{vwap:,.2f}",
              "Price above" if close > vwap else "Price below")
    i4.metric("RSI 14", f"{rsi}",
              "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
    i5.metric("Candle interval", f"{candle_interval}m",
              f"{len(candle_df)} candles" if candle_df is not None else "--")
    st.divider()

# ==============================================================
# 22. GREEKS PANELS
# ==============================================================
if spot and atm and g_ce and g_pe:
    col_ce, col_pe = st.columns(2)

    with col_ce:
        label = "✅ RECOMMENDED" if (signal and signal["direction"] == "CE") else ""
        st.success(f"### {ce_strike:,} CE — Call  {label}")
        p1, p2, p3 = st.columns(3)
        p1.metric("Premium",      f"Rs.{ce_ltp:,.2f}" if ce_ltp else "--")
        p2.metric("Buyer Margin", fmt_inr(ce_ltp * conf["lot_size"] * lots) if ce_ltp else "--")
        p3.metric("OI",           f"{ce_data.get('ce_oi', 0):,}" if ce_data else "--")
        st.divider()
        a, b, c_ = st.columns(3)
        a.metric("Delta",     g_ce["delta"], sign_str(g_ce["delta"]))
        b.metric("Theta/day", g_ce["theta"])
        c_.metric("Gamma",    g_ce["gamma"])
        d_, e_, f_ = st.columns(3)
        d_.metric("Vega", g_ce["vega"])
        e_.metric("IV",   f"{g_ce['iv']:.1f}%")
        f_.metric("Source", greeks_source)

    with col_pe:
        label = "✅ RECOMMENDED" if (signal and signal["direction"] == "PE") else ""
        st.error(f"### {pe_strike:,} PE — Put  {label}")
        p1, p2, p3 = st.columns(3)
        p1.metric("Premium",      f"Rs.{pe_ltp:,.2f}" if pe_ltp else "--")
        p2.metric("Buyer Margin", fmt_inr(pe_ltp * conf["lot_size"] * lots) if pe_ltp else "--")
        p3.metric("OI",           f"{pe_data.get('pe_oi', 0):,}" if pe_data else "--")
        st.divider()
        a, b, c_ = st.columns(3)
        a.metric("Delta",     g_pe["delta"], sign_str(g_pe["delta"]))
        b.metric("Theta/day", g_pe["theta"])
        c_.metric("Gamma",    g_pe["gamma"])
        d_, e_, f_ = st.columns(3)
        d_.metric("Vega", g_pe["vega"])
        e_.metric("IV",   f"{g_pe['iv']:.1f}%")
        f_.metric("Source", greeks_source)

    st.divider()

    # Net summary
    st.markdown("### Net Position Summary")
    net_delta = round(g_ce["delta"] + g_pe["delta"], 4)
    net_theta = round(g_ce["theta"] + g_pe["theta"], 2)
    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("Total Premium", fmt_inr((ce_ltp + pe_ltp) * conf["lot_size"] * lots))
    n2.metric("Net Delta",     net_delta,
              "Neutral" if abs(net_delta) < 0.05 else "Directional")
    n3.metric("Net Theta/day", net_theta)
    n4.metric("Net Gamma",     round(g_ce["gamma"] + g_pe["gamma"], 6))
    n5.metric("Net Vega",      round(g_ce["vega"]  + g_pe["vega"],  2))

    st.divider()

# ==============================================================
# 23. OPTION CHAIN TABLE
# ==============================================================
if chain:
    st.markdown("### Option Chain — ATM ± 3 Strikes")
    rows = []
    for s in sorted(chain.keys()):
        d = chain[s]
        rows.append({
            "Strike":   s,
            "CE LTP":   f"Rs.{d['ce_ltp']:.2f}",
            "CE IV%":   f"{d['ce_iv']:.1f}",
            "CE Delta": f"{d['ce_delta']:.3f}",
            "CE OI":    f"{d['ce_oi']:,}",
            "PE LTP":   f"Rs.{d['pe_ltp']:.2f}",
            "PE IV%":   f"{d['pe_iv']:.1f}",
            "PE Delta": f"{d['pe_delta']:.3f}",
            "PE OI":    f"{d['pe_oi']:,}",
        })
    df_chain = pd.DataFrame(rows)

    def highlight_atm(row):
        color = "background-color: #1a472a; color: white" if row["Strike"] == atm else ""
        return [color] * len(row)

    st.dataframe(df_chain.style.apply(highlight_atm, axis=1),
                 width='stretch', hide_index=True)
    st.divider()

# ==============================================================
# 24. ALL-INDEX OVERVIEW
# ==============================================================
if run_live and all_prices:
    st.markdown("### All Index Prices")
    rows = []
    for idx_name, idx_conf in INDEX_CONFIG.items():
        entry = all_prices.get(idx_conf["response_key"])
        if entry:
            dte_i, exp_i = get_dte(idx_conf["expiry_weekday"])
            rows.append({
                "Index":    idx_name,
                "LTP":      f"Rs.{entry['ltp']:,.2f}",
                "Change %": f"{entry['change_pct']:+.2f}%",
                "Expiry":   exp_i.strftime("%d %b"),
                "DTE":      dte_i,
                "Updated":  entry["ts"].strftime("%H:%M:%S"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

if not run_live:
    st.info("Toggle **Start Live Feed** in the sidebar to begin.")

# ==============================================================
# 25. AUTO-REFRESH
# ==============================================================
if run_live:
    time.sleep(_refresh / 1000)
    st.rerun()
