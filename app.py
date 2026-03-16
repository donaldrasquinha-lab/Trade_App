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
    "prev_chain":  {},     # chain snapshot 30s ago for OI momentum
    "prev_chain_ts": None,
    "candle_ts":   None,
    "candle_df":   None,   # primary timeframe candles
    "candle_df_15": None,  # 15m candles
    "candle_ts_15": None,
    "candle_df_30": None,  # 30m candles
    "candle_ts_30": None,
    "fii_data":     None,
    "opt_candles":  {},
    "opt_candle_ts": None,
    "prev_closes":   {},    # prev day closing prices per instrument
    "prev_closes_ts": None,
    "show_points":   False,  # True = show pts diff, False = show %
    "active_trade":  None,   # current saved trade
    "trade_log":     [],     # list of completed trades
    "trade_saved_at": None,  # datetime trade was saved
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

def fmt_change(ltp, close_price, change_pct, show_pts, pts_diff=None):
    """Returns (display_string, color) for change display."""
    if show_pts:
        pts  = pts_diff if pts_diff is not None else (round(ltp - close_price, 2) if close_price else 0)
        sign = "+" if pts >= 0 else ""
        col  = "#00c853" if pts >= 0 else "#f44336"
        return f"{sign}{pts:.2f} pts", col
    else:
        col = "#00c853" if (change_pct or 0) >= 0 else "#f44336"
        return f"{change_pct:+.2f}%" if change_pct is not None else "", col

def fetch_prev_day_close(token, instrument_keys):
    """
    Fetches yesterday's official NSE closing price for each index.
    Tries HistoryV3Api first, falls back to HistoryApi.
    Stores errors in prev_closes["_errors"] for debug.
    """
    from datetime import timezone as _tz
    prev_closes = {}
    errors      = {}

    now_ist   = datetime.now(_tz.utc) + timedelta(hours=5, minutes=30)
    yesterday = now_ist.date() - timedelta(days=1)
    week_ago  = now_ist.date() - timedelta(days=10)

    conf = upstox_client.Configuration()
    conf.access_token = token
    client = upstox_client.ApiClient(conf)

    for ikey in instrument_keys:
        try:
            # Capture V3 error explicitly for debug
            _v3_err = None
            try:
                hist_api = upstox_client.HistoryV3Api(client)
                res = hist_api.get_historical_candle_data(
                    ikey, "days", "1",
                    str(yesterday), str(week_ago)
                )
            except Exception as e1:
                _v3_err = str(e1)
                # Fallback to legacy HistoryApi
                hist_api = upstox_client.HistoryApi(client)
                res = hist_api.get_historical_candle_data(
                    ikey, "days", "1",
                    str(yesterday), str(week_ago)
                )
            if _v3_err:
                errors[ikey + "_v3_err"] = _v3_err

            if res.status == "success" and res.data and res.data.candles:
                candles        = res.data.candles
                candles_sorted = sorted(candles, key=lambda c: c[0], reverse=True)
                prev_closes[ikey] = float(candles_sorted[0][4])
                prev_closes[ikey + "_debug"] = [
                    {"ts": str(c[0])[:10], "open": c[1], "high": c[2],
                     "low": c[3], "close": c[4]}
                    for c in candles_sorted[:3]
                ]
            else:
                errors[ikey] = f"API returned: status={getattr(res,'status','?')} data={bool(getattr(res,'data',None))}"

        except Exception as e:
            errors[ikey] = str(e)

    if errors:
        prev_closes["_errors"] = errors
    return prev_closes


def _extract_ohlc_val(ohlc_obj, key):
    """Safely extract a value from ohlc whether it is a dict or an object."""
    if ohlc_obj is None:
        return None
    if isinstance(ohlc_obj, dict):
        return ohlc_obj.get(key)
    # SDK object — try attribute directly
    val = getattr(ohlc_obj, key, None)
    if val is not None:
        return val
    # Try to_dict fallback
    if hasattr(ohlc_obj, "to_dict"):
        d = ohlc_obj.to_dict()
        return d.get(key)
    return None


def fetch_all_prices(token):
    try:
        conf = upstox_client.Configuration()
        conf.access_token = token
        api  = upstox_client.MarketQuoteApi(upstox_client.ApiClient(conf))
        keys_str = ",".join(ALL_INSTRUMENT_KEYS + [VIX_INSTRUMENT_KEY])
        res = api.get_market_quote_ohlc(keys_str, "1d", "2.0")
        prices   = {}
        vix      = None
        _debug   = {}   # store raw values for sidebar debug

        if res.status == "success" and res.data:
            for rkey, quote in res.data.items():
                # Get raw dict regardless of SDK version
                if isinstance(quote, dict):
                    q = quote
                elif hasattr(quote, "to_dict"):
                    q = quote.to_dict()
                else:
                    q = {}

                ltp        = q.get("last_price") or getattr(quote, "last_price", None)
                ohlc_raw   = q.get("ohlc") or getattr(quote, "ohlc", None)

                # Extract all OHLC fields
                o_open  = _extract_ohlc_val(ohlc_raw, "open")
                o_close = _extract_ohlc_val(ohlc_raw, "close")
                o_high  = _extract_ohlc_val(ohlc_raw, "high")
                o_low   = _extract_ohlc_val(ohlc_raw, "low")

                _debug[rkey] = {
                    "ltp": ltp, "open": o_open,
                    "close": o_close, "high": o_high, "low": o_low
                }

                if ltp is None:
                    continue

                ltp        = float(ltp)
                # ohlc.close from "1d" interval = previous session official close
                prev_close = float(o_close) if o_close else None
                today_open = float(o_open)  if o_open  else ltp

                # For NSE indices the "1d" OHLC close = current LTP (running close).
                # The only true prev-day reference available intraday is today's OPEN,
                # which NSE sets at the previous session's official close for indices.
                prev_close = today_open   # open = yesterday's official close for NSE indices

                change_pct = round(((ltp - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
                pts_diff   = round(ltp - prev_close, 2) if prev_close else 0.0

                entry = {
                    "ltp":        ltp,
                    "close":      prev_close or ltp,
                    "today_open": today_open,
                    "change_pct": change_pct,
                    "pts_diff":   pts_diff,
                    "ts":         datetime.now(),
                }
                if rkey == VIX_RESPONSE_KEY:
                    vix = ltp
                else:
                    prices[rkey] = entry

        # Store debug info in session state for sidebar display
        try:
            st.session_state["_price_debug"] = _debug
        except Exception:
            pass

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
# 7b. SUPPORT & RESISTANCE ENGINE
#     Works on any candle DataFrame (index or option premium).
#     Uses three methods and merges nearby levels:
#       - Swing highs/lows  (local pivot points)
#       - Round numbers     (psychological levels)
#       - Volume nodes      (high-volume price areas)
# ==============================================================
def compute_snr(df, n_levels=3, merge_pct=0.005):
    """
    Returns {"support": [...], "resistance": [...]}
    Each level: {"price": float, "strength": int, "type": str}
    merge_pct: levels within this % of each other are merged.
    """
    if df is None or len(df) < 6:
        return {"support": [], "resistance": []}

    levels  = []   # list of (price, "support"/"resistance", strength)
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    volumes = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    current = closes[-1]

    # ── Method 1: Swing pivot points ─────────
    # A swing high: high[i] > high[i-1] and high[i] > high[i+1]
    # A swing low:  low[i]  < low[i-1]  and low[i]  < low[i+1]
    for i in range(2, len(df) - 2):
        # Swing high (resistance)
        if highs[i] > highs[i-1] and highs[i] > highs[i+1] and            highs[i] > highs[i-2] and highs[i] > highs[i+2]:
            strength = int(volumes[i] / (np.mean(volumes) + 1e-9) * 2)
            levels.append((highs[i], "resistance", max(1, min(strength, 5))))
        # Swing low (support)
        if lows[i] < lows[i-1] and lows[i] < lows[i+1] and            lows[i] < lows[i-2] and lows[i] < lows[i+2]:
            strength = int(volumes[i] / (np.mean(volumes) + 1e-9) * 2)
            levels.append((lows[i], "support", max(1, min(strength, 5))))

    # ── Method 2: Round number levels ────────
    # For index: multiples of 50/100; for options: multiples of 5/10
    magnitude = current
    if magnitude > 1000:
        step = 50
    elif magnitude > 100:
        step = 10
    else:
        step = 5
    lo = current * 0.97
    hi = current * 1.03
    rn = lo - (lo % step)
    while rn <= hi:
        kind = "support" if rn < current else "resistance"
        levels.append((round(rn, 2), kind, 2))
        rn += step

    # ── Method 3: High-volume nodes ──────────
    if len(df) >= 10:
        # Bin prices into 10 buckets, find top-volume buckets
        price_min = min(lows)
        price_max = max(highs)
        if price_max > price_min:
            bins = np.linspace(price_min, price_max, 11)
            vol_by_bin = np.zeros(10)
            price_of_bin = [(bins[i] + bins[i+1]) / 2 for i in range(10)]
            for i in range(len(df)):
                mid = (highs[i] + lows[i]) / 2
                idx = min(int((mid - price_min) / (price_max - price_min) * 10), 9)
                vol_by_bin[idx] += volumes[i]
            # Top 3 volume bins become S/R
            top_bins = np.argsort(vol_by_bin)[-3:]
            for b in top_bins:
                p    = price_of_bin[b]
                kind = "support" if p < current else "resistance"
                levels.append((round(p, 2), kind, 3))

    # ── Merge nearby levels ───────────────────
    def merge_levels(raw, kind):
        raw = sorted([p for p, k, _ in raw if k == kind])
        merged = []
        i = 0
        while i < len(raw):
            cluster = [raw[i]]
            while i + 1 < len(raw) and (raw[i+1] - raw[i]) / (raw[i] + 1e-9) < merge_pct:
                i += 1
                cluster.append(raw[i])
            merged.append(round(np.mean(cluster), 2))
            i += 1
        return merged

    supports    = merge_levels(levels, "support")
    resistances = merge_levels(levels, "resistance")

    # Keep only levels near current price (within 5%)
    def near(lst):
        return sorted([p for p in lst if abs(p - current) / current < 0.05],
                       key=lambda p: abs(p - current))

    supports    = near(supports)[:n_levels]
    resistances = near(resistances)[:n_levels]

    return {
        "support":    supports,
        "resistance": resistances,
        "current":    current,
    }

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
                    oi_val = int(md.get("oi", 0) or 0)
                    return {
                        "ltp":   float(md.get("ltp", 0) or 0),
                        "oi":    oi_val,
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
# OI MOMENTUM ENGINE
# Compares current vs previous chain OI to determine:
#   Long Buildup   : Price↑ + OI↑  → Buy CE
#   Short Buildup  : Price↓ + OI↑  → Buy PE
#   Short Covering : Price↑ + OI↓  → Caution (exit PE)
#   Long Unwinding : Price↓ + OI↓  → Caution (exit CE)
# Also finds max OI strikes (support/resistance walls)
# ==============================================================
def compute_oi_momentum(chain, prev_chain, current_price, prev_price):
    """
    Returns OI momentum analysis dict.
    chain / prev_chain : {strike: {ce_oi, pe_oi, ce_ltp, pe_ltp, ...}}
    current_price / prev_price : float
    """
    if not chain:
        return None

    # ── Total OI change ──────────────────────────────────────
    total_ce_oi_now  = sum(v.get("ce_oi", 0) for v in chain.values())
    total_pe_oi_now  = sum(v.get("pe_oi", 0) for v in chain.values())
    total_ce_oi_prev = sum(v.get("ce_oi", 0) for v in prev_chain.values()) if prev_chain else total_ce_oi_now
    total_pe_oi_prev = sum(v.get("pe_oi", 0) for v in prev_chain.values()) if prev_chain else total_pe_oi_now

    ce_oi_change = total_ce_oi_now - total_ce_oi_prev
    pe_oi_change = total_pe_oi_now - total_pe_oi_prev

    # Determine dominant OI momentum
    total_oi_now  = total_ce_oi_now  + total_pe_oi_now
    total_oi_prev = total_ce_oi_prev + total_pe_oi_prev
    oi_rising     = total_oi_now > total_oi_prev

    price_rising  = (current_price > prev_price) if prev_price else None

    # ── Decision matrix ──────────────────────────────────────
    if price_rising is True  and oi_rising:
        scenario     = "Long Buildup"
        decision     = "BUY CE"
        decision_col = "#00c853"
        decision_bg  = "#0d3320"
        emoji        = "🟢"
        strength     = "Strong"
    elif price_rising is False and oi_rising:
        scenario     = "Short Buildup"
        decision     = "BUY PE"
        decision_col = "#f44336"
        decision_bg  = "#3d0a0a"
        emoji        = "🔴"
        strength     = "Strong"
    elif price_rising is True  and not oi_rising:
        scenario     = "Short Covering"
        decision     = "CAUTION — Exit PE"
        decision_col = "#ffc107"
        decision_bg  = "#2a2200"
        emoji        = "🟡"
        strength     = "Weak"
    elif price_rising is False and not oi_rising:
        scenario     = "Long Unwinding"
        decision     = "CAUTION — Exit CE"
        decision_col = "#ffc107"
        decision_bg  = "#2a2200"
        emoji        = "🟡"
        strength     = "Weak"
    else:
        scenario     = "Neutral"
        decision     = "WAIT"
        decision_col = "#90caf9"
        decision_bg  = "#0d1f33"
        emoji        = "⚪"
        strength     = "Neutral"

    # ── Max OI strike (support/resistance walls) ─────────────
    max_ce_oi_strike = max(chain.keys(), key=lambda s: chain[s].get("ce_oi", 0)) if chain else None
    max_pe_oi_strike = max(chain.keys(), key=lambda s: chain[s].get("pe_oi", 0)) if chain else None
    max_ce_oi_val    = chain[max_ce_oi_strike].get("ce_oi", 0) if max_ce_oi_strike else 0
    max_pe_oi_val    = chain[max_pe_oi_strike].get("pe_oi", 0) if max_pe_oi_strike else 0

    # Resistance wall: highest CE OI strike (option sellers defend here)
    # Support wall:    highest PE OI strike (option sellers defend here)
    resistance_wall = max_ce_oi_strike  # call writers defend this level
    support_wall    = max_pe_oi_strike  # put writers defend this level

    # ── PCR from chain ───────────────────────────────────────
    pcr_now  = round(total_pe_oi_now  / total_ce_oi_now,  2) if total_ce_oi_now  else None
    pcr_prev = round(total_pe_oi_prev / total_ce_oi_prev, 2) if total_ce_oi_prev else None
    pcr_rising = (pcr_now > pcr_prev) if (pcr_now and pcr_prev) else None

    # ── OI availability and zero-data fallback ──────────────────
    oi_data_available = (total_ce_oi_now > 0 or total_pe_oi_now > 0)

    if not oi_data_available:
        # OI data not returned by API — use price direction only
        if price_rising is True:
            scenario, decision = "Price Rising", "Lean CE (Price signal)"
            decision_col, decision_bg, emoji, strength = "#00c853","#0d3320","🟢","Moderate"
        elif price_rising is False:
            scenario, decision = "Price Falling", "Lean PE (Price signal)"
            decision_col, decision_bg, emoji, strength = "#f44336","#3d0a0a","🔴","Moderate"
        else:
            scenario, decision = "Neutral", "WAIT"
            decision_col, decision_bg, emoji, strength = "#90caf9","#0d1f33","⚪","Neutral"

    # ── Liquidity: top OI strikes — fallback to highest LTP ──────
    if chain:
        oi_values  = sorted([v.get("ce_oi", 0) + v.get("pe_oi", 0)
                             for v in chain.values()], reverse=True)
        top_decile = oi_values[0] * 0.7 if oi_values[0] > 0 else 0
        liquid_strikes = [s for s, v in chain.items()
                         if (v.get("ce_oi", 0) + v.get("pe_oi", 0)) >= top_decile
                         and top_decile > 0]
        if not liquid_strikes:
            # Fallback: strikes with highest combined LTP (most active)
            liquid_strikes = sorted(
                chain.keys(),
                key=lambda s: chain[s].get("ce_ltp",0) + chain[s].get("pe_ltp",0),
                reverse=True
            )[:3]
    else:
        liquid_strikes = []

    return {
        "scenario":         scenario,
        "decision":         decision,
        "decision_col":     decision_col,
        "decision_bg":      decision_bg,
        "emoji":            emoji,
        "strength":         strength,
        "price_rising":     price_rising,
        "oi_rising":        oi_rising,
        "oi_available":     oi_data_available,
        "ce_oi_now":        total_ce_oi_now,
        "pe_oi_now":        total_pe_oi_now,
        "ce_oi_change":     ce_oi_change,
        "pe_oi_change":     pe_oi_change,
        "resistance_wall":  resistance_wall,
        "support_wall":     support_wall,
        "max_ce_oi":        max_ce_oi_val,
        "max_pe_oi":        max_pe_oi_val,
        "pcr":              pcr_now,
        "pcr_prev":         pcr_prev,
        "pcr_rising":       pcr_rising,
        "liquid_strikes":   liquid_strikes,
    }

# ==============================================================
# 9. INDICATORS
# ==============================================================
def compute_indicators(df):
    """
    Computes EMA9, EMA21, VWAP, RSI14, ADX14.
    ADX < 20 = weak/choppy, 20-25 = developing, 25-40 = strong, >40 = very strong.
    +DI > -DI = bullish direction.
    """
    if df is None or len(df) < 5:
        return df, {}

    df = df.copy()

    # EMA
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # VWAP
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["tpv"]  = df["tp"] * df["volume"]
    df["vwap"] = df["tpv"].cumsum() / df["volume"].cumsum()

    # RSI 14
    delta_ = df["close"].diff()
    gain_  = delta_.clip(lower=0)
    loss_  = (-delta_).clip(lower=0)
    avg_g_ = gain_.ewm(com=13, adjust=False).mean()
    avg_l_ = loss_.ewm(com=13, adjust=False).mean()
    rs_    = avg_g_ / avg_l_.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs_))

    # ADX 14 (Wilder method)
    period    = 14
    prev_cl   = df["close"].shift(1)
    tr        = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_cl).abs(),
        (df["low"]  - prev_cl).abs(),
    ], axis=1).max(axis=1)

    up_mv   = df["high"].diff()
    dn_mv   = (-df["low"].diff())
    plus_dm = pd.Series(
        np.where((up_mv > dn_mv) & (up_mv > 0), up_mv, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((dn_mv > up_mv) & (dn_mv > 0), dn_mv, 0.0), index=df.index)

    def wilder(s, n):
        out = pd.Series(np.nan, index=s.index)
        if len(s) < n + 1:
            return out
        out.iloc[n] = s.iloc[1:n+1].sum()
        for i in range(n + 1, len(s)):
            out.iloc[i] = out.iloc[i-1] - out.iloc[i-1] / n + s.iloc[i]
        return out

    tr_w   = wilder(tr,       period)
    pdm_w  = wilder(plus_dm,  period)
    mdm_w  = wilder(minus_dm, period)

    df["+di"] = 100 * pdm_w  / tr_w.replace(0, np.nan)
    df["-di"] = 100 * mdm_w  / tr_w.replace(0, np.nan)

    dx        = 100 * (df["+di"] - df["-di"]).abs() / (df["+di"] + df["-di"]).replace(0, np.nan)
    df["adx"] = dx.ewm(span=period, adjust=False).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    adx_val  = round(float(last["adx"]),  1) if not pd.isna(last["adx"])  else None
    plus_di  = round(float(last["+di"]),  1) if not pd.isna(last["+di"])  else None
    minus_di = round(float(last["-di"]),  1) if not pd.isna(last["-di"])  else None

    if adx_val is None:     adx_label = "Computing"
    elif adx_val >= 40:     adx_label = "Very Strong Trend"
    elif adx_val >= 25:     adx_label = "Strong Trend"
    elif adx_val >= 20:     adx_label = "Developing Trend"
    else:                   adx_label = "Weak / Choppy"

    adx_dir = "Bullish" if (plus_di and minus_di and plus_di > minus_di) else "Bearish"

    return df, {
        "close":      last["close"],
        "ema9":       round(last["ema9"],  2),
        "ema21":      round(last["ema21"], 2),
        "vwap":       round(last["vwap"],  2),
        "rsi":        round(last["rsi"],   1),
        "volume":     int(last["volume"]),
        "ema9_prev":  round(prev["ema9"],  2),
        "ema21_prev": round(prev["ema21"], 2),
        "prev_close": prev["close"],
        "adx":        adx_val,
        "plus_di":    plus_di,
        "minus_di":   minus_di,
        "adx_label":  adx_label,
        "adx_dir":    adx_dir,
    }

# ==============================================================
# 10. SIGNAL ENGINE  (Multi-factor: Technical + Macro)
#
#     LAYER 1 — Technical (max ±8 pts)
#       EMA9 vs EMA21        ±1
#       EMA crossover        ±2  (strong)
#       Price vs VWAP        ±1
#       Price vs EMA9        ±1
#       RSI                  ±1
#       MTF 15m alignment    ±1  (bonus if 15m agrees)
#       MTF 30m alignment    ±1  (bonus if 30m agrees)
#
#     LAYER 2 — Macro/Market (max ±4 pts, fed into same score)
#       India VIX            ±1  (direction bias + confidence cap)
#       Intraday change      ±1  (day trend confirmation)
#       VIX extreme          confidence cap override
#
#     THRESHOLD — dynamic based on market conditions
#       Normal:    score ≥ ±3 triggers recommendation
#       High VIX:  score ≥ ±4 required (harder to trigger)
#       Conflicted macro: score ≥ ±4 required
# ==============================================================
def generate_signal(ind, spot, vix, high_vol_window,
                    change_pct=None, ind_15=None, ind_30=None, oi_score_bonus=0):
    """
    Full multi-factor signal engine.
    ind        : primary timeframe indicators dict
    vix        : India VIX float
    change_pct : intraday % change of index
    ind_15     : 15m indicators dict (optional, for MTF)
    ind_30     : 30m indicators dict (optional, for MTF)
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

    # ════════════════════════════════════════════
    # LAYER 1: TECHNICAL INDICATORS
    # ════════════════════════════════════════════

    # ── 1. EMA9 vs EMA21 trend ────────────────
    if ema9 > ema21:
        score += 1
        reasons.append(("bull", f"EMA9 ({ema9:.0f}) > EMA21 ({ema21:.0f}) — uptrend"))
    else:
        score -= 1
        reasons.append(("bear", f"EMA9 ({ema9:.0f}) < EMA21 ({ema21:.0f}) — downtrend"))

    # ── 2. Fresh EMA crossover ────────────────
    if ind["ema9_prev"] <= ind["ema21_prev"] and ema9 > ema21:
        score += 2
        reasons.append(("bull", "EMA9 just crossed ABOVE EMA21 (golden cross)"))
    elif ind["ema9_prev"] >= ind["ema21_prev"] and ema9 < ema21:
        score -= 2
        reasons.append(("bear", "EMA9 just crossed BELOW EMA21 (death cross)"))

    # ── 3. Price vs VWAP ──────────────────────
    if close > vwap:
        score += 1
        reasons.append(("bull", f"Price ({close:.0f}) above VWAP ({vwap:.0f}) — buyers in control"))
    else:
        score -= 1
        reasons.append(("bear", f"Price ({close:.0f}) below VWAP ({vwap:.0f}) — sellers in control"))

    # ── 4. Price vs EMA9 momentum ─────────────
    if close > ema9:
        score += 1
        reasons.append(("bull", f"Price above EMA9 — positive momentum"))
    else:
        score -= 1
        reasons.append(("bear", f"Price below EMA9 — negative momentum"))

    # ── 5. RSI ────────────────────────────────
    if 55 <= rsi <= 75:
        score += 1
        reasons.append(("bull", f"RSI {rsi} in bullish zone (55–75)"))
    elif 25 <= rsi <= 45:
        score -= 1
        reasons.append(("bear", f"RSI {rsi} in bearish zone (25–45)"))
    elif rsi > 80:
        score -= 1
        reasons.append(("warn", f"RSI {rsi} overbought — risk of reversal, avoid CE"))
    elif rsi < 20:
        score += 1
        reasons.append(("warn", f"RSI {rsi} oversold — possible bounce, favour CE"))

    # ── 6. MTF 15m alignment ──────────────────
    if ind_15:
        mtf15_bull = (ind_15["ema9"] > ind_15["ema21"] and
                      ind_15["close"] > ind_15["vwap"])
        mtf15_bear = (ind_15["ema9"] < ind_15["ema21"] and
                      ind_15["close"] < ind_15["vwap"])
        if mtf15_bull:
            score += 1
            reasons.append(("bull", "15m chart aligned bullish (EMA & VWAP)"))
        elif mtf15_bear:
            score -= 1
            reasons.append(("bear", "15m chart aligned bearish (EMA & VWAP)"))
        else:
            reasons.append(("info", "15m chart: mixed / neutral"))

    # ── 7. MTF 30m alignment ──────────────────
    if ind_30:
        mtf30_bull = (ind_30["ema9"] > ind_30["ema21"] and
                      ind_30["close"] > ind_30["vwap"])
        mtf30_bear = (ind_30["ema9"] < ind_30["ema21"] and
                      ind_30["close"] < ind_30["vwap"])
        if mtf30_bull:
            score += 1
            reasons.append(("bull", "30m chart aligned bullish (EMA & VWAP) — strong confirmation"))
        elif mtf30_bear:
            score -= 1
            reasons.append(("bear", "30m chart aligned bearish (EMA & VWAP) — strong confirmation"))
        else:
            reasons.append(("info", "30m chart: mixed / neutral"))

    # ════════════════════════════════════════════
    # LAYER 2: MACRO / MARKET CONTEXT
    # ════════════════════════════════════════════

    vix_extreme   = False   # True = require higher threshold
    macro_conflict = False  # True = technical & macro disagree

    # ── 8. India VIX ──────────────────────────
    if vix is not None:
        if vix > 22:
            score -= 1       # high fear biases toward PE
            vix_extreme = True
            reasons.append(("bear", f"VIX {vix:.1f} > 22 — high fear, bearish bias (-1), threshold raised"))
        elif vix > 16:
            reasons.append(("warn", f"VIX {vix:.1f} elevated — widen stop-loss by 50%"))
        elif vix < 12:
            score -= 1       # low VIX = complacency, moves may be muted
            reasons.append(("warn", f"VIX {vix:.1f} very low — weak momentum, signal penalised (-1)"))
        else:
            score += 1       # VIX 12–16 = ideal scalping conditions
            reasons.append(("bull", f"VIX {vix:.1f} in ideal range (12–16) — good scalping conditions (+1)"))

    # ── 9. Intraday change ───────────────────
    if change_pct is not None:
        if change_pct > 0.8:
            score += 1
            reasons.append(("bull", f"Intraday +{change_pct:.2f}% — strong positive day (+1)"))
        elif change_pct > 0.3:
            reasons.append(("bull", f"Intraday +{change_pct:.2f}% — mildly positive day"))
        elif change_pct < -0.8:
            score -= 1
            reasons.append(("bear", f"Intraday {change_pct:.2f}% — strong negative day (-1)"))
        elif change_pct < -0.3:
            reasons.append(("bear", f"Intraday {change_pct:.2f}% — mildly negative day"))
        else:
            reasons.append(("info", f"Intraday {change_pct:+.2f}% — flat / sideways day"))

    # ── OI Momentum bonus ────────────────────────
    if oi_score_bonus != 0:
        score += oi_score_bonus
        if oi_score_bonus > 0:
            reasons.append(("bull", f"OI Momentum: Long Buildup — CE positions increasing (+{oi_score_bonus})"))
        else:
            reasons.append(("bear", f"OI Momentum: Short Buildup — PE positions increasing ({oi_score_bonus})"))

    # ════════════════════════════════════════════
    # TRADING WINDOW MODIFIER
    # ════════════════════════════════════════════
    if high_vol_window:
        reasons.append(("info", "Inside high-volatility window (9:15–10:15 AM) — prime scalping time"))
    else:
        score = int(score * 0.7)
        reasons.append(("warn", "Outside prime scalping window — score reduced 30%"))

    # ════════════════════════════════════════════
    # DYNAMIC THRESHOLD
    # Normal market:   ±3 to trigger
    # High VIX:        ±4 required  (volatile = need stronger signal)
    # Macro conflict:  ±4 required  (technicals vs PCR disagreeing)
    # Both:            ±5 required  (very high bar)
    # ════════════════════════════════════════════
    if vix_extreme and macro_conflict:
        threshold = 5
        reasons.append(("warn", "Threshold raised to ±5 (high VIX + macro conflict)"))
    elif vix_extreme or macro_conflict:
        threshold = 4
        reasons.append(("warn", f"Threshold raised to ±4 ({'high VIX' if vix_extreme else 'macro conflict'})"))
    else:
        threshold = 3

    abs_score = abs(score)

    # ── Final recommendation ──────────────────
    if score >= threshold:
        direction      = "CE"
        recommendation = "BUY CE"
        confidence     = "High" if abs_score >= threshold + 1 else "Medium"
        emoji          = "🟢"
    elif score <= -threshold:
        direction      = "PE"
        recommendation = "BUY PE"
        confidence     = "High" if abs_score >= threshold + 1 else "Medium"
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

    # ── Target & stop-loss ────────────────────
    # Widen SL in high VIX conditions
    strike_move = 15
    sl_ratio    = 0.75 if (vix and vix > 16) else 0.5
    if direction == "CE":
        target    = round(close + strike_move, 0)
        stop_loss = round(close - (strike_move * sl_ratio), 0)
    elif direction == "PE":
        target    = round(close - strike_move, 0)
        stop_loss = round(close + (strike_move * sl_ratio), 0)
    else:
        target = stop_loss = None

    rr = round(1 / sl_ratio, 1) if sl_ratio else 2.0

    return {
        "score":          score,
        "threshold":      threshold,
        "direction":      direction,
        "recommendation": recommendation,
        "confidence":     confidence,
        "emoji":          emoji,
        "target":         target,
        "stop_loss":      stop_loss,
        "rr":             rr,
        "vix_extreme":    vix_extreme,
        "macro_conflict": macro_conflict,
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

    candle_interval = st.selectbox("Primary Interval", ["1", "3", "5", "15", "30"], index=0,
                                    format_func=lambda x: f"{x} min",
                                    help="Used for signal. 15m & 30m always shown for confluence.")

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

    st.divider()
    _toggle_label = "Show % Change" if st.session_state.show_points else "Show Points Diff"
    if st.button(_toggle_label):
        st.session_state.show_points = not st.session_state.show_points
        st.rerun()

    # Debug expander — shows raw OHLC values from API
    if st.session_state.get("_price_debug"):
        with st.expander("🔍 Raw OHLC Debug"):
            for k, v in st.session_state["_price_debug"].items():
                st.write(f"**{k}**")
                st.json(v)
    # Prev closes debug
    with st.expander("🔍 Prev Close Debug"):
        pc = st.session_state.get("prev_closes", {})
        if not pc:
            st.write("Not fetched yet — prev_closes is empty")
        else:
            if "_errors" in pc:
                st.error(f"Errors: {pc['_errors']}")
            for k, v in pc.items():
                if k.startswith("_"):
                    continue
                if "_debug" not in k:
                    dbg = pc.get(k + "_debug", [])
                    st.write(f"**{k}** → `{v}`")
                    if dbg:
                        st.json(dbg)

# ==============================================================
# 14. FETCH SPOT PRICES
# ==============================================================
if run_live:
    prices, vix, _ = fetch_all_prices(TOKEN)
    if prices:
        # Fetch actual prev-day closes from historical API (once per session)
        _pc_ts  = st.session_state.prev_closes_ts
        _pc_age = (datetime.now() - _pc_ts).total_seconds() if _pc_ts else 9999
        if _pc_age > 60 or not st.session_state.prev_closes or "_errors" in st.session_state.prev_closes:
            _ikeys = [v["instrument_key"] for v in INDEX_CONFIG.values()]
            _prev  = fetch_prev_day_close(TOKEN, _ikeys)
            if _prev:
                st.session_state.prev_closes    = _prev
                st.session_state.prev_closes_ts = datetime.now()

        # Override change_pct and pts_diff using true prev-day close
        _pc = st.session_state.prev_closes
        for rkey, entry in prices.items():
            _ikey      = rkey.replace(":", "|")
            true_close = _pc.get(_ikey)
            if true_close and true_close > 0:
                ltp_v                = entry["ltp"]
                entry["close"]       = true_close
                entry["pts_diff"]    = round(ltp_v - true_close, 2)
                entry["change_pct"]  = round((ltp_v - true_close) / true_close * 100, 2)

        st.session_state.last_prices = prices
    if vix:
        st.session_state.last_vix = vix

all_prices = st.session_state.last_prices
feed_entry = all_prices.get(conf["response_key"])
spot       = feed_entry["ltp"]       if feed_entry else None
change_pct = feed_entry["change_pct"] if feed_entry else None
data_age   = (datetime.now() - feed_entry["ts"]).total_seconds() if feed_entry else None

# ==============================================================
# 15. FETCH CANDLES  (primary + 15m + 30m)
#     Primary: every 60s
#     15m / 30m: every 5 min (candles change less frequently)
# ==============================================================
candle_df  = st.session_state.candle_df
indicators = {}
candle_err = None

ind_15 = {}   # indicators from 15m chart
ind_30 = {}   # indicators from 30m chart

if run_live and spot:
    # -- Primary timeframe --
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

    # -- 15 minute timeframe --
    age_15 = (
        (datetime.now() - st.session_state.candle_ts_15).total_seconds()
        if st.session_state.candle_ts_15 else 999
    )
    if age_15 >= 300:
        df15, _ = fetch_candles(TOKEN, conf["instrument_key"], 15)
        if df15 is not None and len(df15) >= 5:
            st.session_state.candle_df_15 = df15
            st.session_state.candle_ts_15 = datetime.now()
    if st.session_state.candle_df_15 is not None:
        _, ind_15 = compute_indicators(st.session_state.candle_df_15)

    # -- 30 minute timeframe --
    age_30 = (
        (datetime.now() - st.session_state.candle_ts_30).total_seconds()
        if st.session_state.candle_ts_30 else 999
    )
    if age_30 >= 300:
        df30, _ = fetch_candles(TOKEN, conf["instrument_key"], 30)
        if df30 is not None and len(df30) >= 5:
            st.session_state.candle_df_30 = df30
            st.session_state.candle_ts_30 = datetime.now()
    if st.session_state.candle_df_30 is not None:
        _, ind_30 = compute_indicators(st.session_state.candle_df_30)

# ==============================================================
# 16. FETCH OPTION CHAIN  (every 5s)
# ==============================================================
step  = conf["strike_step"]
atm   = int(round(spot / step) * step) if spot else None
chain = st.session_state.last_chain

if run_live and spot and atm:
    expiry_str = expiry_date.strftime("%Y-%m-%d")

    # ── Display chain: ATM ±3, refresh every 5s ──────────────
    chain_age = (
        (datetime.now() - st.session_state.chain_ts).total_seconds()
        if st.session_state.chain_ts else 999
    )
    if chain_age >= 5:
        new_chain, _ = fetch_option_chain(TOKEN, conf["instrument_key"],
                                           expiry_str, atm, step, n=3)
        if new_chain:
            # Save current chain as prev before overwriting
            if st.session_state.last_chain:
                prev_ts = st.session_state.chain_ts
                # Only rotate to prev if chain is at least 25s old (meaningful OI change)
                if prev_ts and (datetime.now() - prev_ts).total_seconds() >= 25:
                    st.session_state.prev_chain    = dict(st.session_state.last_chain)
                    st.session_state.prev_chain_ts = prev_ts
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
elif spot and atm and ce_strike and pe_strike:
    # Always compute BS Greeks when we have spot — never leave g_ce as None
    g_ce = bs_greeks(spot, ce_strike, dte, r, iv, "call"); g_ce["iv"] = iv_pct
    g_pe = bs_greeks(spot, pe_strike, dte, r, iv, "put");  g_pe["iv"] = iv_pct
    greeks_source = "BS Model"
else:
    g_ce = {"delta": 0.5, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "iv": iv_pct}
    g_pe = {"delta":-0.5, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "iv": iv_pct}
    greeks_source = "Estimated"


# ==============================================================
# 17b. MARKET OUTLOOK ENGINE
#      Derives Bullish/Bearish/Sideways + metric table from
#      live data: VIX, PCR from option chain, RSI, signal score
# ==============================================================
def compute_market_outlook(vix, rsi, signal_score, change_pct):
    """
    Returns outlook dict with overall sentiment and metric rows.
    VIX > 20 = high fear, < 13 = complacency.
    """
    bull_points = 0
    bear_points = 0
    metrics     = []

    # VIX
    if vix is not None:
        if vix > 22:
            bear_points += 2
            vix_sig = ("😨 High Fear / Volatile", "bear")
        elif vix > 16:
            bear_points += 1
            vix_sig = ("😰 Elevated Volatility", "warn")
        elif vix < 12:
            bear_points += 1
            vix_sig = ("😴 Low Fear / Complacency", "warn")
        else:
            bull_points += 1
            vix_sig = ("😌 Normal Volatility", "bull")
        metrics.append(("India VIX", f"{vix:.2f}", vix_sig[0], vix_sig[1]))

    # RSI
    if rsi:
        if rsi > 70:
            bear_points += 1
            rsi_sig = ("📉 Overbought (Bearish Momentum)", "bear")
        elif rsi > 55:
            bull_points += 1
            rsi_sig = ("📈 Bullish Momentum", "bull")
        elif rsi > 45:
            rsi_sig = ("➡️ Neutral Momentum", "neutral")
            pass
        elif rsi > 30:
            bear_points += 1
            rsi_sig = ("📉 Weak / Bearish Momentum", "warn")
        else:
            bear_points += 2
            rsi_sig = ("📉 Oversold (Bearish Momentum)", "bear")
        metrics.append(("Technical RSI", f"{rsi:.0f}", rsi_sig[0], rsi_sig[1]))

    # ADX
    adx_val_ = indicators.get("adx")       if indicators else None
    adx_dir_ = indicators.get("adx_dir",   "") if indicators else ""
    adx_lbl_ = indicators.get("adx_label", "") if indicators else ""
    if adx_val_ is not None:
        if adx_val_ >= 40:
            adx_sig_ = (f"🔥 Very Strong Trend ({adx_dir_}) — watch for exhaustion", "warn")
        elif adx_val_ >= 25:
            if adx_dir_ == "Bullish":
                bull_points += 1
                adx_sig_ = (f"💪 Strong Bullish Trend", "bull")
            else:
                bear_points += 1
                adx_sig_ = (f"💪 Strong Bearish Trend", "bear")
        elif adx_val_ >= 20:
            adx_sig_ = (f"📈 Developing Trend ({adx_dir_})", "neutral")
        else:
            adx_sig_ = ("↔️ Weak / Choppy — range-bound, scalp carefully", "warn")
        metrics.append(("ADX (Trend Strength)", f"{adx_val_:.1f}", adx_sig_[0], adx_sig_[1]))

    # Intraday change
    if change_pct is not None:
        if change_pct > 0.5:
            bull_points += 1
            chg_sig = ("📈 Positive Day", "bull")
        elif change_pct < -0.5:
            bear_points += 1
            chg_sig = ("📉 Negative Day", "bear")
        else:
            chg_sig = ("➡️ Flat Day", "neutral")
        # Show both % and pts in the outlook table
        metrics.append(("Intraday Move", f"{change_pct:+.2f}%", chg_sig[0], chg_sig[1]))

    # Signal score from technical indicators
    if signal_score is not None:
        if signal_score >= 3:
            bull_points += 2
            sig_lbl = ("🟢 Bullish Signal", "bull")
        elif signal_score <= -3:
            bear_points += 2
            sig_lbl = ("🔴 Bearish Signal", "bear")
        else:
            sig_lbl = ("🟡 Mixed Signal", "warn")
        metrics.append(("MTF Signal", f"{signal_score:+d}/6", sig_lbl[0], sig_lbl[1]))

    # Overall verdict
    total = bull_points + bear_points
    if total == 0:
        sentiment = "SIDEWAYS"
        sent_emoji = "↔️"
        sent_color = "#ffc107"
        sent_bg    = "#2a2a00"
    elif bull_points > bear_points * 1.3:
        sentiment = "BULLISH"
        sent_emoji = "📈"
        sent_color = "#00c853"
        sent_bg    = "#0d3320"
    elif bear_points > bull_points * 1.3:
        sentiment = "BEARISH"
        sent_emoji = "📉"
        sent_color = "#f44336"
        sent_bg    = "#3d0a0a"
    else:
        sentiment = "SIDEWAYS"
        sent_emoji = "↔️"
        sent_color = "#ffc107"
        sent_bg    = "#2a2a00"

    return {
        "sentiment":   sentiment,
        "emoji":       sent_emoji,
        "color":       sent_color,
        "bg":          sent_bg,
        "bull_points": bull_points,
        "bear_points": bear_points,
        "metrics":     metrics,
    }


# ==============================================================
# 16b. OPTION PREMIUM S/R  (computed every rerun from index S/R)
#      Scales index support/resistance to option premium levels
#      using delta. Works immediately — no candle data needed.
#      Falls back to simple ±% bands when no candle data yet.
# ==============================================================
# ==============================================================
# OPTION S/R — computed directly from option premium
#
# Three sources merged per option:
#   1. Round premium levels  (e.g. 180, 185, 190 for CE at 186)
#   2. IV-based daily move bands  (±1 sigma intraday range)
#   3. Pivot levels from option OHLC if candle data available
# This is completely independent of the index price.
# ==============================================================

def option_snr(ltp, iv_pct, dte_days, candle_df_arg=None):
    """
    Compute support and resistance directly for an option premium.

    ltp        : current option LTP (Rs.)
    iv_pct     : implied volatility as integer percent (e.g. 15)
    dte_days   : days to expiry
    candle_df_arg: index candle df for pivot extraction (optional)

    Returns {"support": [...], "resistance": [...]}
    """
    if not ltp or ltp <= 0:
        return {"support": [], "resistance": []}

    levels_sup = []
    levels_res = []
    iv = iv_pct / 100.0 if iv_pct else 0.15

    # ── 1. Round number levels ──────────────────────────────
    # Options traders watch round numbers (50, 100, 150, 200...)
    # Choose step based on premium magnitude
    if ltp >= 500:   step = 50
    elif ltp >= 200: step = 25
    elif ltp >= 100: step = 10
    elif ltp >= 50:  step = 5
    else:            step = 2

    base = round(ltp / step) * step
    for i in range(1, 5):
        s = round(base - i * step, 1)
        r = round(base + i * step, 1)
        if 0 < s < ltp:  levels_sup.append(s)
        if r > ltp:      levels_res.append(r)

    # ── 2. IV-based intraday move bands ────────────────────
    # Daily 1-sigma move of the option premium ≈ ltp * iv / sqrt(252)
    # Intraday (half-day) ≈ divide by sqrt(2)
    import math
    daily_sigma = ltp * iv / math.sqrt(252)
    intra_sigma = daily_sigma / math.sqrt(2)

    for mult in [0.5, 1.0, 1.5, 2.0]:
        s = round(ltp - mult * intra_sigma, 1)
        r = round(ltp + mult * intra_sigma, 1)
        if 0 < s < ltp:  levels_sup.append(s)
        if r > ltp:      levels_res.append(r)

    # ── 3. Index candle pivots scaled by delta (if available) ──
    if candle_df_arg is not None and len(candle_df_arg) >= 6:
        highs  = candle_df_arg["high"].values
        lows   = candle_df_arg["low"].values
        closes = candle_df_arg["close"].values
        vols   = candle_df_arg["volume"].values if "volume" in candle_df_arg.columns else None

        idx_cur = closes[-1]
        # Rough delta estimate: option_ltp / spot (simplified)
        rough_delta = min(max(ltp / (idx_cur * 0.05), 0.1), 0.9)

        for i in range(2, len(candle_df_arg) - 2):
            # Swing high → resistance
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                idx_dist = highs[i] - idx_cur
                opt_lvl  = round(ltp + idx_dist * rough_delta, 1)
                if opt_lvl > ltp * 1.005:
                    levels_res.append(opt_lvl)
            # Swing low → support
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                idx_dist = lows[i] - idx_cur
                opt_lvl  = round(ltp + idx_dist * rough_delta, 1)
                if 0 < opt_lvl < ltp * 0.995:
                    levels_sup.append(opt_lvl)

    # ── Deduplicate and sort ─────────────────────────────────
    def clean(lst, above):
        lst = [x for x in lst if x > 0]
        lst = sorted(set(round(x, 1) for x in lst),
                     reverse=not above)
        result = []
        for v in lst:
            if not result:
                result.append(v)
            elif above and (v - result[-1]) / result[-1] > 0.008:
                result.append(v)
            elif not above and (result[-1] - v) / result[-1] > 0.008:
                result.append(v)
        return result[:3]

    return {
        "support":    clean(levels_sup, above=False),
        "resistance": clean(levels_res, above=True),
        "current":    ltp,
    }

# Compute S/R directly per option
ce_snr = {"support": [], "resistance": [], "current": ce_ltp or 0}
pe_snr = {"support": [], "resistance": [], "current": pe_ltp or 0}

if ce_ltp and ce_ltp > 0:
    ce_snr = option_snr(ce_ltp, iv_pct, dte_days, candle_df)
elif spot and ce_strike:
    # Estimate ATM option price via BS when chain not yet loaded
    _ce_est = bs_greeks(spot, ce_strike, dte, r, iv, "call")
    # Rough premium estimate: intrinsic + time value
    _ce_ltp_est = max(round(spot * iv * (dte_days / 365) ** 0.5 * 0.4, 1), 5.0)
    ce_snr = option_snr(_ce_ltp_est, iv_pct, dte_days, candle_df)

if pe_ltp and pe_ltp > 0:
    pe_snr = option_snr(pe_ltp, iv_pct, dte_days, candle_df)
elif spot and pe_strike:
    _pe_ltp_est = max(round(spot * iv * (dte_days / 365) ** 0.5 * 0.4, 1), 5.0)
    pe_snr = option_snr(_pe_ltp_est, iv_pct, dte_days, candle_df)


# Compute OI momentum using current and previous chain
_prev_chain  = st.session_state.prev_chain or {}
_prev_prices = st.session_state.last_prices
_prev_spot   = _prev_prices.get(conf["response_key"], {}).get("close") if _prev_prices else None
_oi_momentum = compute_oi_momentum(chain, _prev_chain, spot, _prev_spot)

# ==============================================================
# 17. GENERATE SIGNAL
# ==============================================================
# Incorporate OI momentum into signal score
_oi_score_bonus = 0
if _oi_momentum:
    if _oi_momentum["decision"] == "BUY CE":
        _oi_score_bonus = 2
    elif _oi_momentum["decision"] == "BUY PE":
        _oi_score_bonus = -2
    elif "Exit CE" in _oi_momentum["decision"]:
        _oi_score_bonus = -1
    elif "Exit PE" in _oi_momentum["decision"]:
        _oi_score_bonus = 1

signal = generate_signal(
    indicators, spot, st.session_state.last_vix, _high_vol,
    change_pct=change_pct,
    ind_15=ind_15,
    ind_30=ind_30,
    oi_score_bonus=_oi_score_bonus,
)


# ==============================================================
# TRADE SAVE LOGIC
# Saves the best option when a strong signal fires.
# Holds it for the selected candle_interval in minutes.
# Auto-expires after the interval, marks as completed.
# ==============================================================
_interval_mins = int(candle_interval)
_now           = datetime.now()

# Check if active trade has expired
if st.session_state.active_trade and st.session_state.trade_saved_at:
    _elapsed = (_now - st.session_state.trade_saved_at).total_seconds() / 60
    if _elapsed >= _interval_mins:
        # Move to trade log with exit info
        _trade = dict(st.session_state.active_trade)
        _trade["exit_time"]  = _now.strftime("%H:%M:%S")
        _trade["exit_price"] = (ce_ltp if _trade["type"] == "CE" else pe_ltp) or _trade["entry_premium"]
        _trade["pnl"]        = round(
            (_trade["exit_price"] - _trade["entry_premium"])
            * (_trade["lots"] * _trade["lot_size"])
            * (1 if _trade["type"] == "CE" else 1),
            2
        )
        _trade["status"]     = "Completed"
        st.session_state.trade_log.insert(0, _trade)
        st.session_state.active_trade  = None
        st.session_state.trade_saved_at = None
        # Keep last 20 trades only
        st.session_state.trade_log = st.session_state.trade_log[:20]

# Auto-save new trade when signal is strong and no active trade
if (signal and spot and atm
        and signal["direction"] in ("CE", "PE")
        and signal["confidence"] in ("High", "Medium")
        and not st.session_state.active_trade
        and run_live):
    _opt_type  = signal["direction"]
    _opt_strike = ce_strike if _opt_type == "CE" else pe_strike
    _opt_ltp   = ce_ltp    if _opt_type == "CE" else pe_ltp
    _opt_delta = g_ce["delta"] if (_opt_type == "CE" and g_ce) else (g_pe["delta"] if g_pe else "--")
    _opt_iv    = g_ce["iv"]    if (_opt_type == "CE" and g_ce) else (g_pe["iv"]    if g_pe else iv_pct)
    if _opt_ltp and _opt_ltp > 0:
        st.session_state.active_trade = {
            "index":         selected_index,
            "type":          _opt_type,
            "strike":        _opt_strike,
            "entry_premium": _opt_ltp,
            "entry_spot":    spot,
            "target_spot":   signal["target"],
            "sl_spot":       signal["stop_loss"],
            "delta":         _opt_delta,
            "iv":            _opt_iv,
            "confidence":    signal["confidence"],
            "score":         signal["score"],
            "scenario":      _oi_momentum["scenario"] if _oi_momentum else "--",
            "lots":          lots,
            "lot_size":      conf["lot_size"],
            "interval_mins": _interval_mins,
            "entry_time":    _now.strftime("%H:%M:%S"),
            "expiry_at":     (_now + pd.Timedelta(minutes=_interval_mins)).strftime("%H:%M:%S"),
            "status":        "Active",
        }
        st.session_state.trade_saved_at = _now

# ==============================================================
# ==============================================================
# 18. PAGE HEADER + MARKET OUTLOOK
# ==============================================================

# Compute outlook
_signal_score = signal["score"] if signal else None
_outlook = compute_market_outlook(
    vix         = st.session_state.last_vix,
    rsi         = indicators.get("rsi") if indicators else None,
    signal_score= _signal_score,
    change_pct  = change_pct,
)

sent = _outlook["sentiment"]
col  = _outlook["color"]

# ADX badge
_adx_val  = indicators.get("adx")       if indicators else None
_adx_lbl  = indicators.get("adx_label") if indicators else None
_adx_dir  = indicators.get("adx_dir")   if indicators else None
_plus_di  = indicators.get("plus_di")   if indicators else None
_minus_di = indicators.get("minus_di")  if indicators else None

if _adx_val is not None:
    if _adx_val >= 40:
        adx_bg, adx_col, adx_icon = "#3d1a00", "#ff6d00", "🔥"
    elif _adx_val >= 25:
        adx_bg  = "#0d3320" if _adx_dir == "Bullish" else "#3d0a0a"
        adx_col = "#00c853" if _adx_dir == "Bullish" else "#f44336"
        adx_icon = "📈" if _adx_dir == "Bullish" else "📉"
    elif _adx_val >= 20:
        adx_bg, adx_col, adx_icon = "#1a1a2e", "#90caf9", "〰️"
    else:
        adx_bg, adx_col, adx_icon = "#2a2a1a", "#ffc107", "↔️"
    di_str    = f"+DI {_plus_di:.0f} / -DI {_minus_di:.0f}" if _plus_di and _minus_di else ""
    adx_badge = (
        f'<span style="background:{adx_bg};color:{adx_col};'
        f'border:1.5px solid {adx_col};border-radius:6px;'
        f'padding:3px 10px;font-size:12px;font-weight:700;white-space:nowrap;">'
        f'{adx_icon} ADX {_adx_val:.0f} — {_adx_lbl} '
        f'<span style="font-size:10px;opacity:0.75;">{di_str}</span>'
        f'</span>'
    )
else:
    adx_badge = ''

# ── Main header layout: 3 columns ──────────────────────────
# Col 1: Index title + Live status (small)
# Col 2: Spot Price (large, prominent)
# Col 3: Market Sentiment badge + ADX
# ──────────────────────────────────────────────────────────
hc1, hc2, hc3 = st.columns([2, 2, 3])

with hc1:
    # Index name + feed status
    _status_icon = "🟢" if (run_live and spot) else ("🟡" if run_live else "⚪")
    _status_txt  = "Live" if (run_live and spot) else ("Connecting..." if run_live else "Paused")
    st.markdown(
        f'<div style="padding-top:4px;">'
        f'<div style="font-size:22px;font-weight:700;color:var(--color-text-primary);">'
        f'{selected_index}</div>'
        f'<div style="font-size:12px;color:#888;margin-top:2px;">'
        f'{_status_icon} {_status_txt} &nbsp;|&nbsp; {greeks_source} Greeks</div>'
        f'<div style="font-size:11px;color:#888;margin-top:1px;">'
        f'Expiry: {expiry_date.strftime("%d %b")} ({dte_days}d)'
        f'{"  |  ATM: " + str(f"{atm:,}") if atm else ""}'
        f'</div></div>',
        unsafe_allow_html=True
    )

with hc2:
    # Spot price — most prominent element
    if spot:
        _close_px = feed_entry.get("close", spot) if feed_entry else spot
        _pts_diff = feed_entry.get("pts_diff") if feed_entry else None
        _chg_str, _chg_col = fmt_change(spot, _close_px, change_pct, st.session_state.show_points, _pts_diff)
        _age_str  = (f"{'< 1s' if data_age < 1 else f'{data_age:.0f}s'} ago"
                     if data_age is not None else "")
        _mode_lbl = "pts" if st.session_state.show_points else "%"
        st.markdown(
            f'<div style="padding-top:2px;">'
            f'<div style="font-size:11px;color:#888;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:2px;color:#000000;">Spot Price</div>'
            f'<div style="font-size:28px;font-weight:700;color:#000000;line-height:1.1;">'
            f'Rs.{spot:,.2f}</div>'
            f'<div style="font-size:13px;margin-top:2px;">'
            f'<span style="color:{_chg_col};font-weight:600;">{_chg_str}</span>'
            f'<span style="color:#888;margin-left:8px;font-size:11px;">{_age_str}</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="padding-top:2px;">'
            '<div style="font-size:11px;color:#888;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:2px;">Spot Price</div>'
            '<div style="font-size:28px;font-weight:700;color:#555;">—</div>'
            '<div style="font-size:11px;color:#888;margin-top:2px;">Enable Live Feed</div>'
            '</div>',
            unsafe_allow_html=True
        )

with hc3:
    # Sentiment badge + ADX + VIX
    vix_val = st.session_state.last_vix
    vix_str = f"VIX: {vix_val:.2f}%" if vix_val else ""
    st.markdown(
        f'<div style="padding-top:4px;">'
        f'<div style="margin-bottom:6px;">'
        f'<span style="background:{_outlook["bg"]};color:{col};'
        f'border:1.5px solid {col};border-radius:6px;'
        f'padding:5px 16px;font-size:16px;font-weight:700;letter-spacing:0.05em;">'
        f'{_outlook["emoji"]} {sent}</span>'
        f'</div>'
        f'<div style="margin-bottom:4px;">{adx_badge}</div>'
        f'<div style="font-size:11px;color:#888;">{vix_str}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.divider()


# ── Market Outlook Summary panel ─────────────────────────────
with st.expander("📊 Market Outlook Summary", expanded=True):
    oc1, oc2 = st.columns([2, 1])
    with oc1:
        # Metric table
        color_map = {
            "bull":    ("#00c853", "#0d3320"),
            "bear":    ("#f44336", "#3d0a0a"),
            "warn":    ("#ffc107", "#2a2200"),
            "neutral": ("#90caf9", "#0d1f33"),
        }
        rows_html = ""
        for metric, value, signal_txt, kind in _outlook["metrics"]:
            fg, bg = color_map.get(kind, ("#ccc", "#1a1a1a"))
            rows_html += (
                f'<tr>'
                f'<td style="padding:8px 12px;color:#333333;font-weight:500;border-bottom:1px solid #ddd;">{metric}</td>'
                f'<td style="padding:8px 12px;color:#000000;font-weight:700;border-bottom:1px solid #333;">{value}</td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #333;">'
                f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:4px;font-size:13px;">'
                f'{signal_txt}</span></td>'
                f'</tr>'
            )
        table_html = (
            f'<table style="width:100%;border-collapse:collapse;font-size:14px;background:white;border-radius:8px;">'
            f'<thead><tr>'
            f'<th style="padding:8px 12px;text-align:left;color:#000;font-weight:700;border-bottom:2px solid #333;">Metric</th>'
            f'<th style="padding:8px 12px;text-align:left;color:#000;font-weight:700;border-bottom:2px solid #333;">Current Value</th>'
            f'<th style="padding:8px 12px;text-align:left;color:#000;font-weight:700;border-bottom:2px solid #333;">Sentiment Signal</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

    with oc2:
        # Overall verdict card
        bull = _outlook["bull_points"]
        bear = _outlook["bear_points"]
        total_pts = bull + bear or 1
        bull_pct = int(bull / total_pts * 100)
        bear_pct = 100 - bull_pct
        col  = _outlook["color"]
        bg   = _outlook["bg"]
        st.markdown(
            f'<div style="background:{bg};border:1.5px solid {col};border-radius:10px;'
            f'padding:16px;text-align:center;">'
            f'<div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:0.1em;">Overall Sentiment</div>'
            f'<div style="font-size:32px;margin:6px 0;">{_outlook["emoji"]}</div>'
            f'<div style="font-size:22px;font-weight:700;color:{col};">{_outlook["sentiment"]}</div>'
            f'<div style="margin-top:12px;font-size:12px;color:#888;">Bull signals: <b style="color:#00c853;">{bull}</b>'
            f'&nbsp;&nbsp;Bear signals: <b style="color:#f44336;">{bear}</b></div>'
            f'<div style="margin-top:10px;background:#111;border-radius:4px;height:8px;overflow:hidden;">'
            f'<div style="width:{bull_pct}%;height:100%;background:#00c853;float:left;"></div>'
            f'<div style="width:{bear_pct}%;height:100%;background:#f44336;float:left;"></div>'
            f'</div>'
            f'<div style="margin-top:4px;font-size:11px;color:#888;">{bull_pct}% bull &nbsp;/&nbsp; {bear_pct}% bear</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.divider()

# ==============================================================
# ACTIVE TRADE CARD + TRADE LOG
# ==============================================================

# ── Active trade sticky card ─────────────────────────────────
_at = st.session_state.active_trade
if _at:
    _elapsed_mins = round(
        (_now - st.session_state.trade_saved_at).total_seconds() / 60, 1
    ) if st.session_state.trade_saved_at else 0
    _remaining    = max(0, _at["interval_mins"] - _elapsed_mins)
    _progress_pct = min(int(_elapsed_mins / _at["interval_mins"] * 100), 100)
    _progress_bar = "█" * (_progress_pct // 10) + "░" * (10 - _progress_pct // 10)
    _cur_premium  = (ce_ltp if _at["type"] == "CE" else pe_ltp) or _at["entry_premium"]
    _cur_pnl      = round((_cur_premium - _at["entry_premium"]) * _at["lots"] * _at["lot_size"], 2)
    _pnl_col      = "#00c853" if _cur_pnl >= 0 else "#f44336"
    _type_col     = "#00c853" if _at["type"] == "CE" else "#f44336"
    _type_bg      = "#0d3320" if _at["type"] == "CE" else "#3d0a0a"

    st.markdown(
        f'<div style="background:{_type_bg};border:2px solid {_type_col};'        f'border-radius:12px;padding:16px 20px;margin-bottom:12px;">'        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">'        f'<div>'        f'<span style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:0.08em;">🔴 Active Trade</span>'        f'<div style="font-size:22px;font-weight:700;color:{_type_col};margin-top:2px;">'        f'{_at["index"]} {_at["strike"]:,} {_at["type"]} &nbsp;'        f'<span style="font-size:13px;background:{_type_bg};border:1px solid {_type_col};'        f'padding:2px 8px;border-radius:4px;">{_at["confidence"]}</span>'        f'</div></div>'        f'<div style="text-align:right;">'        f'<div style="font-size:11px;color:#aaa;">Entry: {_at["entry_time"]} &nbsp;|&nbsp; Expires: {_at["expiry_at"]}</div>'        f'<div style="font-size:11px;color:#aaa;margin-top:2px;">{_at["interval_mins"]}m interval &nbsp;|&nbsp; {_at["lots"]} lot</div>'        f'</div></div>'        f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:10px;">'        f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:8px 10px;">'        f'<div style="font-size:10px;color:#aaa;">Entry Premium</div>'        f'<div style="font-size:16px;font-weight:600;color:white;">Rs.{_at["entry_premium"]:.1f}</div></div>'        f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:8px 10px;">'        f'<div style="font-size:10px;color:#aaa;">Current Premium</div>'        f'<div style="font-size:16px;font-weight:600;color:white;">Rs.{_cur_premium:.1f}</div></div>'        f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:8px 10px;">'        f'<div style="font-size:10px;color:#aaa;">Unrealised P&L</div>'        f'<div style="font-size:16px;font-weight:600;color:{_pnl_col};">'        f'{"+" if _cur_pnl >= 0 else ""}Rs.{_cur_pnl:.0f}</div></div>'        f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:8px 10px;">'        f'<div style="font-size:10px;color:#aaa;">Target Spot</div>'        f'<div style="font-size:16px;font-weight:600;color:#00c853;">{_at["target_spot"] or "--"}</div></div>'        f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:8px 10px;">'        f'<div style="font-size:10px;color:#aaa;">Stop-Loss</div>'        f'<div style="font-size:16px;font-weight:600;color:#f44336;">{_at["sl_spot"] or "--"}</div></div>'        f'<div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:8px 10px;">'        f'<div style="font-size:10px;color:#aaa;">OI Scenario</div>'        f'<div style="font-size:13px;font-weight:600;color:white;">{_at["scenario"]}</div></div>'        f'</div>'        f'<div style="display:flex;align-items:center;gap:10px;">'        f'<span style="font-size:11px;color:#aaa;">Time elapsed:</span>'        f'<span style="font-family:monospace;color:{_type_col};">{_progress_bar}</span>'        f'<span style="font-size:11px;color:#aaa;">{_elapsed_mins:.1f}m / {_at["interval_mins"]}m'        f'&nbsp; ({_remaining:.1f}m remaining)</span>'        f'</div></div>',
        unsafe_allow_html=True
    )

    # Manual exit button
    col_exit, col_void = st.columns([1, 4])
    with col_exit:
        if st.button("🚪 Exit Trade Now", type="primary"):
            _trade = dict(_at)
            _trade["exit_time"]  = _now.strftime("%H:%M:%S")
            _trade["exit_price"] = _cur_premium
            _trade["pnl"]        = _cur_pnl
            _trade["status"]     = "Exited"
            st.session_state.trade_log.insert(0, _trade)
            st.session_state.active_trade   = None
            st.session_state.trade_saved_at = None
            st.session_state.trade_log = st.session_state.trade_log[:20]
            st.rerun()

    st.divider()

# ── Trade Log ────────────────────────────────────────────────
if st.session_state.trade_log:
    with st.expander(f"📋 Trade Log ({len(st.session_state.trade_log)} trades)", expanded=False):
        log_rows = []
        for t in st.session_state.trade_log:
            pnl_str = f'{"+" if t.get("pnl",0) >= 0 else ""}Rs.{t.get("pnl",0):.0f}'
            log_rows.append({
                "Time":     t.get("entry_time", "--"),
                "Index":    t.get("index", "--"),
                "Option":   f'{t.get("strike","")} {t.get("type","")}',
                "Entry":    f'Rs.{t.get("entry_premium",0):.1f}',
                "Exit":     f'Rs.{t.get("exit_price",0):.1f}',
                "P&L":      pnl_str,
                "Status":   t.get("status", "--"),
                "Scenario": t.get("scenario", "--"),
                "Interval": f'{t.get("interval_mins","?")}m',
            })
        df_log = pd.DataFrame(log_rows)

        def color_pnl(val):
            if val.startswith("+"):  return "color: #00c853; font-weight: 600"
            if val.startswith("-"):  return "color: #f44336; font-weight: 600"
            return ""

        st.dataframe(
            df_log.style.map(color_pnl, subset=["P&L"]),
            width="stretch", hide_index=True
        )
        # Summary
        total_pnl = sum(t.get("pnl", 0) for t in st.session_state.trade_log)
        wins      = sum(1 for t in st.session_state.trade_log if t.get("pnl", 0) > 0)
        losses    = sum(1 for t in st.session_state.trade_log if t.get("pnl", 0) < 0)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total P&L",   f'{"+" if total_pnl >= 0 else ""}Rs.{total_pnl:.0f}')
        s2.metric("Trades",      len(st.session_state.trade_log))
        s3.metric("Wins",        wins)
        s4.metric("Losses",      losses)

        if st.button("🗑️ Clear Log"):
            st.session_state.trade_log = []
            st.rerun()

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
# OI MOMENTUM DECISION MATRIX  (below signal, above CE/PE panels)
# ==============================================================
if _oi_momentum and run_live:
    with st.expander("📊 OI Momentum Decision Matrix", expanded=True):
        oi = _oi_momentum

        price_arrow = "↑" if oi["price_rising"] else ("↓" if oi["price_rising"] is False else "→")
        oi_arrow    = "↑" if oi["oi_rising"]    else "↓"

        st.markdown(
            f'<div style="background:{oi["decision_bg"]};border:2px solid {oi["decision_col"]};'            f'border-radius:10px;padding:14px 18px;margin-bottom:12px;">'            f'<div style="display:flex;align-items:center;justify-content:space-between;">'            f'<div>'            f'<div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:0.08em;">OI Momentum Signal</div>'            f'<div style="font-size:22px;font-weight:700;color:{oi["decision_col"]};margin-top:4px;">'            f'{oi["emoji"]} {oi["decision"]}</div>'            f'<div style="font-size:13px;color:#ccc;margin-top:4px;">'            f'Scenario: <b>{oi["scenario"]}</b> &nbsp;|&nbsp; '            f'Price {price_arrow} + OI {oi_arrow} &nbsp;|&nbsp; '            f'Strength: <b style="color:{oi["decision_col"]};">{oi["strength"]}</b>'            f'</div></div>'            f'<div style="text-align:right;font-size:13px;color:#aaa;">'            f'CE OI: <b style="color:#00c853;">{oi["ce_oi_now"]:,}</b><br>'            f'PE OI: <b style="color:#f44336;">{oi["pe_oi_now"]:,}</b><br>'            f'PCR: <b style="color:white;">{oi["pcr"] or "--"}</b>'            f'{"↑" if oi["pcr_rising"] else ("↓" if oi["pcr_rising"] is False else "")}'            f'</div></div></div>',
            unsafe_allow_html=True
        )

        mc1, mc2 = st.columns([3, 2])
        with mc1:
            st.markdown("**Decision Matrix** (active row highlighted)")
            matrix_rows = [
                ("↑ Rising",  "↑ Increasing", "Long Buildup",   "Buy CE",           "#00c853", "#0d3320"),
                ("↓ Falling", "↑ Increasing", "Short Buildup",  "Buy PE",           "#f44336", "#3d0a0a"),
                ("↑ Rising",  "↓ Decreasing", "Short Covering", "Caution — Exit PE","#ffc107", "#2a2200"),
                ("↓ Falling", "↓ Decreasing", "Long Unwinding", "Caution — Exit CE","#ffc107", "#2a2200"),
            ]
            tbl = ('<table style="width:100%;border-collapse:collapse;font-size:13px;">'                   '<thead><tr>'                   '<th style="padding:6px 10px;color:#888;border-bottom:1px solid #333;text-align:left;">Price</th>'                   '<th style="padding:6px 10px;color:#888;border-bottom:1px solid #333;text-align:left;">OI</th>'                   '<th style="padding:6px 10px;color:#888;border-bottom:1px solid #333;text-align:left;">Interpretation</th>'                   '<th style="padding:6px 10px;color:#888;border-bottom:1px solid #333;text-align:left;">Recommendation</th>'                   '</tr></thead><tbody>')
            for price_act, oi_mom, interp, rec_txt, fg, bg in matrix_rows:
                is_cur  = oi["scenario"].replace(" ", "").lower() in interp.replace(" ", "").lower()
                row_bg  = f"background:{bg};" if is_cur else ""
                border  = f"border-left:3px solid {fg};" if is_cur else "border-left:3px solid transparent;"
                tbl += (f'<tr style="{row_bg}{border}">'                        f'<td style="padding:7px 10px;color:#ccc;">{price_act}</td>'                        f'<td style="padding:7px 10px;color:#ccc;">{oi_mom}</td>'                        f'<td style="padding:7px 10px;color:#ccc;">{interp}</td>'                        f'<td style="padding:7px 10px;">'                        f'<span style="background:{bg};color:{fg};padding:2px 8px;'                        f'border-radius:4px;font-weight:600;font-size:12px;">{rec_txt}</span>'                        f'</td></tr>')
            tbl += "</tbody></table>"
            st.markdown(tbl, unsafe_allow_html=True)

        with mc2:
            st.markdown("**OI Walls (Max OI Strikes)**")
            if oi["resistance_wall"] and oi["max_ce_oi"] > 0:
                dist_r = round(oi["resistance_wall"] - (spot or 0), 0)
                sign_r = f"+{dist_r:.0f}" if dist_r >= 0 else f"{dist_r:.0f}"
                st.markdown(
                    f'<div style="background:#3d0a0a;border:1px solid #f44336;border-radius:6px;'                    f'padding:10px 14px;margin-bottom:8px;">'                    f'<div style="font-size:11px;color:#f44336;text-transform:uppercase;">🔴 Resistance Wall</div>'                    f'<div style="font-size:20px;font-weight:700;color:white;">{oi["resistance_wall"]:,}'                    f'<span style="font-size:12px;color:#f44336;margin-left:8px;">({sign_r} pts)</span></div>'                    f'<div style="font-size:12px;color:#aaa;">CE OI: {oi["max_ce_oi"]:,}</div></div>',
                    unsafe_allow_html=True)
            if oi["support_wall"] and oi["max_pe_oi"] > 0:
                dist_s = round(oi["support_wall"] - (spot or 0), 0)
                sign_s = f"+{dist_s:.0f}" if dist_s >= 0 else f"{dist_s:.0f}"
                st.markdown(
                    f'<div style="background:#0d3320;border:1px solid #00c853;border-radius:6px;'                    f'padding:10px 14px;margin-bottom:8px;">'                    f'<div style="font-size:11px;color:#00c853;text-transform:uppercase;">🟢 Support Wall</div>'                    f'<div style="font-size:20px;font-weight:700;color:white;">{oi["support_wall"]:,}'                    f'<span style="font-size:12px;color:#00c853;margin-left:8px;">({sign_s} pts)</span></div>'                    f'<div style="font-size:12px;color:#aaa;">PE OI: {oi["max_pe_oi"]:,}</div></div>',
                    unsafe_allow_html=True)
            if oi["pcr"]:
                pcr_col   = "#00c853" if oi["pcr"] > 1.0 else ("#f44336" if oi["pcr"] < 0.8 else "#ffc107")
                pcr_trend = "↑ Rising (Bullish)" if oi["pcr_rising"] else ("↓ Falling (Bearish)" if oi["pcr_rising"] is False else "Stable")
                st.markdown(
                    f'<div style="background:#1a1a2e;border:1px solid #444;border-radius:6px;'                    f'padding:10px 14px;margin-bottom:8px;">'                    f'<div style="font-size:11px;color:#aaa;text-transform:uppercase;">PCR</div>'                    f'<div style="font-size:20px;font-weight:700;color:{pcr_col};">{oi["pcr"]}</div>'                    f'<div style="font-size:12px;color:#aaa;">{pcr_trend}</div></div>',
                    unsafe_allow_html=True)
            if oi["liquid_strikes"]:
                st.markdown(
                    f'<div style="font-size:12px;color:#aaa;padding:8px;">'                    f'<b>Liquid strikes:</b> {", ".join(str(s) for s in sorted(oi["liquid_strikes"]))}'                    f'</div>', unsafe_allow_html=True)

    st.divider()

# ==============================================================
# 20. RECOMMENDED CE / PE PANELS  (directly below signal card)
# ==============================================================

def build_snr_html(snr, ltp, label_col, tf_label):
    """Build S/R pills HTML string. Always returns something visible."""
    if not ltp or ltp <= 0:
        return ""

    supports    = list(snr.get("support", []))
    resistances = list(snr.get("resistance", []))

    # Guaranteed fallback — always show levels
    if not supports:
        supports = [round(ltp * 0.97, 1), round(ltp * 0.94, 1), round(ltp * 0.91, 1)]
    if not resistances:
        resistances = [round(ltp * 1.03, 1), round(ltp * 1.06, 1), round(ltp * 1.09, 1)]

    def pill(price, kind):
        color = "#00c853" if kind == "S" else "#f44336"
        bg    = "#0d3320" if kind == "S" else "#3d0a0a"
        dist  = round(price - ltp, 1)
        sign  = "+" if dist >= 0 else ""
        return (f'<span style="display:inline-block;background:{bg};color:{color};'
                f'border:1px solid {color};border-radius:4px;padding:3px 8px;'
                f'font-size:12px;font-weight:600;margin:2px;">'
                f'{kind} {price:.1f}'
                f'<span style="font-size:10px;opacity:0.8;margin-left:3px;">({sign}{dist:.1f})</span>'
                f'</span>')

    ltp_pill = (f'<span style="display:inline-block;background:#0d0d2e;color:{label_col};'
                f'border:2px solid {label_col};border-radius:4px;padding:3px 10px;'
                f'font-size:12px;font-weight:700;margin:2px;">LTP {ltp:.1f}</span>')

    html = (f'<div style="margin:8px 0;padding:8px;background:rgba(255,255,255,0.03);'
            f'border-radius:6px;border-left:3px solid {label_col};">'
            f'<div style="font-size:11px;color:#888;margin-bottom:6px;'
            f'text-transform:uppercase;letter-spacing:0.08em;">'
            f'S/R Levels ({tf_label})</div>'
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:2px;">')

    for p in sorted(supports, reverse=True)[:3]:
        html += pill(p, "S")
    html += ltp_pill
    for p in sorted(resistances)[:3]:
        html += pill(p, "R")

    html += "</div></div>"
    return html

if spot and atm:
    col_ce, col_pe = st.columns(2)

    with col_ce:
        label = "✅ RECOMMENDED" if (signal and signal["direction"] == "CE") else ""
        st.success(f"### {ce_strike:,} CE — Call  {label}")
        p1, p2, p3 = st.columns(3)
        p1.metric("Premium",      f"Rs.{ce_ltp:,.2f}" if ce_ltp else "--")
        p2.metric("Buyer Margin", fmt_inr(ce_ltp * conf["lot_size"] * lots) if ce_ltp else "--")
        p3.metric("OI",           f"{ce_data.get('ce_oi', 0):,}" if ce_data else "--")
        if ce_ltp:
            ce_html = build_snr_html(ce_snr, ce_ltp, "#00e676", candle_interval + "m")
            st.markdown(ce_html, unsafe_allow_html=True)
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
        if pe_ltp:
            pe_html = build_snr_html(pe_snr, pe_ltp, "#ff5252", candle_interval + "m")
            st.markdown(pe_html, unsafe_allow_html=True)
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


# ==============================================================
# 23. MULTI-TIMEFRAME INDICATOR PANEL
# ==============================================================
def _ind_row(label, ind, tf, candles_n=None):
    """Render one timeframe indicator row."""
    if not ind:
        st.caption(f"{tf} — waiting for data")
        return
    close = ind["close"]
    ema9  = ind["ema9"]
    ema21 = ind["ema21"]
    vwap  = ind["vwap"]
    rsi   = ind["rsi"]

    trend   = "🟢 Bull" if ema9 > ema21 else "🔴 Bear"
    vs_vwap = "↑ above" if close > vwap else "↓ below"
    rsi_lbl = "OB" if rsi > 70 else ("OS" if rsi < 30 else "OK")

    c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.5, 1.5, 1.5, 1.2, 1.2])
    c1.markdown(f"**{tf}**")
    c2.metric("EMA 9",  f"{ema9:,.1f}",  "▲" if ema9 > close else "▼")
    c3.metric("EMA 21", f"{ema21:,.1f}", trend)
    c4.metric("VWAP",   f"{vwap:,.1f}",  vs_vwap)
    c5.metric("RSI",    f"{rsi}",        rsi_lbl)
    if candles_n:
        c6.metric("Candles", candles_n)

if indicators or ind_15 or ind_30:
    st.markdown("### Technical Indicators — Multi-Timeframe")

    # Confluence score across timeframes
    def _tf_score(ind):
        if not ind: return 0
        s = 0
        if ind["ema9"] > ind["ema21"]: s += 1
        if ind["close"] > ind["vwap"]: s += 1
        if ind["close"] > ind["ema9"]: s += 1
        if 50 < ind["rsi"] < 75:       s += 1
        return s

    s_primary = _tf_score(indicators)
    s_15      = _tf_score(ind_15)
    s_30      = _tf_score(ind_30)
    total     = s_primary + s_15 + s_30
    max_total = 12

    conf_pct  = int(total / max_total * 100)
    conf_bar  = "█" * int(conf_pct / 10) + "░" * (10 - int(conf_pct / 10))
    conf_color = "#00c853" if conf_pct >= 70 else ("#ffc107" if conf_pct >= 40 else "#f44336")

    st.markdown(
        f'<div style="background:#1a1a2e;border-radius:8px;padding:12px 16px;margin-bottom:12px;">'
        f'<span style="color:#aaa;font-size:12px;text-transform:uppercase;">MTF Confluence</span>&nbsp;&nbsp;'
        f'<span style="font-family:monospace;color:{conf_color};font-size:15px;">{conf_bar}</span>&nbsp;&nbsp;'
        f'<b style="color:{conf_color};">{conf_pct}%</b>'
        f'&nbsp;&nbsp;<span style="color:#888;font-size:12px;">'
        f'{candle_interval}m: {s_primary}/4 &nbsp;|&nbsp; 15m: {s_15}/4 &nbsp;|&nbsp; 30m: {s_30}/4'
        f'</span></div>',
        unsafe_allow_html=True
    )

    n_primary = len(candle_df) if candle_df is not None else None
    n_15      = len(st.session_state.candle_df_15) if st.session_state.candle_df_15 is not None else None
    n_30      = len(st.session_state.candle_df_30) if st.session_state.candle_df_30 is not None else None

    _ind_row(f"{candle_interval}m (primary)", indicators, f"{candle_interval}m", n_primary)
    st.markdown('<hr style="margin:4px 0;border-color:#333;">', unsafe_allow_html=True)
    _ind_row("15m", ind_15, "15m", n_15)
    st.markdown('<hr style="margin:4px 0;border-color:#333;">', unsafe_allow_html=True)
    _ind_row("30m", ind_30, "30m", n_30)
    st.divider()



# ==============================================================
# 23b. SUPPORT & RESISTANCE PANEL
# ==============================================================
if candle_df is not None and spot and len(candle_df) >= 6:
    with st.expander(f"📐 Support & Resistance — {selected_index} ({candle_interval}m chart)", expanded=True):
        idx_snr = compute_snr(candle_df, n_levels=5)
        supports    = idx_snr.get("support", [])
        resistances = idx_snr.get("resistance", [])
        current_px  = idx_snr.get("current", spot)

        # Build visual level map
        all_levels = (
            [{"price": p, "type": "R", "color": "#f44336", "bg": "#3d0a0a"} for p in sorted(resistances, reverse=True)] +
            [{"price": current_px, "type": "NOW", "color": "#ffffff", "bg": "#1a1a3e"}] +
            [{"price": p, "type": "S", "color": "#00c853", "bg": "#0d3320"} for p in sorted(supports, reverse=True)]
        )

        # Left column: visual price ladder
        lc, rc = st.columns([1, 1])
        with lc:
            st.markdown("**Price Ladder**")
            for lvl in all_levels:
                p    = lvl["price"]
                t    = lvl["type"]
                col  = lvl["color"]
                bg   = lvl["bg"]
                dist = round(p - current_px, 1)
                sign = f"+{dist}" if dist > 0 else str(dist)
                width = "100%" if t == "NOW" else "80%"
                border = "2px solid #fff" if t == "NOW" else f"1px solid {col}"
                label  = f"🔴 R  {p:,.1f}  ({sign})" if t == "R" else                          f"⚪ NOW  {p:,.1f}" if t == "NOW" else                          f"🟢 S  {p:,.1f}  ({sign})"
                st.markdown(
                    f'<div style="background:{bg};border:{border};border-radius:6px;'
                    f'padding:6px 12px;margin:3px 0;width:{width};font-size:13px;'
                    f'font-weight:{"700" if t=="NOW" else "500"};color:{col};">'
                    f'{label}</div>',
                    unsafe_allow_html=True
                )

        with rc:
            st.markdown("**Key Levels Summary**")
            if resistances:
                nearest_r = min(resistances)
                pts_to_r  = round(nearest_r - current_px, 1)
                st.metric("Nearest Resistance", f"{nearest_r:,.1f}", f"+{pts_to_r} pts")
            if supports:
                nearest_s = max(supports)
                pts_to_s  = round(current_px - nearest_s, 1)
                st.metric("Nearest Support",    f"{nearest_s:,.1f}", f"-{pts_to_s} pts")
            if supports and resistances:
                range_pts = round(min(resistances) - max(supports), 1)
                mid_pt    = round((min(resistances) + max(supports)) / 2, 1)
                st.metric("Trading Range",      f"{range_pts} pts", f"Mid: {mid_pt:,.1f}")

            # Position within range
            if supports and resistances:
                s_price = max(supports)
                r_price = min(resistances)
                if r_price > s_price:
                    pct = round((current_px - s_price) / (r_price - s_price) * 100, 1)
                    bar_fill = int(pct / 10)
                    bar = "█" * bar_fill + "░" * (10 - bar_fill)
                    pos_color = "#f44336" if pct > 75 else ("#00c853" if pct < 25 else "#ffc107")
                    st.markdown(
                        f'<div style="margin-top:12px;">'
                        f'<div style="font-size:12px;color:#aaa;margin-bottom:4px;">Position in range</div>'
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<span style="font-size:11px;color:#00c853;">S</span>'
                        f'<span style="font-family:monospace;color:{pos_color};font-size:14px;">{bar}</span>'
                        f'<span style="font-size:11px;color:#f44336;">R</span>'
                        f'<b style="color:{pos_color};margin-left:4px;">{pct}%</b>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

        st.divider()

        # Option premium S/R table
        if ce_ltp or pe_ltp:
            st.markdown("**Option Premium S/R Levels**")
            op1, op2 = st.columns(2)
            with op1:
                st.markdown(f"🟢 **{ce_strike} CE** (LTP: Rs.{ce_ltp:.2f})")
                if ce_snr.get("resistance"):
                    for r in sorted(ce_snr["resistance"])[:3]:
                        dist = round(r - ce_ltp, 2)
                        st.markdown(
                            f'<span style="background:#3d0a0a;color:#f44336;border-radius:4px;'
                            f'padding:2px 8px;font-size:12px;margin:2px;">R {r:.1f} (+{dist:.1f})</span>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<span style="background:#1a1a3e;color:white;border:1px solid white;'
                    f'border-radius:4px;padding:2px 8px;font-size:12px;margin:2px;">LTP {ce_ltp:.1f}</span>',
                    unsafe_allow_html=True)
                if ce_snr.get("support"):
                    for s in sorted(ce_snr["support"], reverse=True)[:3]:
                        dist = round(ce_ltp - s, 2)
                        st.markdown(
                            f'<span style="background:#0d3320;color:#00c853;border-radius:4px;'
                            f'padding:2px 8px;font-size:12px;margin:2px;">S {s:.1f} (-{dist:.1f})</span>',
                            unsafe_allow_html=True)
            with op2:
                st.markdown(f"🔴 **{pe_strike} PE** (LTP: Rs.{pe_ltp:.2f})")
                if pe_snr.get("resistance"):
                    for r in sorted(pe_snr["resistance"])[:3]:
                        dist = round(r - pe_ltp, 2)
                        st.markdown(
                            f'<span style="background:#3d0a0a;color:#f44336;border-radius:4px;'
                            f'padding:2px 8px;font-size:12px;margin:2px;">R {r:.1f} (+{dist:.1f})</span>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<span style="background:#1a1a3e;color:white;border:1px solid white;'
                    f'border-radius:4px;padding:2px 8px;font-size:12px;margin:2px;">LTP {pe_ltp:.1f}</span>',
                    unsafe_allow_html=True)
                if pe_snr.get("support"):
                    for s in sorted(pe_snr["support"], reverse=True)[:3]:
                        dist = round(pe_ltp - s, 2)
                        st.markdown(
                            f'<span style="background:#0d3320;color:#00c853;border-radius:4px;'
                            f'padding:2px 8px;font-size:12px;margin:2px;">S {s:.1f} (-{dist:.1f})</span>',
                            unsafe_allow_html=True)

# ==============================================================
# 24. OPTION CHAIN TABLE
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
# 25. ALL-INDEX OVERVIEW
# ==============================================================
if run_live and all_prices:
    st.markdown("### All Index Prices")
    _col_label = "Change (pts)" if st.session_state.show_points else "Change %"
    rows = []
    for idx_name, idx_conf in INDEX_CONFIG.items():
        entry = all_prices.get(idx_conf["response_key"])
        if entry:
            dte_i, exp_i = get_dte(idx_conf["expiry_weekday"])
            _ltp   = entry["ltp"]
            _close = entry.get("close", _ltp)
            _cpct  = entry.get("change_pct", 0)
            _pts   = entry.get("pts_diff")
            _chg, _ = fmt_change(_ltp, _close, _cpct, st.session_state.show_points, _pts)
            rows.append({
                "Index":       idx_name,
                "LTP":         f"Rs.{_ltp:,.2f}",
                _col_label:    _chg,
                "Expiry":      exp_i.strftime("%d %b"),
                "DTE":         dte_i,
                "Updated":     entry["ts"].strftime("%H:%M:%S"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

if not run_live:
    st.info("Toggle **Start Live Feed** in the sidebar to begin.")

# ==============================================================
# 26. AUTO-REFRESH
# ==============================================================
if run_live:
    time.sleep(_refresh / 1000)
    st.rerun()
