# ==============================================================
# 10. MAIN UI LOGIC
# ==============================================================

# Placeholder for the live data display
dashboard_placeholder = st.empty()

if run_live:
    start_feed()
    
    # Simple loop to refresh UI. In a production app, consider 
    # st.fragment (if on Streamlit 1.37+) or a dedicated refresh component.
    while True:
        with dashboard_placeholder.container():
            # 1. Connection Status Header
            status = _ws_status[0]
            if "live" in status:
                st.success(f"● Live Feed: {selected_index}")
            elif "error" in status:
                st.error(f"Status: {status}")
            else:
                st.warning(f"Status: {status}...")

            # 2. Get Current Data for Selected Index
            ikey = conf["instrument_key"]
            data = _price_feed.get(ikey)

            if not data:
                st.info("Waiting for first tick...")
            else:
                ltp = data["ltp"]
                change = data["change_pct"]
                ts = data["ts"].strftime("%H:%M:%S")

                # 3. Calculate Option Strikes
                step = conf["strike_step"]
                atm_strike = round(ltp / step) * step
                
                # ITM for Call is lower strike; ITM for Put is higher strike
                ce_strike = atm_strike - (strike_offset * step)
                pe_strike = atm_strike + (strike_offset * step)

                # 4. Display Overview
                c1, c2, c3 = st.columns(3)
                c1.metric("Spot Price", f"₹{ltp:,.2f}", f"{sign_str(change)}%")
                c2.metric("Selected Strike Mode", strike_mode)
                c3.metric("Last Update", ts)

                st.divider()

                # 5. Scalper Cards (CE vs PE)
                col_ce, col_pe = st.columns(2)

                with col_ce:
                    st.subheader(f"CALL: {ce_strike}")
                    g_ce = greeks(ltp, ce_strike, dte, r/100, iv, "call")
                    
                    # Layout for Greeks
                    gc1, gc2 = st.columns(2)
                    gc1.metric("Delta", g_ce["delta"])
                    gc1.metric("Theta", g_ce["theta"])
                    gc2.metric("Gamma", g_ce["gamma"])
                    gc2.metric("Vega", g_ce["vega"])
                    
                    qty = lots * conf["lot_size"]
                    st.info(f"Exposure: {qty} units | Delta Qty: {round(qty * g_ce['delta'])}")

                with col_pe:
                    st.subheader(f"PUT: {pe_strike}")
                    g_pe = greeks(ltp, pe_strike, dte, r/100, iv, "put")
                    
                    gc1, gc2 = st.columns(2)
                    gc1.metric("Delta", g_pe["delta"])
                    gc1.metric("Theta", g_pe["theta"])
                    gc2.metric("Gamma", g_pe["gamma"])
                    gc2.metric("Vega", g_pe["vega"])
                    
                    qty = lots * conf["lot_size"]
                    st.info(f"Exposure: {qty} units | Delta Qty: {round(qty * abs(g_pe['delta']))}")

        time.sleep(0.5) # Refresh rate (500ms)
else:
    st.info("Toggle 'Start Live Feed' in the sidebar to begin.")
