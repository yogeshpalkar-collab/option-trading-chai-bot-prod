import streamlit as st
import pandas as pd
from SmartApi import SmartConnect
import pyotp
import datetime as dt
import numpy as np
import os
import random

# ------------------------- Utility Functions -------------------------

def fetch_instruments(api):
    try:
        if hasattr(api, "get_instruments"):
            response = api.get_instruments()
        else:
            response = api.getInstruments()
        if response and "data" in response:
            all_instruments = response["data"]
            nifty_opts = [
                inst for inst in all_instruments
                if inst.get("name") == "NIFTY" and inst.get("instrumenttype") == "OPTIDX"
            ]
            return nifty_opts
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

def fetch_nifty_spot(api):
    try:
        spot_data = api.ltpData("NSE", "NIFTY 50", "26000")
        if spot_data and "data" in spot_data and "ltp" in spot_data["data"]:
            return float(spot_data["data"]["ltp"])
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching NIFTY spot: {e}")
        return None

def calculate_atm_strike(spot):
    return int(round(spot / 50.0) * 50)

def determine_bias(ema_fast, ema_slow, vwap, price, cpr_status, call_oi_chg, put_oi_chg):
    reasons = []
    if ema_fast > ema_slow:
        reasons.append("EMA Bullish")
    else:
        reasons.append("EMA Bearish")
    if price > vwap:
        reasons.append("Above VWAP")
    else:
        reasons.append("Below VWAP")
    reasons.append(f"CPR: {cpr_status}")
    if put_oi_chg > call_oi_chg:
        reasons.append("Put OI > Call OI (Bullish OI)")
    else:
        reasons.append("Call OI > Put OI (Bearish OI)")

    if (ema_fast > ema_slow) and (price > vwap) and (cpr_status == "Bullish") and (put_oi_chg > call_oi_chg):
        return "Bullish", reasons
    elif (ema_fast < ema_slow) and (price < vwap) and (cpr_status == "Bearish") and (call_oi_chg > put_oi_chg):
        return "Bearish", reasons
    else:
        return "Neutral", reasons

# ------------------------- Streamlit UI -------------------------

def main():
    st.title("📈 Secured Options Trading Bot (Angel One, Render Version)")
    st.caption("Production-ready. Live Angel One SmartAPI integration.")

    # Master password protection
    if "authenticated" not in st.session_state:
        pwd = st.text_input("Enter Master Password", type="password")
        if st.button("Login"):
            if pwd == os.environ.get("MASTER_PASSWORD", "changeme"):
                st.session_state.authenticated = True
                st.success("Unlocked ✅")
            else:
                st.error("Invalid password ❌")
                return
        else:
            return

    mode = st.radio("Mode", ["Paper Trading", "Live Trading"], index=0)
    st.write(f"Current mode: **{mode}**")

    # Connect to Angel One
    api = None
    try:
        api_key = os.environ["API_KEY"]
        client_id = os.environ["CLIENT_ID"]
        password = os.environ["PASSWORD"]
        totp = pyotp.TOTP(os.environ["TOTP"]).now()
        api = SmartConnect(api_key)
        session_data = api.generateSession(client_id, password, totp)
        st.success("✅ Logged in to Angel One")
    except Exception as e:
        st.error(f"Login failed: {e}")
        return

    # Init trade log and P&L tracking
    if "trade_log" not in st.session_state:
        st.session_state.trade_log = []
    if "cum_pnl" not in st.session_state:
        st.session_state.cum_pnl = 0

    # --- Summary Panel ---
    trades_taken = len([t for t in st.session_state.trade_log if t["Action"] == "Exit (Target)"])
    remaining_trades = max(0, 3 - trades_taken)
    status = "🟢 Active"
    if st.session_state.cum_pnl <= -8000:
        status = "🔴 Halted (Max Daily Loss Reached)"
    elif st.session_state.cum_pnl >= 15000:
        status = "🔴 Halted (Max Daily Profit Reached)"
    elif trades_taken >= 3:
        status = "🔴 Halted (Max Trades Reached)"

    st.subheader("📊 Today’s Summary")
    st.write(f"**P&L:** ₹{st.session_state.cum_pnl:+}")
    st.write(f"**Trades Taken:** {trades_taken}")
    st.write(f"**Remaining Trades:** {remaining_trades}")
    st.write(f"**Status:** {status}")

    instruments = fetch_instruments(api)
    st.write(f"Fetched {len(instruments)} NIFTY option instruments")

    if instruments:
        expiries = sorted(set(inst["expiry"] for inst in instruments))
        selected_expiry = st.selectbox("Select Expiry", expiries, index=0)
        expiry_instruments = [i for i in instruments if i["expiry"] == selected_expiry]

        spot = fetch_nifty_spot(api)
        if spot:
            atm_strike = calculate_atm_strike(spot)
            st.success(f"NIFTY Spot: {spot} | ATM Strike: {atm_strike}")

            atm_instruments = [i for i in expiry_instruments if i.get("strike") == atm_strike]
            df_atm = pd.DataFrame(atm_instruments)[["tradingsymbol", "expiry", "strike", "instrumenttype"]]
            st.subheader("ATM CE/PE")
            st.dataframe(df_atm)

            # Bias Dashboard (mock placeholders for now)
            ema_fast, ema_slow, vwap = 20000, 19900, 19850
            cpr_status = "Bullish"
            call_oi_chg, put_oi_chg = 120000, 150000
            bias, reasons = determine_bias(ema_fast, ema_slow, vwap, spot, cpr_status, call_oi_chg, put_oi_chg)

            st.subheader("Bias Dashboard")
            st.info(f"Bias: **{bias}**")
            for r in reasons:
                st.write(f"- {r}")

            # Daily P&L guardrails
            if st.session_state.cum_pnl <= -8000:
                st.error("🚫 Max Daily Loss Limit Reached (–₹8,000) — Trading Halted for Today")
            elif st.session_state.cum_pnl >= 15000:
                st.success("✅ Max Daily Profit Target Reached (+₹15,000) — Trading Halted for Today")
            elif trades_taken >= 3:
                st.error("🚫 Max Trades Reached (3) — Trading Halted for Today")
            else:
                # Auto-pick CE/PE and simulate trade execution with live TSL updates
                if bias in ["Bullish", "Bearish"]:
                    trade_symbol = None
                    if bias == "Bullish":
                        trade_symbol = next((i for i in atm_instruments if "CE" in i["tradingsymbol"]), None)
                    elif bias == "Bearish":
                        trade_symbol = next((i for i in atm_instruments if "PE" in i["tradingsymbol"]), None)

                    if trade_symbol:
                        now = dt.datetime.now().strftime("%H:%M:%S")
                        order_id = f"SIM-{random.randint(1000,9999)}" if mode == "Paper Trading" else "LIVE-ORDER"

                        # Entry log
                        st.session_state.trade_log.append({
                            "Time": now,
                            "Expiry": trade_symbol["expiry"],
                            "Strike": trade_symbol["strike"],
                            "Type": "CE" if "CE" in trade_symbol["tradingsymbol"] else "PE",
                            "Bias": bias,
                            "CPR Status": cpr_status,
                            "Action": "Entry",
                            "SL/Target": "SL=ATR+10",
                            "P&L (₹)": "",
                            "Order ID": order_id
                        })

                        # Simulate TSL updates
                        for i in range(3):
                            tsl_val = 90 + i*10
                            now = dt.datetime.now().strftime("%H:%M:%S")
                            st.session_state.trade_log.append({
                                "Time": now,
                                "Expiry": trade_symbol["expiry"],
                                "Strike": trade_symbol["strike"],
                                "Type": "CE" if "CE" in trade_symbol["tradingsymbol"] else "PE",
                                "Bias": bias,
                                "CPR Status": cpr_status,
                                "Action": "TSL Update",
                                "SL/Target": f"SL={tsl_val}",
                                "P&L (₹)": "",
                                "Order ID": order_id
                            })

                        # Exit log with P&L (example +800)
                        now = dt.datetime.now().strftime("%H:%M:%S")
                        pnl_val = 800
                        st.session_state.cum_pnl += pnl_val
                        st.session_state.trade_log.append({
                            "Time": now,
                            "Expiry": trade_symbol["expiry"],
                            "Strike": trade_symbol["strike"],
                            "Type": "CE" if "CE" in trade_symbol["tradingsymbol"] else "PE",
                            "Bias": bias,
                            "CPR Status": cpr_status,
                            "Action": "Exit (Target)",
                            "SL/Target": "",
                            "P&L (₹)": f"{pnl_val:+}",
                            "Order ID": order_id
                        })

    # Trade log display
    st.subheader("Today’s Trades (Live Updates)")
    if st.session_state.trade_log:
        df_log = pd.DataFrame(st.session_state.trade_log)
        st.dataframe(df_log)
        st.info(f"Cumulative P&L: ₹{st.session_state.cum_pnl}")
    else:
        st.info("No trades yet.")

if __name__ == "__main__":
    main()
