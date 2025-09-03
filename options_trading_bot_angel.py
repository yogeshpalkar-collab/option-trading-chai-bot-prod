import streamlit as st
import pandas as pd
from SmartApi import SmartConnect
import pyotp
import datetime as dt
import os
import requests

# ------------------------- Utility Functions -------------------------

def fetch_instruments(api):
    try:
        if hasattr(api, "get_instruments"):
            response = api.get_instruments()
        elif hasattr(api, "getInstruments"):
            response = api.getInstruments()
        elif hasattr(api, "get_exchange_instruments"):
            response = api.get_exchange_instruments("NFO")
        else:
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            response = requests.get(url).json()

        if isinstance(response, dict) and "data" in response:
            all_instruments = response["data"]
        elif isinstance(response, list):
            all_instruments = response
        else:
            return []

        nifty_opts = [
            inst for inst in all_instruments
            if inst.get("name") == "NIFTY" and inst.get("instrumenttype") == "OPTIDX"
        ]
        return nifty_opts
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

def fetch_nifty_spot(api):
    try:
        spot_data = api.ltpData("NSE", "NIFTY 50", "26000")
        if spot_data and "data" in spot_data and "ltp" in spot_data["data"]:
            return float(spot_data["data"]["ltp"])
        return None
    except Exception as e:
        st.error(f"Error fetching NIFTY spot: {e}")
        return None

def calculate_atm_strike(spot):
    return int(round(spot / 50.0) * 50)

# ------------------------- Streamlit UI -------------------------

def main():
    st.set_page_config(page_title="Options Trading Bot", layout="wide")

    # --- Fast /ping endpoint for uptime checks ---
    query_params = st.experimental_get_query_params()
    if query_params.get("ping") == ["1"]:
        st.write("pong")
        return

    st.title("📈 Secured Options Trading Bot (Angel One, Render Version)")
    st.caption("Production-ready. Live Angel One SmartAPI integration.")

    # Persistent state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "api" not in st.session_state:
        st.session_state.api = None
    if "trade_log" not in st.session_state:
        st.session_state.trade_log = []
    if "cum_pnl" not in st.session_state:
        st.session_state.cum_pnl = 0
    if "expiry" not in st.session_state:
        st.session_state.expiry = None

    # Login
    if not st.session_state.authenticated:
        pwd = st.text_input("Enter Master Password", type="password")
        if st.button("Login"):
            if pwd == os.environ.get("MASTER_PASSWORD", "changeme"):
                try:
                    api = SmartConnect(os.environ["API_KEY"])
                    totp = pyotp.TOTP(os.environ["TOTP"]).now()
                    api.generateSession(os.environ["CLIENT_ID"], os.environ["PASSWORD"], totp)
                    st.session_state.authenticated = True
                    st.session_state.api = api
                    st.success("Unlocked ✅")
                except Exception as e:
                    st.error(f"Login failed: {e}")
                    return
            else:
                st.error("Invalid password ❌")
                return
        else:
            return
    else:
        api = st.session_state.api

    mode = st.radio("Mode", ["Paper Trading", "Live Trading"], index=0)
    st.write(f"Current mode: **{mode}**")

    # Summary panel
    trades_taken = len([t for t in st.session_state.trade_log if "Exit" in t["Action"]])
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

    # Instruments & expiry
    instruments = fetch_instruments(api)
    st.write(f"Fetched {len(instruments)} NIFTY option instruments")

    if instruments:
        expiries = sorted(set(inst["expiry"] for inst in instruments))
        if st.session_state.expiry not in expiries:
            st.session_state.expiry = expiries[0]

        selected_expiry = st.selectbox(
            "Select Expiry",
            expiries,
            index=expiries.index(st.session_state.expiry),
            key="expiry_select"
        )
        st.session_state.expiry = selected_expiry
        expiry_instruments = [i for i in instruments if i["expiry"] == selected_expiry]

        spot = fetch_nifty_spot(api)
        if spot:
            atm_strike = calculate_atm_strike(spot)
            st.success(f"NIFTY Spot: {spot} | ATM Strike: {atm_strike}")

            atm_instruments = [i for i in expiry_instruments if i.get("strike") == atm_strike]
            if atm_instruments:
                df_atm = pd.DataFrame(atm_instruments)[["tradingsymbol", "expiry", "strike", "instrumenttype"]]
                st.subheader("ATM CE/PE")
                st.dataframe(df_atm)

            # Bias dashboard (always visible)
            st.subheader("Bias Dashboard")
            st.info("Bias: Neutral (demo placeholder)")
            st.write("- EMA check pending")
            st.write("- VWAP check pending")
            st.write("- CPR check pending")
            st.write("- OI Change check pending")

    # Trade log (always visible)
    st.subheader("Today’s Trades (Live Updates)")
    if st.session_state.trade_log:
        df_log = pd.DataFrame(st.session_state.trade_log)
        st.dataframe(df_log)
        st.info(f"Cumulative P&L: ₹{st.session_state.cum_pnl}")
    else:
        st.info("No trades yet (waiting for setup).")

if __name__ == "__main__":
    main()
