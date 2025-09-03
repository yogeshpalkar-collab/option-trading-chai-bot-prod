import streamlit as st
import pandas as pd
from SmartApi import SmartConnect
import pyotp
import datetime as dt
import os

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
            raise AttributeError("SmartConnect has no instruments method available")

        # Handle both dict-with-data and direct list responses
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
    try:
        api_key = os.environ["API_KEY"]
        client_id = os.environ["CLIENT_ID"]
        password = os.environ["PASSWORD"]
        totp = pyotp.TOTP(os.environ["TOTP"]).now()
        api = SmartConnect(api_key)
        api.generateSession(client_id, password, totp)
        st.success("✅ Logged in to Angel One")
    except Exception as e:
        st.error(f"Login failed: {e}")
        return

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
            if atm_instruments:
                df_atm = pd.DataFrame(atm_instruments)[["tradingsymbol", "expiry", "strike", "instrumenttype"]]
                st.subheader("ATM CE/PE")
                st.dataframe(df_atm)

if __name__ == "__main__":
    main()
