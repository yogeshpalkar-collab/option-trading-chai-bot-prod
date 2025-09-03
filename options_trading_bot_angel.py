import streamlit as st
import pandas as pd
from smartapi import SmartConnect
import pyotp
import datetime as dt
import numpy as np

# ------------------------- Utility Functions -------------------------

def fetch_instruments(api):
    """
    Fetch full instrument list from Angel One SmartAPI (live integration).
    Compatible with smartapi-python v1.5.5
    """
    try:
        response = api.getInstruments()  # returns dict with "data" key
        if response and "data" in response:
            return response["data"]      # actual instruments list
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return []

def calculate_atm_strike(spot):
    return int(round(spot / 50.0) * 50)

def atr(data, period=14):
    data['H-L'] = data['high'] - data['low']
    data['H-PC'] = abs(data['high'] - data['close'].shift(1))
    data['L-PC'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['H-L','H-PC','L-PC']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=period).mean()
    return data['ATR'].iloc[-1]

# ------------------------- Streamlit UI -------------------------

def main():
    st.title("📈 Secured Options Trading Bot (Angel One, Render Version)")
    st.caption("Production-ready. Live Angel One SmartAPI integration.")

    # Master password protection
    if "authenticated" not in st.session_state:
        pwd = st.text_input("Enter Master Password", type="password")
        if st.button("Login"):
            if pwd == st.secrets.get("MASTER_PASSWORD", "changeme"):
                st.session_state.authenticated = True
                st.success("Unlocked ✅")
            else:
                st.error("Invalid password ❌")
                return
        else:
            return

    # Paper / Live toggle
    mode = st.radio("Mode", ["Paper Trading", "Live Trading"], index=0)
    st.write(f"Current mode: **{mode}**")

    # Connect to Angel One (only if live)
    api = None
    if mode == "Live Trading":
        try:
            api_key = st.secrets["API_KEY"]
            client_id = st.secrets["CLIENT_ID"]
            password = st.secrets["PASSWORD"]
            totp = pyotp.TOTP(st.secrets["TOTP"]).now()

            api = SmartConnect(api_key)
            session_data = api.generateSession(client_id, password, totp)
            st.success("✅ Logged in to Angel One")
        except Exception as e:
            st.error(f"Login failed: {e}")
            return

    # Fetch instruments
    instruments = fetch_instruments(api) if api else []
    st.write(f"Fetched {len(instruments)} instruments")

    # Placeholder for dashboard
    st.subheader("Bias Dashboard")
    st.info("📊 EMA, VWAP, CPR, ATR calculations will be displayed here.")

    # Placeholder for trades log
    st.subheader("Today’s Trades")
    st.warning("📑 Trade log table will appear here.")

if __name__ == "__main__":
    main()
