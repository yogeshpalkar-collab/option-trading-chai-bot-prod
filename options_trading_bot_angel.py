import streamlit as st
import pandas as pd
from SmartApi import SmartConnect
import pyotp
import datetime as dt
import os
import pkg_resources

API_KEY = os.getenv("API_KEY")
CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP = os.getenv("TOTP")
MASTER_PASSWORD = os.getenv("MASTER_PASSWORD")

trade_count = 0
trade_engine_status = "❌ Disabled"
instrument_source = None
instrument_last_updated = None
selected_expiry = None

def get_smartapi_package_info():
    try:
        version = pkg_resources.get_distribution("smartapi-python").version
        return f"📦 SmartAPI Package: smartapi-python (v{version})"
    except Exception as e:
        return f"📦 SmartAPI Package check failed: {e}"

def login_smartapi():
    try:
        smartApi = SmartConnect(api_key=API_KEY)
        token = pyotp.TOTP(TOTP).now()
        data = smartApi.generateSession(CLIENT_ID, PASSWORD, token)
        if not data['status']:
            st.error(f"Login failed: {data['message']}")
            return None
        return smartApi
    except Exception as e:
        st.error(f"SmartAPI login error: {e}")
        return None

def refresh_instruments(smartApi):
    global instrument_source, instrument_last_updated
    today = dt.date.today().strftime("%Y%m%d")
    csv_file = f"instruments_{today}.csv"

    # Try loading today's CSV first
    if os.path.exists(csv_file):
        instrument_source = f"🟡 Instruments loaded from today's CSV ({csv_file})"
        instrument_last_updated = dt.datetime.fromtimestamp(os.path.getmtime(csv_file)).strftime("%Y-%m-%d %H:%M:%S")
        return pd.read_csv(csv_file)

    # Fetch from API (no stubs, no fake fallbacks)
    try:
        instruments = smartApi.get_instrument_master()
        df = pd.DataFrame(instruments)
        if df.empty:
            st.error("❌ Angel API returned no instruments.")
            return None
        df.to_csv(csv_file, index=False)
        instrument_source = "🟢 Instruments loaded via API (fresh)"
        instrument_last_updated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return df
    except Exception as e:
        st.error(f"Instrument fetch failed: {e}")
        return None

def get_expiry_dropdown(instruments):
    try:
        nifty_opts = instruments[(instruments["name"] == "NIFTY") & (instruments["instrumenttype"] == "OPTIDX")]
        expiries = sorted(nifty_opts["expiry"].unique())
        if not expiries:
            st.error("No NIFTY option expiries found in instruments.")
            return None

        today = dt.date.today()
        weekly_candidates = []
        for e in expiries:
            e_date = pd.to_datetime(e).date()
            if e_date.weekday() == 1 and e_date >= today:
                weekly_candidates.append(e_date)

        if weekly_candidates:
            nearest_weekly = min(weekly_candidates)
            default_index = expiries.index(str(nearest_weekly))
        else:
            default_index = 0

        selected = st.sidebar.selectbox("Select Expiry", expiries, index=default_index)
        return selected
    except Exception as e:
        st.error(f"Error building expiry dropdown: {e}")
        return None

def get_market_status():
    now = dt.datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now.weekday() >= 5:
        return "🔴 Market CLOSED (Weekend)", False
    elif market_open <= now <= market_close:
        return f"🟢 Market OPEN (NSE: {now.strftime('%Y-%m-%d %H:%M:%S')})", True
    else:
        return f"🔴 Market CLOSED (NSE: {now.strftime('%Y-%m-%d %H:%M:%S')})", False

def update_trade_engine_status(market_open):
    global trade_engine_status, trade_count
    if not market_open:
        trade_engine_status = "❌ Trade Engine DISABLED (Market Closed)"
    elif trade_count >= 3:
        trade_engine_status = "🟡 Trade Engine DISABLED (Max trades reached)"
    elif dt.datetime.now().hour >= 15:
        trade_engine_status = "🟡 Trade Engine DISABLED (After 3 PM)"
    else:
        trade_engine_status = "🟢 Trade Engine ENABLED"

def main():
    st.title("Options Trading Bot (Angel One) - Secured v3 Render Final Clean Version")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        password_input = st.text_input("Enter Master Password", type="password")
        if st.button("Unlock Bot"):
            if password_input == MASTER_PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Access granted. Bot unlocked.")
            else:
                st.error("Invalid password. Access denied.")
        st.stop()

    mode = st.sidebar.radio("Mode", ["Paper", "Live"], index=0)
    status_msg, market_open = get_market_status()
    st.info(status_msg)

    update_trade_engine_status(market_open)
    st.info(trade_engine_status)

    st.info(get_smartapi_package_info())

    smartApi = login_smartapi()
    if smartApi is None:
        st.stop()

    instruments = refresh_instruments(smartApi)
    if instruments is None:
        st.stop()

    if instrument_source:
        if instrument_last_updated:
            st.info(f"{instrument_source} | Last updated: {instrument_last_updated}")
        else:
            st.info(instrument_source)

    selected_expiry = get_expiry_dropdown(instruments)
    if not selected_expiry:
        st.stop()

    if "ENABLED" not in trade_engine_status:
        st.warning("Trade Engine is disabled. Bot will not place trades now.")

    # ... bias calc, trade engine, log ...

if __name__ == "__main__":
    main()
