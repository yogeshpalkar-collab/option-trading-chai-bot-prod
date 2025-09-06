import os
import streamlit as st
from logzero import logger
from smartapi_wrapper import SmartAPIWrapper
from options_trading_bot_angel import run_once

st.set_page_config(page_title="Final Minimal Bot", layout="wide")
st.title("Final Minimal Options Trading Bot (Locked Rules)")

# MODE TOGGLE
DEFAULT_PAPER = os.getenv("PAPER_MODE","true").lower() in ("1","true","yes")
MASTER_PASSWORD_ENV = os.getenv("MASTER_PASSWORD", None)
paper_mode = st.sidebar.checkbox("Paper mode (default)", value=DEFAULT_PAPER)
live_enabled = False
if not paper_mode:
    pw = st.sidebar.text_input("MASTER PASSWORD to enable LIVE", type="password")
    if pw and MASTER_PASSWORD_ENV and pw == MASTER_PASSWORD_ENV:
        st.sidebar.success("LIVE enabled")
        live_enabled = True
    else:
        st.sidebar.error("MASTER_PASSWORD not set or incorrect; LIVE not enabled")

EFFECTIVE_MODE = "LIVE" if live_enabled else "PAPER"
st.sidebar.markdown(f"**Effective Mode:** {EFFECTIVE_MODE}")

st.write("This bot uses EMA9/EMA21, VWAP, CPR, ATR(14), RSI and has locked risk rules:")
st.markdown("- Max trades per day = 3 (hard limit)")
st.markdown("- DEFAULT_LOTS configurable via env DEFAULT_LOTS (default=1)")
st.markdown("- TARGET_POINTS configurable via env (default=10)")
st.markdown("- PAPER/LIVE use the same API call path; logs are tagged accordingly.")

if st.button("Run one cycle (strategy)"):
    api_key = os.getenv('ANGEL_API_KEY')
    client_id = os.getenv('ANGEL_CLIENT_ID')
    password = os.getenv('ANGEL_PASSWORD')
    totp = os.getenv('ANGEL_TOTP')
    if not api_key or not client_id or not password:
        st.error("Missing credentials in environment")
    else:
        try:
            st.info("Running one cycle — check logs/trade_logs for results")
            run_once()
            st.success("Cycle completed (check trade_logs)")
        except Exception as e:
            st.exception(e)
