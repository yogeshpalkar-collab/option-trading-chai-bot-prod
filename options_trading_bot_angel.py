import streamlit as st
import pandas as pd
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
import datetime as dt
import logzero
from logzero import logger
import json
import os

API_KEY = os.getenv("API_KEY")
CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP = os.getenv("TOTP")

live_data = []
oi_data = {"CE_OI": None, "PE_OI": None, "CE_OI_prev": None, "PE_OI_prev": None}
trades = []
trade_count = 0
traded_strikes = set()
mode = st.sidebar.radio("Mode", ["Paper", "Live"], index=0)  # Default = Paper

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

def fetch_instruments(smartApi):
    try:
        instruments = smartApi.getInstruments()
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return None

def interpret_oi_change():
    try:
        ce_oi = oi_data["CE_OI"]
        pe_oi = oi_data["PE_OI"]
        ce_prev = oi_data["CE_OI_prev"]
        pe_prev = oi_data["PE_OI_prev"]
        if None in [ce_oi, pe_oi, ce_prev, pe_prev]:
            return "Neutral OI"
        ce_change = ce_oi - ce_prev
        pe_change = pe_oi - pe_prev
        if pe_change > ce_change:
            return "Bullish OI"
        elif ce_change > pe_change:
            return "Bearish OI"
        else:
            return "Neutral OI"
    except Exception as e:
        return f"Error in OI change: {e}"

def calculate_bias(df):
    try:
        ema9 = df['close'].ewm(span=9).mean().iloc[-1]
        ema21 = df['close'].ewm(span=21).mean().iloc[-1]

        df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_vol_price'] / df['cum_vol']
        vwap = df['vwap'].iloc[-1]

        high = df['close'].max()
        low = df['close'].min()
        close = df['close'].iloc[-1]
        pivot = (high + low + close) / 3
        bc = pivot - (high - low) / 2
        tc = pivot + (high - low) / 2

        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = (df['close'] - df['prev_close']).abs()
        atr = df['tr1'].rolling(14).mean().iloc[-1]

        last_close = df['close'].iloc[-1]
        oi_signal = interpret_oi_change()

        if ema9 > ema21 and last_close > vwap and "Bullish" in oi_signal:
            if last_close > tc:
                return "Bullish"
            else:
                return "Bullish (CPR relaxed)"
        elif ema9 < ema21 and last_close < vwap and "Bearish" in oi_signal:
            if last_close < bc:
                return "Bearish"
            else:
                return "Bearish (CPR relaxed)"
        else:
            return "Neutral"
    except Exception as e:
        return f"Error in bias calculation: {e}"

def execute_trade(bias, spot_price, instruments, smartApi):
    global trades, trade_count, traded_strikes
    if trade_count >= 3:
        return "Max trades reached"
    if dt.datetime.now().hour >= 15:
        return "No trades after 3 PM"

    atm_strike = round(spot_price / 50) * 50
    option_type = "CE" if "Bullish" in bias else "PE"
    strike_key = f"{atm_strike}{option_type}"
    if strike_key in traded_strikes:
        return "Repeat strike blocked"

    # SL = ATR(14)+10; here assume ATR ~15 for placeholder; actual ATR should come from bias calc
    sl_buffer = 25
    entry_price = spot_price
    sl_price = entry_price - sl_buffer if option_type == "CE" else entry_price + sl_buffer
    target_price = entry_price + 10 if option_type == "CE" else entry_price - 10

    trade = {
        "time": dt.datetime.now(),
        "strike": atm_strike,
        "type": option_type,
        "entry": entry_price,
        "sl": sl_price,
        "target": target_price,
        "tsl": sl_price,
        "status": "OPEN",
        "exit": None,
        "pnl": None,
        "reason": None,
        "order_id": None if mode == "Paper" else "LIVE_ORDER_ID"
    }

    trades.append(trade)
    traded_strikes.add(strike_key)
    trade_count += 1

    return f"Trade executed: {strike_key} at {entry_price}"

def update_trades(latest_price):
    global trades
    for trade in trades:
        if trade["status"] != "OPEN":
            continue
        if trade["type"] == "CE":
            if latest_price > trade["tsl"] + 5:
                trade["tsl"] = latest_price - 5
            if latest_price <= trade["tsl"]:
                trade["status"] = "CLOSED"
                trade["exit"] = trade["tsl"]
                trade["pnl"] = trade["exit"] - trade["entry"]
                trade["reason"] = "TSL Hit"
            elif latest_price >= trade["target"]:
                trade["tsl"] = latest_price - 5
                trade["reason"] = "Target Relax Mode"
        else:
            if latest_price < trade["tsl"] - 5:
                trade["tsl"] = latest_price + 5
            if latest_price >= trade["tsl"]:
                trade["status"] = "CLOSED"
                trade["exit"] = trade["tsl"]
                trade["pnl"] = trade["entry"] - trade["exit"]
                trade["reason"] = "TSL Hit"
            elif latest_price <= trade["target"]:
                trade["tsl"] = latest_price + 5
                trade["reason"] = "Target Relax Mode"

def start_websocket(smartApi, client_id, feed_token, instrument_token):
    global live_data
    try:
        sws = SmartWebSocketV2(api_key=API_KEY, client_id=client_id,
                               feed_token=feed_token, jwt_token=smartApi.jwt_token)

        def on_data(wsapp, message):
            global live_data
            try:
                data = json.loads(message)
                if "last_traded_price" in data:
                    ltp = data["last_traded_price"]
                    ts = dt.datetime.now()
                    live_data.append({"time": ts, "close": ltp, "volume": 1})
                    if len(live_data) > 500:
                        live_data = live_data[-500:]
                    update_trades(ltp)
            except Exception as e:
                logger.error(f"Error in on_data: {e}")

        def on_open(wsapp):
            logger.info("WebSocket connected. Subscribing to NIFTY spot...")
            sws.subscribe([{"exchangeType": 1, "token": instrument_token}])

        def on_error(wsapp, error):
            st.error(f"WebSocket error: {error}")

        def on_close(wsapp):
            logger.warning("WebSocket closed.")

        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = on_error
        sws.on_close = on_close

        sws.connect(threaded=True)

    except Exception as e:
        st.error(f"Error starting WebSocket: {e}")

def main():
    st.title("Options Trading Bot (Angel One) - Secured v3 Render Final Engine")

    smartApi = login_smartapi()
    if smartApi is None:
        st.stop()

    instruments = fetch_instruments(smartApi)
    if instruments is None:
        st.stop()

    feed_token = smartApi.getfeedToken()
    nifty_row = instruments[(instruments['name'] == 'NIFTY') & (instruments['exch_seg'] == 'NSE')]
    if nifty_row.empty:
        st.error("NIFTY instrument not found in instruments list.")
        st.stop()

    nifty_token = int(nifty_row.iloc[0]['token'])
    start_websocket(smartApi, CLIENT_ID, feed_token, nifty_token)

    if live_data:
        df = pd.DataFrame(live_data)
        bias = calculate_bias(df)
        st.subheader(f"Bias: {bias}")

        if "Bullish" in bias or "Bearish" in bias:
            msg = execute_trade(bias, df['close'].iloc[-1], instruments, smartApi)
            st.info(msg)

    st.subheader("Today's Trades")
    if trades:
        df_trades = pd.DataFrame(trades)
        st.dataframe(df_trades)
    else:
        st.write("No trades yet.")

if __name__ == "__main__":
    main()
