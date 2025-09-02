import datetime, os
import pandas as pd
import streamlit as st
from SmartApi import SmartConnect  # Official Angel One SmartApi

MAX_TRADES = 3
TARGET = 10
NO_TRADE_AFTER = datetime.time(15, 0)
LOTS_PER_TRADE = 4
LOT_SIZE = 75

trade_log = []

def login_angel(api_key, client_id, password, totp):
    obj = SmartConnect(api_key=api_key)
    session = obj.generateSession(client_id, password, totp)
    return obj, session

def fetch_instruments(api):
    return pd.DataFrame(api.getInstruments("NFO"))

def get_month_expiries(instruments, symbol="NIFTY"):
    df = instruments[instruments["name"] == symbol].copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    today = datetime.date.today()
    return sorted([e for e in df["expiry"].unique() if e >= today and e.month == today.month])

def compute_indicators(df):
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA21"] = df["close"].ewm(span=21).mean()
    df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["PP"] = (df["high"] + df["low"] + df["close"]) / 3
    df["BC"] = (df["high"] + df["low"]) / 2
    df["TC"] = 2 * df["PP"] - df["BC"]
    df["H-L"] = df["high"] - df["low"]
    df["H-C"] = abs(df["high"] - df["close"].shift())
    df["L-C"] = abs(df["low"] - df["close"].shift())
    df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()
    return df

def check_bias(df):
    latest = df.iloc[-1]
    reasons = []
    cpr_status = "Aligned"
    bias = "Neutral"

    if latest["EMA9"] > latest["EMA21"] and latest["EMA9"] > latest["VWAP"]:
        bias = "Bullish"; reasons.append("EMA9 > EMA21 & VWAP")
    elif latest["EMA9"] < latest["EMA21"] and latest["EMA9"] < latest["VWAP"]:
        bias = "Bearish"; reasons.append("EMA9 < EMA21 & VWAP")

    atr_now = latest["ATR"]
    if latest["close"] > latest["TC"]:
        if bias == "Bullish": reasons.append("Above CPR")
        else: cpr_status = "Mismatch (trend reversal, ATR rising)" if atr_now > df["ATR"].mean() else "Mismatch (trend reversal, ATR shrinking)"
    elif latest["close"] < latest["BC"]:
        if bias == "Bearish": reasons.append("Below CPR")
        else: cpr_status = "Mismatch (trend reversal, ATR rising)" if atr_now > df["ATR"].mean() else "Mismatch (trend reversal, ATR shrinking)"
    else:
        reasons.append("Inside CPR")

    return bias, ", ".join(reasons), cpr_status

def add_trade(signal, bias, reasons, cpr_status, expiry, strike, entry, sl, tsl, exit_price, status, order_id=None):
    pnl = (exit_price - entry) * LOT_SIZE * LOTS_PER_TRADE if signal == "GO CALL" else (entry - exit_price) * LOT_SIZE * LOTS_PER_TRADE
    trade_log.append({
        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
        "Signal": signal,
        "Bias": bias,
        "Reasons": reasons,
        "CPR Status": cpr_status,
        "Expiry": expiry,
        "Strike": strike,
        "Entry Price": entry,
        "SL": sl,
        "TSL": tsl,
        "Exit Price": exit_price,
        "Status": status,
        "P&L (₹)": round(pnl,2),
        "Order ID": order_id if order_id else "-"
    })

def main():
    st.title("Options Trading Bot (Angel One) - NIFTY Only")
    MASTER_PASSWORD = os.getenv("MASTER_PASSWORD")
    API_KEY = os.getenv("API_KEY")
    CLIENT_ID = os.getenv("CLIENT_ID")
    PASSWORD = os.getenv("PASSWORD")
    TOTP = os.getenv("TOTP")

    pw = st.text_input("Enter Master Password", type="password")
    if pw != MASTER_PASSWORD:
        st.warning("Enter correct master password to continue.")
        st.stop()

    mode = st.radio("Select Mode", ["Paper Trading", "Live Trading"], index=0)
    paper = (mode == "Paper Trading")

    api, _ = login_angel(API_KEY, CLIENT_ID, PASSWORD, TOTP)
    instruments = fetch_instruments(api)
    expiries = get_month_expiries(instruments)
    expiry_choice = st.selectbox("Select Expiry", expiries)

    # Example candle for indicators (replace with live candles if needed)
    df = pd.DataFrame([{"datetime": datetime.datetime.now(),"open":20000,"high":20050,"low":19950,"close":20010,"volume":100000}])
    df = compute_indicators(df)
    bias, reasons, cpr_status = check_bias(df)

    # Fetch NIFTY spot LTP (token 99926000 is for NIFTY index spot)
    try:
        ltp_data = api.ltpData("NSE", "NIFTY", "99926000")
        spot_price = ltp_data["data"]["ltp"]
        atm_strike = round(spot_price / 50) * 50
    except Exception as e:
        st.error(f"Error fetching LTP: {e}")
        spot_price = 20000
        atm_strike = 20000

    st.subheader(f"GO: {'CALL' if bias=='Bullish' else 'PUT' if bias=='Bearish' else 'NO GO'}")
    st.caption(f"ATM Strike: {atm_strike} (Spot: {spot_price})")
    st.subheader(f"Market Bias: {bias}")
    st.caption(f"Reasons: {reasons}")
    st.caption(f"CPR Status: {cpr_status}")

    # Placeholder trade execution
    if st.button("Execute Trade"):
        signal = "GO CALL" if bias == "Bullish" else "GO PUT" if bias == "Bearish" else "NO GO"
        if signal != "NO GO":
            entry_price = spot_price
            sl = entry_price - 20 if signal == "GO CALL" else entry_price + 20
            tsl = sl
            exit_price = entry_price + TARGET if signal == "GO CALL" else entry_price - TARGET
            status = "Target"
            order_id = None

            if not paper:
                try:
                    order = api.placeOrder(
                        variety="NORMAL",
                        tradingsymbol=f"NIFTY{expiry_choice.strftime('%d%b').upper()}{atm_strike}{'CE' if signal=='GO CALL' else 'PE'}",
                        symboltoken="12345",
                        transactiontype="BUY",
                        exchange="NFO",
                        ordertype="MARKET",
                        producttype="INTRADAY",
                        duration="DAY",
                        price=0,
                        squareoff=0,
                        stoploss=0,
                        quantity=LOTS_PER_TRADE * LOT_SIZE
                    )
                    order_id = order.get("orderid", None)
                except Exception as e:
                    st.error(f"Order placement failed: {e}")

            add_trade(signal, bias, reasons, cpr_status, expiry_choice, atm_strike, entry_price, sl, tsl, exit_price, status, order_id)

    if trade_log:
        st.subheader("Today's Trades")
        st.dataframe(pd.DataFrame(trade_log))

if __name__ == "__main__":
    main()
