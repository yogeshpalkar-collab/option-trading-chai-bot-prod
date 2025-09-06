import os, csv, math, time
from datetime import datetime, time as dtime
import pandas as pd
from logzero import logger
from smartapi_wrapper import SmartAPIWrapper

# Configurable lots via env, default 1
DEFAULT_LOTS = int(os.getenv("DEFAULT_LOTS", "1"))
# Hard cap: max trades per day = 3 (NOT configurable)
MAX_TRADES_PER_DAY = 3
NO_TRADE_AFTER = os.getenv("NO_TRADE_AFTER", "15:00")  # HH:MM IST
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() in ("1","true","yes")

TARGET_POINTS = int(os.getenv("TARGET_POINTS", "10"))
PARTIAL_BOOK_PERCENT = int(os.getenv("PARTIAL_BOOK_PERCENT", "0"))  # Default 0 = no partial

def round_to_nearest_50(x):
    return int(round(x/50.0))*50

def compute_indicators(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df.get('volume', pd.Series([0]*len(df))).astype(float)
    # EMA
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    # VWAP
    tp = (df['high'] + df['low'] + df['close'])/3.0
    vwap = (tp * df['volume']).sum() / max(1, df['volume'].sum())
    # ATR(14)
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1).fillna(df['close'].iloc[0])
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1] if len(tr)>=14 else tr.mean()
    # RSI(14)
    delta = df['close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / down.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1] if not rsi.empty else 50
    # CPR (using last completed candle)
    prev = df.iloc[-2] if len(df)>=2 else df.iloc[-1]
    pivot = (prev['high'] + prev['low'] + prev['close'])/3.0
    bc = (prev['high'] + prev['low'])/2.0
    tc = pivot - (bc - pivot)
    cpr = {"BC": bc, "TC": tc, "pivot": pivot}
    return {
        "ema9": df['ema9'].iloc[-1],
        "ema21": df['ema21'].iloc[-1],
        "vwap": vwap,
        "atr14": float(atr14 if not pd.isna(atr14) else 0.0),
        "rsi": float(rsi_val if not pd.isna(rsi_val) else 50.0),
        "cpr": cpr,
        "last_close": float(df['close'].iloc[-1])
    }

def within_trading_hours():
    now = datetime.now().time()
    cutoff_h, cutoff_m = map(int, NO_TRADE_AFTER.split(":"))
    cutoff = dtime(hour=cutoff_h, minute=cutoff_m)
    return now < cutoff

def get_today_trade_count():
    logs_dir = os.path.join(os.getcwd(), "trade_logs")
    fname = os.path.join(logs_dir, f"trade_log_{datetime.now().date()}.csv")
    if not os.path.exists(fname):
        return 0
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        return max(0, len(rows)-1)

def build_order_payload(tradingsymbol, symboltoken, transactiontype, quantity, price=None):
    payload = {
        "variety": "NORMAL",
        "tradingsymbol": tradingsymbol,
        "symboltoken": str(symboltoken),
        "transactiontype": transactiontype,
        "exchange": "NFO",
        "ordertype": "MARKET" if price is None else "LIMIT",
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": str(price) if price is not None else "0",
        "quantity": str(quantity)
    }
    return payload

def run_once():
    cfg = {
        'API_KEY': os.getenv('ANGEL_API_KEY'),
        'CLIENT_ID': os.getenv('ANGEL_CLIENT_ID'),
        'PASSWORD': os.getenv('ANGEL_PASSWORD'),
        'TOTP': os.getenv('ANGEL_TOTP'),
        'PAPER_MODE': PAPER_MODE
    }
    if not cfg['API_KEY'] or not cfg['CLIENT_ID'] or not cfg['PASSWORD']:
        logger.error("Missing ANGEL credentials in environment. Aborting.")
        return
    if not within_trading_hours():
        logger.info("Outside trading hours or past NO_TRADE_AFTER. Aborting cycle.")
        return

    today_trades = get_today_trade_count()
    if today_trades >= MAX_TRADES_PER_DAY:
        logger.info("Max trades reached for today (%d). Aborting.", today_trades)
        return

    sm = SmartAPIWrapper(cfg['API_KEY'])
    sm.login(cfg['CLIENT_ID'], cfg['PASSWORD'], cfg.get('TOTP'))

    instruments = sm.get_instruments('NFO', symbol_filter='NIFTY') or []
    logger.info("Fetched instruments: %d", len(instruments))

    # Fetch candles for underlying if possible
    sample_token = None
    for ins in instruments:
        if ins and (ins.get('tradingsymbol') or '').upper().startswith('NIFTY'):
            sample_token = ins.get('symboltoken') or ins.get('token') or ins.get('instrument_token') or ins.get('symbolToken')
            break
    df = None
    if sample_token:
        try:
            params = {"exchange":"NSE","symboltoken":str(sample_token),"interval":"ONE_MINUTE","fromdate":(datetime.now().strftime("%Y-%m-%d 09:15")),"todate":(datetime.now().strftime("%Y-%m-%d %H:%M"))}
            resp = sm.get_candle_data(params)
            if resp and resp.get('data'):
                df = pd.DataFrame(resp['data'], columns=["time","o","h","l","c","v"])
                df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}, inplace=True)
        except Exception as e:
            logger.warning("Candle fetch failed: %s", e)

    if df is None or df.empty:
        logger.warning("No candle data available; using placeholder recent price.")
        last_close = 23300.0
        indicators = {"last_close": last_close, "ema9": last_close, "ema21": last_close, "vwap": last_close, "atr14": 10.0, "rsi":50.0, "cpr": {"BC": last_close-5, "TC": last_close+5}}
    else:
        indicators = compute_indicators(df)

    spot = indicators['last_close']
    atm = round_to_nearest_50(spot)
    logger.info("Spot: %s ATM: %s indicators: %s", spot, atm, indicators)

    go_long = (indicators['last_close'] > indicators['vwap']) and (indicators['ema9'] > indicators['ema21']) and (indicators['last_close'] > indicators['cpr']['TC']) and (indicators['rsi'] < 70)
    go_short = (indicators['last_close'] < indicators['vwap']) and (indicators['ema9'] < indicators['ema21']) and (indicators['last_close'] < indicators['cpr']['BC']) and (indicators['rsi'] > 30)

    if not (go_long or go_short):
        logger.info("No GO condition met. Exiting cycle.")
        return

    side = "BUY" if go_long else "SELL"
    opt_type = "CE" if go_long else "PE"
    target_strike = f"NIFTY{atm}{opt_type}"

    chosen_token = "000000"
    chosen_symbol = target_strike
    for ins in instruments:
        ts = (ins.get('tradingsymbol') or ins.get('name') or '').upper()
        if str(atm) in ts and opt_type in ts:
            chosen_token = ins.get('symboltoken') or ins.get('token') or ins.get('instrument_token') or ins.get('symbolToken') or ins.get('symbol_token') or chosen_token
            chosen_symbol = ts
            break

    qty = DEFAULT_LOTS
    order_payload = build_order_payload(chosen_symbol, chosen_token, side, qty)

    # compute SL/target
    atr = indicators.get('atr14', 0.0) or 0.0
    sl_buffer = max(10, int(atr) + 10)
    target_pts = TARGET_POINTS

    log_entry = {"timestamp": datetime.now().isoformat(), "mode": ("PAPER" if PAPER_MODE else "LIVE"), "payload": order_payload, "indicators": indicators, "sl_buffer": sl_buffer, "target_pts": target_pts}

    sm2 = SmartAPIWrapper(cfg['API_KEY'])
    sm2.login(cfg['CLIENT_ID'], cfg['PASSWORD'], cfg.get('TOTP'))
    try:
        resp = sm2.place_order(order_payload)
        log_entry['response'] = resp
        logger.info("Order response: %s", resp)
    except Exception as e:
        log_entry['error'] = str(e)
        logger.exception("Order placement failed: %s", e)

    # write trade log CSV
    os.makedirs("trade_logs", exist_ok=True)
    fname = f"trade_logs/trade_log_{datetime.now().date()}.csv"
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline='') as f:
        import csv
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp','mode','tradingsymbol','symboltoken','quantity','response','error','indicators','sl_buffer','target_pts'])
        writer.writerow([log_entry.get('timestamp'), log_entry.get('mode'), chosen_symbol, chosen_token, qty, str(log_entry.get('response')), str(log_entry.get('error')), str(log_entry.get('indicators')), log_entry.get('sl_buffer'), log_entry.get('target_pts')])

    logger.info("Trade cycle complete. Check trade_logs for details.")

if __name__ == "__main__":
    run_once()
