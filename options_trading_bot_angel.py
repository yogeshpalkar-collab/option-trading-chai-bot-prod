
# options_trading_bot_angel.py
# Updated trading logic:
# - CE: EMA9 > EMA20 and EMA9 > VWAP, plus OI increase required
# - PE: EMA9 < EMA20 and EMA9 < VWAP, plus OI increase required
# - OI change computed from consecutive LTP/OI reads stored in bot state
# Environment vars used: API_KEY, CLIENT_ID, PASSWORD, MASTER_PASSWORD, TOTP
import streamlit as st
import os, json, threading, time, traceback
from datetime import datetime, timedelta, time as dtime
import pandas as pd
import numpy as np

# SmartAPI import
SMARTAPI_AVAILABLE = False
try:
    from smartapi import SmartConnect
    SMARTAPI_AVAILABLE = True
except Exception:
    try:
        from SmartApi.smartConnect import SmartConnect
        SMARTAPI_AVAILABLE = True
    except Exception:
        SMARTAPI_AVAILABLE = False

# pyotp for TOTP
try:
    import pyotp
    PYOTP_AVAILABLE = True
except Exception:
    PYOTP_AVAILABLE = False

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def round_to_nearest(x, base=50):
    return int(base * round(float(x) / base))

def dtstr(dt):
    return dt.strftime("%Y-%m-%d %H:%M")

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def cpr_calc(df):
    h, l, c = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
    bc = (h + l + c) / 3.0
    tc = bc + (h - l)
    return {'BC': bc, 'TC': tc, 'width': abs(tc - bc)}

class AngelClient:
    def __init__(self, api_key, client_id, password, master_password=None, totp_secret=None):
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.master_password = master_password
        self.totp_secret = totp_secret
        self.client = None
        self.logged_in = False

    def connect(self):
        if not SMARTAPI_AVAILABLE:
            raise RuntimeError("smartapi-python not installed in environment.")
        self.client = SmartConnect(api_key=self.api_key)
        totp = None
        if self.totp_secret and PYOTP_AVAILABLE:
            try:
                totp = pyotp.TOTP(self.totp_secret).now()
            except Exception:
                totp = None
        if totp:
            data = self.client.generateSession(self.client_id, self.password, totp)
        else:
            data = self.client.generateSession(self.client_id, self.password)
        if not data or not data.get('data'):
            raise RuntimeError(f"Login failed: {data}")
        token = data['data'].get('jwtToken') or data['data'].get('refreshToken') or data['data']
        try:
            self.client.set_access_token(token)
        except Exception:
            pass
        self.logged_in = True
        return data

    def get_instruments(self):
        raw = self.client.getInstruments()
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        try:
            return pd.DataFrame(raw)
        except Exception:
            return raw

    def get_ltp(self, exchange, tradingsymbol, symboltoken):
        if hasattr(self.client, 'ltpData'):
            return self.client.ltpData(exchange, tradingsymbol, str(symboltoken))
        if hasattr(self.client, 'getLtp'):
            return self.client.getLtp(exchange, tradingsymbol, str(symboltoken))
        raise RuntimeError("No LTP method found in SmartAPI client.")

    def get_candle_data(self, exchange, symboltoken, interval='ONE_MINUTE', fromdate=None, todate=None):
        params = {"exchange": exchange, "symboltoken": str(symboltoken), "interval": interval, "fromdate": fromdate, "todate": todate}
        return self.client.getCandleData(params)

    def place_order(self, order_params):
        return self.client.placeOrder(order_params=order_params)

class LiveBot(threading.Thread):
    def __init__(self, angel_client, config, expiry_getter, ui_logger):
        super().__init__(daemon=True)
        self.angel = angel_client
        self.config = config
        self.expiry_getter = expiry_getter
        self.stop_event = threading.Event()
        self.trades = []
        self.ui_log = ui_logger
        # store last seen OI per symboltoken to compute change
        self.last_oi = {}

    def stop(self):
        self.stop_event.set()

    def ui(self, *args, **kwargs):
        try:
            self.ui_log(*args, **kwargs)
        except Exception:
            pass

    def find_underlying_token(self, instruments_df, underlying_symbol='NIFTY'):
        df = instruments_df.copy()
        candidates = []
        for col in ['tradingsymbol', 'name', 'symbol']:
            if col in df.columns:
                candidates.extend(df[df[col].str.contains(underlying_symbol, case=False, na=False)].to_dict('records'))
        if candidates:
            return candidates[0]
        return None

    def find_option_instrument(self, instruments_df, expiry_str, strike, right):
        df = instruments_df.copy()
        col_map = {c.lower(): c for c in df.columns}
        try:
            if 'expiry' in col_map and 'strike' in col_map and 'instrumenttype' in col_map:
                cond = (df[col_map['expiry']].astype(str) == expiry_str) & (df[col_map['strike']].astype(float) == float(strike)) & (df[col_map['instrumenttype']].astype(str).str.contains(right))
                res = df[cond]
                if len(res) > 0:
                    return res.iloc[0].to_dict()
            if 'tradingsymbol' in col_map:
                subs = df[df[col_map['tradingsymbol']].astype(str).str.contains(str(strike)) & df[col_map['tradingsymbol']].astype(str).str.contains(right)]
                if 'expiry' in col_map:
                    subs = subs[subs[col_map['expiry']].astype(str).str.contains(expiry_str)]
                if len(subs) > 0:
                    return subs.iloc[0].to_dict()
        except Exception:
            pass
        return None

    def compute_indicators_from_candles(self, candles_df):
        df = candles_df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}).copy()
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['atr14'] = atr(df['high'], df['low'], df['close'], 14)
        df['vwap'] = vwap(df)
        return df

    def fetch_oi_from_ltp(self, instrument_record):
        # Try to read OI from ltpData/getLtp response; different SDK versions use different keys.
        symboltoken = instrument_record.get('symboltoken') or instrument_record.get('token') or instrument_record.get('instrument_token')
        try:
            resp = self.angel.get_ltp(instrument_record.get('exchange','NFO'), instrument_record.get('tradingsymbol') or instrument_record.get('symbol'), symboltoken)
            oi = None
            if isinstance(resp, dict) and 'data' in resp:
                dat = resp['data']
                if isinstance(dat, dict) and ('oi' in dat or 'openInterest' in dat or 'open_interest' in dat):
                    oi = dat.get('oi') or dat.get('openInterest') or dat.get('open_interest')
                elif isinstance(dat, list) and len(dat) and isinstance(dat[0], dict):
                    oi = dat[0].get('oi') or dat[0].get('openInterest') or dat[0].get('open_interest')
            # fallback: try common numeric in resp
            if oi is None:
                try:
                    oi = float(str(resp))
                except Exception:
                    oi = None
            return float(oi) if oi is not None else None
        except Exception as e:
            self.ui(f"[warn] fetch_oi_from_ltp failed: {e}")
            return None

    def run_once_for_option(self, option_token_info, exchange='NFO'):
        symboltoken = option_token_info.get('symboltoken') or option_token_info.get('token') or option_token_info.get('instrument_token')
        if not symboltoken:
            raise RuntimeError("Option symboltoken not found in instrument record.")
        todate = datetime.now()
        fromdate = todate - timedelta(minutes=200)
        try:
            raw = self.angel.get_candle_data(exchange, symboltoken, interval='ONE_MINUTE', fromdate=dtstr(fromdate), todate=dtstr(todate))
        except Exception as e:
            raise RuntimeError(f"getCandleData failed: {e}")
        df = None
        try:
            if isinstance(raw, dict) and 'data' in raw:
                data_rows = raw['data']
            else:
                data_rows = raw
            if len(data_rows) and isinstance(data_rows[0], (list, tuple)):
                df = pd.DataFrame(data_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
            elif len(data_rows) and isinstance(data_rows[0], dict):
                df = pd.json_normalize(data_rows)
                possible = ['timestamp','open','high','low','close','volume']
                df = df[[c for c in df.columns if c in possible]].copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                raise RuntimeError("Unhandled candle data format.")
        except Exception as e:
            raise RuntimeError(f"Failed to parse candle data: {e}")
        df = df.dropna().reset_index(drop=True)
        if df is None or df.shape[0] < 5:
            raise RuntimeError("Insufficient candle data for indicators.")
        ind_df = self.compute_indicators_from_candles(df)
        latest = ind_df.iloc[-1]
        cprv = cpr_calc(df)
        # compute OI change
        current_oi = self.fetch_oi_from_ltp(option_token_info) or 0.0
        prev_oi = self.last_oi.get(str(symboltoken))
        oi_change_pct = None
        if prev_oi is not None and prev_oi != 0:
            oi_change_pct = ((current_oi - prev_oi) / abs(prev_oi)) * 100.0
        # update last_oi store
        self.last_oi[str(symboltoken)] = current_oi
        # Decision logic with EMA9 vs EMA20 and VWAP plus OI increase requirement
        bias = 'Neutral'
        if latest['close'] > latest['ema9'] and latest['ema9'] > latest['ema20'] and latest['ema9'] > latest['vwap']:
            bias = 'Bullish'
        elif latest['close'] < latest['ema9'] and latest['ema9'] < latest['ema20'] and latest['ema9'] < latest['vwap']:
            bias = 'Bearish'
        decision = None
        sl = float(latest['atr14']) + 10
        # require OI change positive to confirm trade (configurable threshold)
        threshold = float(self.config['trade_rules'].get('oi_change_threshold_percent', 0.0))
        oi_ok = oi_change_pct is not None and oi_change_pct > threshold
        # If prev_oi missing, we treat OI as not confirmed (conservative)
        if bias == 'Bullish' and oi_ok:
            decision = {'type': 'BUY_CALL', 'sl': sl, 'oi_change_pct': oi_change_pct}
        elif bias == 'Bearish' and oi_ok:
            decision = {'type': 'BUY_PUT', 'sl': sl, 'oi_change_pct': oi_change_pct}
        return decision, bias, cprv, ind_df

    def build_order_params(self, instrument_record, decision, lots_multiplier=1):
        tradingsymbol = instrument_record.get('tradingsymbol') or instrument_record.get('symbol') or instrument_record.get('name')
        symboltoken = instrument_record.get('symboltoken') or instrument_record.get('token') or instrument_record.get('instrument_token')
        lot_size = instrument_record.get('lot_size') or instrument_record.get('lotsize') or instrument_record.get('lotSize') or 1
        qty = int(int(lot_size) * int(self.config['trade_rules'].get('lots_per_trade', 1)) * int(lots_multiplier))
        tx = 'BUY'
        exchange = instrument_record.get('exchange') or 'NFO'
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": tradingsymbol,
            "symboltoken": str(symboltoken),
            "transactiontype": tx,
            "exchange": exchange,
            "ordertype": "MARKET",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": "0",
            "quantity": qty
        }
        return order_params

    def run(self):
        try:
            instr_df = None
            while not self.stop_event.is_set():
                try:
                    if self.angel and not getattr(self.angel, 'logged_in', False):
                        self.ui("[info] reconnecting...")
                        self.angel.connect()
                    if self.angel:
                        raw_instr = self.angel.get_instruments()
                        if isinstance(raw_instr, pd.DataFrame):
                            instr_df = raw_instr
                        elif isinstance(raw_instr, (list, tuple)):
                            instr_df = pd.DataFrame(raw_instr)
                        else:
                            try:
                                instr_df = pd.DataFrame(raw_instr)
                            except Exception:
                                instr_df = None
                    if instr_df is None:
                        self.ui("[error] Could not download instruments. Retrying in 30s.")
                        time.sleep(30)
                        continue
                    und = self.find_underlying_token(instr_df, 'NIFTY')
                    if not und:
                        self.ui("[error] Could not find underlying NIFTY in instruments. Retrying.")
                        time.sleep(30); continue
                    try:
                        spot_resp = self.angel.get_ltp(und.get('exchange','NSE'), und.get('tradingsymbol', und.get('symbol', 'NIFTY')), und.get('symboltoken'))
                        spot = None
                        if isinstance(spot_resp, dict) and 'data' in spot_resp:
                            dat = spot_resp['data']
                            if isinstance(dat, dict) and ('lastprice' in dat or 'ltp' in dat):
                                spot = float(dat.get('lastprice') or dat.get('ltp'))
                            elif isinstance(dat, list) and len(dat) and isinstance(dat[0], dict):
                                spot = float(dat[0].get('lastprice') or dat[0].get('ltp') or dat[0].get('ltpPrice'))
                        if spot is None:
                            try:
                                spot = float(str(spot_resp))
                            except Exception:
                                spot = None
                        if spot is None:
                            raise RuntimeError(f"Could not parse spot LTP from: {spot_resp}")
                    except Exception as e:
                        self.ui(f"[error] failed to fetch underlying LTP: {e}")
                        time.sleep(10); continue
                    atm_strike = round_to_nearest(spot, base=50)
                    expiry = self.expiry_getter()
                    if not expiry:
                        self.ui("[warn] No expiry selected in UI; waiting for selection.")
                        time.sleep(5); continue
                    ce_rec = self.find_option_instrument(instr_df, expiry, atm_strike, 'CE')
                    pe_rec = self.find_option_instrument(instr_df, expiry, atm_strike, 'PE')
                    if ce_rec is None and pe_rec is None:
                        self.ui(f"[error] Could not find ATM instruments for strike {atm_strike} expiry {expiry}.")
                        time.sleep(20); continue
                    for candidate in [ce_rec, pe_rec]:
                        if candidate is None: continue
                        try:
                            decision, bias, cprv, ind_df = self.run_once_for_option(candidate, exchange=candidate.get('exchange','NFO'))
                        except Exception as e:
                            self.ui(f"[error] Candles/indicators failed for {candidate.get('tradingsymbol')}: {e}")
                            continue
                        self.ui(f"[info] Decision test for {candidate.get('tradingsymbol')}: bias={bias}, CPRwidth={cprv['width']:.2f}")
                        if decision:
                            order_params = self.build_order_params(candidate, decision)
                            try:
                                if self.angel and self.angel.logged_in:
                                    res = self.angel.place_order(order_params)
                                else:
                                    res = {"paper": True, "order_params": order_params}
                            except Exception as e:
                                res = {"error": str(e)}
                            res_record = {"timestamp": datetime.now().isoformat(), "decision": decision, "order_result": res}
                            self.trades.append(res_record)
                            self.ui(f"[order] {res_record}")
                            break
                    time.sleep(5)
                except Exception as ee:
                    self.ui("[exception] " + str(ee))
                    self.ui(traceback.format_exc())
                    time.sleep(5)
        except Exception as main_e:
            self.ui("[fatal] Bot main loop crashed: " + str(main_e))
            self.ui(traceback.format_exc())

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title='Secured Trading Bot v3 (Angel Live) v2', layout='wide')
st.title("Secured Trading Bot v3 — Live (Angel One) v2")

cfg_path = os.path.join(os.path.dirname(__file__), 'secured_trading_bot_v3_render_final_clean_json_expiry_safe.json')
if not os.path.exists(cfg_path):
    st.error("Missing config JSON. Place secured_trading_bot_v3_render_final_clean_json_expiry_safe.json in app folder.")
    st.stop()
config = load_config(cfg_path)

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Paper", "Live"], index=0)
    st.markdown("**Expiry selector (populated from Angel instruments when connected)**")
    expiries = []
    can_fetch = SMARTAPI_AVAILABLE and os.getenv('API_KEY') and os.getenv('CLIENT_ID') and os.getenv('PASSWORD')
    if can_fetch:
        try:
            tmp_client = AngelClient(os.getenv('API_KEY'), os.getenv('CLIENT_ID'), os.getenv('PASSWORD'), master_password=os.getenv('MASTER_PASSWORD'), totp_secret=os.getenv('TOTP'))
            tmp_client.connect()
            inst_df = tmp_client.get_instruments()
            if isinstance(inst_df, pd.DataFrame) and 'expiry' in inst_df.columns:
                expiries = sorted(inst_df['expiry'].dropna().unique().tolist())
            elif isinstance(inst_df, pd.DataFrame) and 'tradingsymbol' in inst_df.columns:
                expiries = sorted({str(s).split('_')[-1] for s in inst_df['tradingsymbol'].dropna().unique().tolist()})
        except Exception as e:
            st.warning("Could not fetch instruments: " + str(e))
            expiries = []
    else:
        if not SMARTAPI_AVAILABLE:
            st.info("smartapi-python not installed in environment. Install it to enable live mode and expiry fetch.")
        else:
            st.info("Set API_KEY, CLIENT_ID and PASSWORD in environment for live mode.")

    if expiries:
        selected_expiry = st.selectbox("Expiry", options=expiries)
    else:
        selected_expiry = st.selectbox("Expiry", options=["-- no expiry loaded --"])

    start = st.button("Start Bot")
    stop = st.button("Stop Bot")
    st.markdown("Paper and Live run identical strategy logic. Live submits orders via Angel SmartAPI when 'Live' is selected and credentials are valid. OI change must be positive to confirm trade.")

# main layout
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Bot status & last logs")
    log_area = st.empty()
with col2:
    st.subheader("Trades")
    trades_area = st.empty()

# session-managed bot
if 'bot' not in st.session_state:
    st.session_state['bot'] = None
    st.session_state['log_msgs'] = []

def ui_logger(msg):
    st.session_state['log_msgs'].append(f"{datetime.now().isoformat()} - {msg}")
    st.session_state['log_msgs'] = st.session_state['log_msgs'][-200:]
    log_area.text("\n".join(st.session_state['log_msgs'][-50:]))

def expiry_getter():
    return selected_expiry if selected_expiry != "-- no expiry loaded --" else None

if start:
    if st.session_state['bot'] is not None and st.session_state['bot'].is_alive():
        st.info("Bot already running.")
    else:
        angel_client = None
        if mode == 'Live':
            if not SMARTAPI_AVAILABLE:
                st.error("smartapi-python not installed; cannot enable Live mode.")
            elif not (os.getenv('API_KEY') and os.getenv('CLIENT_ID') and os.getenv('PASSWORD')):
                st.error("Missing API_KEY/CLIENT_ID/PASSWORD in environment; cannot enable Live.")
            else:
                try:
                    angel_client = AngelClient(os.getenv('API_KEY'), os.getenv('CLIENT_ID'), os.getenv('PASSWORD'), master_password=os.getenv('MASTER_PASSWORD'), totp_secret=os.getenv('TOTP'))
                    angel_client.connect()
                    st.success("Connected to Angel SmartAPI (Live enabled).")
                except Exception as e:
                    st.error("Could not connect to Angel SmartAPI: " + str(e))
                    angel_client = None
        bot_thread = LiveBot(angel_client, config, expiry_getter, ui_logger)
        st.session_state['bot'] = bot_thread
        bot_thread.start()
        st.success(f"Bot started in {'Live' if angel_client and angel_client.logged_in else 'Paper'} mode.")

if stop:
    if st.session_state.get('bot') is not None:
        st.session_state['bot'].stop()
        st.session_state['bot'] = None
        st.success("Bot stopped.")
    else:
        st.info("Bot not running.")

# refresh UI once
if st.session_state.get('bot') is None:
    log_area.info("Bot not running. Use sidebar to start.")
else:
    trades_area.dataframe(pd.json_normalize(st.session_state['bot'].trades[::-1]) if st.session_state['bot'].trades else pd.DataFrame())
    log_area.text("\n".join(st.session_state['log_msgs'][-50:] if st.session_state['log_msgs'] else ["No logs yet."]))
