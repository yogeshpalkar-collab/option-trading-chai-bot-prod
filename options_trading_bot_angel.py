
# options_trading_bot_angel.py
# Updated trading logic:
# - CE: EMA9 > EMA20 and EMA9 > VWAP, plus OI increase required
# - PE: EMA9 < EMA20 and EMA9 < VWAP, plus OI increase required
# - OI change computed from consecutive LTP/OI reads stored in bot state
# Environment vars used: API_KEY, CLIENT_ID, PASSWORD, MASTER_PASSWORD, TOTP
import streamlit as st
import os, json, threading, time, traceback, datetime
try:
    import websocket
    WS_CLIENT_AVAILABLE = True
except Exception:
    WS_CLIENT_AVAILABLE = False
from queue import Queue, Empty

from datetime import datetime, timedelta, time as dtime
import pandas as pd
import numpy as np



# --- Safe shutdown helpers (compatible with Streamlit) ---
import threading
_stop_event = threading.Event()

def get_stop_event():
    return _stop_event

def safe_register_signals(handler):
    """Register OS signals only if running in main thread. In worker threads (Streamlit)
    we skip registering signal handlers to avoid exceptions."""
    try:
        if threading.current_thread() is threading.main_thread():
            import signal as _signal_module
            _signal_module.signal(_signal_module.SIGINT, lambda *a, **k: handler())
            _signal_module.signal(_signal_module.SIGTERM, lambda *a, **k: handler())
            print('[INFO] Signal handlers registered (main thread).')
        else:
            print('[INFO] Running in worker thread; skipping OS signal registration.')
    except Exception as e:
        print('[WARN] safe_register_signals failed:', e)

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




# --- WebSocket helper for SmartAPI ticks (best-effort) ---
class SmartAPIWebsocketClient:
    \"\"\"Best-effort SmartAPI websocket client wrapper.
    Tries to use SmartConnect websocket if available; otherwise uses websocket-client to connect.
    It exposes a simple subscribe(tokens) and a thread that pushes ticks to a Queue for consumer.\"\"\"
    def __init__(self, angel_client):
        self.angel = angel_client
        self.ws = None
        self.thread = None
        self.queue = Queue()
        self.subscribed = set()
        self.stop_event = threading.Event()

    def _on_message(self, ws, message):
        try:
            # SmartAPI websockets often send JSON messages; push to queue
            data = json.loads(message)
            self.queue.put(data)
        except Exception:
            # push raw if JSON fails
            self.queue.put(message)

    def _on_open(self, ws):
        # optionally log
        try:
            print('[WS] connection opened')
        except Exception:
            pass

    def _on_error(self, ws, err):
        try:
            print('[WS] error', err)
        except Exception:
            pass

    def _on_close(self, ws, code, reason):
        try:
            print('[WS] closed', code, reason)
        except Exception:
            pass

    def start(self, tokens):
        # tokens: list of symboltoken strings
        if not WS_CLIENT_AVAILABLE:
            raise RuntimeError('websocket-client not installed in environment.')
        # Attempt to use SmartConnect built-in websocket if available
        try:
            client = getattr(self.angel, 'client', None)
            if client and hasattr(client, 'ws_connect'):
                # some SmartAPI SDKs provide ws_connect — try to use it
                self.ws = client.ws_connect(tokens)
                # SDK may return a wrapper; push a small reader thread if possible
                self.thread = threading.Thread(target=self._sdk_ws_reader, daemon=True)
                self.thread.start()
                return
        except Exception as e:
            print('[WS] SmartConnect.ws_connect failed:', e)
        # Fallback: use websocket-client to connect to a guessed SmartAPI websocket endpoint
        try:
            # Construct websocket URL based on known SmartAPI patterns (best-effort). If it fails, exception will be raised.
            ws_url = "wss://ws.smartapi.angelone.in/v2"  # common endpoint pattern; may vary per account/region
            headers = []
            # If token available, add auth header; some endpoints require access token in params
            token = None
            try:
                token = self.angel.client.get_access_token() if hasattr(self.angel.client, 'get_access_token') else None
            except Exception:
                token = None
            if token:
                ws_url += f"?token={token}"
            self.ws = websocket.WebSocketApp(ws_url,
                                             on_message=self._on_message,
                                             on_open=self._on_open,
                                             on_error=self._on_error,
                                             on_close=self._on_close,
                                             header=headers)
            self.thread = threading.Thread(target=self._run_ws, daemon=True)
            self.thread.start()
            # send initial subscribe once open (subscribe format may vary)
            # We'll add subscriptions via self.subscribe()
        except Exception as e:
            raise RuntimeError(f'WebSocket start failed: {e}')

    def _run_ws(self):
        try:
            self.ws.run_forever()
        except Exception as e:
            print('[WS] run_forever failed:', e)

    def _sdk_ws_reader(self):
        # Best-effort: read messages from SDK wrapper and push to queue
        try:
            while not self.stop_event.is_set():
                try:
                    msg = self.ws.recv()
                    if msg:
                        self.queue.put(msg)
                except Exception:
                    time.sleep(0.1)
        except Exception as e:
            print('[WS] sdk reader stopped:', e)

    def subscribe(self, tokens):
        # tokens: iterable of symboltoken strings
        for t in tokens:
            self.subscribed.add(str(t))
        # Try SDK subscribe if available
        try:
            client = getattr(self.angel, 'client', None)
            if client and hasattr(client, 'ws_subscribe'):
                client.ws_subscribe(list(self.subscribed))
                return
        except Exception as e:
            print('[WS] sdk subscribe failed:', e)
        # If using websocket-client, send a subscribe message (format depends on provider)
        try:
            if self.ws and hasattr(self.ws, 'send'):
                msg = json.dumps({"action": "subscribe", "params": {"tokens": list(self.subscribed)}})
                try:
                    self.ws.send(msg)
                except Exception:
                    pass
        except Exception:
            pass

    def get_tick(self, timeout=0.2):
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        try:
            self.stop_event.set()
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass
            if self.thread and self.thread.is_alive():
                try:
                    self.thread.join(timeout=1)
                except Exception:
                    pass
        except Exception:
            pass
class LiveBot(threading.Thread):
    def __init__(self, angel_client, config, expiry_getter, ui_logger, settings_getter=None):
        # Accept settings_getter via kwargs for runtime controls (will be injected during start)

        super().__init__(daemon=True)
        self.angel = angel_client
        self.config = config
        self.expiry_getter = expiry_getter
        self.stop_event = get_stop_event()
        self.trades = []
        self.ui_log = ui_logger
        # store last seen OI per symboltoken to compute change
        self.last_oi = {}


def _get_runtime_setting(self, key, default):

def _fetch_candles_either_ws_or_rest(self, instrument_record, exchange='NFO'):
    \"\"\"Fetch candles using websocket ticks when enabled in runtime settings; otherwise fall back to REST getCandleData.\"\"\"
    use_ws = False
    try:
        use_ws = bool(self._get_runtime_setting('use_websocket_ticks', False))
    except Exception:
        use_ws = False
    # Try websocket path
    if use_ws and WS_CLIENT_AVAILABLE:
        try:
            # Start websocket client if not present
            if not hasattr(self, 'ws_client') or self.ws_client is None:
                self.ws_client = SmartAPIWebsocketClient(self.angel)
            # subscribe to instrument token
            token = instrument_record.get('symboltoken') or instrument_record.get('token') or instrument_record.get('instrument_token')
            if token is None:
                raise RuntimeError('No symboltoken for websocket subscribe')
            self.ws_client.start([str(token)])
            self.ws_client.subscribe([str(token)])
            # collect ticks for up to 10 seconds to assemble short candles; this is a best-effort approach
            ticks = []
            start = time.time()
            while time.time() - start < 10:
                tick = self.ws_client.get_tick(timeout=1.0)
                if tick:
                    ticks.append(tick)
            # attempt to convert ticks into candles (best-effort, depends on tick format)
            # Expected tick format: dict with 'last_price','open','high','low','close','volume','timestamp' or similar
            rows = []
            for t in ticks:
                if isinstance(t, dict):
                    # try to extract price and volume
                    ts = t.get('timestamp') or t.get('time') or t.get('t')
                    price = t.get('last_price') or t.get('ltp') or t.get('price') or t.get('l')
                    vol = t.get('volume') or t.get('v') or 0
                    if ts is None or price is None:
                        continue
                    rows.append({'timestamp': pd.to_datetime(ts), 'open': price, 'high': price, 'low': price, 'close': price, 'volume': vol})
            if len(rows) >= 3:
                df = pd.DataFrame(rows).dropna().reset_index(drop=True)
                return df
        except Exception as e:
            self.ui(f'[warn] websocket tick fetch failed, falling back to REST candles: {e}')
            try:
                if hasattr(self, 'ws_client') and self.ws_client:
                    self.ws_client.stop()
            except Exception:
                pass
    # REST fallback - use get_candle_data as before
    try:
        symboltoken = instrument_record.get('symboltoken') or instrument_record.get('token') or instrument_record.get('instrument_token')
        todate = datetime.datetime.now()
        fromdate = todate - datetime.timedelta(minutes=200)
        raw = self.angel.get_candle_data(exchange, symboltoken, interval='ONE_MINUTE', fromdate=dtstr(fromdate), todate=dtstr(todate))
        # parse as before
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
            raise RuntimeError('Unhandled candle data format.')
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        raise RuntimeError(f'Failed to fetch candles by REST: {e}')

    try:
        if callable(getattr(self, 'settings_getter', None)):
            s = self.settings_getter() or {}
            return s.get(key, default)
    except Exception:
        pass
    # fallback to config (older behavior) or default
    try:
        return self.config.get('trade_rules', {}).get(key, default)
    except Exception:
        return default

    def stop(self):
        self.stop_event.set()
        try:
            path = _export_trades_to_csv_helper(self.trades)
            if path:
                self.ui(f'[info] Trades exported to {path} on stop')
        except Exception:
            pass

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

    

# --- consolidated decision logic enforcing EMA/VWAP/OI/CPR (with reversal relaxation) ---
# Variables available: ind_df, latest, cprv, current_oi, prev_oi, oi_change_pct, sl, bias, oi_ok (maybe)
# Compute EMA relations and VWAP checks explicitly
ema9 = float(latest.get('ema9', 0.0)) if 'ema9' in latest.index else None
ema20 = float(latest.get('ema20', 0.0)) if 'ema20' in latest.index else None
vwap_val = float(latest.get('vwap', 0.0)) if 'vwap' in latest.index else None
price = float(latest.get('close', 0.0))
# EMA/VWAP rules
ema_vwap_ok = False
if ema9 is not None and ema20 is not None and vwap_val is not None:
    if bias == 'Bullish':
        ema_vwap_ok = (ema9 > ema20) and (ema9 > vwap_val)
    elif bias == 'Bearish':
        ema_vwap_ok = (ema9 < ema20) and (ema9 < vwap_val)
else:
    # conservative default: require EMA relation only if VWAP missing
    if ema9 is not None and ema20 is not None:
        if bias == 'Bullish': ema_vwap_ok = ema9 > ema20
        if bias == 'Bearish': ema_vwap_ok = ema9 < ema20

# OI change enforcement (explicit)
oi_threshold = float(self._get_runtime_setting('oi_change_threshold_percent', 0.0))
oi_ok_local = False
try:
    # oi_change_pct may be None if previous OI missing; require positive and above threshold
    if oi_change_pct is not None:
        oi_ok_local = (oi_change_pct > oi_threshold)
except Exception:
    oi_ok_local = False

# CPR logic (use previously computed cpr_pass via the reversal-aware block)
# Recompute CPR pass using existing config and reversal detection for safety
tr_cfg = self.config.get('trade_rules', {})
cpr_max_points = float(tr_cfg.get('cpr_max_width_points', 50.0))
cpr_max_factor = float(tr_cfg.get('cpr_max_width_factor_of_atr', 2.0))
cpr_relax_factor = float(tr_cfg.get('cpr_relax_factor', 2.0))
require_price_vs_tc_bc = bool(tr_cfg.get('cpr_require_price_vs_tc_bc', True))

cpr_width = float(cprv.get('width', 0.0))
atr_latest = float(latest.get('atr14', 0.0)) if 'atr14' in latest.index else 0.0

# detect recent EMA cross (reversal) and recent price crossing BC/TC
ema_diff = ind_df['ema9'] - ind_df['ema20'] if 'ema9' in ind_df.columns and 'ema20' in ind_df.columns else None
recent_cross = False
if ema_diff is not None and len(ema_diff) >= 4:
    try:
        last_signs = (ema_diff.iloc[-4:] > 0).astype(int)
        recent_cross = (last_signs.max() != last_signs.min())
    except Exception:
        recent_cross = False
price_cross_recent = False
try:
    closes = ind_df['close'].iloc[-4:]
    bc = float(cprv.get('BC', 0.0))
    tc = float(cprv.get('TC', 0.0))
    if len(closes) >= 2:
        crossed_bc = ((closes < bc).astype(int).diff().abs().sum() > 0)
        crossed_tc = ((closes < tc).astype(int).diff().abs().sum() > 0)
        price_cross_recent = bool(crossed_bc or crossed_tc)
except Exception:
    price_cross_recent = False

reversal_signal = recent_cross or price_cross_recent

effective_max_points = cpr_max_points * (cpr_relax_factor if reversal_signal else 1.0)
effective_max_factor = cpr_max_factor * (cpr_relax_factor if reversal_signal else 1.0)

cpr_ok_abs = (cpr_width <= effective_max_points)
cpr_ok_rel = True
if atr_latest and effective_max_factor > 0:
    cpr_ok_rel = (cpr_width <= (effective_max_factor * atr_latest))
cpr_ok = cpr_ok_abs and cpr_ok_rel

bc = float(cprv.get('BC', 0.0))
tc = float(cprv.get('TC', 0.0))

cpr_direction_ok = True
if require_price_vs_tc_bc:
    if bias == 'Bullish':
        if reversal_signal:
            cpr_direction_ok = (price >= bc) or cpr_ok
        else:
            cpr_direction_ok = (price > tc) or (price > bc and cpr_ok)
    elif bias == 'Bearish':
        if reversal_signal:
            cpr_direction_ok = (price <= tc) or cpr_ok
        else:
            cpr_direction_ok = (price < bc) or (price < tc and cpr_ok)
    else:
        cpr_direction_ok = False

cpr_pass = cpr_ok and cpr_direction_ok

# Final combined gate: EMA/VWAP ok AND OI ok AND CPR pass
final_decision = None
if bias == 'Bullish' and ema_vwap_ok and oi_ok_local and cpr_pass:
    final_decision = {'type': 'BUY_CALL', 'sl': sl, 'oi_change_pct': oi_change_pct, 'cpr_width': cpr_width, 'reversal_relaxed': reversal_signal}
elif bias == 'Bearish' and ema_vwap_ok and oi_ok_local and cpr_pass:
    final_decision = {'type': 'BUY_PUT', 'sl': sl, 'oi_change_pct': oi_change_pct, 'cpr_width': cpr_width, 'reversal_relaxed': reversal_signal}
else:
    final_decision = None

decision = final_decision

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
            while not get_stop_event().is_set():
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
                    self.ui(f"[info] ATM strike: {atm_strike}")
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


def _export_trades_to_csv_helper(trades):
    try:
        import pandas as _pd, os as _os, datetime as _dt
        if not trades:
            return None
        df = _pd.json_normalize(trades[::-1])
        fname = f"trades_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = _os.path.join('/mnt/data', fname)
        df.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        return None


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

# --- Runtime tuning controls (no JSON edits) ---
st.markdown('---')
st.subheader('Runtime gates (live tuning)')
# OI threshold percent (default 0.0)
oi_thr = st.slider('OI increase threshold %', -50.0, 200.0, 0.0, step=0.5, help='Require OI percent change greater than this to allow trade.')
cpr_points = st.slider('CPR max width (points)', 0, 500, 50, step=1, help='Absolute CPR width cap.')
cpr_factor = st.slider('CPR max-factor of ATR', 0.1, 10.0, 2.0, step=0.1, help='Relative cap: CPR width <= factor * ATR.')
relax_toggle = st.checkbox('Relax CPR on reversal', value=False)
relax_factor = st.slider('CPR relax factor', 1.0, 5.0, 2.0, step=0.1, help='When reversal detected, CPR thresholds multiplied by this.')
apply_runtime = st.button('Apply runtime settings')
if 'applied_runtime_settings' not in st.session_state:
    st.session_state['applied_runtime_settings'] = {'oi_change_threshold_percent': 0.0, 'cpr_max_width_points': 50.0, 'cpr_max_width_factor_of_atr': 2.0, 'cpr_relax_factor': 2.0, 'cpr_require_price_vs_tc_bc': True}
if apply_runtime:
    st.session_state['applied_runtime_settings'] = {
            'oi_change_threshold_percent': float(oi_thr),
            'cpr_max_width_points': int(cpr_points),
            'cpr_max_width_factor_of_atr': float(cpr_factor),
            'cpr_relax_factor': float(relax_factor),
            'cpr_require_price_vs_tc_bc': True,
            'use_websocket_ticks': bool(use_ws)
        }
    st.success('Applied runtime settings.')
# show current applied settings
st.write('Active runtime settings:', st.session_state.get('applied_runtime_settings'))

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

# Auto-start behavior: start bot automatically in Paper mode unless Live selected and credentials exist.
_auto_start_on_launch = True
if _auto_start_on_launch and (st.session_state.get('bot') is None or not getattr(st.session_state.get('bot'), 'is_alive', lambda: False)()):
    try:
        # reuse the same start logic: create angel_client only if Live selected
        angel_client = None
        paper = (mode == 'Paper')
        if mode == 'Live' and SMARTAPI_AVAILABLE and os.getenv('API_KEY') and os.getenv('CLIENT_ID') and os.getenv('PASSWORD'):
            try:
                angel_client = AngelClient(os.getenv('API_KEY'), os.getenv('CLIENT_ID'), os.getenv('PASSWORD'), master_password=os.getenv('MASTER_PASSWORD'), totp_secret=os.getenv('TOTP'))
                angel_client.connect()
            except Exception as e:
                st.session_state['log_msgs'].append(f"{datetime.datetime.now().isoformat()} - [warn] Auto-start live connection failed: {e}")
                angel_client = None
                paper = True
        executor = OrderExecutor(api_client=(angel_client.client if angel_client else None), paper_mode=paper) if 'OrderExecutor' in globals() else None
        bot_thread = LiveBot(angel_client, config, expiry_getter, ui_logger, settings_getter=lambda: st.session_state.get('applied_runtime_settings', {}))
        st.session_state['bot'] = bot_thread
        bot_thread.start()
        st.session_state['log_msgs'].append(f"{datetime.datetime.now().isoformat()} - [info] Bot auto-started in {'Live' if angel_client and getattr(angel_client,'logged_in',False) else 'Paper'} mode.")
    except Exception as e:
        st.session_state['log_msgs'].append(f"{datetime.datetime.now().isoformat()} - [error] Auto-start failed: {e}")

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
        bot_thread = LiveBot(angel_client, config, expiry_getter, ui_logger, settings_getter=lambda: st.session_state.get('applied_runtime_settings', {}))
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


# --- Export trades button (writes CSV to disk and shows download link) ---
def _export_trades_to_csv():
    bot = st.session_state.get('bot')
    if not bot or not getattr(bot, 'trades', None):
        st.warning('No trades to export.')
        return None
    import pandas as _pd, os as _os, datetime as _dt
    df = _pd.json_normalize(bot.trades[::-1])
    fname = f"trades_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = _os.path.join('/mnt/data', fname)
    df.to_csv(out_path, index=False)
    return out_path

with st.sidebar:
    st.markdown('---')
    if st.button('Export trades to CSV'):
        path = _export_trades_to_csv()
        if path:
            st.success('Trades exported: ' + path)
            st.markdown(f'[Download exported trades]({path})')
