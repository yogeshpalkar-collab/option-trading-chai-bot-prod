
# options_trading_bot_angel.py - Clean consolidated app with SmartAPI websocket auto-subscribe to NIFTY ATM
import streamlit as st
import time, math, re, json, traceback
from collections import deque

st.set_page_config(page_title="Options Trading Bot (Angel) - ATM WS", layout="wide")

# ----------------- Utilities -----------------
def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default
def norm_sym(s):
    return str(s).upper() if s is not None else ""

# ----------------- Indicator engine (minimal for ATR) -----------------
class IndicatorEngine:
    def __init__(self):
        self._ema_short = None
        self._ema_long = None
        self._atr = None
        self._prev_close = None
        self._atr_alpha = 2.0 / (14 + 1)
    def update_tick(self, price, high=None, low=None, close=None, volume=0, oi=None, timestamp=None):
        try:
            p = float(price)
        except Exception:
            return
        if self._ema_short is None:
            self._ema_short = p; self._ema_long = p
        else:
            alpha_s = 2.0/(9+1); alpha_l = 2.0/(21+1)
            self._ema_short = alpha_s * p + (1-alpha_s)*self._ema_short
            self._ema_long = alpha_l * p + (1-alpha_l)*self._ema_long
        # ATR approximate using high/low/prev_close if given
        if high is not None and low is not None:
            tr = max(high - low, abs(high - (self._prev_close if self._prev_close is not None else close or p)), abs(low - (self._prev_close if self._prev_close is not None else close or p)))
            if self._atr is None:
                self._atr = tr
            else:
                self._atr = self._atr_alpha * tr + (1-self._atr_alpha) * self._atr
            if close is not None:
                self._prev_close = close
    def atr(self): return self._atr
    def ema_short(self): return self._ema_short
    def ema_long(self): return self._ema_long

# ----------------- Lot size & instruments helpers -----------------
def get_instrument_master(conn):
    """Best-effort: retrieve instrument list/dictionary from SmartAPI client"""
    if conn is None: return None
    candidates = ["get_instruments", "get_instrument_master", "getInstrumentMaster", "get_instruments_master", "instruments", "get_master_contracts", "get_master"]
    for name in candidates:
        fn = getattr(conn, name, None)
        if callable(fn):
            try:
                res = fn()
                # normalize dict/list
                if isinstance(res, dict) and res.get("data"):
                    return res.get("data")
                if isinstance(res, list):
                    return res
                # sometimes returns dict with instruments key
                if isinstance(res, dict) and res.get("instruments"):
                    return res.get("instruments")
            except Exception:
                continue
    return None

def get_lot_size_from_item(item):
    """Try common keys in an instrument item to extract lot size"""
    if not isinstance(item, dict): return None
    for k in ("lotsize","lotSize","lot_size","contractSize","contract_size","lot","LotSize","quantity","qty","multiplier"):
        if k in item and item[k]:
            try:
                return int(item[k])
            except Exception:
                try:
                    return int(float(item[k]))
                except Exception:
                    pass
    # nested meta
    meta = item.get("meta") or item.get("instrument") or item.get("details") or {}
    if isinstance(meta, dict):
        for k in ("lotsize","lotSize","lot_size","contractSize","contract_size","lot","quantity"):
            if k in meta and meta[k]:
                try:
                    return int(meta[k])
                except Exception:
                    try: return int(float(meta[k]))
                    except Exception: pass
    return None

def find_atm_instrument(conn, underlying_symbol="NIFTY", option_type_preference="CE"):
    """
    Find nearest ATM option instrument for underlying_symbol.
    Returns dict with keys: tradingsymbol, token, lot_size, strike, expiry, option_type
    Uses instrument master from API.
    """
    master = get_instrument_master(conn)
    if not master:
        return None
    # Get current underlying spot price from API if available
    spot = None
    # try to find nifty underlying quote via conn (best-effort)
    try:
        # try common methods
        for fn in ("get_quote", "getLTP", "ltp", "quote", "get_quote_data"):
            f = getattr(conn, fn, None)
            if callable(f):
                try:
                    # best-effort call
                    res = f(underlying_symbol)
                    if isinstance(res, dict):
                        # try typical fields
                        for key in ("lastprice","lastPrice","ltp","lastTradedPrice","close"):
                            if res.get(key) is not None:
                                spot = safe_float(res.get(key)); break
                    break
                except Exception:
                    continue
    except Exception:
        pass
    # fallback to scanning master for underlying 'NIFTY' underlying entries and compute median strike
    strikes = []
    candidates = []
    for item in master:
        try:
            ts = str(item.get("tradingsymbol") or item.get("symbol") or item.get("instrument") or "")
            if underlying_symbol.upper() in ts.upper() and ("CE" in ts.upper() or "PE" in ts.upper()):
                candidates.append(item)
                # extract strike number from tradingsymbol by regex
                m = re.search(r"(\d{3,6})", ts)
                if m:
                    try:
                        strikes.append(int(m.group(1)))
                    except Exception:
                        pass
        except Exception:
            continue
    if not candidates:
        return None
    # if spot not found, approximate spot using median strike list or choose nearest current time-based ATM (last known)
    if spot is None and strikes:
        # try to approximate using average of strikes
        try:
            spot = float(sum(strikes)/len(strikes))
        except Exception:
            spot = strikes[0]
    # choose nearest strike to spot, prefer option_type_preference
    best = None; best_diff = 1e12
    for item in candidates:
        ts = str(item.get("tradingsymbol") or item.get("symbol") or "")
        m = re.search(r"(\d{3,6})", ts)
        if not m: continue
        try:
            strike = int(m.group(1))
        except Exception:
            continue
        diff = abs(strike - float(spot))
        # prefer option type (CE/PE)
        ot = "CE" if "CE" in ts.upper() else ("PE" if "PE" in ts.upper() else None)
        pref_bonus = 0 if ot == option_type_preference else 1000000
        score = diff + pref_bonus
        if score < best_diff:
            best_diff = score; best = (item, strike, ot)
    if not best: return None
    item, strike, ot = best
    token = item.get("token") or item.get("instrumentToken") or item.get("tokenId") or item.get("instrument_token") or item.get("token_no") or item.get("token_id") or None
    lot = get_lot_size_from_item(item) or get_lot_size_from_item(item.get("meta") or {})
    expiry = item.get("expiry") or item.get("expiryDate") or item.get("expiry_date") or None
    return {"tradingsymbol": str(item.get("tradingsymbol") or item.get("symbol") or ""), "token": token, "lot_size": lot, "strike": strike, "expiry": expiry, "option_type": ot, "raw": item}

# ----------------- Broker & AutoTrader (simplified) -----------------
class PaperBroker:
    def __init__(self):
        self.market_prices = {}  # symbol -> price
    def place_order(self, symbol, qty, side, price=None, order_type="IOC", immediate_fill=False):
        # simulate immediate fill at market price if known
        filled = False; fill_price = None
        if order_type == "IOC":
            p = self.market_prices.get(symbol)
            if p is not None:
                filled = True; fill_price = p
        return {"order_id": f"P{int(time.time())}", "symbol": symbol, "qty": qty, "side": side, "filled": filled, "fill_price": fill_price, "order_type": order_type}

class LiveBroker(PaperBroker):
    """
    LiveBroker wraps a SmartAPI client when available. This implementation attempts to call common SmartAPI
    order placement methods in a best-effort manner and falls back to PaperBroker behaviour if calls fail.
    """
    def __init__(self, conn=None):
        super().__init__()
        self.conn = conn

    def place_order(self, symbol, qty, side, price=None, order_type="IOC", immediate_fill=False):
        # Try to use SmartAPI client methods if available
        conn = self.conn or (__import__('streamlit').session_state.get('smart_conn') if 'streamlit' in globals() else None)
        if conn is not None:
            try:
                # Some SmartAPI variants expose a method 'place_order' or 'placeOrder' or under 'order' namespace.
                # We'll try a few method names and parameter shapes.
                candidate_methods = [
                    ('place_order', {'tradingsymbol': symbol, 'quantity': qty, 'transactiontype': side, 'price': price, 'ordertype': order_type}),
                    ('placeOrder', {'tradingsymbol': symbol, 'quantity': qty, 'transactiontype': side, 'price': price, 'ordertype': order_type}),
                    ('order_place', {'symbol': symbol, 'qty': qty, 'side': side, 'price': price}),
                    ('placeOrderV2', {'symbol': symbol, 'quantity': qty, 'side': side, 'price': price}),
                ]
                for name, params in candidate_methods:
                    fn = getattr(conn, name, None)
                    if callable(fn):
                        try:
                            # Try calling with kwargs first
                            try:
                                res = fn(**{k:v for k,v in params.items() if v is not None})
                            except TypeError:
                                # fallback to positional
                                res = fn(symbol, qty, side, price)
                            # If result looks like an order dict, return it
                            if isinstance(res, dict) and res:
                                return res
                            # otherwise wrap in a dict
                            return {"order_id": str(res), "raw": res}
                        except Exception:
                            continue
                # Some clients expose 'placeOrder' under conn.order or similar
                if hasattr(conn, "placeOrder") and callable(getattr(conn, "placeOrder")):
                    try:
                        res = conn.placeOrder(symbol, qty, side, price)
                        return {"order_id": str(res), "raw": res}
                    except Exception:
                        pass
                # Try a generic 'order' attribute with 'placeOrder' method
                order_ns = getattr(conn, 'order', None) or getattr(conn, 'orders', None)
                if order_ns and hasattr(order_ns, 'placeOrder'):
                    try:
                        res = order_ns.placeOrder(symbol, qty, side, price)
                        return {"order_id": str(res), "raw": res}
                    except Exception:
                        pass
            except Exception:
                pass
        # Fallback to PaperBroker behaviour (simulate IOC immediate fill if market price known)
        return super().place_order(symbol, qty, side, price=price, order_type=order_type, immediate_fill=immediate_fill)

    def query_order(self, order_id):
        conn = self.conn or (__import__('streamlit').session_state.get('smart_conn') if 'streamlit' in globals() else None)
        if conn is not None:
            for name in ("query_order", "queryOrder", "get_order", "order_status", "getOrderById"):
                fn = getattr(conn, name, None)
                if callable(fn):
                    try:
                        return fn(order_id)
                    except Exception:
                        continue
        return super().query_order(order_id)



class SmartAPIProxy:
    """
    Proxy that mimics SmartAPI client methods for dry-run / paper mode.
    Methods mirror common SmartAPI method names used by LiveBroker, but do NOT send network requests.
    Returns structures similar to SmartAPI so code paths are exercised identically.
    """
    def __init__(self, underlying_conn=None):
        self.underlying = underlying_conn  # real conn optional (not used for network in dry-run)
        self._orders = {}
        self._next = 1

    def place_order(self, **kwargs):
        # Accept kwargs or positional style via compatibility
        oid = f"PAPERPROXY{int(time.time())}{self._next}"; self._next += 1
        # Compose an order-like dict
        order = {"order_id": oid, "status": "filled", "filled": True, "fill_price": kwargs.get("price") or None, "raw": {"proxy": True, "request": kwargs}}
        self._orders[oid] = order
        return order

    def placeOrder(self, *args, **kwargs):
        # map positional args to a dict if possible
        try:
            if args and len(args) >= 3:
                symbol, qty, side = args[0], args[1], args[2]
                return self.place_order(tradingsymbol=symbol, quantity=qty, transactiontype=side, **kwargs)
        except Exception:
            pass
        return self.place_order(**kwargs)

    def order_place(self, *args, **kwargs):
        return self.placeOrder(*args, **kwargs)

    def query_order(self, order_id):
        return self._orders.get(order_id) or {"order_id": order_id, "status": "unknown"}

    def queryOrder(self, order_id):
        return self.query_order(order_id)

    def get_order(self, order_id):
        return self.query_order(order_id)

    # Minimal getfeedToken pass-through
    def getfeedToken(self):
        if self.underlying and hasattr(self.underlying, 'getfeedToken'):
            try:
                return self.underlying.getfeedToken()
            except Exception:
                pass
        return None

class AutoTrader:
    def __init__(self, broker):
        self.broker = broker
        self.positions = []
    
    def open_position(self, symbol, side, entry_price, lots, lot_size, sl_price, tp_price):
        # Create and register a new position; tag with whether it was opened via PROXY or LIVE API
        pos = {"order_id": f"pos{int(time.time())}", "symbol": symbol, "side": side, "entry_price": float(entry_price), "lots": lots, "lot_size": lot_size, "qty": int(lots*lot_size), "sl_price": float(sl_price) if sl_price is not None else None, "tp_price": float(tp_price) if tp_price is not None else None, "locked": False, "trailing": False, "highest_price": float(entry_price), "lowest_price": float(entry_price), "active": True}
        # detect whether broker.conn is a proxy
        try:
            br = getattr(self, 'broker', None)
            conn = getattr(br, 'conn', None) if br is not None else None
            via = "LIVE API"
            try:
                # if SmartAPIProxy class exists in globals, use isinstance check; else fallback to attr check
                if conn is not None and globals().get('SmartAPIProxy') is not None and isinstance(conn, SmartAPIProxy):
                    via = "PROXY (PAPER)"
                elif conn is not None and getattr(conn, '__class__', None) and getattr(conn.__class__, '__name__', '').lower().find('proxy')!=-1:
                    via = "PROXY (PAPER)"
            except Exception:
                pass
            pos['order_via'] = via
        except Exception:
            pos['order_via'] = "UNKNOWN"
        self.positions.append(pos)
        try:
            import streamlit as _st
            _st.session_state.setdefault('logs', []).append(f"Opened position {pos['order_id']} {side} {symbol} @ {entry_price} VIA {pos.get('order_via')}")
        except Exception:
            pass
        return pos

    def close_position(self, pos, price):
        try:
            pos["active"] = False; pos["closed_at"] = price
            st.session_state.setdefault('logs', []).append(f"Closed {pos['order_id']} @ {price}")
        except Exception:
            pass
    def monitor_positions(self, market_map, lock_buffer=0.5):
        for p in list(self.positions):
            if not p.get("active", True): continue
            sym = p["symbol"]; m = market_map.get(sym)
            if m is None: continue
            # update high/low
            if p["side"].upper()=="BUY": p["highest_price"] = max(p.get("highest_price", m), m)
            else: p["lowest_price"] = min(p.get("lowest_price", m), m)
            # compute profit per lot
            if p["side"].upper()=="BUY": profit_per_lot = (m - p["entry_price"]) * p["lot_size"]
            else: profit_per_lot = (p["entry_price"] - m) * p["lot_size"]
            # lock TP
            if not p.get("locked", False) and p.get("tp_price") is not None:
                if p["side"].upper()=="BUY" and m >= p["tp_price"]:
                    p["locked"] = True; p["sl_price"] = p["tp_price"] - lock_buffer; p["trailing"] = True
                if p["side"].upper()=="SELL" and m <= p["tp_price"]:
                    p["locked"] = True; p["sl_price"] = p["tp_price"] + lock_buffer; p["trailing"] = True
            # dynamic trailing
            if p.get("trailing", False):
                try:
                    if profit_per_lot < 20: base_step = 3.0
                    elif profit_per_lot < 30: base_step = 6.0
                    else: base_step = 10.0
                except Exception:
                    base_step = 3.0
                step = base_step
                try:
                    eng = st.session_state.get('indicator_engine')
                    atr = eng.atr() if eng and hasattr(eng,'atr') else None
                    if atr is not None: step = max(base_step, float(atr)*0.4)
                except Exception:
                    pass
                if p["side"].upper()=="BUY":
                    new_sl = p.get("highest_price", m) - step
                    if new_sl > p.get("sl_price", -1e9): p["sl_price"] = new_sl
                else:
                    new_sl = p.get("lowest_price", m) + step
                    if new_sl < p.get("sl_price", 1e9): p["sl_price"] = new_sl
            # check SL
            if p["side"].upper()=="BUY" and m <= p.get("sl_price", -1e9):
                self.close_position(p, m)
            if p["side"].upper()=="SELL" and m >= p.get("sl_price", 1e9):
                self.close_position(p, m)

# ----------------- Session init -----------------
st.session_state.setdefault('paper_broker', PaperBroker())

# --- Auto-run smoke test once per session (Paper mode only) ---
try:
    if not st.session_state.get('smoke_auto_ran', False):
        # Only auto-run in Paper mode to avoid accidental live orders
        try:
            default_mode = st.session_state.get('mode_override') if st.session_state.get('mode_override') else None
        except Exception:
            default_mode = None
        # If UI 'mode' variable exists, prefer it; otherwise assume Paper
        try:
            ui_mode = locals().get('mode', None)
        except Exception:
            ui_mode = None
        run_in_paper = True
        if ui_mode is not None:
            run_in_paper = (ui_mode == 'Paper')
        # Run auto-smoke only if Paper
        if run_in_paper:
            st.session_state.setdefault('logs', []).append(f"AUTO-SMOKE START {time.strftime('%X')}")
            # attempt to reuse the smoke button logic to place one deterministic entry
            try:
                # determine symbol: prefer ATM subscription, else last tick or default
                atm_sym = None
                try:
                    subs = st.session_state.get('ws_subscriptions', [])
                    if isinstance(subs, list) and subs:
                        atm_sym = subs[-1]
                    elif isinstance(subs, str) and subs:
                        atm_sym = subs
                except Exception:
                    atm_sym = None
                if not atm_sym:
                    atm_sym = st.session_state.get('last_tick', {}).get('symbol') or 'NIFTY24000CE'
                # lot size from cache or API proxy
                lot_map = st.session_state.get('lot_size_map', {})
                lot = lot_map.get(atm_sym) or lot_map.get(str(atm_sym).upper()) or None
                if not lot:
                    try:
                        conn = st.session_state.get('smart_conn', None)
                        inst = None
                        if globals().get('find_atm_instrument'):
                            try:
                                inst = find_atm_instrument(conn, underlying_symbol='NIFTY')
                            except Exception:
                                inst = None
                        if inst and isinstance(inst, dict) and inst.get('lot_size'):
                            lot = inst.get('lot_size')
                            st.session_state.setdefault('lot_size_map', {})[atm_sym] = int(lot)
                    except Exception:
                        lot = None
                if not lot:
                    st.session_state['logs'].append("AUTO-SMOKE ABORT: lot size missing for " + str(atm_sym))
                else:
                    aut = st.session_state.get('autotrader')
                    if aut is None:
                        st.session_state['logs'].append("AUTO-SMOKE ABORT: autotrader missing")
                    else:
                        try:
                            filled = bool(order.get('filled') or (isinstance(order.get('status'), str) and order.get('status').lower()=='filled') or (isinstance(order.get('raw'), dict) and order.get('raw').get('proxy')==True))
                        except Exception:
                            filled = False
                        entry_price = None
                        try:
                            entry_price = order.get('fill_price') or order.get('fillPrice') or (order.get('raw',{}) or {}).get('request',{}).get('price') or st.session_state.get('paper_broker').market_prices.get(atm_sym) or None
                        except Exception:
                            entry_price = None
                        if not entry_price:
                            entry_price = st.session_state.get('last_tick', {}).get('ltp') or 0.0
                        if filled:
                            st.session_state['logs'].append(f"AUTO-SMOKE: order filled at {entry_price}; opening position")
                            aut.open_position(order, "BUY", atm_sym, float(entry_price), lots=4, lot_size=int(lot), sl_price=float(entry_price)-20.0, tp_price=float(entry_price)+(10.0/float(lot)))
                        else:
                            st.session_state['logs'].append("AUTO-SMOKE: order not filled; opening simulated position for lifecycle test")
                            aut.open_position({"order_id":"SIMAUTO"+str(int(time.time()))}, "BUY", atm_sym, float(entry_price), lots=4, lot_size=int(lot), sl_price=float(entry_price)-20.0, tp_price=float(entry_price)+(10.0/float(lot)))
                        # call monitor once
                        try:
                            broker_map = broker.market_prices if hasattr(broker,'market_prices') else {}
                            aut.monitor_positions(broker_map, lock_buffer=st.session_state.get('lock_buffer', 0.5))
                            st.session_state['logs'].append("AUTO-SMOKE: monitor_positions executed")
                        except Exception as e:
                            st.session_state['logs'].append("AUTO-SMOKE: monitor error: " + str(e))
            except Exception as e:
                st.session_state['logs'].append("AUTO-SMOKE ERROR: " + str(e))
        st.session_state['smoke_auto_ran'] = True
except Exception:
    pass
st.session_state.setdefault('live_broker', LiveBroker(conn=None))
st.session_state.setdefault('autotrader', AutoTrader(st.session_state['paper_broker']))
st.session_state.setdefault('indicator_engine', IndicatorEngine())
st.session_state.setdefault('ws_messages', [])

# ----------------- UI -----------------
st.title("Options Trading Bot — ATM WebSocket Auto-subscribe")
col1, col2 = st.columns([3,1])
with col2:
    st.markdown("### Controls")
    smart_api_key = st.text_input("SmartAPI API Key", type="password")
    smart_client_id = st.text_input("SmartAPI Client ID")
    smart_password = st.text_input("SmartAPI Password", type="password")
    connect = st.button("Connect SmartAPI")
    disconnect = st.button("Disconnect SmartAPI")
    st.markdown("---")
    st.markdown("### WebSocket")
    feed_token = st.text_input("Feed token (leave blank to auto)", value="")
    ws_task = st.selectbox("Feed task", ["mw","sfi","dp"], index=0)
    connect_ws = st.button("Connect WebSocket (auto-sub ATM)")

# --- Smoke test: place one deterministic entry and show lifecycle logs ---
smoke = st.button("Run smoke test (place 1 entry and show log)")
if smoke:
    try:
        st.session_state.setdefault('logs', [])
        st.session_state['logs'].append(f"SMOKE TEST START {time.strftime('%X')}")
        # determine symbol: prefer ATM subscription, else symbol input
        atm_sym = None
        try:
            subs = st.session_state.get('ws_subscriptions', [])
            if isinstance(subs, list) and subs:
                atm_sym = subs[-1]  # last subscribed
            elif isinstance(subs, str) and subs:
                atm_sym = subs
        except Exception:
            atm_sym = None
        if not atm_sym:
            atm_sym = st.session_state.get('last_tick', {}).get('symbol') or globals().get('symbol', None) or 'NIFTY24000CE'
        # determine lot size (from cache or API)
        lot_map = st.session_state.get('lot_size_map', {})
        lot = lot_map.get(atm_sym) or lot_map.get(str(atm_sym).upper()) or None
        if not lot:
            # try to find ATM via API
            try:
                conn = st.session_state.get('smart_conn', None)
                inst = None
                try:
                    from inspect import getsource
                    # use find_atm_instrument if available
                    inst = globals().get('find_atm_instrument') and find_atm_instrument(conn, underlying_symbol='NIFTY') or None
                except Exception:
                    inst = None
                if inst and isinstance(inst, dict) and inst.get('lot_size'):
                    lot = inst.get('lot_size')
                    st.session_state.setdefault('lot_size_map', {})[atm_sym] = int(lot)
            except Exception:
                lot = None
        if not lot:
            st.error("SMOKE TEST ABORT: lot size not found for symbol " + str(atm_sym))
            st.session_state['logs'].append("SMOKE TEST ABORT: lot size missing")
        else:
            # place order via autotrader.broker (uses proxy in Paper mode)
            aut = st.session_state.get('autotrader')
            if aut is None:
                st.error("AutoTrader not initialized")
                st.session_state['logs'].append("SMOKE TEST ABORT: autotrader missing")
            else:
                # if order filled or proxy returns filled, open position
                try:
                    filled = bool(order.get('filled') or order.get('status')=='filled' or order.get('raw',{}).get('proxy')==True)
                except Exception:
                    filled = False
                entry_price = None
                try:
                    entry_price = order.get('fill_price') or order.get('fillPrice') or order.get('raw',{}).get('request',{}).get('price') or st.session_state.get('paper_broker').market_prices.get(atm_sym) or None
                except Exception:
                    entry_price = None
                if not entry_price:
                    # fallback to last_tick ltp
                    entry_price = st.session_state.get('last_tick', {}).get('ltp') or 0.0
                if filled:
                    st.session_state['logs'].append(f"SMOKE: order filled at {entry_price}; opening position")
                    pos = aut.open_position(order, "BUY", atm_sym, float(entry_price), lots=4, lot_size=int(lot), sl_price=float(entry_price)-20.0, tp_price=float(entry_price)+(10.0/float(lot)))
                    st.write("Opened position:", pos)
                else:
                    st.session_state['logs'].append("SMOKE: order not filled (simulated), opening position anyway for lifecycle test")
                    # open position anyway to test lifecycle
                    pos = aut.open_position({"order_id":"SIMSMOKE"+str(int(time.time()))}, "BUY", atm_sym, float(entry_price), lots=4, lot_size=int(lot), sl_price=float(entry_price)-20.0, tp_price=float(entry_price)+(10.0/float(lot)))
                    st.write("Opened simulated position:", pos)
                # call monitor once to exercise trailing/SL
                try:
                    broker_map = broker.market_prices if hasattr(broker, 'market_prices') else {}
                    aut.monitor_positions(broker_map, lock_buffer=st.session_state.get('lock_buffer', 0.5))
                    st.session_state['logs'].append("SMOKE: monitor_positions executed")
                except Exception as e:
                    st.session_state['logs'].append("SMOKE: monitor_positions error: " + str(e))
                st.success("Smoke test completed; check logs and TSL table")
    except Exception as e:
        st.error("Smoke test failed: " + str(e))
    disconnect_ws = st.button("Disconnect WebSocket")
    st.markdown("---")
    st.write("Logs:"); st.write(st.session_state.get('logs', [])[-5:])

with col1:
    st.markdown("### Debug")
    st.write("Last tick:", st.session_state.get('last_tick'))
    st.write("Detected subscriptions:", st.session_state.get('ws_subscriptions'))
    st.write("WS messages (recent):")
    msgs = st.session_state.get('ws_messages', [])[:20]
    for m in msgs:
        ts = time.strftime("%H:%M:%S", time.localtime(m.get('ts', time.time())))
        st.write(f"{ts} {str(m.get('raw'))[:200]}")

# ----------------- SmartAPI connection logic -----------------
if connect:
    try:
        SmartConnect = None
        try:
            from smartapi import SmartConnect as SmartConnect1
            SmartConnect = SmartConnect1
        except Exception:
            try:
                from SmartApi import SmartConnect as SmartConnect2
                SmartConnect = SmartConnect2
            except Exception:
                SmartConnect = None
        if SmartConnect is None:
            st.error("smartapi client not installed in environment. Install smartapi-python package.")
        else:
            try:
                # Attempt typical constructor patterns
                try:
                    conn = SmartConnect(api_key=smart_api_key)
                except TypeError:
                    try:
                        conn = SmartConnect(smart_api_key)
                    except Exception:
                        conn = SmartConnect(api_key=smart_api_key)
                # attempt session creation if method exists
                try:
                    if hasattr(conn, 'generateSession') and smart_client_id and smart_password:
                        conn.generateSession(smart_client_id, smart_password)
                except Exception:
                    pass
                st.session_state['smart_conn'] = conn
                st.success("SmartAPI connection object stored.")
            except Exception as e:
                st.error(f"SmartAPI connect failed: {e}")
    except Exception as e:
        st.error(f"Connect error: {e}")

if disconnect:
    st.session_state.pop('smart_conn', None); st.success("Disconnected SmartAPI session object.")

# ----------------- WebSocket wiring -----------------
def _on_message(ws, message):
    try:
        payload = message
        if isinstance(message, (bytes, bytearray)):
            try: payload = message.decode('utf-8')
            except Exception: payload = str(message)
        st.session_state.setdefault('ws_messages', []).insert(0, {'ts': time.time(), 'raw': payload})
        # keep max 200
        if len(st.session_state['ws_messages'])>200: st.session_state['ws_messages'].pop()
        # try parse JSON
        data = None
        try:
            data = json.loads(payload)
        except Exception:
            data = payload
        st.session_state['last_tick'] = data
        # extract ticks and update market_prices
        ticks = []
        if isinstance(data, dict) and data.get('data') and isinstance(data.get('data'), list):
            ticks = data.get('data')
        elif isinstance(data, list):
            ticks = data
        elif isinstance(data, dict) and ('ltp' in data or 'lastPrice' in data):
            ticks = [data]
        for it in ticks:
            try:
                sym = it.get('symbol') or it.get('tradingsymbol') or it.get('instrument') or it.get('token') or it.get('instrumentToken') or it.get('t')
                ltp = it.get('ltp') or it.get('lastPrice') or it.get('price')
                if sym and ltp is not None:
                    try:
                        pb = st.session_state.get('paper_broker')
                        if pb and hasattr(pb,'market_prices'):
                            pb.market_prices[str(sym)] = float(ltp)
                    except Exception:
                        pass
            except Exception:
                pass
        # route to autotrader for immediate fills
        try:
            aut = st.session_state.get('autotrader')
            if aut:
                broker = aut.broker
                market_map = {}
                try: market_map = broker.market_prices if hasattr(broker,'market_prices') else {}
                except Exception: market_map = {}
                aut.monitor_positions(market_map, lock_buffer=st.session_state.get('lock_buffer', 0.5))
        except Exception:
            pass
    except Exception:
        pass

# Connect WebSocket and auto-subscribe ATM logic
if connect_ws:
    try:
        SmartWebSocket = None
        try:
            from smartapi import SmartWebSocket as SmartWebSocket1
            SmartWebSocket = SmartWebSocket1
        except Exception:
            try:
                from SmartApi.smartWebSocket import SmartWebSocket as SmartWebSocket2
                SmartWebSocket = SmartWebSocket2
            except Exception:
                SmartWebSocket = None
        if SmartWebSocket is None:
            st.error("SmartWebSocket class not available in installed SmartAPI package.")
        else:
            conn = st.session_state.get('smart_conn', None)
            # determine feed token: if user provided feed_token use it else try conn.getfeedToken()
            ft = feed_token or None
            if not ft and conn is not None:
                try:
                    if hasattr(conn, 'getfeedToken'):
                        res = conn.getfeedToken()
                        if isinstance(res, dict) and res.get('data'): ft = res.get('data')
                        else: ft = res
                except Exception:
                    ft = ft
            # create ws object
            try:
                wsobj = SmartWebSocket(ft, st.session_state.get('smart_credentials', {}).get('client_id', None))
            except Exception as e:
                st.error(f"Failed to create SmartWebSocket: {e}"); wsobj = None
            if wsobj is not None:
                # attach handler
                try:
                    wsobj.on_message = _on_message
                except Exception:
                    try:
                        wsobj.set_on_message(_on_message)
                    except Exception:
                        pass
                st.session_state['smart_ws'] = wsobj
                # find ATM instrument and auto-subscribe
                try:
                    conn = st.session_state.get('smart_conn', None)
                    atm = find_atm_instrument(conn, underlying_symbol="NIFTY", option_type_preference="CE")
                    sub = ""
                    if atm and atm.get('tradingsymbol'):
                        sub = str(atm.get('tradingsymbol'))
                        st.session_state.setdefault('ws_subscriptions', []).append(sub)
                        # if ws has subscribe method, call it; else send basic message
                        try:
                            if hasattr(wsobj,'subscribe'):
                                wsobj.subscribe(sub)
                            elif hasattr(wsobj,'send'):
                                try:
                                    wsobj.send(json.dumps({"task": ws_task, "type": "subscribe", "tokens": sub}))
                                except Exception:
                                    try: wsobj.send(sub)
                                    except Exception: pass
                        except Exception:
                            pass
                        st.success(f"Auto-subscribed to ATM: {sub}")
                        # also cache detected lot size
                        try:
                            if atm.get('lot_size'):
                                st.session_state.setdefault('lot_size_map', {})[sub] = int(atm.get('lot_size'))
                        except Exception:
                            pass
                    else:
                        st.warning("ATM instrument not found via API; please provide subscribe token manually.")
                except Exception as e:
                    st.warning(f"Auto-subscribe/ATM lookup failed: {e}")
                # attempt to start ws
                try:
                    if hasattr(wsobj,'connect'):
                        wsobj.connect()
                    elif hasattr(wsobj,'run_forever'):
                        wsobj.run_forever()
                    elif hasattr(wsobj,'start'):
                        wsobj.start()
                    st.success("WebSocket started (best-effort).")
                except Exception as e:
                    st.warning(f"Starting websocket raised: {e}")
    except Exception as e:
        st.error(f"connect_ws error: {e}")

if disconnect_ws:
    try:
        ws = st.session_state.pop('smart_ws', None)
        st.session_state.pop('ws_subscriptions', None)
        if ws is not None:
            try:
                if hasattr(ws,'close'): ws.close()
                elif hasattr(ws,'disconnect'): ws.disconnect()
            except Exception:
                pass
        st.success("WebSocket disconnected.")
    except Exception as e:
        st.error(f"disconnect error: {e}")

# ----------------- Simple manual entry & demo tick -----------------
st.markdown('### Manual / Demo')
colA, colB = st.columns(2)
with colA:
    symbol = st.text_input("Symbol", value="NIFTY24000CE")
    lots = st.number_input("Lots", value=4, min_value=4, max_value=4, step=1, disabled=True)
    place = st.button("Place demo position")
with colB:
    inject = st.button("Inject demo tick for symbol")
    demo_price = st.number_input("Demo price", value=100.0)
if place:
    # require lot size from cache or API
    lot = st.session_state.get('lot_size_map', {}).get(symbol)
    if not lot:
        conn = st.session_state.get('smart_conn')
        inst = find_atm_instrument(conn, underlying_symbol="NIFTY")
        lot = inst.get('lot_size') if inst else None
    if not lot:
        st.error("No lot size; cannot place position.")
    else:
        entry = st.session_state.get('paper_broker').market_prices.get(symbol, demo_price)
        tp = entry + (10.0 / float(lot))
        sl = entry - 20.0
        st.session_state['autotrader'].open_position(symbol, "BUY", entry, lots, lot, sl, tp)
        st.success(f"Placed demo position {symbol} @ {entry} lots={lots} lot_size={lot}")
if inject:
    # inject tick into engine and broker maps and call autotrader.monitor_positions
    st.session_state['paper_broker'].market_prices[symbol] = float(demo_price)
    st.session_state['last_tick'] = {"symbol": symbol, "ltp": float(demo_price)}
    try:
        st.session_state['indicator_engine'].update_tick(price=demo_price, high=demo_price, low=demo_price, close=demo_price, volume=1, oi=0, timestamp=time.time())
    except Exception:
        pass
    try:
        st.session_state['autotrader'].monitor_positions(st.session_state['paper_broker'].market_prices, lock_buffer=st.session_state.get('lock_buffer', 0.5))
    except Exception:
        pass

# ----------------- TSL table -----------------
st.markdown("### TSL & Profit table")
try:
    import pandas as pd
    rows = []
    for p in st.session_state['autotrader'].positions:
        try:
            sym = p.get('symbol'); entry = p.get('entry_price'); lot = p.get('lot_size',1); qty = p.get('qty',0)
            cur = st.session_state['paper_broker'].market_prices.get(sym) or st.session_state.get('last_tick', {}).get('ltp') or entry
            side = p.get('side','BUY')
            profit_per_lot = (cur-entry)*lot if side=='BUY' else (entry-cur)*lot
            total = profit_per_lot * (qty/lot if lot else 0)
            total_invested = float(entry) * float(qty if qty else 0)
            rows.append({"order_id": p.get('order_id'), "symbol": sym, "side": side, "entry": entry, "current": cur, "sl": p.get('sl_price'), "tp": p.get('tp_price'), "locked": p.get('locked'), "profit_per_lot": round(profit_per_lot,2), "total_profit": round(total,2), "order_via": p.get("order_via", "")})
        except Exception:
            continue
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)
    else:
        st.write("No active positions")
except Exception as e:
    st.write("TSL table error:", e)

st.markdown("### Notes")
st.write("This app auto-subscribes to NIFTY ATM on websocket connect if SmartAPI client and websocket classes are available. It fetches lot size from instrument master and caches it. Websocket messages are routed to the paper broker market_prices and AutoTrader.monitor_positions for immediate paper fills.")

# end
