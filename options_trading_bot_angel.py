#!/usr/bin/env python3
"""
options_trading_bot_angel.py
Consolidated secured trading bot (Streamlit) with SmartAPI wiring, Paper/Live parity,
AutoTrader (opt-in), safety checks (daily loss, per-trade risk), and UI debug/logs.
No hard-coded credentials. Use environment variables for Live mode.
"""

from __future__ import annotations
import os
import time
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
import threading
from collections import deque

# Dependency flags
SMARTAPI_AVAILABLE = False
PYOTP_AVAILABLE = False
try:
    from SmartApi import SmartConnect  # type: ignore
    SMARTAPI_AVAILABLE = True
except Exception:
    SMARTAPI_AVAILABLE = False

try:
    import pyotp  # type: ignore
    PYOTP_AVAILABLE = True
except Exception:
    PYOTP_AVAILABLE = False

# --- SmartAPI adapter helpers ---
def create_smartapi_connection(api_key: str, client_id: str, password: str, totp_secret: str = None, client_code: str = None):
    if not SMARTAPI_AVAILABLE:
        raise RuntimeError("smartapi-python not installed in environment")
    from SmartApi import SmartConnect  # type: ignore
    conn = SmartConnect(api_key=api_key)
    otp = None
    if totp_secret and PYOTP_AVAILABLE:
        try:
            import pyotp as _pyotp  # type: ignore
            otp = _pyotp.TOTP(totp_secret).now()
        except Exception:
            otp = None
    data = conn.generateSession(client_id, password, otp)
    feed_token = None
    try:
        feed_token = conn.getfeedToken()
    except Exception:
        try:
            feed_token = conn.getFeedToken()
        except Exception:
            feed_token = None
    if client_code is None:
        try:
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
                client_code = data["data"].get("clientCode") or data["data"].get("client_code") or client_id
        except Exception:
            client_code = client_id
    return conn, feed_token, client_code

def start_smartapi_websockets(conn, feed_token, client_code, paper_broker=None, live_broker=None, subscribe_tokens=None):
    result = {"tick_ws": None, "order_ws": None, "errors": []}
    try:
        # Try imports in multiple possible locations
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore
        except Exception:
            try:
                from SmartApi import SmartWebSocketV2  # type: ignore
            except Exception:
                SmartWebSocketV2 = None
    except Exception:
        SmartWebSocketV2 = None

    try:
        try:
            from SmartApi.smartWebSocketOrderUpdate import SmartWebSocketOrderUpdate  # type: ignore
        except Exception:
            try:
                from SmartApi import SmartWebSocketOrderUpdate  # type: ignore
            except Exception:
                SmartWebSocketOrderUpdate = None
    except Exception:
        SmartWebSocketOrderUpdate = None

    def _parse_tick(payload):
        if not payload:
            return None, None
        try:
            if isinstance(payload, dict):
                data = payload.get("data") if "data" in payload else payload
                if isinstance(data, dict):
                    sym = data.get("symbol") or data.get("scrip") or data.get("instrument") or data.get("tradingsymbol")
                    ltp = data.get("lastTradedPrice") or data.get("ltp") or data.get("last_price") or data.get("ltpPrice")
                    if ltp is None:
                        for k in ("lastTradedPrice","ltp","last_price","price","ltpPrice"):
                            if k in data and data[k] is not None:
                                ltp = data[k]; break
                    return sym, float(ltp) if ltp is not None else None
            if isinstance(payload, (list, tuple)) and len(payload)>0:
                return _parse_tick(payload[0])
        except Exception:
            return None, None
        return None, None

    # Tick websocket
    try:
        if SmartWebSocketV2 is not None:
            try:
                tws = SmartWebSocketV2(conn, feed_token, client_code)
            except Exception:
                try:
                    tws = SmartWebSocketV2(feed_token, conn)
                except Exception:
                    try:
                        tws = SmartWebSocketV2(conn)
                    except Exception:
                        tws = None
            if tws is not None:
                def _on_tick(data):
                    try:
                        sym, ltp = _parse_tick(data)
                        if sym is None:
                            return
                        # update broker caches
                        if paper_broker is not None:
                            paper_broker.market_prices[sym] = float(ltp) if ltp is not None else None
                            for oid, o in list(paper_broker.orders.items()):
                                if o["symbol"] == sym and o["status"] == "OPEN" and ltp is not None:
                                    paper_broker._try_fill_against_price(oid, float(ltp))
                        if live_broker is not None and hasattr(live_broker, "market_prices"):
                            live_broker.market_prices[sym] = float(ltp) if ltp is not None else None
                        try:
                            import streamlit as st
                            st.session_state['last_tick'] = {"symbol": sym, "ltp": ltp, "raw": data}

# --- Indicator update & bias calculation (best-effort) ---
try:
    import streamlit as st
    engine = st.session_state.get('indicator_engine', None)
    if engine is None:
        # create with defaults and persisted UI config if present
        cfg = st.session_state.get('bias_config', {})
        engine = ensure_indicator_engine(cfg)
    if engine is not None and sym is not None and ltp is not None:
        vol = None
        oi = None
        try:
            if isinstance(data, dict):
                d = data.get('data') if 'data' in data else data
                vol = d.get('volume') or d.get('v') or d.get('vol') or 0.0
                oi = d.get('oi') or d.get('openInterest') or d.get('open_interest') or None
            elif isinstance(data, (list,tuple)) and len(data)>0:
                first = data[0]
                if isinstance(first, dict):
                    vol = first.get('volume') or first.get('v') or first.get('vol') or 0.0
                    oi = first.get('oi') or first.get('openInterest') or None
        except Exception:
            pass
        try:
            engine.update_tick(price=ltp, volume=vol or 0.0, timestamp=round(time.time()), oi=oi)
            cfg = st.session_state.get('bias_config', {})
            bias = engine.get_bias(cfg)
            st.session_state['bias'] = bias
            st.session_state['last_indicators'] = {
                'ema9': engine.ema_short(), 'ema21': engine.ema_long(), 'vwap': engine.vwap(), 'atr': engine.atr(), 'oi_ch': engine.oi_change_pct(), 'cpr': engine.cpr_levels()
            }

# --- TSL levels & Real Profit table (auto) ---
try:
    import streamlit as st, pandas as pd
    aut = None
    for k, v in st.session_state.items():
        try:
            if hasattr(v, "positions") and isinstance(getattr(v, "positions"), list):
                aut = v
                break
        except Exception:
            continue
    pos_list = []
    if aut is not None:
        for p in getattr(aut, "positions", []):
            try:
                sym = p.get("symbol")
                entry = float(p.get("entry_price", p.get("entry", 0.0) or 0.0))
                sl = p.get("sl_price")
                tp = p.get("tp_price")
                lot_size = int(p.get("lot_size", p.get("lot_size", 1) or 1))
                qty = float(p.get("qty", 0) or 0)
                # try to fetch current price from broker.market_prices if available
                cur = None
                try:
                    broker = getattr(aut, "broker", None)
                    if broker is not None and hasattr(broker, "market_prices"):
                        mp = getattr(broker, "market_prices", None)
                        if isinstance(mp, dict):
                            cur = mp.get(sym) or mp.get(str(sym)) or None
                except Exception:
                    cur = None
                # fallback to last_tick in session_state
                if cur is None:
                    lt = st.session_state.get("last_tick")
                    if isinstance(lt, dict) and lt.get("symbol") == sym:
                        cur = lt.get("ltp")
                # fallback to indicator engine fast EMA as proxy
                if cur is None:
                    eng = st.session_state.get("indicator_engine", None)
                    try:
                        cur = eng.ema_short() if eng is not None and hasattr(eng, "ema_short") else None
                    except Exception:
                        cur = None
                if cur is None:
                    cur = entry
                # profit per lot (currency)
                side = str(p.get("side", "BUY")).upper()
                if side == "BUY":
                    profit_per_lot = (float(cur) - entry) * float(lot_size)
                else:
                    profit_per_lot = (entry - float(cur)) * float(lot_size)
                lots_count = (qty / lot_size) if lot_size else qty
                total_profit = profit_per_lot * lots_count
                pos_list.append({
                    "order_id": p.get("order_id"),
                    "symbol": sym,
                    "side": side,
                    "entry_price": round(entry, 4),
                    "current_price": round(float(cur), 4),
                    "sl_price": round(float(sl), 4) if sl is not None else None,
                    "tp_price": round(float(tp), 4) if tp is not None else None,
                    "locked": bool(p.get("locked", False)),
                    "profit_per_lot": round(profit_per_lot, 2),
                    "total_profit": round(total_profit, 2),
                    "lots_count": round(lots_count, 2)
                })
            except Exception:
                continue
    # render table
    if pos_list:
        df = pd.DataFrame(pos_list)
        try:
            from ace_tools import display_dataframe_to_user
            display_dataframe_to_user("TSL & Profit Table", df)
        except Exception:
            st.dataframe(df)
    else:
        st.write("No active positions to show TSL / profit for.")
except Exception as e:
    st.write("TSL table error:", e)
        except Exception:
            pass
except Exception:
    pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                # attach callback
                if hasattr(tws, "on_data"):
                    try:
                        tws.on_data = _on_tick
                    except Exception:
                        pass
                elif hasattr(tws, "on_message"):
                    try:
                        tws.on_message = lambda ws, msg: _on_tick(msg)
                    except Exception:
                        pass
                result["tick_ws"] = tws
                # subscribe tokens if provided
                if subscribe_tokens and hasattr(tws, "subscribe"):
                    try:
                        tws.subscribe(1, 1, subscribe_tokens)
                    except Exception:
                        try:
                            tws.subscribe(subscribe_tokens)
                        except Exception:
                            pass
    except Exception as e:
        result["errors"].append(str(e))

    # Order update websocket
    try:
        if SmartWebSocketOrderUpdate is not None:
            try:
                ows = SmartWebSocketOrderUpdate(conn)
            except Exception:
                try:
                    ows = SmartWebSocketOrderUpdate(conn, feed_token)
                except Exception:
                    try:
                        ows = SmartWebSocketOrderUpdate(feed_token, conn)
                    except Exception:
                        ows = None
            if ows is not None:
                def _on_order_update(data):
                    try:
                        import streamlit as st
                        st.session_state['last_order_update'] = data
                    except Exception:
                        pass
                    if paper_broker is not None and isinstance(data, dict):
                        oid = data.get('order_id') or data.get('orderId') or data.get('orderID') or data.get('orderNumber')
                        status = data.get('status') or data.get('order_status') or data.get('orderStatus') or data.get('statusMessage')
                        if oid and status and oid in paper_broker.orders:
                            s = str(status).upper()
                            if 'FILL' in s or 'COMPLETE' in s or 'TRADE' in s:
                                fp = data.get('fill_price') or data.get('avg_fill_price') or data.get('avg_price') or None
                                fq = data.get('filled_qty') or data.get('filledQty') or data.get('quantity') or None
                                try:
                                    fq = int(fq) if fq is not None else paper_broker.orders[oid]['qty']
                                except Exception:
                                    fq = paper_broker.orders[oid]['qty']
                                paper_broker._fill(oid, float(fp) if fp is not None else paper_broker.orders[oid]['price'], fq)
                            elif 'CANCEL' in s:
                                paper_broker.orders[oid]['status'] = 'CANCELLED'
                if hasattr(ows, "on_data"):
                    ows.on_data = _on_order_update
                elif hasattr(ows, "on_message"):
                    ows.on_message = lambda ws, msg: _on_order_update(msg)
                result["order_ws"] = ows
    except Exception as e:
        result["errors"].append(str(e))

    return result



# --- Indicator Engine: EMA, VWAP, CPR (relaxed), ATR, OI Change ---
from collections import deque as _deque
class IndicatorEngine:
    def __init__(self, ema_short_n=9, ema_long_n=21, atr_n=14, vwap_window_secs=None, cpr_padding_pct=0.0, oi_lookback=2):
        self.ema_short_n = ema_short_n
        self.ema_long_n = ema_long_n
        self._ema_short = None
        self._ema_long = None
        self._ema_alpha_short = 2.0 / (ema_short_n + 1)
        self._ema_alpha_long = 2.0 / (ema_long_n + 1)
        self.atr_n = atr_n
        self._atr = None
        self._prev_close = None
        self._atr_alpha = 2.0 / (atr_n + 1)
        self.vwap_window_secs = vwap_window_secs
        self._vwap_vp = 0.0
        self._vwap_v = 0.0
        self._vwap_queue = _deque() if vwap_window_secs else None
        self._pivot_high = None
        self._pivot_low = None
        self._pivot_close = None
        self.cpr_padding_pct = cpr_padding_pct
        self.oi_lookback = oi_lookback
        self._oi_deque = _deque(maxlen=oi_lookback)
        self._recent_high = None
        self._recent_low = None

    def update_tick(self, price: float, volume: float = 0.0, timestamp: float = None, oi: float = None, high: float = None, low: float = None, close: float = None):
        try:
            p = float(price)
        except Exception:
            return
        if self._ema_short is None:
            self._ema_short = p
            self._ema_long = p
        else:
            self._ema_short = (self._ema_alpha_short * p) + (1 - self._ema_alpha_short) * self._ema_short
            self._ema_long = (self._ema_alpha_long * p) + (1 - self._ema_alpha_long) * self._ema_long
        if self.vwap_window_secs and timestamp is not None:
            pv = p * float(volume or 0.0)
            self._vwap_queue.append((timestamp, pv, float(volume or 0.0)))
            self._vwap_vp += pv
            self._vwap_v += float(volume or 0.0)
            cutoff = timestamp - self.vwap_window_secs
            while self._vwap_queue and self._vwap_queue[0][0] < cutoff:
                _, pv_old, v_old = self._vwap_queue.popleft()
                self._vwap_vp -= pv_old
                self._vwap_v -= v_old
        else:
            self._vwap_vp += p * float(volume or 0.0)
            self._vwap_v += float(volume or 0.0)
        if high is not None and low is not None:
            tr = max(high - low,
                     abs(high - (self._prev_close if self._prev_close is not None else (close if close is not None else p))),
                     abs(low - (self._prev_close if self._prev_close is not None else (close if close is not None else p))))
            if self._atr is None:
                self._atr = tr
            else:
                self._atr = (self._atr_alpha * tr) + (1 - self._atr_alpha) * self._atr
            if close is not None:
                self._prev_close = close
        if high is not None:
            if self._pivot_high is None or high > self._pivot_high:
                self._pivot_high = high
        if low is not None:
            if self._pivot_low is None or low < self._pivot_low:
                self._pivot_low = low
        if close is not None:
            self._pivot_close = close
        if oi is not None:
            try:
                self._oi_deque.append(float(oi))
            except Exception:
                pass

    def ema_short(self):
        return self._ema_short
    def ema_long(self):
        return self._ema_long
    def vwap(self):
        return (self._vwap_vp / self._vwap_v) if (self._vwap_v and self._vwap_v > 0) else None
    def atr(self):
        return self._atr
    def cpr_levels(self):
        if self._pivot_high is None or self._pivot_low is None or self._pivot_close is None:
            return None
        pivot = (self._pivot_high + self._pivot_low + self._pivot_close) / 3.0
        bc = (self._pivot_high + self._pivot_low) / 2.0
        tc = pivot + (pivot - bc)
        pad = abs(tc - bc) * self.cpr_padding_pct if self.cpr_padding_pct else 0.0
        return {"BC": bc - pad, "TC": tc + pad, "Pivot": pivot}
    def oi_change_pct(self):
        if len(self._oi_deque) < 2:
            return None
        prev = self._oi_deque[-2]
        curr = self._oi_deque[-1]
        if prev == 0:
            return None
        return (curr - prev) / prev

# --- TickAggregator ---
class TickAggregator:
    def __init__(self, engine=None, symbol=None):
        self.engine = engine
        self.symbol = symbol
        self.current_min = None
        self.ohlcv = None
    def ingest_tick(self, price, volume=0.0, oi=None, timestamp=None):
        try:
            ts = float(timestamp or time.time())
            minute = int(ts // 60)
            p = float(price)
            v = float(volume or 0.0)
        except Exception:
            return
        if self.current_min is None:
            self.current_min = minute
            self.ohlcv = {"open": p, "high": p, "low": p, "close": p, "volume": v, "oi": oi}
            return
        if minute == self.current_min:
            self.ohlcv["high"] = max(self.ohlcv["high"], p)
            self.ohlcv["low"] = min(self.ohlcv["low"], p)
            self.ohlcv["close"] = p
            self.ohlcv["volume"] += v
            if oi is not None:
                self.ohlcv["oi"] = oi
            return
        else:
            try:
                if self.engine is not None and self.ohlcv is not None:
                    try:
                        self.engine.update_tick(price=self.ohlcv["close"],
                                                volume=self.ohlcv["volume"],
                                                timestamp=self.current_min * 60,
                                                oi=self.ohlcv.get("oi", None),
                                                high=self.ohlcv["high"],
                                                low=self.ohlcv["low"],
                                                close=self.ohlcv["close"])
                    except Exception:
                        pass
            except Exception:
                pass
            self.current_min = minute
            self.ohlcv = {"open": p, "high": p, "low": p, "close": p, "volume": v, "oi": oi}
            return

# ensure helper to create engine in session when Streamlit is present
def ensure_indicator_engine(defaults=None):
    try:
        import streamlit as st
    except Exception:
        return None
    if 'indicator_engine' not in st.session_state:
        cfg = defaults or {}
        engine = IndicatorEngine(ema_short_n=cfg.get("ema_short_n",9), ema_long_n=cfg.get("ema_long_n",21),
                                 atr_n=cfg.get("atr_n",14), vwap_window_secs=cfg.get("vwap_window_secs", None),
                                 cpr_padding_pct=cfg.get("cpr_padding_pct",0.0), oi_lookback=cfg.get("oi_lookback",2))
        st.session_state['indicator_engine'] = engine
        try:
            st.session_state.setdefault('tick_aggregator', TickAggregator(engine=engine))
        except Exception:
            pass
    return st.session_state.get('indicator_engine')



# --- Brokers ---
class PaperBroker:
    def __init__(self, smart_conn=None):
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.next_id = 1
        self.trades: List[Dict[str, Any]] = []
        self.smart_conn = smart_conn
        self.market_prices: Dict[str, float] = {}
        self._poller_running = False

    def place_order(self, symbol: str, qty: int, side: str, price: float, order_type: str = "LIMIT", **kwargs) -> Dict[str, Any]:
        order_id = f"P{self.next_id}"
        self.next_id += 1
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "qty": int(qty),
            "side": side,
            "price": float(price),
            "order_type": order_type,
            "status": "OPEN",
            "timestamp": datetime.utcnow().isoformat(),
            "filled_qty": 0,
            "avg_fill_price": None,
        }
        self.orders[order_id] = order
        # try immediate match against last known price
        last_price = self.market_prices.get(symbol)
        if last_price is not None:
            self._try_fill_against_price(order_id, last_price)
        return order

    def _try_fill_against_price(self, order_id: str, market_price: float):
        order = self.orders.get(order_id)
        if not order or order["status"] != "OPEN":
            return
        side = order["side"].upper()
        price = float(order["price"])
        if side == "BUY" and market_price <= price:
            self._fill(order_id, market_price, order["qty"])
        elif side == "SELL" and market_price >= price:
            self._fill(order_id, market_price, order["qty"])

    def _fill(self, order_id: str, fill_price: float, filled_qty: int):
        order = self.orders.get(order_id)
        if not order:
            return
        order["status"] = "FILLED"
        order["filled_qty"] = int(filled_qty)
        order["avg_fill_price"] = float(fill_price)
        self.trades.append({
            "order_id": order_id,
            "symbol": order["symbol"],
            "qty": order["qty"],
            "side": order["side"],
            "price": float(fill_price),
            "time": datetime.utcnow().isoformat()
        })
        # update global session-state PnL if streamlit available
        try:
            import streamlit as st
            pnl = float(st.session_state.get('pnl_today', 0.0))
            sign = -1 if order["side"].upper() == "BUY" else 1
            trade_pnl = sign * float(order["filled_qty"]) * float(order["avg_fill_price"] or order["price"] or 0)
            st.session_state['pnl_today'] = pnl + trade_pnl
            st.session_state['_last_autotrade_order'] = order
        except Exception:
            pass

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        order = self.orders.get(order_id)
        if not order:
            return {"error": "order_not_found"}
        if order["status"] in ("FILLED", "CANCELLED"):
            return {"error": "cannot_cancel"}
        order["status"] = "CANCELLED"
        return {"success": True, "order_id": order_id}

    def get_orders(self) -> List[Dict[str, Any]]:
        return list(self.orders.values())

    def get_trades(self) -> List[Dict[str, Any]]:
        return list(self.trades)
        # positions: list of dicts {order_id, symbol, qty, side, entry_price, sl_price, tp_price, trailing_active, highest_price, lowest_price}
        self.positions = []


    def start_order_status_poller(self, smart_conn, interval: float = 1.0):
        if not SMARTAPI_AVAILABLE or smart_conn is None:
            return False
        if getattr(self, "_poller_running", False):
            return True
        self._poller_running = True
        def _loop():
            while getattr(self, "_poller_running", False):
                try:
                    open_orders = [o for o in list(self.orders.values()) if o["status"] == "OPEN"]
                    symbols = set(o["symbol"] for o in open_orders)
                    for sym in symbols:
                        # try to fetch LTP via common SmartAPI calls
                        price = None
                        for name in ("getLTP","get_ltp","getQuote","get_quotes","getQuoteLtp","ltp"):
                            func = getattr(smart_conn, name, None)
                            if callable(func):
                                try:
                                    res = func(sym)
                                    if isinstance(res, (int,float)):
                                        price = float(res); break
                                    if isinstance(res, dict):
                                        for k in ("ltp","lastTradedPrice","last_price","price"):
                                            if k in res and res[k] is not None:
                                                price = float(res[k]); break
                                        if price is not None: break
                                except Exception:
                                    continue
                        if price is not None:
                            self.market_prices[sym] = price
                            for oid, o in list(self.orders.items()):
                                if o["symbol"] == sym and o["status"] == "OPEN":
                                    self._try_fill_against_price(oid, price)
                except Exception:
                    pass
                time.sleep(interval)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self._poller_thread = t
        return True

    def stop_order_status_poller(self):
        if getattr(self, "_poller_running", False):
            self._poller_running = False
            try:
                self._poller_thread.join(timeout=1.0)
            except Exception:
                pass
            return True
        return False

class LiveBroker:
    def __init__(self, api_key: str, client_id: str, password: str, totp_secret: Optional[str] = None):
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.totp_secret = totp_secret
        self.conn = None
        self._connected = False
        self.market_prices: Dict[str, float] = {}

    def connect(self):
        if not SMARTAPI_AVAILABLE:
            raise RuntimeError("SmartAPI library not available")
        from SmartApi import SmartConnect  # type: ignore
        conn = SmartConnect(api_key=self.api_key)
        otp = None
        if self.totp_secret and PYOTP_AVAILABLE:
            try:
                import pyotp as _pyotp  # type: ignore
                otp = _pyotp.TOTP(self.totp_secret).now()
            except Exception:
                otp = None
        data = conn.generateSession(self.client_id, self.password, otp)
        self.conn = conn
        self._connected = True
        return data

    def place_order(self, *args, **kwargs):
        if not self._connected or self.conn is None:
            raise RuntimeError("Not connected to SmartAPI. Call connect() first.")
        return self.conn.placeOrder(*args, **kwargs)

    def get_order_status(self, order_id: str) -> dict:
        if not self._connected or self.conn is None:
            raise RuntimeError("Not connected")
        # try common names
        for name in ("orderBook","getOrderBook","getOrderStatus","order_status","getOrder"):
            func = getattr(self.conn, name, None)
            if callable(func):
                try:
                    res = func(order_id)
                    return res
                except Exception:
                    try:
                        res = func([order_id])
                        return res[0] if isinstance(res, list) and res else res
                    except Exception:
                        continue
        raise RuntimeError("Could not fetch order status via SmartAPI")

# --- AutoTrader ---
class AutoTrader:
    def __init__(self, broker, get_market_price_callable, strategy_callable, max_trades_per_day=3, per_trade_max_risk=5000.0, daily_loss_limit=10000.0):
        self.broker = broker
        self.get_price = get_market_price_callable
        self.strategy = strategy_callable
        self.max_trades = max_trades_per_day
        self.per_trade_max_risk = float(per_trade_max_risk)
        self.daily_loss_limit = float(daily_loss_limit)
        self.trades_today = 0
        self.traded_strikes = set()
        self._running = False
        self._thread = None
        self.log = deque(maxlen=200)
        # pnl tracking (mirror in streamlit session_state when available)
        self.pnl_today = 0.0

    def start(self):
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if not self._running:
            return True
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        return True

    def reset_daily_counters(self):
        self.trades_today = 0
        self.traded_strikes.clear()
        self.pnl_today = 0.0
        try:
            import streamlit as st
            st.session_state['pnl_today'] = 0.0
        except Exception:
            pass

    def _log(self, msg):
        ts = datetime.utcnow().isoformat()
        entry = f"{ts} - {msg}"
        self.log.appendleft(entry)
        try:
            import streamlit as st
            st.session_state.setdefault('autotrader_log', deque(maxlen=500))
            st.session_state['autotrader_log'].appendleft(entry)
        except Exception:
            pass

    def _loop(self):
        while self._running:
            # Position monitor: check existing positions for SL/TP/trailing triggers
            try:
                # build set of symbols to check
                for p in list(self.positions):
                    if not p.get("active", False):
                        continue
                    sym = p.get("symbol")
                    # get latest market price
                    mprice = None
                    try:
                        mprice = float(self.get_price(sym)) if callable(self.get_price) else None
                    except Exception:
                        mprice = None
                    if mprice is None and hasattr(self.broker, "market_prices"):
                        mprice = self.broker.market_prices.get(sym)
                    if mprice is None:
                        continue
                    side_p = p.get("side", "BUY").upper()
                    # update highest/lowest for trailing calculations
                    try:
                        if mprice > p.get("highest_price", mprice):
                            p["highest_price"] = mprice
                        if mprice < p.get("lowest_price", mprice):
                            p["lowest_price"] = mprice
                    except Exception:
                        pass
                    # trailing logic: for BUY, trail below highest by trailing_step; for SELL, trail above lowest by trailing_step
                    
try:
    if p.get("trailing", False):
        # Dynamic trailing step selection (hardcoded tiers: 3 / 6 / 10 points)
        try:
            lot_sz = int(p.get("lot_size", 1)) if p.get("lot_size", 1) else 1
        except Exception:
            lot_sz = 1
        try:
            entry = float(p.get("entry_price", p.get("avg_entry_price", p.get("entry", 0.0)) or 0.0))
        except Exception:
            entry = float(p.get("entry_price", 0.0) or 0.0)
        # compute profit per lot in currency units
        try:
            if p.get("side", "BUY").upper() == "BUY":
                profit_per_lot = (mprice - entry) * float(lot_sz)
            else:
                profit_per_lot = (entry - mprice) * float(lot_sz)
        except Exception:
            profit_per_lot = 0.0
        # select step based on profit_per_lot thresholds (hardcoded)
        if profit_per_lot < 20.0:
                                base_step = 3.0
                            elif profit_per_lot < 30.0:
                                base_step = 6.0
                            else:
                                base_step = 10.0
                            # ATR scaling: widen step in high-volatility sessions (factor hardcoded = 0.4)
                            try:
                                atr_val = float(self.get_price and 0 or 0)  # placeholder
                            except Exception:
                                atr_val = None
                            try:
                                # try to get ATR from indicator engine if available in session_state
                                import streamlit as st
                                eng = st.session_state.get('indicator_engine', None)
                                if eng is not None:
                                    atr_val = eng.atr() or atr_val
                            except Exception:
                                pass
                            try:
                                if atr_val is not None:
                                    atr_scaled = float(atr_val) * 0.4
                                    step = max(base_step, atr_scaled)
                                else:
                                    step = base_step
                            except Exception:
                                step = base_step
        # apply trailing using selected step
        if p.get("side", "BUY").upper() == "BUY":
            new_sl = p.get("highest_price", mprice) - step
            if new_sl > p.get("sl_price", -1e9):
                p["sl_price"] = new_sl
        else:
            new_sl = p.get("lowest_price", mprice) + step
            if new_sl < p.get("sl_price", 1e9):
                p["sl_price"] = new_sl


    # Now check for SL or TP conditions: we only close on SL (which may be at TP if locked) or if a manual close is triggered.
    try:
        # If price hits SL (including when SL was moved to TP), close position
        if mprice <= p.get("sl_price", -1e9):
            try:
                close_qty = p.get("qty", 0)
                close_order = self.broker.place_order(symbol=sym, qty=close_qty, side="SELL", price=float(mprice), order_type="IOC", immediate_fill=False)
                p["active"] = False
                self._log(f"Position {p.get('order_id')} SL hit; placed close order {close_order.get('order_id')} at {mprice}")
            except Exception as e:
                self._log(f"Failed to place SL close order for {sym}: {e}")
    except Exception:
        pass

else:
    # SELL position: when TP reached (price <= tp_price), lock profit by moving SL to TP and enable trailing
    try:
        tp_val = p.get("tp_price", None)
        sl_val = p.get("sl_price", None)
        if tp_val is not None and not p.get("locked", False) and mprice <= tp_val:
            p["locked"] = True
                                try:
                                    import streamlit as st
                                    lb = float(st.session_state.get('lock_buffer', 0.5))
                                except Exception:
                                    lb = 0.5
                                # set SL slightly below TP by lock_buffer points (for BUY). For SELL handled symmetrically later.
                                try:
                                    p["sl_price"] = float(tp_val) - float(lb)
                                except Exception:
                                    p["sl_price"] = float(tp_val)
 - 0.5  # buffer applied
            p["trailing"] = True
            if "trailing_step" not in p:
                p["trailing_step"] = int(p.get("trailing_step", 5))
            self._log(f"Position {p.get('order_id')} reached base TP (SELL); locking profit at {p['sl_price']} and enabling trailing")
    except Exception as e:
        self._log(f"Error when locking TP for SELL {p.get('order_id')}: {e}")

    # Close only when SL is hit (which may be at TP if locked)
    try:
        if mprice >= p.get("sl_price", 1e9):
            try:
                close_qty = p.get("qty", 0)
                close_order = self.broker.place_order(symbol=sym, qty=close_qty, side="BUY", price=float(mprice), order_type="IOC", immediate_fill=False)
                p["active"] = False
                self._log(f"Position {p.get('order_id')} SL hit; placed close order {close_order.get('order_id')} at {mprice}")
            except Exception as e:
                self._log(f"Failed to place SL close order for {sym}: {e}")
    except Exception:
        pass
except Exception:
                        pass
            except Exception:
                pass

            try:
                now = datetime.now()
                # stop after 15:00 local exchange time approx
                if now.hour >= 15:
                    self._log("Market past 15:00 — auto-trader paused for the day.")
                    time.sleep(5)
                    continue

                if self.trades_today >= self.max_trades:
                    time.sleep(1)
                    continue

                # check daily loss limit from session_state if present
                try:
                    import streamlit as st
                    pnl_today = float(st.session_state.get('pnl_today', 0.0))
try:
    if float(st.session_state.get('pnl_today', 0.0)) >= float(DAILY_PROFIT_CAP):
        self._log(f"Daily profit cap reached ({st.session_state.get('pnl_today')}): stopping AutoTrader for the day.")
        time.sleep(2)
        continue
except Exception:
    pass

                except Exception:
                    pnl_today = self.pnl_today
                if pnl_today <= -abs(self.daily_loss_limit):
                    self._log(f"Daily loss limit reached ({pnl_today}) — pausing AutoTrader for the day.")
                    time.sleep(2)
                    continue

                sig = self.strategy()
                if not sig or sig.get("action") is None:
                    time.sleep(0.5)
                    continue

                symbol = sig.get("symbol")
                lots = int(sig.get("lots", 1))
                side = sig.get("action").upper()
                qty = lots * int(sig.get("lot_size", 1))
                strike_key = symbol
                if strike_key in self.traded_strikes:
                    self._log(f"Skipping {symbol} — already traded this strike today.")
                    time.sleep(0.5)
                    continue

                market_price = sig.get("price") or (self.get_price(symbol) if callable(self.get_price) else None)
                if market_price is None:
                    self._log(f"No market price available for {symbol}; skipping.")
                    time.sleep(0.5)
                    continue

                # simplistic per-trade risk check placeholder (user can adapt to real Greeks)
                est_risk = abs(float(market_price) * qty)
                if est_risk > self.per_trade_max_risk and self.per_trade_max_risk > 0:
                    self._log(f"Estimated trade risk {est_risk} exceeds per-trade max {self.per_trade_max_risk}; skipping.")
                    time.sleep(0.5)
                    continue

                try:
                    order = self.broker.place_order(symbol=symbol, qty=qty, side=side, price=float(market_price), order_type="IOC", immediate_fill=False)
                    self.trades_today += 1
                    self.traded_strikes.add(strike_key)
                    self._log(f"Placed {side} order {order.get('order_id')} for {symbol} qty={qty} price={market_price}")
                    # record position with SL/TP and trailing settings
                    try:
                        slp = float(sl_points) if 'sl_points' in locals() else 20.0
                        tpp = float(tp_points) if 'tp_points' in locals() else 40.0
                        trailing = bool(trailing_enabled) if 'trailing_enabled' in locals() else False
                        tstep = int(trailing_step) if 'trailing_step' in locals() else 5
                        entry = float(market_price)
                        if side.upper() == "BUY":
                            sl_price = entry - slp
                            tp_price = entry + tpp
                        else:
                            sl_price = entry + slp
                            tp_price = entry - tpp
                        
# record position with SL/TP and trailing settings; TP set to ₹10 per lot (price movement = 10 / lot_size)
try:
    slp = float(sl_points) if 'sl_points' in locals() else 20.0
    lot_sz = int(lot_size) if 'lot_size' in locals() and int(lot_size) > 0 else 1
    tp_move_per_contract = 10.0 / float(lot_sz)
    trailing = bool(trailing_enabled) if 'trailing_enabled' in locals() else False
    tstep = int(trailing_step) if 'trailing_step' in locals() else 5
    entry = float(market_price)
    if side.upper() == "BUY":
        sl_price = entry - slp
        tp_price = entry + tp_move_per_contract
    else:
        sl_price = entry + slp
        tp_price = entry - tp_move_per_contract
    pos = {"order_id": order.get("order_id"), "symbol": symbol, "qty": qty, "side": side.upper(), "entry_price": entry, "sl_price": sl_price, "tp_price": tp_price, "trailing": trailing, "trailing_step": tstep, "active": True, "highest_price": entry, "lowest_price": entry, "lot_size": lot_sz}
    self.positions.append(pos)
except Exception:
    pass

                    except Exception:
                        pass

                    # persist last order for UI
                    try:
                        import streamlit as st
                        st.session_state['_last_autotrade_order'] = order
                    except Exception:
                        pass
                    # if order filled immediately, update pnl (PaperBroker._fill updates session_state)
                except Exception as e:
                    self._log(f"Failed to place order for {symbol}: {e}")
            except Exception as e:
                self._log(f"AutoTrader error: {e}")
            time.sleep(0.5)

# --- Instrument discovery helper ---
def discover_atm_token(conn, underlying_symbol: str = "NIFTY", expiry: str = None):
    candidates = ["get_instruments", "get_instrument_master", "getInstrumentMaster", "searchscrip", "searchScrip", "instruments", "get_instruments_master"]
    for name in candidates:
        try:
            func = getattr(conn, name, None)
            if not callable(func):
                continue
            try:
                res = func(underlying_symbol) if name.lower().startswith("search") else func()
            except TypeError:
                try:
                    res = func([underlying_symbol])
                except Exception:
                    res = func()
            if isinstance(res, list):
                candidates_list = []
                for item in res:
                    if not isinstance(item, dict):
                        continue
                    s = item.get('symbol') or item.get('tradingsymbol') or item.get('name') or item.get('instrument')
                    if not s:
                        continue
                    if underlying_symbol.upper() in str(s).upper():
                        candidates_list.append(item)
                if candidates_list:
                    try:
                        spot = None
                        for m in ("getLTP","get_ltp","getQuote","getQuoteLtp"):
                            f = getattr(conn, m, None)
                            if callable(f):
                                try:
                                    sp = f(underlying_symbol)
                                    if isinstance(sp, (int, float)):
                                        spot = float(sp); break
                                    if isinstance(sp, dict):
                                        for k in ("ltp","lastTradedPrice","last_price"):
                                            if k in sp and sp[k] is not None:
                                                spot = float(sp[k]); break
                                        if spot is not None: break
                                except Exception:
                                    continue
                        if spot is not None:
                            best = None; best_diff = None
                            for item in candidates_list:
                                strike = item.get('strikePrice') or item.get('strike') or item.get('strike_price')
                                if strike is None:
                                    continue
                                try:
                                    diff = abs(float(strike) - float(spot))
                                    if best is None or diff < best_diff:
                                        best = item; best_diff = diff
                                except Exception:
                                    continue
                            if best is not None:
                                return best.get('token') or best.get('tokenId') or best.get('instrument') or best.get('tradingsymbol') or best.get('symbol')
                    except Exception:
                        return candidates_list[0].get('token') or candidates_list[0].get('tradingsymbol') or candidates_list[0].get('symbol')
            elif isinstance(res, dict):
                data = res.get('data') or res.get('instruments') or res
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    return item.get('token') or item.get('tradingsymbol') or item.get('symbol')
        except Exception:
            continue
    return None




def get_lot_size_from_conn(conn, symbol: str = None):
    """
    Aggressive, cached lot-size lookup for Angel One SmartAPI connection.
    - Checks a wide variety of possible keys and nested structures.
    - Caches results in streamlit session_state['lot_size_map'] to avoid repeated heavy lookups.
    - Returns integer lot_size or None.
    """
    try:
        import streamlit as st
    except Exception:
        st = None
    # session cache
    try:
        if st is not None:
            st.session_state.setdefault('lot_size_map', {})
            cache = st.session_state['lot_size_map']
        else:
            cache = {}
    except Exception:
        cache = {}
    if symbol is not None and cache.get(symbol):
        try:
            return int(cache[symbol])
        except Exception:
            pass
    if conn is None:
        return None

    # common candidate function names that return instruments or metadata
    candidates = ["get_instruments", "get_instrument_master", "getInstrumentMaster", "get_instruments_master",
                  "instruments", "get_master_contracts", "get_master", "market_quote", "instrumentsMaster", "masterContracts"]
    # keys to try at top-level and nested
    lot_keys = ["lotSize","lot_size","contractSize","contract_size","lot","LotSize","LOT_SIZE","quantity","qty","multiplier","contractMultiplier","tickSize","lotSizeInUnits"]
    nested_paths = [["meta"], ["instrument"], ["details"], ["contract"], ["instrumentDetails"], ["data","meta"]]

    try:
        found = None
        # try quick API call to get specific instrument if method exists (some SmartAPI variants have searchScrip/searchscrip)
        quick_try_names = ["searchscrip","searchScrip","searchInstrument","search_instrument","getInstrumentBySymbol","getInstrument"]
        for qn in quick_try_names:
            func = getattr(conn, qn, None)
            if callable(func) and symbol is not None:
                try:
                    res = func(symbol)
                    # normalize to list/dict
                    if isinstance(res, dict) and res.get("data"):
                        res_list = res.get("data")
                    elif isinstance(res, list):
                        res_list = res
                    else:
                        res_list = [res]
                    for item in res_list:
                        if not isinstance(item, dict):
                            continue
                        for k in lot_keys:
                            if k in item and item[k]:
                                try:
                                    found = int(item[k])
                                    break
                                except Exception:
                                    try:
                                        found = int(float(item[k]))
                                        break
                                    except Exception:
                                        pass
                        if found:
                            break
                    if found:
                        break
                except Exception:
                    pass

        # fallback to scanning larger instrument master lists
        if not found:
            for name in candidates:
                func = getattr(conn, name, None)
                if not callable(func):
                    continue
                try:
                    res = func()
                except TypeError:
                    try:
                        res = func(symbol)
                    except Exception:
                        try:
                            res = func([symbol])
                        except Exception:
                            continue
                except Exception:
                    continue
                data_list = None
                if isinstance(res, dict):
                    data_list = res.get("data") or res.get("instruments") or res.get("results") or res.get("records") or None
                elif isinstance(res, list):
                    data_list = res
                if not data_list:
                    continue
                for item in data_list:
                    if not isinstance(item, dict):
                        continue
                    ts = str(item.get("tradingsymbol") or item.get("symbol") or item.get("instrument") or item.get("name") or "")
                    # quick match by symbol substring if provided
                    if symbol and symbol.upper() not in ts.upper() and ts.upper() not in symbol.upper():
                        # continue scanning, but still try to capture lot size from any item if not symbol matched
                        pass
                    # check direct keys
                    for k in lot_keys:
                        if k in item and item[k]:
                            try:
                                found = int(item[k])
                                break
                            except Exception:
                                try:
                                    found = int(float(item[k]))
                                    break
                                except Exception:
                                    pass
                    if found:
                        break
                    # check nested paths
                    for path in nested_paths:
                        cur = item
                        try:
                            for p in path:
                                if isinstance(cur, dict) and p in cur:
                                    cur = cur[p]
                                else:
                                    cur = None
                                    break
                            if isinstance(cur, dict):
                                for k in lot_keys:
                                    if k in cur and cur[k]:
                                        try:
                                            found = int(cur[k])
                                            break
                                        except Exception:
                                            try:
                                                found = int(float(cur[k]))
                                                break
                                            except Exception:
                                                pass
                        except Exception:
                            pass
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
        if found:
            try:
                if st is not None and symbol:
                    st.session_state.setdefault('lot_size_map', {})[symbol] = int(found)
            except Exception:
                pass
            return int(found)
    except Exception:
        pass
    return None
    candidates = ["get_instruments", "get_instrument_master", "getInstrumentMaster", "get_instruments_master", "instruments", "get_master_contracts"]
    try:
        for name in candidates:
            func = getattr(conn, name, None)
            if not callable(func):
                continue
            try:
                res = func()
            except TypeError:
                try:
                    # some methods accept symbol/name
                    res = func(symbol)
                except Exception:
                    try:
                        res = func([symbol])
                    except Exception:
                        continue
            except Exception:
                continue
            # res may be list or dict with 'data'
            data_list = None
            if isinstance(res, dict):
                data_list = res.get("data") or res.get("instruments") or res.get("results") or None
            elif isinstance(res, list):
                data_list = res
            if not data_list:
                continue
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                # match by tradingsymbol / symbol / instrument names
                ts = item.get("tradingsymbol") or item.get("symbol") or item.get("instrument") or item.get("name")
                if not ts or not symbol:
                    # if symbol not provided, try to pick first and return lot size
                    pass
                else:
                    if str(symbol).upper() in str(ts).upper() or str(ts).upper() in str(symbol).upper():
                        # found candidate, extract lot keys
                        for k in ("lotSize","lot_size","lot_size_contract","lot","contractSize","contract_size"):
                            if k in item and item[k]:
                                try:
                                    return int(item[k])
                                except Exception:
                                    try:
                                        return int(float(item[k]))
                                    except Exception:
                                        pass
                        # sometimes quantity is nested under 'instrument' or 'meta'
                        if "meta" in item and isinstance(item["meta"], dict):
                            for k in ("lotSize","lot_size","contractSize"):
                                if k in item["meta"]:
                                    try:
                                        return int(item["meta"][k])
                                    except Exception:
                                        pass
                # if symbol not provided, attempt to find common lot size keys and return first found
                for k in ("lotSize","lot_size","contractSize","contract_size","lot"):
                    if k in item and item[k]:
                        try:
                            return int(item[k])
                        except Exception:
                            try:
                                return int(float(item[k]))
                            except Exception:
                                pass
    except Exception:
        return None
    return None



# --- Streamlit app ---
def build_streamlit_app():
    try:
        import streamlit as st
        import pandas as pd
    except Exception as e:
        raise RuntimeError("Streamlit/pandas must be installed to run the dashboard") from e

    st.set_page_config(page_title="Options Trading Bot (Angel One) - Secured v3", layout="wide")
    st.title("Options Trading Bot — Secured v3 (Paper/Live)")

    # Sidebar: connection + automation controls
    st.sidebar.header("Connection & Automation")
    mode = st.sidebar.selectbox("Mode", ["Paper", "Live"], index=0)
    paper_mode = (mode == "Paper")

    st.sidebar.markdown("Provide Live credentials using environment variables (recommended):\n- ANGEL_API_KEY\n- ANGEL_CLIENT_ID\n- ANGEL_PASSWORD\n- ANGEL_TOTP_SECRET (optional)\n- ANGEL_CLIENT_CODE (optional)")

    st.sidebar.header("Automation & Logs")
    auto_trade = st.sidebar.checkbox("Enable Auto Trade (must enable explicitly)", value=False)
    show_logs = st.sidebar.checkbox("Show Tick & Decision Logs", value=True)
    max_trades = st.sidebar.number_input("Max trades per day (auto-trader)", value=3, min_value=1, step=1)
    # Safety controls moved to main UI as dropdowns (preset options + custom)
    st.markdown("### Safety Controls (select preset or choose Custom)")

    # Profit / Loss controls for AutoTrader

    # Lock buffer: when TP hit, move SL to TP - lock_buffer to give breathing room
    lock_buffer = st.number_input("Lock buffer (points)", value=0.5, min_value=0.0, step=0.1)
    sl_points = st.number_input("Default Stop-Loss (points)", value=20, min_value=0, step=1)
    tp_points = st.number_input("Default Target / Take-Profit (points)", value=40, min_value=0, step=1)
    trailing_enabled = st.checkbox("Enable Trailing Stop-Loss", value=False)
    trailing_step = st.number_input("Trailing step (points)", value=5, min_value=1, step=1)
    dl_choice = st.selectbox("Daily loss limit", ["Off", "₹1,000", "₹5,000", "₹8,000", "Custom"], index=3)
    if dl_choice == "Off":
        daily_loss_limit = 0.0
    elif dl_choice == "₹1,000":
        daily_loss_limit = 1000.0
    elif dl_choice == "₹5,000":
        daily_loss_limit = 5000.0
    elif dl_choice == "₹10,000":
        daily_loss_limit = 10000.0
    else:
        daily_loss_limit = float(st.number_input("Custom daily loss limit (₹)", value=10000, min_value=0, step=100))

    ptr_choice = st.selectbox("Per-trade max risk", ["Off", "₹500", "₹1,000", "₹5,000", "Custom"], index=3)
    if ptr_choice == "Off":
        per_trade_max_risk = 0.0
    elif ptr_choice == "₹500":
        per_trade_max_risk = 500.0
    elif ptr_choice == "₹1,000":
        per_trade_max_risk = 1000.0
    elif ptr_choice == "₹5,000":
        per_trade_max_risk = 5000.0
    else:
        per_trade_max_risk = float(st.number_input("Custom per-trade max risk (₹)", value=5000, min_value=0, step=100))

    reset_counters = st.button("Reset daily counters (trades & PnL)")
    if reset_counters:
        st.session_state['pnl_today'] = 0.0
        if '_autotrader' in st.session_state and hasattr(st.session_state['_autotrader'], 'reset_daily_counters'):
            st.session_state['_autotrader'].reset_daily_counters()
        st.success("Counters reset.")

    # Daily profit cap: stop trading when reached
    DAILY_PROFIT_CAP = 15000.0
    st.write(f"Daily profit cap (auto-stop): ₹{int(DAILY_PROFIT_CAP)}")

    # Quick market inputs (these are auto-filled by feed/discovery when possible)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        spot = st.number_input("Spot LTP (for ATM calc)", value=22000.0, step=1.0)
        lot_size = st.number_input("Lot size (units per lot)", value=1, step=1)
    with col2:
        ema9 = st.number_input("EMA(9)", value=21990.0, step=0.1)
        ema21 = st.number_input("EMA(21)", value=21980.0, step=0.1)
    with col3:
        vwap = st.number_input("VWAP", value=21995.0, step=0.1)

    bias = "NEUTRAL"
    if ema9 > ema21 and vwap > ema21:
        bias = "BULLISH"
    elif ema9 < ema21 and vwap < ema21:
        bias = "BEARISH"
    st.metric("Bias", bias)
    atm = int(round(spot / 50) * 50)
    st.write(f"ATM Strike (nearest 50): **{atm}**")

    # Broker setup
    broker = None
    if paper_mode:
        broker = PaperBroker()
        st.sidebar.success("Paper trading active — orders simulated via feed when available.")
    else:
        api_key = os.getenv("ANGEL_API_KEY", "")
        client_id = os.getenv("ANGEL_CLIENT_ID", "")
        password = os.getenv("ANGEL_PASSWORD", "")
        totp = os.getenv("ANGEL_TOTP_SECRET", None)
        client_code = os.getenv("ANGEL_CLIENT_CODE", None)
        if not api_key or not client_id or not password:
            st.sidebar.error("Missing ANGEL API credentials in environment variables for Live mode.")
            st.stop()
        broker = LiveBroker(api_key, client_id, password, totp)
        try:
            conn_data = broker.connect()
            st.sidebar.success("Connected to Angel One SmartAPI (live).")
        except Exception as e:
            st.sidebar.error(f"Live connection failed: {e}")
            st.stop()

    # Attempt SmartAPI wiring automatically if env vars present
    try:
        api_key = os.getenv("ANGEL_API_KEY", "") or None
        client_id = os.getenv("ANGEL_CLIENT_ID", "") or None
        password = os.getenv("ANGEL_PASSWORD", "") or None
        totp = os.getenv("ANGEL_TOTP_SECRET", None)
        client_code = os.getenv("ANGEL_CLIENT_CODE", None)
        if SMARTAPI_AVAILABLE and api_key and client_id and password:
            try:
                conn, feed_token, client_code = create_smartapi_connection(api_key, client_id, password, totp, client_code)
                st.session_state['ws_status'] = 'SmartAPI connected'
                try:
                    import streamlit as st
                    st.session_state['smart_conn'] = conn

try:
    st.session_state.setdefault('lot_size_map', {})
except Exception:
    pass
                    st.session_state['smart_client_code'] = client_code
                except Exception:
                    pass

                # try to discover ATM token and subscribe (best-effort)
                try:
                    token = discover_atm_token(conn, "NIFTY")
                    subs = [token] if token else None
                except Exception:
                    subs = None
                try:
                    ws_objs = start_smartapi_websockets(conn, feed_token, client_code, paper_broker=broker if isinstance(broker, PaperBroker) else None, live_broker=broker if not isinstance(broker, PaperBroker) else None, subscribe_tokens=subs)
                    st.session_state['ws_status'] = 'Websockets started' if (ws_objs.get('tick_ws') or ws_objs.get('order_ws')) else st.session_state.get('ws_status', 'No websockets')
                    st.session_state['ws_objs'] = ws_objs
                    # if paper mode and ws started, start poller
                    try:
                        if isinstance(broker, PaperBroker):
                            broker.smart_conn = conn
                            broker.start_order_status_poller(conn, interval=1.0)
                    except Exception:
                        pass
                except Exception as e:
                    st.session_state['ws_status'] = f'Websocket start failed: {e}'
            except Exception as e:
                st.session_state['ws_status'] = f'SmartAPI connection failed: {e}'
    except Exception:
        pass

    # Trade controls (manual)
    st.subheader("Place Trade (single)")
    colA, colB, colC, colD = st.columns([2,1,1,1])
    with colA:
        symbol = st.text_input("Symbol (eg NIFTY23SEPxxxxCE)", value=f"NIFTY{int(atm)}CE")
    with colB:
        side = st.selectbox("Side", ["BUY", "SELL"], index=0)
    with colC:
        qty_lots = st.number_input("Lots", value=4, min_value=1, step=1)  # default 4 lots
            # Show detected lot size from API if available
            try:
                import streamlit as st
                conn = st.session_state.get("smart_conn", None)
                detected_lot = None
                if conn is not None:
                    try:
                        detected_lot = get_lot_size_from_conn(conn, symbol)
                    except Exception:
                        detected_lot = None
                if detected_lot:
                    st.write(f"Detected lot size (from API): {detected_lot}")
                else:
                    st.write("Lot size not detected from API — auto-trading will be paused until available.")
            except Exception:
                pass

    with colD:
        price = st.number_input("Limit Price", value=20.0, step=0.1)

    if not auto_trade:
        place = st.button("Place Order")
        
if place:
    # require lot size from API for manual orders as well
    try:
        import streamlit as st
        conn = st.session_state.get('smart_conn', None)
        lot_sz = None
        if conn is not None:
            try:
                lot_sz = get_lot_size_from_conn(conn, symbol)
            except Exception:
                lot_sz = None
        if not lot_sz:
            st.error(f"Lot size not available from API for {symbol}. Manual order blocked. Please ensure SmartAPI connection and instrument metadata are available.")
        else:
            qty = int(qty_lots * int(lot_sz))
            order = broker.place_order(symbol=symbol, qty=qty, side=side, price=price, order_type="IOC", immediate_fill=False)
            st.success(f"Order placed: {order.get('order_id', order)}")
            st.json(order)
    except Exception as e:
        st.error(f"Failed to place order: {e}")

    else:
        st.info("Auto Trade enabled — manual placing disabled.")

    # Orders and trade table
    st.subheader("Orders / Trades")
    try:
        orders = broker.get_orders() if hasattr(broker, "get_orders") else []
        df_orders = pd.DataFrame(orders)
        if df_orders.empty:
            st.info("No orders placed yet.")
        else:
            st.dataframe(df_orders)
    except Exception:
        st.info("Orders not available.")

    # Debug / feed panel
    st.subheader("Feed & Order Debug")
    last_tick = st.session_state.get('last_tick', None)
    last_order_update = st.session_state.get('last_order_update', None)
    if last_tick:
        st.write("Last tick:", last_tick)
    else:
        st.info("No ticks received yet (check SmartAPI feed connection).")
    if last_order_update:
        st.write("Last order update:", last_order_update)
    else:
        st.info("No order updates received yet.")
    ws_status = st.session_state.get('ws_status', 'Not started')
    st.write("WebSocket status:", ws_status)

    last_auto = st.session_state.get('_last_autotrade_order', None)
    if last_auto:
        st.info(f"Last auto order: {last_auto.get('order_id')} — {last_auto.get('side')} {last_auto.get('symbol')} qty={last_auto.get('qty')} price={last_auto.get('price')} status={last_auto.get('status')}")

    # AutoTrader wiring
    def _get_market_price(sym):
        try:
            if hasattr(broker, "market_prices") and broker.market_prices.get(sym) is not None:
                return float(broker.market_prices.get(sym))
            if not paper_mode and isinstance(broker, LiveBroker) and getattr(broker, "conn", None) is not None:
                lb = broker
                res = None
                for name in ("getLTP", "get_ltp", "getQuote", "get_quotes", "getQuoteLtp"):
                    func = getattr(lb.conn, name, None)
                    if callable(func):
                        try:
                            r = func(sym)
                            if isinstance(r, (int, float)):
                                res = float(r)
                                break
                            if isinstance(r, dict):
                                for k in ("ltp", "last_price", "lastTradedPrice"):
                                    if k in r and r[k] is not None:
                                        res = float(r[k])
                                        break
                                if res is not None:
                                    break
                        except Exception:
                            continue
                if res is not None:
                    return res
        except Exception:
            pass
        return None

    
def _simple_strategy():
        try:
            _bias = bias
            atm_strike = atm
            if _bias == "BULLISH":
                action = "BUY"
                instr = f"NIFTY{int(atm_strike)}CE"
            elif _bias == "BEARISH":
                action = "SELL"
                instr = f"NIFTY{int(atm_strike)}PE"
            else:
                return {"action": None}
            mprice = _get_market_price(instr) or price
            # Require lot size from SmartAPI only
            try:
                import streamlit as st
                conn = st.session_state.get('smart_conn', None)
                lot_sz = None
                if conn is not None:
                    try:
                        lot_sz = get_lot_size_from_conn(conn, instr)
                    except Exception:
                        lot_sz = None
                if not lot_sz:
                    # Signal to UI that lot size is missing from API and skip trading
                    try:
                        st.session_state['lot_size_missing'] = True
                        st.session_state['lot_size_error_msg'] = f"Lot size not available from API for {instr}. Auto-trading paused until API provides lot size."
                    except Exception:
                        pass
                    return {"action": None}
                else:
                    try:
                        st.session_state['lot_size_missing'] = False
                        st.session_state.pop('lot_size_error_msg', None)
                    except Exception:
                        pass
            except Exception:
                return {"action": None}
            return {"action": action, "symbol": instr, "price": mprice, "lots": int(qty_lots), "lot_size": int(lot_sz)}
        except Exception:
            return {"action": None}


    # Start/stop AutoTrader
    if auto_trade:
        if '_autotrader' not in st.session_state:
            at = AutoTrader(broker, _get_market_price, _simple_strategy, max_trades_per_day=int(max_trades), per_trade_max_risk=float(per_trade_max_risk), daily_loss_limit=float(daily_loss_limit))
            st.session_state['_autotrader'] = at
            at.start()
            st.sidebar.success("AutoTrader started")
        else:
            # update config
            at = st.session_state['_autotrader']
            at.max_trades = int(max_trades)
            at.per_trade_max_risk = float(per_trade_max_risk)
            at.daily_loss_limit = float(daily_loss_limit)
    else:
        if '_autotrader' in st.session_state:
            try:
                st.session_state['_autotrader'].stop()
            except Exception:
                pass
            st.session_state.pop('_autotrader', None)
            st.sidebar.info("AutoTrader stopped")

    # Export trade log
    if st.button("Export Trade Log"):
        try:
            df = pd.DataFrame(broker.get_orders())
            csv = df.to_csv(index=False)
            import base64
            b64 = base64.b64encode(csv.encode()).decode()
            href = f"data:file/csv;base64,{b64}"
            st.markdown(f"[Download trade log]({href})")
        except Exception:
            st.error("Failed to export trade log.")

if __name__ == "__main__":
    # When run directly, start the Streamlit app
    build_streamlit_app()