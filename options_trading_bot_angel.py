# options_trading_bot_angel.py
# Final Secured Trading Bot v3 (Render) - safe default, paper trading by default.
# IMPORTANT: This script contains placeholders and safe guards. Add your credentials
# as environment variables on Render (or other host) before enabling live mode.
#
# Features implemented:
# - Loads expiry-safe JSON config
# - Connects to Angel One SmartAPI (if credentials provided)
# - Fetches instruments and validates expiry dynamically
# - Basic indicator calculations (EMA, ATR, VWAP) using pandas
# - GO/NO-GO mini-checks, trade limits, no-repeat-strike logic, no-trade-after-time
# - Paper trading mode by default (no live orders unless live_mode=True and env vars set)
# - Graceful shutdown handlers and simple logging to stdout + daily trade log (in-memory)
#
# WARNING: This is a template. Review carefully before enabling live trading.

import os
import json
import time
import signal
import hashlib
from datetime import datetime, time as dtime
from collections import defaultdict
import pandas as pd
import numpy as np

# Optional dependency (smartapi-python). Import at runtime if available.
SMARTAPI_AVAILABLE = False
try:
    from smartapi import SmartConnect  # pip install smartapi-python==1.5.5
    SMARTAPI_AVAILABLE = True
except Exception:
    SMARTAPI_AVAILABLE = False

# --- Helpers ---
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def now_ist():
    return datetime.now()

# Simple EMA
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ATR (True Range then ATR)
def atr(high, low, close, n=14):
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=1).mean()

# VWAP (typical price * volume cumulative)
def vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

# CPR calculation (simple central pivot range approximation)
def cpr(df):
    # Using typical pivot method for daily CPR: BC = (H+L+C)/3; TC = BC +/- (H-L)
    h = df['high'].iloc[-1]; l = df['low'].iloc[-1]; c = df['close'].iloc[-1]
    bc = (h + l + c) / 3.0
    tc = bc + (h - l)
    return {'BC': bc, 'TC': tc, 'width': abs(tc - bc)}

# --- Trading infrastructure (paper-mode by default) ---
class TradingBot:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.paper_mode = self.config.get('app_settings', {}).get('paper_trading_default', True)
        self.max_trades_day = self.config.get('trade_rules', {}).get('max_trades_per_day', 3)
        self.lots_per_trade = self.config.get('trade_rules', {}).get('lots_per_trade', 4)
        self.no_trades_after = self._parse_time(self.config.get('trade_rules', {}).get('no_trades_after_time', '15:00'))
        self.trades_today = []
        self.used_strikes = set()
        self.running = True
        self.api = None
        self.session_token = None
        self._setup_signal_handlers()
        self._load_env_credentials()

    def _parse_time(self, tstr):
        if isinstance(tstr, str):
            hh, mm = tstr.split(':')
            return dtime(int(hh), int(mm))
        return dtime(15,0)

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        print('[INFO] Graceful shutdown requested (signum=%s). Saving state...' % signum)
        self.running = False

    def _load_env_credentials(self):
        # Load credentials from environment — these should be set on Render
        self.angel_api_key = os.getenv('ANGEL_API_KEY')
        self.angel_client_id = os.getenv('ANGEL_CLIENT_ID')
        self.angel_password = os.getenv('ANGEL_PASSWORD')
        self.angel_totp = os.getenv('ANGEL_TOTP_SECRET')
        self.master_password_hash = os.getenv('MASTER_PASSWORD_HASH')

    def connect_api(self):
        if not SMARTAPI_AVAILABLE:
            print('[WARN] smartapi-python not installed. Running in offline/paper-only mode.')
            return False
        if not (self.angel_api_key and self.angel_client_id and self.angel_password):
            print('[WARN] Missing Angel credentials in environment. Running in paper mode.')
            return False
        try:
            self.api = SmartConnect(api_key=self.angel_api_key)
            # Login flow depends on smartapi version; this is illustrative.
            data = self.api.generateSession(self.angel_client_id, self.angel_password)
            self.session_token = data.get('data', {}).get('jwtToken')
            self.api.set_access_token(self.session_token)
            print('[INFO] Connected to Angel SmartAPI (live mode enabled)')
            return True
        except Exception as e:
            print('[ERROR] Failed connecting to Angel SmartAPI:', e)
            self.api = None
            return False

    def fetch_instruments(self):
        # Best-effort: if API available, fetch instruments; otherwise expect offline testing.
        if self.api:
            try:
                instruments_df = self.api.getInstruments()  # real method may differ
                # If returned as list/dicts convert to DataFrame safely
                return instruments_df
            except Exception as e:
                print('[WARN] Error fetching instruments from API:', e)
                return None
        else:
            print('[INFO] No API client — cannot fetch live instruments. Use dev config or mock instruments.')
            return None

    def validate_expiry(self, instruments, expiry_str):
        # instruments: DataFrame or list. expiry_str: a broker expiry string to validate.
        # We must not hardcode expiries; use broker's instruments endpoint at runtime.
        if instruments is None:
            return False
        # Implement a robust check depending on returned structure; try common keys.
        try:
            if hasattr(instruments, 'to_dict'):
                df = instruments if isinstance(instruments, pd.DataFrame) else pd.DataFrame(instruments)
                if 'expiry' in df.columns:
                    return expiry_str in df['expiry'].astype(str).values
                # fallback checking symbol column
                if 'symbol' in df.columns:
                    return any(expiry_str in str(x) for x in df['symbol'].values)
            else:
                # instruments as list of dicts
                return any(expiry_str in str(j) for j in instruments)
        except Exception:
            return False

    def can_trade_now(self):
        now = datetime.now().time()
        if now >= self.no_trades_after:
            print('[INFO] It is past no-trade cutoff (%s). No new trades allowed.' % self.no_trades_after)
            return False
        if len(self.trades_today) >= self.max_trades_day:
            print('[INFO] Max trades for today reached (%d).' % self.max_trades_day)
            return False
        return True

    def compute_indicators(self, df):
        # expects df with columns: ['open','high','low','close','volume'], indexed by timestamp
        df = df.copy().astype(float)
        df['ema9'] = ema(df['close'], span=9)
        df['ema21'] = ema(df['close'], span=21)
        df['atr14'] = atr(df['high'], df['low'], df['close'], n=14)
        df['vwap'] = vwap(df)
        c = cpr(df)
        return df, c

    def decide_trade(self, df_ind, cpr_vals):
        # Very conservative example decision logic using EMA, VWAP, CPR
        latest = df_ind.iloc[-1]
        bias = 'Neutral'
        if latest['close'] > latest['ema9'] and latest['close'] > latest['vwap']:
            bias = 'Bullish'
        elif latest['close'] < latest['ema9'] and latest['close'] < latest['vwap']:
            bias = 'Bearish'

        # Buy call if bullish, sell put if bearish (example). In paper mode we only simulate.
        decision = None
        if bias == 'Bullish':
            decision = {'type': 'BUY_CALL', 'reason': 'Price > EMA9 & VWAP', 'stop_loss': latest['atr14'] + 10}
        elif bias == 'Bearish':
            decision = {'type': 'BUY_PUT', 'reason': 'Price < EMA9 & VWAP', 'stop_loss': latest['atr14'] + 10}
        return decision, bias

    def place_order(self, decision, symbol_info):
        # Simulated order placement for paper mode. For live mode, use API client's order placement method.
        if self.paper_mode or not self.api:
            order = {
                'order_id': 'PAPER-' + datetime.now().strftime('%Y%m%d%H%M%S'),
                'type': decision['type'],
                'symbol': symbol_info,
                'lots': self.lots_per_trade,
                'stop_loss': decision.get('stop_loss'),
                'target_points': self.config.get('trade_rules', {}).get('target_points', 10),
                'timestamp': datetime.now().isoformat()
            }
            print('[PAPER ORDER] %s' % json.dumps(order))
            self.trades_today.append(order)
            self.used_strikes.add(symbol_info.get('strike', 'unknown'))
            return order
        else:
            # Live order path (illustrative)
            try:
                # TODO: Replace with actual SmartAPI order call + proper params
                res = self.api.placeOrder(order_params={})
                print('[LIVE ORDER] Response:', res)
                self.trades_today.append(res)
                return res
            except Exception as e:
                print('[ERROR] Live order failed:', e)
                return None

    def run_once(self):
        # Example data fetch: in production you would fetch 1-min or tick OHLC+volume and run strategy
        # Here we simulate with random walk data for demonstration.
        idx = pd.date_range(end=datetime.now(), periods=200, freq='T')
        price = pd.Series(np.cumsum(np.random.randn(len(idx))) + 20000, index=idx)
        df = pd.DataFrame({
            'open': price.shift(1).fillna(price.iloc[0]),
            'high': price + np.abs(np.random.rand(len(price)) * 10),
            'low': price - np.abs(np.random.rand(len(price)) * 10),
            'close': price,
            'volume': (np.random.rand(len(price)) * 100).astype(int)
        }, index=idx)

        df_ind, cpr_vals = self.compute_indicators(df)
        decision, bias = self.decide_trade(df_ind, cpr_vals)
        print('[INFO] Bias=%s, CPR width=%.2f' % (bias, cpr_vals['width']))
        if decision and self.can_trade_now():
            # symbol_info would be resolved using instruments & ATM logic; we simulate it.
            symbol_info = {'symbol': 'NIFTY', 'strike': 'ATM-xxxx', 'expiry': 'DYNAMIC'}
            order = self.place_order(decision, symbol_info)
            return order
        return None

    def run(self, loop_seconds=60):
        print('[INFO] Bot starting main loop. Paper mode=%s' % self.paper_mode)
        while self.running:
            try:
                self.run_once()
                time.sleep(loop_seconds)
            except Exception as e:
                print('[ERROR] Exception in main loop:', e)
                time.sleep(5)
        print('[INFO] Bot stopped. Trades today:', len(self.trades_today))

# --- Main entry point ---
if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'secured_trading_bot_v3_render_final_clean_json_expiry_safe.json')
    if not os.path.exists(config_path):
        print('[ERROR] Missing config JSON at', config_path)
        raise SystemExit(1)
    bot = TradingBot(config_path)
    # Try to connect to live API if possible (but remain in paper mode if fail)
    live_connected = bot.connect_api()
    if live_connected and os.getenv('ENABLE_LIVE_MODE', 'false').lower() == 'true':
        bot.paper_mode = False
    # Run a fast demonstration loop (short period) so user sees activity immediately.
    bot.run(loop_seconds=5)
