# Options Trading Bot (Angel One) - Secured v3 Render Final Engine (Smart Expiry + Logzero + Diagnostic)

## Overview
Final production version with:
- Real-time NIFTY feed via Angel One SmartAPI WebSocket.
- Full bias logic: EMA(9/21), VWAP, CPR (relaxed for reversals), ATR, OI Change.
- Trade engine:
  - ATM CE/PE entries
  - SL = ATR(14)+10
  - TSL enabled
  - Target Relax Mode (capture >10 profit if price spikes)
  - Max 3 trades/day, no repeat strike, no trades after 3PM
- Paper mode (default) and Live mode behave identically.
- ✅ Instruments auto-refresh daily with API and CSV fallback.
- ✅ Dashboard banners: Market Status, Trade Engine Status, Instrument Source + Timestamp.
- ✅ Master Password protection (via `MASTER_PASSWORD` env variable).
- ✅ Expiry dropdown with nearest Tuesday weekly auto-selected (fallback to monthly).
- ✅ SmartAPI support with corrected order: `get_instrument_master()` first, fallback to `getInstruments()`.
- ✅ Explicit error if neither method exists: **SmartAPI version mismatch**.
- ✅ NEW: SmartAPI package/version diagnostic banner shown on dashboard.

## Files
- options_trading_bot_angel.py → bot script
- requirements.txt → dependencies
- README.md → this file

## Render Deployment
1. Upload all files to Render.
2. Set environment variables: API_KEY, CLIENT_ID, PASSWORD, TOTP, MASTER_PASSWORD.
3. In **Build Command**, paste this:
   ```bash
   pip uninstall -y SmartAPI smartapi-python && pip install smartapi-python==1.5.5 && pip install -r requirements.txt
   ```
   This guarantees removal of any wrong SmartAPI package and clean install of the correct version.
4. In **Start Command**, set:
   ```bash
   streamlit run options_trading_bot_angel.py --server.port 10000 --server.address 0.0.0.0
   ```
5. Deploy. Default mode starts as Paper Trading.
