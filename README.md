# Options Trading Bot (Angel One) - Secured v3 Render Final Engine (Smart Expiry + Corrected SmartAPI Fallback)

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

## Files
- options_trading_bot_angel.py → bot script
- requirements.txt → dependencies
- README.md → this file

## Deployment
1. Upload all files to Render or Streamlit Cloud.
2. Set environment variables: API_KEY, CLIENT_ID, PASSWORD, TOTP, MASTER_PASSWORD.
3. Start command:
   ```bash
   streamlit run options_trading_bot_angel.py --server.port 10000 --server.address 0.0.0.0
   ```
4. Default mode starts as Paper Trading.
