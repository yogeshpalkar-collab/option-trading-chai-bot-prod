# Options Trading Bot (Angel One) - Secured v3 Render Final Clean Version

## Overview
Final clean production version with:
- Real-time NIFTY feed via Angel One SmartAPI WebSocket.
- Full bias logic: EMA(9/21), VWAP, CPR (relaxed for reversals), ATR, OI Change.
- Trade engine:
  - ATM CE/PE entries
  - SL = ATR(14)+10
  - TSL enabled
  - Target Relax Mode (capture >10 profit if price spikes)
  - Max 3 trades/day, no repeat strike, no trades after 3PM
- Paper mode (default) and Live mode behave identically.
- ✅ Instruments fetched only via Angel API (no stubs, no fake fallbacks).
- ✅ Expiry dropdown built only from real NIFTY option expiries.
- ✅ Dashboard banners: Market Status, Trade Engine Status, Instrument Source + Timestamp.
- ✅ Master Password protection (via `MASTER_PASSWORD` env variable).
- ✅ SmartAPI diagnostic banner to show exact installed package & version.

## Files
- options_trading_bot_angel.py → bot script
- requirements.txt → dependencies
- README.md → this file

## Render Deployment
1. Upload all files to Render.
2. Set environment variables: API_KEY, CLIENT_ID, PASSWORD, TOTP, MASTER_PASSWORD.
3. In **Build Command**, paste this:
   ```bash
   pip uninstall -y SmartAPI smartapi-python && pip install --no-cache-dir smartapi-python==1.5.5 && pip install -r requirements.txt
   ```
4. In **Start Command**, set:
   ```bash
   streamlit run options_trading_bot_angel.py --server.port 10000 --server.address 0.0.0.0
   ```
5. Deploy. Default mode starts as Paper Trading.
