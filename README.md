# Options Trading Bot (Angel One) - Secured v3 Render Final Engine

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

## Files
- options_trading_bot_angel.py → bot script
- requirements.txt → dependencies
- README.md → this file

## Deployment
1. Upload to Render or Streamlit Cloud.
2. Set environment variables: API_KEY, CLIENT_ID, PASSWORD, TOTP.
3. Deploy. Default mode starts as Paper Trading.
