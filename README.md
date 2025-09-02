# Options Trading Bot (Angel One) - Final Production

## Features
- Uses official Angel One SmartApi (no stubs)
- Master password protection
- Angel One login via env vars
- Paper/Live toggle (default Paper)
- Expiry dropdown (weekly + monthly from Angel)
- ATR(14)+10 Stop Loss, TSL, Target = 10 pts
- Max 3 trades/day, no repeat strikes, no trades after 3PM
- Bias Dashboard (EMA, VWAP, CPR, ATR)
- CPR treated as advisory (trend reversal flag if mismatch)
- CPR Status shown in dashboard and logged in trade table
- Trade Log Table with P&L in ₹ and Order ID (for live trades)
- No Simulate Trade button
- Dynamic ATM Strike calculation using Angel LTP (nearest 50)

## Deployment (Render)
- Build Command:
  pip install -r requirements.txt
- Start Command:
  streamlit run options_trading_bot_angel.py --server.port 10000 --server.address 0.0.0.0

## Required Environment Variables
- MASTER_PASSWORD
- API_KEY
- CLIENT_ID
- PASSWORD
- TOTP
