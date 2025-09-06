# Options Trading Bot (Angel One) - Final Bundle

## Overview
This is the final production-ready bundle of the Options Trading Bot.

- Auto-subscribes to **NIFTY ATM** using Angel One SmartAPI WebSocket
- **Paper and Live mode** (identical lifecycle via SmartAPIProxy for Paper)
- **Lots selectable via dropdown (1 to 50)**
- **ATR-based trailing stop-loss** hardcoded (factor = 0.4, tiers = 3 / 6 / 10)
- **LiveBroker** wired to SmartAPI order placement
- **TSL & Profit table** shows `order_via` and `total_invested`
- **Highlighted banner** showing current Mode and last order route

## Setup
### Install dependencies
```bash
pip install -r requirements.txt
```

### Required Environment Variables (for Live Mode)
- `API_KEY` (Angel One API key)
- `CLIENT_ID` (your client id / username)
- `PASSWORD` (login password)
- `TOTP` (time-based OTP if enabled)

### Run in Streamlit
```bash
streamlit run options_trading_bot_angel.py
```

## Modes
- **Paper**: default, uses WebSocket feed + SmartAPIProxy (no real trades)
- **Live**: places orders via SmartAPI (ensure credentials are correct)
