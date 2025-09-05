# Secured Trading Bot v3 (Render)

This is the final secured trading bot for Angel One, deployed on Render.

## Contents
- `options_trading_bot_angel.py` → Main bot script (template & safe defaults)
- `requirements.txt` → Python dependencies
- `README.md` → Setup & usage guide
- `secured_trading_bot_v3_render_final_clean_json_expiry_safe.json` → Expiry-safe config

## Setup (Render)
1. Upload these files to your Render service.
2. Add environment variables:
   - ANGEL_API_KEY
   - ANGEL_CLIENT_ID
   - ANGEL_PASSWORD
   - ANGEL_TOTP_SECRET
   - MASTER_PASSWORD_HASH
   - (Optional) ENABLE_LIVE_MODE=true to allow live orders when credentials are present.
3. Deploy the service. Default mode = Paper Trading.

## Notes
- JSON config ensures expiry-safe handling (no hardcoded dates).
- Bot stops trading after 3:00 PM IST.
- Max 3 trades/day, no repeat strike, SL = ATR(14)+10, TSL active.
- This is a template: review and replace simulated data fetch with real OHLC+volume feed & ATM strike resolution.
