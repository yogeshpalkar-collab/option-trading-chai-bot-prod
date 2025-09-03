# Secured Options Trading Bot v3 (Render, Production)

## 🚀 Features
- Real Angel One SmartAPI v1.5.5 integration (no stubs)
- Master password protection
- Environment variable based secrets (API_KEY, CLIENT_ID, PASSWORD, TOTP, MASTER_PASSWORD)
- Paper/Live trading toggle (default = Paper)
- Expiry dropdown, 4 lots per trade, dynamic lot size
- SL = ATR(14) + 10 points, Trailing SL, Target = 10 points
- Max 3 trades/day, no repeat strike, no trades after 3 PM
- Bias dashboard (EMA, VWAP, CPR, ATR)
- Trade log table with CPR status, P&L in ₹, Order ID for live trades

## 🛠 Setup (Local)
1. Clone repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add `.streamlit/secrets.toml` with:
   ```toml
   API_KEY = "your_api_key"
   CLIENT_ID = "your_client_id"
   PASSWORD = "your_password"
   TOTP = "your_totp_secret"
   MASTER_PASSWORD = "your_master_password"
   ```
3. Run:
   ```bash
   streamlit run options_trading_bot_angel.py
   ```

## ☁️ Deployment (Render)
1. Push repo to GitHub
2. Create new Render Web Service
3. Set build command:
   ```bash
   pip install -r requirements.txt
   ```
4. Set run command:
   ```bash
   streamlit run options_trading_bot_angel.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add environment variables (same as secrets.toml above)

## ✅ Notes
- Default mode = Paper Trading
- Live mode requires valid Angel One credentials
- No stubs, full production-ready SmartAPI integration
