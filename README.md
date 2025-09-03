# Secured Options Trading Bot v3 (Render, Persistent, Ultimate Wrapper, /ping Endpoint)

## 🚀 Features
- Expiry dropdown + ATM detection (remembers selection across refreshes)
- Bias Dashboard (always shown, even if Neutral)
- Auto CE/PE selection & trade execution
- Unified Paper/Live behaviour
- Risk rules: 4 lots, ATR+10 SL, Target=10, TSL, 3 trades/day, no repeat strike, no trades after 3 PM
- Trade log always visible (shows 'No trades yet' if empty)
- Auto-refresh every 5 seconds
- 🚫 Hard stop at –₹8,000/day
- ✅ Lock-in at +₹15,000/day
- 📊 Summary panel (P&L, trades, status)
- 🛠 Ultimate instruments wrapper (SmartApi methods + JSON fallback)
- 🔒 Persistent Login & Expiry memory
- ⚡ New: `/ping` endpoint for uptime monitors

## 🛠 Ping Endpoint
Use this URL for keep-alive pings:  
```
https://options-trading-bot-angelv2.onrender.com/?ping=1
```
It responds with `pong` instantly without requiring login.
