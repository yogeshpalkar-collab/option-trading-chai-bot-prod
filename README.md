# Secured Options Trading Bot v3 (Render, Production, Auto-Refresh, Max Loss & Max Profit, Summary Panel)

## 🚀 Features
- Expiry dropdown + ATM detection
- Bias Dashboard (EMA, VWAP, CPR, ATR, OI Change)
- Auto CE/PE selection & trade execution
- Unified Paper/Live behaviour
- Risk rules: 4 lots, ATR+10 SL, Target=10, TSL, 3 trades/day, no repeat strike, no trades after 3 PM
- Trade log with **live TSL updates**
- Auto-refresh every 5 seconds for live updates
- 🚫 Hard-coded Max Daily Loss = –₹8,000 (trading halted after breach)
- ✅ Hard-coded Max Daily Profit = +₹15,000 (trading halted after hit)
- 📊 **Summary Panel**: shows P&L, trades taken, remaining slots, and status (Active/Halted)

## Notes
- Paper mode logs simulated Order IDs (`SIM-xxxx`)
- Live mode logs real Angel One Order IDs
