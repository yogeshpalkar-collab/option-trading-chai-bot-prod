# Secured Options Trading Bot v3 (Render, Summary Final, Ultimate Wrapper)

## 🚀 Features
- Expiry dropdown + ATM detection
- Bias Dashboard (EMA, VWAP, CPR, ATR, OI Change)
- Auto CE/PE selection & trade execution
- Unified Paper/Live behaviour
- Risk rules: 4 lots, ATR+10 SL, Target=10, TSL, 3 trades/day, no repeat strike, no trades after 3 PM
- Trade log with **live TSL updates**
- Auto-refresh every 5 seconds
- 🚫 Hard stop at –₹8,000/day
- ✅ Lock-in at +₹15,000/day
- 📊 Summary panel (P&L, trades, status)
- 🛠 **Ultimate instruments wrapper**:  
  1. `get_instruments()`  
  2. `getInstruments()`  
  3. `get_exchange_instruments("NFO")`  
  4. Official JSON fallback (`OpenAPIScripMaster.json`)

## Notes
- Guaranteed to fetch instruments even if SmartApi changes
- Paper mode logs simulated Order IDs (`SIM-xxxx`)
- Live mode logs real Angel One Order IDs
