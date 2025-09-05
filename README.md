
Secured Trading Bot v3 (Render) - Streamlit Dashboard (env names fixed) v2

Trading logic updated per user request:
- CE trade requires EMA(9) > EMA(20) AND EMA(9) > VWAP, plus OI increase confirmation
- PE trade requires EMA(9) < EMA(20) AND EMA(9) < VWAP, plus OI increase confirmation
- OI change computed from consecutive LTP/OI reads and must be > oi_change_threshold_percent (configurable)
- Uses environment variables: API_KEY, CLIENT_ID, PASSWORD, MASTER_PASSWORD, TOTP
- Live submits via Angel SmartAPI placeOrder with realistic payload mapping. If SmartAPI returns an error, the UI logs it for mapping adjustments.
