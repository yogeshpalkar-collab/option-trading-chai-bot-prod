import os
from logzero import logger

try:
    from SmartApi.smartConnect import SmartConnect
except Exception:
    try:
        from smartapi import SmartConnect
    except Exception:
        try:
            from smartapi.smartConnect import SmartConnect
        except Exception as e:
            raise ImportError("smartapi library not found. Install smartapi-python.") from e

class SmartAPIWrapper:
    def __init__(self, api_key):
        self.client = SmartConnect(api_key=api_key)
        self.auth = None

    def login(self, client_id, password, totp=None):
        try:
            if totp:
                import pyotp
                totp_code = pyotp.TOTP(totp).now()
                resp = self.client.generateSession(client_id, password, totp_code)
            else:
                resp = self.client.generateSession(client_id, password)
            if not resp or not resp.get('data'):
                logger.error("Login failed: %s", resp)
                raise RuntimeError("SmartAPI login failed")
            self.auth = resp['data']
            logger.info("SmartApi login OK")
            return resp['data']
        except Exception as e:
            logger.exception("Login exception: %s", e)
            raise

    def place_order(self, order_params):
        try:
            res = self.client.placeOrder(order_params)
            logger.info("Order placed, response: %s", res)
            return res
        except Exception as e:
            logger.exception("Order failed: %s", e)
            raise

    def get_instruments(self, exchange='NFO', symbol_filter=None):
        try:
            if hasattr(self.client, 'get_instruments'):
                instruments = self.client.get_instruments(exchange)
            elif hasattr(self.client, 'getAllInstrumentTokens'):
                instruments = self.client.getAllInstrumentTokens(exchange)
            elif hasattr(self.client, 'getMaster'):
                instruments = self.client.getMaster(exchange)
            else:
                instruments = []
        except Exception:
            instruments = []
        if symbol_filter and instruments:
            try:
                filtered = [ins for ins in instruments if symbol_filter.upper() in (ins.get('tradingsymbol') or ins.get('name') or '')]
                return filtered or instruments
            except Exception:
                return instruments
        return instruments

    def get_candle_data(self, params):
        try:
            return self.client.getCandleData(params)
        except Exception as e:
            logger.exception("Candle fetch failed: %s", e)
            raise
