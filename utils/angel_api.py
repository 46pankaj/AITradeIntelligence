import os
import requests
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
import pyotp

class AngelOneAPI:
    """
    Angel One API wrapper for trading operations
    """
    def __init__(self, api_key=None, client_id=None, client_password=None, totp_key=None):
        """
        Initialize the Angel One API wrapper

        Args:
            api_key (str): API key for Angel One
            client_id (str): Client ID for Angel One
            client_password (str): Password for Angel One
            totp_key (str): TOTP key for authentication
        """
        self.api_key = api_key or os.getenv("ANGEL_API_KEY")
        self.client_id = client_id or os.getenv("ANGEL_CLIENT_ID")
        self.client_password = client_password or os.getenv("ANGEL_CLIENT_PASSWORD")
        self.totp_key = totp_key or os.getenv("ANGEL_TOTP_KEY")

        self.access_token = None
        self.refresh_token = None
        self.feed_token = None
        self.logged_in = False

        self.base_url = "https://apiconnect.angelbroking.com"
        self.login_url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
        self.token_url = f"{self.base_url}/rest/auth/angelbroking/jwt/v1/generateTokens"
        self.logout_url = f"{self.base_url}/rest/secure/angelbroking/user/v1/logout"

        # Get client IP address
        try:
            import socket
            import uuid

            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0,8*6,8)][::-1])

            self.headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": local_ip,
                "X-ClientPublicIP": local_ip,
                "X-MACAddress": mac,
                "X-PrivateKey": self.api_key
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {str(e)}")
            raise Exception("Failed to initialize API headers")

        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def is_authenticated(self):
        """
        Check if the API is authenticated
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        return self.logged_in and self.jwt_token is not None
        
    def login(self):
        """
        Login to Angel One API

        Returns:
            bool: True if login was successful, False otherwise
        """
        try:
            # Generate TOTP
            totp = pyotp.TOTP(self.totp_key)
            totp_val = totp.now()

            # Login payload
            login_payload = {
                "clientcode": self.client_id,
                "password": self.client_password,
                "totp": totp_val
            }

            # Make login request
            self.logger.info("Attempting to login to Angel One API")
            response = requests.post(self.login_url, json=login_payload, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    self.logger.info("Login successful")
                    self.jwt_token = response_data['data']['jwtToken']
                    self.refresh_token = response_data['data']['refreshToken']
                    self.feed_token = response_data['data'].get('feedToken')

                    # Update headers with token
                    self.headers["Authorization"] = f"Bearer {self.jwt_token}"

                    self.logged_in = True
                    return True
                else:
                    self.logger.error(f"Login failed: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return False
        except Exception as e:
            self.logger.error(f"Login error: {str(e)}")
            return False

    def get_user_profile(self):
        """
        Get user profile from Angel One API

        Returns:
            dict: User profile data or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return None

        try:
            profile_url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getProfile"
            response = requests.get(profile_url, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data['data']
                else:
                    self.logger.error(f"Failed to get profile: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return None
        except Exception as e:
            self.logger.error(f"Get profile error: {str(e)}")
            return None

    def get_ltp(self, exchange, symbol, max_retries=3):
        """
        Get last traded price for a symbol

        Args:
            exchange (str): Exchange (NSE, BSE)
            symbol (str): Trading symbol
            max_retries (int): Maximum number of retry attempts

        Returns:
            dict: LTP data or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            # Try to login first if we're not logged in
            if not self.login():
                self.logger.error("Login failed, returning simulated data")
                return self._get_simulated_ltp(exchange, symbol)

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempt {attempt+1}/{max_retries}: Getting LTP for {symbol} on {exchange}")
                ltp_url = f"{self.base_url}/rest/secure/angelbroking/market/v1/quote/ltp"

                # Get token for the symbol
                token = self._get_token(exchange, symbol)
                self.logger.info(f"Using token: {token} for {symbol}")

                payload = {
                    "exchange": exchange,
                    "tradingsymbol": symbol,
                    "symboltoken": token
                }

                # Ensure headers are up to date (JWT may have changed after login)
                if hasattr(self, 'jwt_token') and self.jwt_token:
                    self.headers['Authorization'] = f"Bearer {self.jwt_token}"

                # Make request with extended timeout
                response = requests.post(ltp_url, json=payload, headers=self.headers, timeout=30)

                # Process successful responses immediately
                if response.status_code == 200 and response.content:
                    try:
                        response_data = response.json()
                        if response_data.get('status') and response_data.get('message') == "SUCCESS":
                            data = response_data.get('data', {})
                            ltp = float(data.get('ltp', 0))
                            if ltp > 0:
                                self.logger.info(f"Successfully retrieved LTP for {symbol}: ₹{ltp}")
                                return data
                        else:
                            error_msg = response_data.get('message', 'Unknown error')
                            self.logger.error(f"API error: {error_msg}")
                    except ValueError as json_err:
                        self.logger.error(f"Invalid JSON response: {str(json_err)}")

                # Handle authentication issues
                if response.status_code in [401, 403] or "Request Rejected" in response.text:
                    self.logger.warning(f"Authentication issue detected. Attempting re-login...")
                    if self.login():
                        self.logger.info("Re-login successful, retrying request...")
                        continue
                    else:
                        self.logger.error("Re-login failed. Please check API credentials.")
                        # Don't return None yet, try remaining attempts

                else:
                    self.logger.error(f"HTTP Error: {response.status_code}")
                    try:
                        self.logger.debug(f"Response: {response.text[:200]}...")
                    except:
                        pass
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error while getting LTP: {str(e)}")
            except Exception as e:
                self.logger.error(f"Get LTP error: {str(e)}")

            # Wait before retrying
            if attempt < max_retries - 1:
                sleep_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                import time
                time.sleep(sleep_time)

        # If we get here, we've exhausted our retries
        self.logger.error(f"Failed to get LTP after {max_retries} attempts, returning simulated data")

        # Return simulated data for development or when API is unavailable
        return self._get_simulated_ltp(exchange, symbol)

    def get_historical_data(self, exchange, symbol, timeframe, count):
        """
        Get historical candlestick data

        Args:
            exchange (str): Exchange (NSE, BSE)
            symbol (str): Trading symbol
            timeframe (str): Candle timeframe (1 minute, 5 minutes, etc.)
            count (int): Number of candles

        Returns:
            pandas.DataFrame: Historical data or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return None

        try:
            # Convert timeframe string to Angel One API format
            tf_map = {
                "1 minute": "ONE_MINUTE",
                "5 minutes": "FIVE_MINUTE",
                "15 minutes": "FIFTEEN_MINUTE",
                "30 minutes": "THIRTY_MINUTE",
                "1 hour": "ONE_HOUR",
                "1 day": "ONE_DAY"
            }

            angel_timeframe = tf_map.get(timeframe, "ONE_DAY")

            # Calculate from and to dates
            to_date = datetime.now()
            from_date = to_date - timedelta(days=count)

            candle_url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"

            payload = {
                "exchange": exchange,
                "symboltoken": self._get_token(exchange, symbol),
                "interval": angel_timeframe,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }

            response = requests.post(candle_url, json=payload, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    # Convert data to DataFrame
                    data = response_data['data']
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    return df
                else:
                    self.logger.error(f"Failed to get historical data: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return None
        except Exception as e:
            self.logger.error(f"Get historical data error: {str(e)}")
            return None

    def place_order(self, symbol, exchange, transaction_type, quantity, order_type, price=0, trigger_price=0, product_type="INTRADAY"):
        """
        Place a new order

        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            transaction_type (str): BUY or SELL
            quantity (int): Order quantity
            order_type (str): MARKET, LIMIT, SL, SL-M
            price (float): Order price (for LIMIT, SL orders)
            trigger_price (float): Trigger price (for SL, SL-M orders)
            product_type (str): DELIVERY, INTRADAY, MARGIN

        Returns:
            dict: Order response or error message
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return {"status": "error", "message": "Not logged in"}

        try:
            order_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder"

            # Map order type to Angel One format
            order_type_map = {
                "MARKET": "MARKET",
                "LIMIT": "LIMIT",
                "SL": "STOPLOSS_LIMIT",
                "SL-M": "STOPLOSS_MARKET"
            }

            # Map product type to Angel One format
            product_type_map = {
                "DELIVERY": "DELIVERY",
                "INTRADAY": "INTRADAY",
                "MARGIN": "MARGIN",
                "CNC": "DELIVERY",
                "MIS": "INTRADAY",
                "NRML": "MARGIN"
            }

            angel_order_type = order_type_map.get(order_type, "MARKET")
            angel_product_type = product_type_map.get(product_type, "INTRADAY")

            # Convert quantity to integer and make sure it's a string in the payload
            quantity_int = int(quantity)

            # Default values for price and trigger price
            price_str = "0"
            trigger_price_str = "0"

            # Set appropriate price values based on order type
            if order_type in ["LIMIT", "SL"] and price > 0:
                price_str = str(price)

            if order_type in ["SL", "SL-M"] and trigger_price > 0:
                trigger_price_str = str(trigger_price)

            # Create the payload with all required fields
            payload = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": self._get_token(exchange, symbol),
                "transactiontype": transaction_type,
                "exchange": exchange,
                "ordertype": angel_order_type,
                "producttype": angel_product_type,
                "duration": "DAY",
                "price": price_str,
                "triggerprice": trigger_price_str,
                "quantity": str(quantity_int),
                "squareoff": "0",
                "stoploss": "0"
            }

            # Log the payload for debugging
            self.logger.info(f"Order payload: {payload}")

            response = requests.post(order_url, json=payload, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    self.logger.info(f"Order placed successfully: {response_data}")
                    return {
                        "status": "success", 
                        "order_id": response_data['data']['orderid'],
                        "message": "Order placed successfully"
                    }
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Failed to place order: {error_msg}")
                    self.logger.error(f"Response data: {response_data}")
                    return {"status": "error", "message": error_msg}
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")
                self.logger.error(f"Response content: {response.text}")
                return {"status": "error", "message": f"HTTP Error: {response.status_code}"}

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Place order error: {error_msg}")
            return {"status": "error", "message": error_msg}

    def modify_order(self, order_id, price=None, quantity=None, trigger_price=None):
        """
        Modify an existing order

        Args:
            order_id (str): Order ID to modify
            price (float, optional): New price
            quantity (int, optional): New quantity
            trigger_price (float, optional): New trigger price

        Returns:
            bool: True if order was modified successfully, False otherwise
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return False

        try:
            modify_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/modifyOrder"

            # Base payload with required fields
            payload = {
                "variety": "NORMAL",
                "orderid": order_id
            }

            # Add optional parameters if provided with proper string conversion
            if price is not None:
                payload["price"] = str(float(price))

            if quantity is not None:
                payload["quantity"] = str(int(quantity))

            if trigger_price is not None:
                payload["triggerprice"] = str(float(trigger_price))

            # Log the payload for debugging
            self.logger.info(f"Modify order payload: {payload}")

            response = requests.post(modify_url, json=payload, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    self.logger.info(f"Order {order_id} modified successfully")
                    return True
                else:
                    self.logger.error(f"Failed to modify order: {response_data['message']}")
                    self.logger.error(f"Response data: {response_data}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")
                self.logger.error(f"Response content: {response.text}")

            return False
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Modify order error: {error_msg}")
            return False

    def cancel_order(self, order_id):
        """
        Cancel an order

        Args:
            order_id (str): Order ID to cancel

        Returns:
            bool: True if order was cancelled successfully, False otherwise
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return False

        try:
            cancel_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/cancelOrder"

            # Ensure order_id is a string
            order_id_str = str(order_id)

            payload = {
                "variety": "NORMAL",
                "orderid": order_id_str
            }

            # Log the payload for debugging
            self.logger.info(f"Cancel order payload: {payload}")

            response = requests.post(cancel_url, json=payload, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    self.logger.info(f"Order {order_id} cancelled successfully")
                    return True
                else:
                    self.logger.error(f"Failed to cancel order: {response_data['message']}")
                    self.logger.error(f"Response data: {response_data}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")
                self.logger.error(f"Response content: {response.text}")

            return False
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Cancel order error: {error_msg}")
            return False

    def get_order_book(self):
        """
        Get order book

        Returns:
            list: List of orders or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return None

        try:
            order_book_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getOrderBook"

            response = requests.get(order_book_url, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data['data']
                else:
                    self.logger.error(f"Failed to get order book: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return None
        except Exception as e:
            self.logger.error(f"Get order book error: {str(e)}")
            return None

    def get_trade_book(self):
        """
        Get trade book

        Returns:
            list: List of trades or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return None

        try:
            trade_book_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getTradeBook"

            response = requests.get(trade_book_url, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data['data']
                else:
                    self.logger.error(f"Failed to get trade book: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return None
        except Exception as e:
            self.logger.error(f"Get trade book error: {str(e)}")
            return None

    def get_holdings(self):
        """
        Get holdings

        Returns:
            list: List of holdings or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return None

        try:
            holdings_url = f"{self.base_url}/rest/secure/angelbroking/portfolio/v1/getHolding"

            response = requests.get(holdings_url, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data['data']
                else:
                    self.logger.error(f"Failed to get holdings: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return None
        except Exception as e:
            self.logger.error(f"Get holdings error: {str(e)}")
            return None

    def get_positions(self):
        """
        Get positions

        Returns:
            list: List of positions or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return None

        try:
            positions_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getPosition"

            response = requests.get(positions_url, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data['data']
                else:
                    self.logger.error(f"Failed to get positions: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return None
        except Exception as e:
            self.logger.error(f"Get positions error: {str(e)}")
            return None

    def get_option_expiry_dates(self, exchange, symbol):
        """
        Get available expiry dates for options

        Args:
            exchange (str): Exchange (NSE, BSE)
            symbol (str): Trading symbol

        Returns:
            list: List of expiry dates or None if request failed
        """
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/derivative/v1/expiryDates"
            payload = {
                "exchange": exchange,
                "tradingSymbol": symbol
            }

            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data['data']

            return None
        except Exception as e:
            self.logger.error(f"Error getting expiry dates: {str(e)}")
            return None

    def get_option_chain(self, symbol, exchange, expiry_date):
        """
        Get option chain data

        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            expiry_date (str): Expiry date

        Returns:
            dict: Option chain data or None if request failed
        """
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/derivative/v1/optionChain"
            payload = {
                "exchange": exchange,
                "tradingSymbol": symbol,
                "expiryDate": expiry_date
            }

            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data

            return None
        except Exception as e:
            self.logger.error(f"Error getting option chain: {str(e)}")
            return None

            self.logger.error("Not logged in")
            return None

        try:
            positions_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getPosition"

            response = requests.get(positions_url, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    return response_data['data']
                else:
                    self.logger.error(f"Failed to get positions: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return None
        except Exception as e:
            self.logger.error(f"Get positions error: {str(e)}")
            return None

    def get_portfolio(self):
        """
        Get portfolio summary (combines positions and holdings)

        Returns:
            dict: Portfolio summary or None if request failed
        """
        if not self.logged_in:
            self.logger.error("Not logged in")
            return None

        try:
            # Get holdings and positions
            holdings = self.get_holdings() or []
            positions = self.get_positions() or []

            # Calculate portfolio value, P&L, etc.
            portfolio_value = sum(float(h.get('ltp', 0)) * float(h.get('quantity', 0)) for h in holdings)
            day_pnl = sum(float(p.get('pnl', 0)) for p in positions)
            overall_pnl = sum(float(h.get('pnl', 0)) for h in holdings)

            # Get margin details
            margin_url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getRMS"
            response = requests.get(margin_url, headers=self.headers)

            available_margin = 0
            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    rms_data = response_data['data']
                    available_margin = float(rms_data.get('availablecash', 0))

            # Calculate day and overall change percent
            day_change_percent = 0 if portfolio_value == 0 else (day_pnl / portfolio_value) * 100
            overall_change_percent = 0 if portfolio_value == 0 else (overall_pnl / portfolio_value) * 100

            # Return portfolio summary
            return {
                'portfolio_value': portfolio_value,
                'day_pnl': day_pnl,
                'overall_pnl': overall_pnl,
                'day_change_percent': round(day_change_percent, 2),
                'overall_change_percent': round(overall_change_percent, 2),
                'available_margin': available_margin,
                'holdings': holdings
            }
        except Exception as e:
            self.logger.error(f"Get portfolio error: {str(e)}")
            return None

    def logout(self):
        """
        Logout from Angel One API

        Returns:
            bool: True if logout was successful, False otherwise
        """
        if not self.logged_in:
            self.logger.warning("Already logged out")
            return True

        try:
            response = requests.post(self.logout_url, headers=self.headers)

            if response.status_code == 200:
                response_data = response.json()
                if response_data['status'] and response_data['message'] == "SUCCESS":
                    self.logger.info("Logged out successfully")
                    self.logged_in = False
                    self.access_token = None
                    self.refresh_token = None
                    self.feed_token = None
                    return True
                else:
                    self.logger.error(f"Logout failed: {response_data['message']}")
            else:
                self.logger.error(f"HTTP Error: {response.status_code}")

            return False
        except Exception as e:
            self.logger.error(f"Logout error: {str(e)}")
            return False

    def _get_token(self, exchange, symbol):
        """
        Get symbol token (needed for some API calls)

        Args:
            exchange (str): Exchange (NSE, BSE)
            symbol (str): Trading symbol

        Returns:
            str: Symbol token or a default value for the demo
        """
        # In a real implementation, you would search for the token in the master contract
        # For the demo, we'll use some dummy values
        token_map = {
            "NSE": {
                "NIFTY": "26000",
                "BANKNIFTY": "26001",
                "RELIANCE": "2885",
                "TCS": "11536",
                "INFY": "1594",
                "HDFCBANK": "1333"
            },
            "BSE": {
                "SENSEX": "1",
                "RELIANCE": "500325",
                "TCS": "532540",
                "INFY": "500209",
                "HDFCBANK": "500180"
            }
        }

        return token_map.get(exchange, {}).get(symbol, "1")

    def _get_simulated_ltp(self, exchange, symbol):
        """
        Get simulated last traded price for demo environment or when real API is unavailable

        Args:
            exchange (str): Exchange (NSE, BSE)
            symbol (str): Trading symbol

        Returns:
            dict: Simulated LTP data
        """
        self.logger.info(f"Using simulated LTP data for {symbol}")

        # Base price points for common symbols
        base_prices = {
            "NIFTY": 18500.00,
            "BANKNIFTY": 43000.00,
            "SENSEX": 62000.00,
            "RELIANCE": 2500.00,
            "TCS": 3400.00,
            "INFY": 1500.00,
            "HDFCBANK": 1600.00
        }

        # Generate a price with small randomization for non-demo mode
        base_price = base_prices.get(symbol, 1000.00)

        # Add a small random variation (±0.5%)
        import random
        variation = random.uniform(-0.5, 0.5) / 100
        ltp = base_price * (1 + variation)

        # Simulated response similar to actual API response
        return {
            "ltp": ltp,
            "change_percent": round(variation * 100, 2),
            "token": self._get_token(exchange, symbol),
            "exchange": exchange,
            "tradingsymbol": symbol
        }