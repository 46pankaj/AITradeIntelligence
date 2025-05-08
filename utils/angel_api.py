import os
import requests
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
import pyotp
import time
from typing import Optional, Dict, List, Union

class AngelOneAPI:
    """
    Complete Angel One API wrapper with all original methods and improved error handling
    """
    def __init__(self, api_key=None, client_id=None, client_password=None, totp_key=None):
        """
        Initialize the Angel One API wrapper with enhanced error handling
        
        Args:
            api_key (str): API key for Angel One
            client_id (str): Client ID for Angel One
            client_password (str): Password for Angel One
            totp_key (str): TOTP key for authentication
        """
        # Initialize configuration
        self.api_key = api_key or os.getenv("ANGEL_API_KEY")
        self.client_id = client_id or os.getenv("ANGEL_CLIENT_ID")
        self.client_password = client_password or os.getenv("ANGEL_CLIENT_PASSWORD")
        self.totp_key = totp_key or os.getenv("ANGEL_TOTP_KEY")

        # Authentication state
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        self.logged_in = False

        # API endpoints
        self.base_url = "https://apiconnect.angelbroking.com"
        self.login_url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
        self.token_url = f"{self.base_url}/rest/auth/angelbroking/jwt/v1/generateTokens"
        self.logout_url = f"{self.base_url}/rest/secure/angelbroking/user/v1/logout"

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize headers with system information
        self.headers = self._initialize_headers()

    def _initialize_headers(self) -> Dict[str, str]:
        """Initialize request headers with system information"""
        try:
            import socket
            import uuid
            
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                          for elements in range(0, 8*6, 8)][::-1])

            return {
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
            self.logger.error(f"Header initialization error: {str(e)}")
            return {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-PrivateKey": self.api_key
            }

    def is_authenticated(self) -> bool:
        """Check if the API is authenticated"""
        return self.logged_in and self.jwt_token is not None

    def login(self, max_retries: int = 3) -> bool:
        """
        Login to Angel One API with retry logic
        
        Args:
            max_retries (int): Maximum number of login attempts
            
        Returns:
            bool: True if login succeeded
        """
        for attempt in range(max_retries):
            try:
                # Generate TOTP
                totp = pyotp.TOTP(self.totp_key)
                totp_val = totp.now()

                # Prepare login payload
                login_payload = {
                    "clientcode": self.client_id,
                    "password": self.client_password,
                    "totp": totp_val
                }

                self.logger.info(f"Login attempt {attempt + 1}/{max_retries}")
                response = requests.post(
                    self.login_url,
                    json=login_payload,
                    headers=self.headers,
                    timeout=10
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('status') and response_data.get('message') == "SUCCESS":
                        self.jwt_token = response_data['data']['jwtToken']
                        self.refresh_token = response_data['data']['refreshToken']
                        self.feed_token = response_data['data'].get('feedToken')
                        self.headers["Authorization"] = f"Bearer {self.jwt_token}"
                        self.logged_in = True
                        self.logger.info("Login successful")
                        return True
                    else:
                        error_msg = response_data.get('message', 'Unknown error')
                        self.logger.error(f"Login failed: {error_msg}")
                else:
                    self.logger.error(f"HTTP Error {response.status_code} during login")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error during login: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected login error: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        self.logger.error("Maximum login attempts reached")
        return False

    def get_user_profile(self) -> Optional[Dict]:
        """Get user profile with error handling"""
        if not self.is_authenticated():
            self.logger.error("Authentication required")
            return None

        try:
            profile_url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getProfile"
            response = requests.get(profile_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    return response_data['data']
                else:
                    self.logger.error(f"Profile error: {response_data.get('message', 'Unknown error')}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} getting profile")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting profile: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting profile: {str(e)}")

        return None

    def get_ltp(self, exchange: str, symbol: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Get last traded price with enhanced error handling and retry logic
        
        Args:
            exchange (str): Exchange (NSE, BSE)
            symbol (str): Trading symbol
            max_retries (int): Maximum retry attempts
            
        Returns:
            dict: LTP data or None if failed
        """
        if not self.is_authenticated() and not self.login():
            self.logger.warning("Using simulated LTP data due to authentication failure")
            return self._get_simulated_ltp(exchange, symbol)

        for attempt in range(max_retries):
            try:
                ltp_url = f"{self.base_url}/rest/secure/angelbroking/market/v1/quote/ltp"
                token = self._get_token(exchange, symbol)

                payload = {
                    "exchange": exchange,
                    "tradingsymbol": symbol,
                    "symboltoken": token
                }

                response = requests.post(
                    ltp_url,
                    json=payload,
                    headers=self.headers,
                    timeout=15
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('status') and response_data.get('message') == "SUCCESS":
                        return response_data['data']
                    else:
                        error_msg = response_data.get('message', 'Unknown error')
                        self.logger.error(f"LTP error: {error_msg}")
                else:
                    self.logger.error(f"HTTP Error {response.status_code} getting LTP")

                # Handle authentication errors
                if response.status_code in [401, 403]:
                    self.logger.warning("Authentication expired, attempting relogin...")
                    if self.login():
                        continue  # Retry with new token
                    break

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error getting LTP: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error getting LTP: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        self.logger.warning("Falling back to simulated LTP data")
        return self._get_simulated_ltp(exchange, symbol)

    def get_historical_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        count: int,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data with improved error handling
        
        Args:
            exchange (str): Exchange (NSE, BSE)
            symbol (str): Trading symbol
            timeframe (str): Timeframe string
            count (int): Number of candles
            max_retries (int): Maximum retry attempts
            
        Returns:
            pd.DataFrame: Historical data or None
        """
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        # Map timeframes to API format
        timeframe_map = {
            "1 minute": "ONE_MINUTE",
            "5 minutes": "FIVE_MINUTE",
            "15 minutes": "FIFTEEN_MINUTE",
            "30 minutes": "THIRTY_MINUTE",
            "1 hour": "ONE_HOUR",
            "1 day": "ONE_DAY"
        }

        angel_timeframe = timeframe_map.get(timeframe, "ONE_DAY")
        to_date = datetime.now()
        from_date = to_date - timedelta(days=count)

        for attempt in range(max_retries):
            try:
                candle_url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
                token = self._get_token(exchange, symbol)

                payload = {
                    "exchange": exchange,
                    "symboltoken": token,
                    "interval": angel_timeframe,
                    "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                    "todate": to_date.strftime("%Y-%m-%d %H:%M")
                }

                response = requests.post(
                    candle_url,
                    json=payload,
                    headers=self.headers,
                    timeout=20
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('status') and response_data.get('message') == "SUCCESS":
                        data = response_data['data']
                        df = pd.DataFrame(
                            data,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        return df
                    else:
                        error_msg = response_data.get('message', 'Unknown error')
                        self.logger.error(f"Historical data error: {error_msg}")
                else:
                    self.logger.error(f"HTTP Error {response.status_code} getting historical data")

                # Handle authentication errors
                if response.status_code in [401, 403]:
                    self.logger.warning("Authentication expired, attempting relogin...")
                    if self.login():
                        continue  # Retry with new token
                    break

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error getting historical data: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error getting historical data: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return None

    def place_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0,
        trigger_price: float = 0,
        product_type: str = "INTRADAY",
        max_retries: int = 3
    ) -> Dict[str, str]:
        """
        Place order with enhanced validation and error handling
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            transaction_type (str): BUY/SELL
            quantity (int): Order quantity
            order_type (str): MARKET/LIMIT/SL/SL-M
            price (float): Order price
            trigger_price (float): Trigger price
            product_type (str): INTRADAY/DELIVERY/MARGIN
            max_retries (int): Maximum retry attempts
            
        Returns:
            dict: Order response with status
        """
        if not self.is_authenticated() and not self.login():
            return {"status": "error", "message": "Authentication failed"}

        # Validate inputs
        if transaction_type.upper() not in ["BUY", "SELL"]:
            return {"status": "error", "message": "Invalid transaction type"}
        
        if quantity <= 0:
            return {"status": "error", "message": "Quantity must be positive"}

        # Map order types
        order_type_map = {
            "MARKET": "MARKET",
            "LIMIT": "LIMIT",
            "SL": "STOPLOSS_LIMIT",
            "SL-M": "STOPLOSS_MARKET"
        }

        product_type_map = {
            "DELIVERY": "DELIVERY",
            "INTRADAY": "INTRADAY",
            "MARGIN": "MARGIN",
            "CNC": "DELIVERY",
            "MIS": "INTRADAY",
            "NRML": "MARGIN"
        }

        angel_order_type = order_type_map.get(order_type.upper(), "MARKET")
        angel_product_type = product_type_map.get(product_type.upper(), "INTRADAY")

        for attempt in range(max_retries):
            try:
                order_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder"
                token = self._get_token(exchange, symbol)

                payload = {
                    "variety": "NORMAL",
                    "tradingsymbol": symbol,
                    "symboltoken": token,
                    "transactiontype": transaction_type.upper(),
                    "exchange": exchange,
                    "ordertype": angel_order_type,
                    "producttype": angel_product_type,
                    "duration": "DAY",
                    "price": str(price) if price > 0 else "0",
                    "triggerprice": str(trigger_price) if trigger_price > 0 else "0",
                    "quantity": str(quantity),
                    "squareoff": "0",
                    "stoploss": "0"
                }

                response = requests.post(
                    order_url,
                    json=payload,
                    headers=self.headers,
                    timeout=20
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('status') and response_data.get('message') == "SUCCESS":
                        return {
                            "status": "success",
                            "order_id": response_data['data']['orderid'],
                            "message": "Order placed successfully"
                        }
                    else:
                        error_msg = response_data.get('message', 'Unknown error')
                        self.logger.error(f"Order placement error: {error_msg}")
                        return {"status": "error", "message": error_msg}
                else:
                    self.logger.error(f"HTTP Error {response.status_code} placing order")

                # Handle authentication errors
                if response.status_code in [401, 403]:
                    self.logger.warning("Authentication expired, attempting relogin...")
                    if self.login():
                        continue  # Retry with new token
                    break

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error placing order: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error placing order: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return {"status": "error", "message": "Failed after multiple attempts"}

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        trigger_price: Optional[float] = None,
        max_retries: int = 3
    ) -> bool:
        """
        Modify order with enhanced validation
        
        Args:
            order_id (str): Order ID to modify
            price (float): New price
            quantity (int): New quantity
            trigger_price (float): New trigger price
            max_retries (int): Maximum retry attempts
            
        Returns:
            bool: True if modification succeeded
        """
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return False

        # Validate at least one parameter is being modified
        if all(param is None for param in [price, quantity, trigger_price]):
            self.logger.error("No modification parameters provided")
            return False

        for attempt in range(max_retries):
            try:
                modify_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/modifyOrder"
                payload = {"variety": "NORMAL", "orderid": str(order_id)}

                if price is not None:
                    payload["price"] = str(price)
                if quantity is not None:
                    payload["quantity"] = str(quantity)
                if trigger_price is not None:
                    payload["triggerprice"] = str(trigger_price)

                response = requests.post(
                    modify_url,
                    json=payload,
                    headers=self.headers,
                    timeout=15
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('status') and response_data.get('message') == "SUCCESS":
                        return True
                    else:
                        error_msg = response_data.get('message', 'Unknown error')
                        self.logger.error(f"Modify order error: {error_msg}")
                else:
                    self.logger.error(f"HTTP Error {response.status_code} modifying order")

                # Handle authentication errors
                if response.status_code in [401, 403]:
                    self.logger.warning("Authentication expired, attempting relogin...")
                    if self.login():
                        continue  # Retry with new token
                    break

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error modifying order: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error modifying order: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return False

    def cancel_order(self, order_id: str, max_retries: int = 3) -> bool:
        """
        Cancel order with enhanced error handling
        
        Args:
            order_id (str): Order ID to cancel
            max_retries (int): Maximum retry attempts
            
        Returns:
            bool: True if cancellation succeeded
        """
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return False

        for attempt in range(max_retries):
            try:
                cancel_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/cancelOrder"
                payload = {
                    "variety": "NORMAL",
                    "orderid": str(order_id)
                }

                response = requests.post(
                    cancel_url,
                    json=payload,
                    headers=self.headers,
                    timeout=15
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get('status') and response_data.get('message') == "SUCCESS":
                        return True
                    else:
                        error_msg = response_data.get('message', 'Unknown error')
                        self.logger.error(f"Cancel order error: {error_msg}")
                else:
                    self.logger.error(f"HTTP Error {response.status_code} canceling order")

                # Handle authentication errors
                if response.status_code in [401, 403]:
                    self.logger.warning("Authentication expired, attempting relogin...")
                    if self.login():
                        continue  # Retry with new token
                    break

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error canceling order: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error canceling order: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return False

    def get_order_book(self) -> Optional[List[Dict]]:
        """Get order book with error handling"""
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        try:
            order_book_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getOrderBook"
            response = requests.get(order_book_url, headers=self.headers, timeout=15)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    return response_data['data']
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Order book error: {error_msg}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} getting order book")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting order book: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting order book: {str(e)}")

        return None

    def get_trade_book(self) -> Optional[List[Dict]]:
        """Get trade book with error handling"""
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        try:
            trade_book_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getTradeBook"
            response = requests.get(trade_book_url, headers=self.headers, timeout=15)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    return response_data['data']
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Trade book error: {error_msg}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} getting trade book")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting trade book: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting trade book: {str(e)}")

        return None

    def get_holdings(self) -> Optional[List[Dict]]:
        """Get holdings with error handling"""
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        try:
            holdings_url = f"{self.base_url}/rest/secure/angelbroking/portfolio/v1/getHolding"
            response = requests.get(holdings_url, headers=self.headers, timeout=15)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    return response_data['data']
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Holdings error: {error_msg}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} getting holdings")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting holdings: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting holdings: {str(e)}")

        return None

    def get_positions(self) -> Optional[List[Dict]]:
        """Get positions with error handling"""
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        try:
            positions_url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getPosition"
            response = requests.get(positions_url, headers=self.headers, timeout=15)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    return response_data['data']
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Positions error: {error_msg}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} getting positions")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting positions: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting positions: {str(e)}")

        return None

    def get_option_expiry_dates(self, exchange: str, symbol: str) -> Optional[List[str]]:
        """Get option expiry dates with error handling"""
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        try:
            url = f"{self.base_url}/rest/secure/angelbroking/derivative/v1/expiryDates"
            payload = {
                "exchange": exchange,
                "tradingSymbol": symbol
            }

            response = requests.post(url, json=payload, headers=self.headers, timeout=15)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    return response_data['data']
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Expiry dates error: {error_msg}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} getting expiry dates")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting expiry dates: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting expiry dates: {str(e)}")

        return None

    def get_option_chain(
        self,
        symbol: str,
        exchange: str,
        expiry_date: str
    ) -> Optional[Dict]:
        """Get option chain with error handling"""
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        try:
            url = f"{self.base_url}/rest/secure/angelbroking/derivative/v1/optionChain"
            payload = {
                "exchange": exchange,
                "tradingSymbol": symbol,
                "expiryDate": expiry_date
            }

            response = requests.post(url, json=payload, headers=self.headers, timeout=20)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    return response_data
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Option chain error: {error_msg}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} getting option chain")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting option chain: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting option chain: {str(e)}")

        return None

    def get_portfolio(self) -> Optional[Dict]:
        """Get portfolio summary with error handling"""
        if not self.is_authenticated() and not self.login():
            self.logger.error("Authentication failed")
            return None

        try:
            # Get holdings and positions
            holdings = self.get_holdings() or []
            positions = self.get_positions() or []

            # Calculate portfolio metrics
            portfolio_value = sum(
                float(h.get('ltp', 0)) * float(h.get('quantity', 0)) 
                for h in holdings
            )
            day_pnl = sum(float(p.get('pnl', 0)) for p in positions)
            overall_pnl = sum(float(h.get('pnl', 0)) for h in holdings)

            # Get available margin
            available_margin = 0
            margin_url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getRMS"
            margin_response = requests.get(margin_url, headers=self.headers, timeout=15)

            if margin_response.status_code == 200:
                margin_data = margin_response.json()
                if margin_data.get('status') and margin_data.get('message') == "SUCCESS":
                    available_margin = float(margin_data['data'].get('availablecash', 0))

            # Calculate percentages
            day_change_percent = 0 if portfolio_value == 0 else (day_pnl / portfolio_value) * 100
            overall_change_percent = 0 if portfolio_value == 0 else (overall_pnl / portfolio_value) * 100

            return {
                'portfolio_value': portfolio_value,
                'day_pnl': day_pnl,
                'overall_pnl': overall_pnl,
                'day_change_percent': round(day_change_percent, 2),
                'overall_change_percent': round(overall_change_percent, 2),
                'available_margin': available_margin,
                'holdings': holdings,
                'positions': positions
            }

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error getting portfolio: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting portfolio: {str(e)}")

        return None

    def logout(self) -> bool:
        """Logout with error handling"""
        if not self.is_authenticated():
            self.logger.warning("Already logged out")
            return True

        try:
            response = requests.post(self.logout_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') and response_data.get('message') == "SUCCESS":
                    self._clear_auth_state()
                    return True
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    self.logger.error(f"Logout error: {error_msg}")
            else:
                self.logger.error(f"HTTP Error {response.status_code} during logout")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during logout: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during logout: {str(e)}")

        return False

    def _clear_auth_state(self):
        """Clear authentication state"""
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        self.logged_in = False
        if "Authorization" in self.headers:
            del self.headers["Authorization"]

    def _get_token(self, exchange: str, symbol: str) -> str:
        """Get symbol token (simplified for demo)"""
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

    def _get_simulated_ltp(self, exchange: str, symbol: str) -> Dict:
        """Generate simulated LTP data"""
        base_prices = {
            "NIFTY": 18500.00,
            "BANKNIFTY": 43000.00,
            "SENSEX": 62000.00,
            "RELIANCE": 2500.00,
            "TCS": 3400.00,
            "INFY": 1500.00,
            "HDFCBANK": 1600.00
        }

        base_price = base_prices.get(symbol, 1000.00)
        variation = random.uniform(-0.5, 0.5) / 100
        ltp = base_price * (1 + variation)

        return {
            "ltp": round(ltp, 2),
            "change_percent": round(variation * 100, 2),
            "token": self._get_token(exchange, symbol),
            "exchange": exchange,
            "tradingsymbol": symbol,
            "simulated": True
        }

# Example usage
if __name__ == "__main__":
    # Initialize with environment variables or direct parameters
    api = AngelOneAPI()
    
    # Login
    if api.login():
        print("Login successful")
        
        # Get LTP
        ltp_data = api.get_ltp("NSE", "RELIANCE")
        print(f"Reliance LTP: {ltp_data}")
        
        # Get historical data
        hist_data = api.get_historical_data("NSE", "NIFTY", "1 day", 10)
        print(f"NIFTY historical data:\n{hist_data.tail()}")
        
        # Logout
        api.logout()
    else:
        print("Login failed")
