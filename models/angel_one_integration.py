"""
Angel One Integration Layer for AI-Based Trading Platform

This module provides functionality to interact with Angel One's trading APIs,
handling authentication, order placement, order status checks, and position management.
"""

import os
import json
import time
import requests
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("angel_one_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("angel_one_integration")

# Load environment variables
load_dotenv()

@dataclass
class OrderParams:
    """Data class for order parameters"""
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"
    order_type: str  # "MARKET", "LIMIT", etc.
    product_type: str  # "DELIVERY", "INTRADAY", etc.
    price: float = 0.0  # Optional for MARKET orders
    trigger_price: float = 0.0  # For SL and SL-M orders
    disclosed_quantity: int = 0
    validity: str = "DAY"
    tag: str = ""  # Custom tag for tracking
    target: float = 0.0  # For bracket orders
    stoploss: float = 0.0  # For bracket orders


class AngelOneAPI:
    """Angel One API integration class"""
    
    # API endpoints
    BASE_URL = "https://apiconnect.angelbroking.com"
    LOGIN_URL = f"{BASE_URL}/rest/auth/angelbroking/user/v1/loginByPassword"
    REFRESH_TOKEN_URL = f"{BASE_URL}/rest/auth/angelbroking/jwt/v1/generateTokens"
    ORDER_PLACEMENT_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/placeOrder"
    ORDER_STATUS_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/details"
    ORDER_BOOK_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/getOrderBook"
    TRADE_BOOK_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/getTradeBook"
    POSITIONS_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/getPosition"
    HISTORICAL_DATA_URL = f"{BASE_URL}/rest/secure/angelbroking/historical/v1/getCandleData"
    MODIFY_ORDER_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/modifyOrder"
    CANCEL_ORDER_URL = f"{BASE_URL}/rest/secure/angelbroking/order/v1/cancelOrder"
    
    def __init__(self, api_key: str = None, client_id: str = None, password: str = None, 
                 token_path: str = "angel_one_token.json", auto_refresh: bool = True):
        """
        Initialize Angel One API connection
        
        Args:
            api_key (str): API key from Angel One
            client_id (str): Angel One client ID
            password (str): Angel One password
            token_path (str): Path to store authentication tokens
            auto_refresh (bool): Whether to auto-refresh tokens
        """
        # Use environment variables if not provided
        self.api_key = api_key or os.getenv("ANGEL_ONE_API_KEY")
        self.client_id = client_id or os.getenv("ANGEL_ONE_CLIENT_ID")
        self.password = password or os.getenv("ANGEL_ONE_PASSWORD")
        
        if not all([self.api_key, self.client_id, self.password]):
            raise ValueError("API key, client ID, and password are required")
        
        self.token_path = token_path
        self.auto_refresh = auto_refresh
        self.session = requests.Session()
        
        # Auth tokens
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
        # Headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "CLIENT_LOCAL_IP",
            "X-ClientPublicIP": "CLIENT_PUBLIC_IP",
            "X-MACAddress": "MAC_ADDRESS",
            "X-PrivateKey": self.api_key
        }
        
        # Load saved tokens if available
        self._load_tokens()
        
        # Auto-login if tokens not available or expired
        if not self.is_authenticated():
            self.login()
    
    def _load_tokens(self) -> None:
        """Load authentication tokens from file if available"""
        try:
            if os.path.exists(self.token_path):
                with open(self.token_path, 'r') as f:
                    token_data = json.load(f)
                    self.access_token = token_data.get('access_token')
                    self.refresh_token = token_data.get('refresh_token')
                    self.token_expiry = datetime.fromisoformat(token_data.get('expiry', '2000-01-01T00:00:00'))
                    
                    if self.access_token:
                        self.headers["Authorization"] = f"Bearer {self.access_token}"
                        
                    logger.info("Loaded authentication tokens from file")
        except Exception as e:
            logger.warning(f"Failed to load tokens: {str(e)}")
    
    def _save_tokens(self) -> None:
        """Save authentication tokens to file"""
        try:
            token_data = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'expiry': self.token_expiry.isoformat() if self.token_expiry else None
            }
            
            with open(self.token_path, 'w') as f:
                json.dump(token_data, f)
                
            logger.info("Saved authentication tokens to file")
        except Exception as e:
            logger.warning(f"Failed to save tokens: {str(e)}")
    
    def is_authenticated(self) -> bool:
        """Check if the current session is authenticated and tokens are valid"""
        if not self.access_token or not self.token_expiry:
            return False
            
        # Check if token is expired or about to expire in 5 minutes
        if datetime.now() >= self.token_expiry - timedelta(minutes=5):
            if self.auto_refresh and self.refresh_token:
                return self.refresh_access_token()
            return False
            
        return True
    
    def login(self) -> bool:
        """
        Login to Angel One API
        
        Returns:
            bool: Whether login was successful
        """
        try:
            payload = {
                "clientcode": self.client_id,
                "password": self.password
            }
            
            response = requests.post(
                self.LOGIN_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] and data['message'] == 'SUCCESS':
                    self.refresh_token = data['data']['refreshToken']
                    self.access_token = data['data']['jwtToken']
                    self.token_expiry = datetime.now() + timedelta(days=1)  # Tokens typically valid for 24 hours
                    
                    # Update authorization header
                    self.headers["Authorization"] = f"Bearer {self.access_token}"
                    
                    # Save tokens
                    self._save_tokens()
                    
                    logger.info("Successfully logged in to Angel One API")
                    return True
                else:
                    logger.error(f"Login failed: {data['message']}")
            else:
                logger.error(f"Login request failed with status code {response.status_code}: {response.text}")
                
            return False
        except Exception as e:
            logger.error(f"Login exception: {str(e)}")
            return False
    
    def refresh_access_token(self) -> bool:
        """
        Refresh access token using refresh token
        
        Returns:
            bool: Whether token refresh was successful
        """
        try:
            payload = {
                "refreshToken": self.refresh_token
            }
            
            response = requests.post(
                self.REFRESH_TOKEN_URL,
                headers={h: v for h, v in self.headers.items() if h != 'Authorization'},
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] and data['message'] == 'SUCCESS':
                    self.access_token = data['data']['jwtToken']
                    self.token_expiry = datetime.now() + timedelta(days=1)
                    
                    # Update authorization header
                    self.headers["Authorization"] = f"Bearer {self.access_token}"
                    
                    # Save tokens
                    self._save_tokens()
                    
                    logger.info("Successfully refreshed access token")
                    return True
                else:
                    logger.error(f"Token refresh failed: {data['message']}")
            else:
                logger.error(f"Token refresh request failed with status code {response.status_code}: {response.text}")
                
            return False
        except Exception as e:
            logger.error(f"Token refresh exception: {str(e)}")
            return False
    
    def _ensure_authenticated(self) -> bool:
        """Ensure that the API is authenticated before making requests"""
        if not self.is_authenticated():
            logger.info("Authentication required. Attempting to login...")
            return self.login()
        return True
    
    def place_order(self, order: OrderParams) -> Dict[str, Any]:
        """
        Place a new order on Angel One
        
        Args:
            order (OrderParams): Order parameters
            
        Returns:
            Dict: Response containing order details or error
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            # Prepare order request payload
            payload = {
                "variety": "NORMAL",  # NORMAL, AMO, STOPLOSS
                "tradingsymbol": order.symbol,
                "symboltoken": self._get_token_for_symbol(order.symbol),
                "transactiontype": order.side,
                "exchange": self._get_exchange_for_symbol(order.symbol),
                "ordertype": order.order_type,
                "producttype": order.product_type,
                "duration": order.validity,
                "price": str(order.price) if order.price else "0",
                "squareoff": str(order.target) if order.target else "0",
                "stoploss": str(order.stoploss) if order.stoploss else "0",
                "quantity": str(order.quantity),
                "triggerprice": str(order.trigger_price) if order.trigger_price else "0",
                "tag": order.tag
            }
            
            # Make the request
            response = requests.post(
                self.ORDER_PLACEMENT_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Order placed successfully: {data['data']['orderid']}")
                    return {"status": True, "order_id": data['data']['orderid'], "message": "Order placed successfully"}
                else:
                    logger.error(f"Order placement failed: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Order request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Order placement exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}
    
    def _get_token_for_symbol(self, symbol: str) -> str:
        """
        Get the token for a given symbol
        (For a production system, this would likely be implemented with a lookup table or API call)
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            str: Symbol token
        """
        # This is a simplified implementation
        # In a real-world scenario, you would maintain a mapping or fetch from Angel One's API
        symbol_tokens = {
            "RELIANCE-EQ": "2885",
            "INFY-EQ": "1594",
            "TCS-EQ": "11536",
            "HDFCBANK-EQ": "1333",
            # Add more symbols as needed
        }
        
        return symbol_tokens.get(symbol, "")
    
    def _get_exchange_for_symbol(self, symbol: str) -> str:
        """
        Determine the exchange for a given symbol based on suffix or other logic
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            str: Exchange identifier (NSE, BSE, etc.)
        """
        if symbol.endswith("-EQ"):
            return "NSE"
        elif symbol.endswith("-BE"):
            return "BSE"
        elif "FUT" in symbol:
            return "NFO"  # NSE Futures & Options
        elif "OPT" in symbol:
            return "NFO"  # NSE Futures & Options
        else:
            return "NSE"  # Default to NSE
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific order
        
        Args:
            order_id (str): Order ID to check
            
        Returns:
            Dict: Order status details
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            payload = {
                "orderid": order_id
            }
            
            response = requests.post(
                self.ORDER_STATUS_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Retrieved order status for order {order_id}")
                    return {"status": True, "data": data['data']}
                else:
                    logger.error(f"Failed to get order status: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Order status request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Order status exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}
    
    def get_order_book(self) -> Dict[str, Any]:
        """
        Get the complete order book for the day
        
        Returns:
            Dict: All orders for the day
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            response = requests.get(
                self.ORDER_BOOK_URL,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Retrieved order book with {len(data['data'])} orders")
                    return {"status": True, "data": data['data']}
                else:
                    logger.error(f"Failed to get order book: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Order book request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Order book exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}
    
    def get_trade_book(self) -> Dict[str, Any]:
        """
        Get the complete trade book for the day
        
        Returns:
            Dict: All trades for the day
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            response = requests.get(
                self.TRADE_BOOK_URL,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Retrieved trade book with {len(data['data'])} trades")
                    return {"status": True, "data": data['data']}
                else:
                    logger.error(f"Failed to get trade book: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Trade book request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Trade book exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get all current positions
        
        Returns:
            Dict: Current positions data
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            response = requests.get(
                self.POSITIONS_URL,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Retrieved positions data with {len(data['data'])} positions")
                    return {"status": True, "data": data['data']}
                else:
                    logger.error(f"Failed to get positions: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Positions request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Positions exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}
    
    def modify_order(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """
        Modify an existing order
        
        Args:
            order_id (str): Order ID to modify
            **kwargs: Parameters to modify (price, quantity, etc.)
            
        Returns:
            Dict: Response with modification status
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            # Get current order details
            order_details = self.get_order_status(order_id)
            if not order_details['status']:
                return order_details
            
            order_data = order_details['data']
            
            # Prepare order modification payload
            payload = {
                "orderid": order_id,
                "variety": order_data.get('variety', 'NORMAL'),
                "tradingsymbol": order_data.get('tradingsymbol'),
                "symboltoken": order_data.get('symboltoken'),
                "exchange": order_data.get('exchange'),
                "transactiontype": order_data.get('transactiontype'),
                "ordertype": kwargs.get('order_type', order_data.get('ordertype')),
                "producttype": order_data.get('producttype'),
                "duration": kwargs.get('validity', order_data.get('duration')),
                "price": kwargs.get('price', order_data.get('price')),
                "triggerprice": kwargs.get('trigger_price', order_data.get('triggerprice')),
                "quantity": kwargs.get('quantity', order_data.get('quantity'))
            }
            
            response = requests.post(
                self.MODIFY_ORDER_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Order {order_id} modified successfully")
                    return {"status": True, "message": "Order modified successfully"}
                else:
                    logger.error(f"Order modification failed: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Order modification request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Order modification exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            Dict: Response with cancellation status
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            # Prepare cancellation payload
            payload = {
                "orderid": order_id,
                "variety": "NORMAL"  # Assuming normal order variety
            }
            
            response = requests.post(
                self.CANCEL_ORDER_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Order {order_id} cancelled successfully")
                    return {"status": True, "message": "Order cancelled successfully"}
                else:
                    logger.error(f"Order cancellation failed: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Order cancellation request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Order cancellation exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}
    
    def get_historical_data(self, symbol: str, timeframe: str, from_date: str, to_date: str) -> Dict[str, Any]:
        """
        Get historical candlestick data
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Candle timeframe (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, etc.)
            from_date (str): Start date in format "YYYY-MM-DD HH:MM:SS"
            to_date (str): End date in format "YYYY-MM-DD HH:MM:SS"
            
        Returns:
            Dict: Historical candlestick data
        """
        if not self._ensure_authenticated():
            return {"status": False, "message": "Authentication failed"}
        
        try:
            payload = {
                "exchange": self._get_exchange_for_symbol(symbol),
                "symboltoken": self._get_token_for_symbol(symbol),
                "interval": timeframe,
                "fromdate": from_date,
                "todate": to_date
            }
            
            response = requests.post(
                self.HISTORICAL_DATA_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status']:
                    logger.info(f"Retrieved historical data for {symbol} from {from_date} to {to_date}")
                    
                    # Convert to pandas DataFrame for easier analysis
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]
                    df = pd.DataFrame(data['data'], columns=columns)
                    
                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    # Convert price and volume columns to numeric
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col])
                        
                    return {"status": True, "data": df}
                else:
                    logger.error(f"Failed to get historical data: {data['message']}")
                    return {"status": False, "message": data['message']}
            else:
                error_msg = f"Historical data request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"status": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Historical data exception: {str(e)}"
            logger.error(error_msg)
            return {"status": False, "message": error_msg}


class AngelOneIntegrationLayer:
    """
    Main integration layer class that handles communication between trading platform and Angel One
    """
    
    def __init__(self, api_key: str = None, client_id: str = None, password: str = None):
        """
        Initialize the integration layer
        
        Args:
            api_key (str): Angel One API key
            client_id (str): Angel One client ID
            password (str): Angel One password
        """
        self.api = AngelOneAPI(api_key, client_id, password)
        self.logger = logging.getLogger("angel_one_integration_layer")
        
        # Order tracking
        self.pending_orders = {}
        self.executed_orders = {}
        self.failed_orders = {}
    
    def execute_trade(self, order_params: OrderParams) -> Dict[str, Any]:
        """
        Execute a trade using Angel One API
        
        Args:
            order_params (OrderParams): Order parameters
            
        Returns:
            Dict: Trade execution result
        """
        # Validate order parameters
        if not self._validate_order(order_params):
            error_msg = "Invalid order parameters"
            self.logger.error(error_msg)
            return {"status": False, "message": error_msg}
            
        # Place the order
        order_result = self.api.place_order(order_params)
        
        if order_result["status"]:
            order_id = order_result["order_id"]
            self.pending_orders[order_id] = {
                "params": order_params,
                "time": datetime.now(),
                "status": "PENDING"
            }
            
            self.logger.info(f"Order {order_id} placed and added to pending orders")
            return {"status": True, "order_id": order_id, "message": "Order placed successfully"}
        else:
            self.logger.error(f"Failed to place order: {order_result['message']}")
            return order_result
    
    def _validate_order(self, order: OrderParams) -> bool:
        """
        Validate order parameters
        
        Args:
            order (OrderParams): Order parameters to validate
            
        Returns:
            bool: Whether the order parameters are valid
        """
        # Basic validations
        if not order.symbol:
            self.logger.error("Missing symbol in order parameters")
            return False
            
        if order.quantity <= 0:
            self.logger.error(f"Invalid quantity: {order.quantity}")
            return False
            
        if order.side not in ["BUY", "SELL"]:
            self.logger.error(f"Invalid side: {order.side}")
            return False
            
        if order.order_type not in ["MARKET", "LIMIT", "STOPLOSS", "STOPLOSS_MARKET"]:
            self.logger.error(f"Invalid order type: {order.order_type}")
            return False
            
        if order.product_type not in ["DELIVERY", "INTRADAY", "MARGIN", "BRACKET"]:
            self.logger.error(f"Invalid product type: {order.product_type}")
            return False
            
        # Additional validations based on order type
        if order.order_type == "LIMIT" and order.price <= 0:
            self.logger.error(f"Limit order requires a valid price > 0: {order.price}")
            return False
            
        if order.order_type in ["STOPLOSS", "STOPLOSS_MARKET"] and order.trigger_price <= 0:
            self.logger.error(f"Stop loss order requires a valid trigger price > 0: {order.trigger_price}")
            return False
            
        return True
    
    def update_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Update the status of a specific order
        
        Args:
            order_id (str): Order ID to update
            
        Returns:
            Dict: Updated order status
        """
        if order_id not in self.pending_orders and order_id not in self.executed_orders:
            error_msg = f"Order {order_id} not found in tracking"
            self.logger.error(error_msg)
            return {"status": False, "message": error_msg}
            
        # Get current status from Angel One
        order_status = self.api.get_order_status(order_id)
        
        if not order_status["status"]:
            self.logger.error(f"Failed to get status for order {order_id}: {order_status['message']}")
            return order_status
            
        # Update our tracking
        status = order_status["data"]["status"]
        
        if status in ["COMPLETE", "CANCELED", "REJECTED"]:
            if order_id in self.pending_orders:
                # Move from pending to executed/failed
                order_details = self.pending_orders.pop(order_id)
                
                if status == "COMPLETE":
                    self.executed_orders[order_id] = {
                        **order_details,
                        "status": status,
                        "execution_time": datetime.now(),
                        "execution_price": float(order_status["data"].get("average_price", 0)),
                        "filled_quantity": int(order_status["data"].get("filled_quantity", 0))
                    }
                    self.logger.info(f"Order {order_id} executed successfully")
                else:
                    self.failed_orders[order_id] = {
                        **order_details,
                        "status": status,
                        "failure_time": datetime.now(),
                        "reason": order_status["data"].get("status_message", "")
                    }
                    self.logger.info(f"Order {order_id} {status.lower()}")
        else:
            # Update pending order status
            if order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = status
                self.logger.info(f"Order {order_id} status updated to {status}")
                
        return {"status": True, "data": order_status["data"]}
    
    def check_pending_orders(self) -> Dict[str, Any]:
        """
        Check and update all pending orders
        
        Returns:
            Dict: Status of pending orders update
        """
        if not self.pending_orders:
            return {"status": True, "message": "No pending orders to update"}
            
        results = {}
        for order_id in list(self.pending_orders.keys()):
            order_status = self.update_order_status(order_id)
            results[order_id] = order_status
            
        return {"status": True, "data": results}
    
    def get_portfolio_positions(self) -> Dict[str, Any]:
        """
        Get current portfolio positions
        
        Returns:
            Dict: Current positions
        """
        return self.api.get_positions()
    
    def get_order_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the order history tracked by this integration layer
        
        Returns:
            Dict: All tracked orders categorized by status
        """
        return {
            "pending": self.pending_orders,
            "executed": self.executed_orders,
            "failed": self.failed_orders
        }
    
    def modify_order_price(self, order_id: str, new_price: float) -> Dict[str, Any]:
        """
        Modify the price of an existing order
        
        Args:
            order_id (str): Order ID to modify
            new_price (float): New price for the order
            
        Returns:
            Dict: Modification result
        """
        return self.api.modify_order(order_id, price=new_price)
    
    def modify_order_quantity(self, order_id: str, new_quantity: int) -> Dict[str, Any]:
        """
        Modify the quantity of an existing order
        
        Args:
            order_id (str): Order ID to modify
            new_quantity (int): New quantity for the order
            
        Returns:
            Dict: Modification result
        """
        return self.api.modify_order(order_id, quantity=new_quantity)
    
    def cancel_pending_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel a pending order
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            Dict: Cancellation result
        """
        if order_id not in self.pending_orders:
            error_msg = f"Order {order_id} not found in pending orders"
            self.logger.error(error_msg)
            return {"status": False, "message": error_msg}
            
        # Cancel the order
        cancel_result = self.api.cancel_order(order_id)
        
        if cancel_result["status"]:
            # Update status will move it to failed orders as needed
            self.update_order_status(order_id)
            
        return cancel_result
    
    def get_market_data(self, symbol: str, timeframe: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical market data for a symbol
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Candle timeframe (ONE_MINUTE, FIVE_MINUTE, etc.)
            days (int): Number of days of history to retrieve
            
        Returns:
            Dict: Historical market data
        """
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        
        return self.api.get_historical_data(symbol, timeframe, from_date, to_date)
    
    def bulk_order_placement(self, orders: List[OrderParams]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Place multiple orders in bulk
        
        Args:
            orders (List[OrderParams]): List of order parameters
            
        Returns:
            Dict: Results of each order placement
        """
        results = {
            "successful": [],
            "failed": []
        }
        
        for order in orders:
            order_result = self.execute_trade(order)
            
            if order_result["status"]:
                results["successful"].append({
                    "order_id": order_result["order_id"],
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity
                })
            else:
                results["failed"].append({
                    "params": {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity
                    },
                    "error": order_result["message"]
                })
                
        return results


# Example usage
if __name__ == "__main__":
    # Create a .env file with your credentials or pass them directly
    integration = AngelOneIntegrationLayer()
    
    # Create an order
    order = OrderParams(
        symbol="RELIANCE-EQ",
        quantity=1,
        side="BUY",
        order_type="MARKET",
        product_type="INTRADAY"
    )
    
    # Place the order
    result = integration.execute_trade(order)
    print(f"Order result: {result}")
    
    if result["status"]:
        order_id = result["order_id"]
        
        # Check order status
        status = integration.update_order_status(order_id)
        print(f"Order status: {status}")
        
        # Get positions
        positions = integration.get_portfolio_positions()
        print(f"Current positions: {positions}")
