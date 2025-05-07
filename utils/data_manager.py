import logging
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import random
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    """
    Class for managing data operations for trading strategies
    """
    def __init__(self, api=None):
        """
        Initialize the data manager
        
        Args:
            api: API client for fetching market data
        """
        self.api = api
        self.cached_data = {}
        self.strategies_file = "data/strategies.json"
        self.market_data_file = "data/market_data.json"
        
        # Create data directories if they don't exist
        os.makedirs(os.path.dirname(self.strategies_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.market_data_file), exist_ok=True)
        
        # Load existing strategies if available
        self.strategies = self._load_strategies()
        
        # Check if API is properly initialized
        self.api_available = self._check_api_availability()
        if not self.api_available:
            logger.warning("API client is not available. Using simulated data for all operations.")
    
    def _check_api_availability(self):
        """
        Check if the API client is properly initialized and responsive
        
        Returns:
            bool: True if API is available, False otherwise
        """
        if self.api is None:
            return False
            
        try:
            # Try a simple API call to verify connection
            # This should be adjusted based on actual API implementation
            test_result = self.api.ping() if hasattr(self.api, 'ping') else True
            return test_result is not None
        except Exception as e:
            logger.error(f"API availability check failed: {str(e)}")
            return False
    
    def get_historical_data(self, symbol, exchange, timeframe="1 day", days_back=60):
        """
        Get historical market data for a symbol
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            timeframe (str): Candle timeframe (1 minute, 5 minutes, etc.)
            days_back (int): Number of days of historical data to fetch
            
        Returns:
            pandas.DataFrame: Historical OHLCV data
        """
        try:
            cache_key = f"{symbol}_{exchange}_{timeframe}_{days_back}"
            
            # Check if data is already cached and not expired
            if cache_key in self.cached_data:
                cache_time = self.cached_data.get(f"{cache_key}_timestamp")
                # Cache expiration set to 1 hour for historical data
                if cache_time and (datetime.now() - cache_time).total_seconds() < 3600:
                    logger.info(f"Using cached data for {symbol} on {exchange}")
                    return self.cached_data[cache_key]
                else:
                    logger.info(f"Cached data expired for {symbol} on {exchange}")
            
            # Try to get data from the API if available
            if self.api_available:
                logger.info(f"Fetching historical data for {symbol} on {exchange}")
                
                try:
                    # Convert timeframe to API format
                    api_timeframe = self._convert_timeframe_format(timeframe)
                    
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_back)
                    
                    # Format dates
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    
                    # Call API to get historical data with proper error handling
                    data = self.api.get_historical_data(exchange, symbol, api_timeframe, days_back)
                    
                    if data is not None and len(data) > 0:
                        # Convert to DataFrame
                        df = pd.DataFrame(data)
                        
                        # Cache the data with timestamp
                        self.cached_data[cache_key] = df
                        self.cached_data[f"{cache_key}_timestamp"] = datetime.now()
                        
                        logger.info(f"Retrieved {len(df)} data points for {symbol} on {exchange}")
                        
                        # Add technical indicators
                        self._add_technical_indicators(df)
                        
                        return df
                    else:
                        logger.warning(f"API returned empty data for {symbol} on {exchange}")
                
                except Exception as e:
                    logger.error(f"Error fetching historical data from API for {symbol} on {exchange}: {str(e)}")
                    # Continue to fallback behavior
            else:
                logger.info(f"API not available, using simulated data for {symbol} on {exchange}")
            
            # Generate simulated data as fallback
            logger.warning(f"Falling back to simulated data for {symbol} on {exchange}")
            
            # Generate simulated data if API data is not available
            df = self._generate_simulated_data(symbol, exchange, timeframe, days_back)
            
            # Cache the simulated data with timestamp
            self.cached_data[cache_key] = df
            self.cached_data[f"{cache_key}_timestamp"] = datetime.now()
            
            return df
        
        except Exception as e:
            logger.error(f"Critical error in get_historical_data for {symbol} on {exchange}: {str(e)}")
            # Return empty DataFrame as last resort
            return pd.DataFrame()
    
    def _convert_timeframe_format(self, timeframe):
        """
        Convert human-readable timeframe to API format
        
        Args:
            timeframe (str): Human-readable timeframe
            
        Returns:
            str: API timeframe format
        """
        timeframe_map = {
            "1 day": "1D",
            "1 hour": "1H",
            "30 minutes": "30M",
            "15 minutes": "15M",
            "5 minutes": "5M",
            "1 minute": "1M"
        }
        
        return timeframe_map.get(timeframe, "1D")
    
    def get_market_sentiment(self, symbol, exchange):
        """
        Get market sentiment data for a symbol
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: Sentiment data
        """
        try:
            # Check if API is available and has sentiment capability
            if self.api_available and hasattr(self.api, 'get_sentiment'):
                try:
                    # Try to get sentiment data from API
                    sentiment = self.api.get_sentiment(exchange, symbol)
                    if sentiment and isinstance(sentiment, dict):
                        return sentiment
                except Exception as e:
                    logger.error(f"Error getting sentiment data from API: {str(e)}")
            
            # In a production system, this would call a sentiment analysis service
            # For now, we'll just generate random sentiment data
            sentiment = {
                'bullish': random.uniform(0, 1),
                'bearish': random.uniform(0, 1),
                'neutral': random.uniform(0, 1),
            }
            
            # Normalize to sum to 1
            total = sum(sentiment.values())
            for key in sentiment:
                sentiment[key] /= total
            
            return sentiment
        
        except Exception as e:
            logger.error(f"Error getting market sentiment for {symbol} on {exchange}: {str(e)}")
            # Return balanced sentiment as fallback
            return {'bullish': 0.33, 'bearish': 0.33, 'neutral': 0.34}
    
    def get_open_interest_data(self, symbol, exchange):
        """
        Get open interest data for a symbol
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: Open interest data
        """
        try:
            # Check if API is available and has OI capability
            if self.api_available and hasattr(self.api, 'get_open_interest'):
                try:
                    # Try to get OI data from API
                    oi_data = self.api.get_open_interest(exchange, symbol)
                    if oi_data and isinstance(oi_data, dict):
                        return oi_data
                except Exception as e:
                    logger.error(f"Error getting open interest data from API: {str(e)}")
            
            # In a production system, this would call an API to get open interest data
            # For now, we'll just generate random open interest data
            oi_data = {
                'call_oi': random.randint(10000, 1000000),
                'put_oi': random.randint(10000, 1000000),
                'call_oi_change': random.uniform(-0.1, 0.1),
                'put_oi_change': random.uniform(-0.1, 0.1),
                'pcr': random.uniform(0.5, 1.5)
            }
            
            return oi_data
        
        except Exception as e:
            logger.error(f"Error getting open interest data for {symbol} on {exchange}: {str(e)}")
            # Return default OI data as fallback
            return {
                'call_oi': 0,
                'put_oi': 0,
                'call_oi_change': 0,
                'put_oi_change': 0,
                'pcr': 1.0
            }
    
    def _generate_simulated_data(self, symbol, exchange, timeframe="1 day", days_back=60):
        """
        Generate simulated OHLCV data for testing
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            timeframe (str): Candle timeframe
            days_back (int): Number of days of data to generate
            
        Returns:
            pandas.DataFrame: Simulated OHLCV data
        """
        # Set random seed based on symbol for consistent results
        seed_value = sum(ord(c) for c in symbol)
        np.random.seed(seed_value)
        
        # Determine number of data points based on timeframe
        if timeframe == "1 day":
            num_points = days_back
        elif timeframe == "1 hour":
            num_points = days_back * 8  # 8 hours per day
        elif timeframe == "30 minutes":
            num_points = days_back * 16  # 16 30-minute periods per day
        elif timeframe == "15 minutes":
            num_points = days_back * 32  # 32 15-minute periods per day
        elif timeframe == "5 minutes":
            num_points = days_back * 96  # 96 5-minute periods per day
        elif timeframe == "1 minute":
            num_points = days_back * 480  # 480 1-minute periods per day
        else:
            num_points = days_back  # Default to daily
        
        # Generate dates
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(num_points)]
        dates.reverse()  # Oldest first
        
        # Generate price data
        # Start with a base price appropriate for the symbol
        if symbol in ["NIFTY", "BANKNIFTY", "SENSEX"]:
            base_price = 18000 if symbol == "NIFTY" else 40000 if symbol == "BANKNIFTY" else 60000
            volatility = 0.01  # 1% daily volatility
        else:
            base_price = 1000  # For individual stocks
            volatility = 0.02  # 2% daily volatility
        
        # Generate a random walk with drift
        drift = 0.0001  # Slight positive drift
        prices = [base_price]
        
        for i in range(1, num_points):
            daily_return = np.random.normal(drift, volatility)
            prices.append(prices[-1] * (1 + daily_return))
        
        # Generate OHLC data
        data = []
        
        for i in range(num_points):
            base = prices[i]
            high_pct = np.random.uniform(0, 0.01)  # Up to 1% higher
            low_pct = np.random.uniform(0, 0.01)   # Up to 1% lower
            
            open_price = base * (1 + np.random.uniform(-0.005, 0.005))
            high_price = max(open_price, base) * (1 + high_pct)
            low_price = min(open_price, base) * (1 - low_pct)
            close_price = base
            
            # Generate volume
            volume = int(np.random.gamma(9.0, 1000))
            
            data.append({
                'timestamp': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add technical indicators
        self._add_technical_indicators(df)
        
        return df
    
    def _load_strategies(self):
        """
        Load strategies from file
        
        Returns:
            dict: Strategies by ID
        """
        try:
            if os.path.exists(self.strategies_file):
                with open(self.strategies_file, 'r') as f:
                    strategies = json.load(f)
                logger.info(f"Loaded {len(strategies)} strategies")
                return strategies
            else:
                logger.info("No strategies file found, creating a new one")
                return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in strategies file, creating new file")
            return {}
        except Exception as e:
            logger.error(f"Error loading strategies: {str(e)}")
            return {}
    
    def _save_strategies(self):
        """
        Save strategies to file
        
        Returns:
            bool: Success status
        """
        try:
            # Use temporary file for atomic write
            temp_file = f"{self.strategies_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.strategies, f, indent=4)
            
            # If successful, replace the original file
            os.replace(temp_file, self.strategies_file)
            
            logger.info(f"Saved {len(self.strategies)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error saving strategies: {str(e)}")
            return False
    
    def save_strategy(self, strategy):
        """
        Save a strategy
        
        Args:
            strategy (dict): Strategy data
        
        Returns:
            str: Strategy ID
        """
        try:
            # Validate strategy data
            if not isinstance(strategy, dict):
                logger.error("Invalid strategy data: not a dictionary")
                return None
            
            # Generate a new ID if not provided
            if 'id' not in strategy:
                strategy_id = str(uuid.uuid4())
                strategy['id'] = strategy_id
            else:
                strategy_id = strategy['id']
            
            # Add timestamps
            if 'created_at' not in strategy:
                strategy['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            strategy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to dictionary
            self.strategies[strategy_id] = strategy
            
            # Save to file
            success = self._save_strategies()
            
            if success:
                logger.info(f"Saved strategy: {strategy_id}")
                return strategy_id
            else:
                logger.error(f"Failed to save strategy to file: {strategy_id}")
                return None
        
        except Exception as e:
            logger.error(f"Error saving strategy: {str(e)}")
            return None
    
    def get_strategy(self, strategy_id):
        """
        Get a strategy by ID
        
        Args:
            strategy_id (str): Strategy ID
        
        Returns:
            dict: Strategy data or None if not found
        """
        try:
            if not strategy_id:
                logger.error("Cannot get strategy with empty ID")
                return None
                
            return self.strategies.get(strategy_id)
        except Exception as e:
            logger.error(f"Error getting strategy: {str(e)}")
            return None
    
    def get_all_strategies(self):
        """
        Get all strategies
        
        Returns:
            list: List of strategies
        """
        try:
            return list(self.strategies.values())
        except Exception as e:
            logger.error(f"Error getting all strategies: {str(e)}")
            return []
            
    def load_all_strategies(self):
        """
        Load and return all strategies
        
        Returns:
            list: List of strategies
        """
        try:
            # Refresh from file if needed
            self.strategies = self._load_strategies()
            return list(self.strategies.values())
        except Exception as e:
            logger.error(f"Error loading all strategies: {str(e)}")
            return []
            
    def update_strategy(self, strategy):
        """
        Update an existing strategy
        
        Args:
            strategy (dict): Strategy data with ID
            
        Returns:
            bool: Success status
        """
        try:
            if not isinstance(strategy, dict):
                logger.error("Invalid strategy data: not a dictionary")
                return False
                
            if 'id' not in strategy:
                logger.error("Cannot update strategy without ID")
                return False
                
            strategy_id = strategy['id']
            
            if strategy_id not in self.strategies:
                logger.warning(f"Strategy not found for update: {strategy_id}")
                return False
                
            # Update timestamp
            strategy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Preserve creation timestamp
            if 'created_at' in self.strategies[strategy_id]:
                strategy['created_at'] = self.strategies[strategy_id]['created_at']
            
            # Update in dict
            self.strategies[strategy_id] = strategy
            
            # Save to file
            success = self._save_strategies()
            
            if success:
                logger.info(f"Updated strategy: {strategy_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error updating strategy: {str(e)}")
            return False
    
    def delete_strategy(self, strategy_id):
        """
        Delete a strategy
        
        Args:
            strategy_id (str): Strategy ID
        
        Returns:
            bool: Success status
        """
        try:
            if not strategy_id:
                logger.error("Cannot delete strategy with empty ID")
                return False
                
            if strategy_id in self.strategies:
                del self.strategies[strategy_id]
                success = self._save_strategies()
                
                if success:
                    logger.info(f"Deleted strategy: {strategy_id}")
                    return True
                else:
                    logger.error(f"Failed to save strategies file after deletion")
                    return False
            else:
                logger.warning(f"Strategy not found for deletion: {strategy_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting strategy: {str(e)}")
            return False
    
    def get_market_price(self, symbol, exchange):
        """
        Get current market price for a symbol with improved caching and fallback
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: Price data with ltp (last traded price) and other info
        """
        try:
            cache_key = f"{symbol}_{exchange}_ltp"
            
            # Check if API is available
            if self.api_available:
                try:
                    logger.info(f"Fetching market price for {symbol} on {exchange}")
                    price_data = self.api.get_ltp(exchange, symbol)
                    
                    if price_data and 'ltp' in price_data:
                        # Add a timestamp if not present
                        if 'timestamp' not in price_data:
                            price_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Save to market data cache file
                        self._save_market_price(symbol, exchange, price_data)
                        
                        return price_data
                    else:
                        logger.warning(f"API returned invalid price data for {symbol} on {exchange}")
                except Exception as e:
                    logger.error(f"Error fetching market price from API for {symbol} on {exchange}: {str(e)}")
                    # Continue to fallback behavior
            else:
                logger.info(f"API not available, checking cached price for {symbol} on {exchange}")
            
            # Try to load from cache file with proper expiration check
            cached_price = self._load_market_price(symbol, exchange)
            if cached_price:
                logger.info(f"Using cached market price for {symbol} on {exchange}")
                
                # Check if the cached price is recent enough (within market hours)
                if 'timestamp' in cached_price:
                    timestamp = datetime.strptime(cached_price['timestamp'], '%Y-%m-%d %H:%M:%S')
                    now = datetime.now()
                    
                    # If it's the same trading day and within trading hours (9:15 AM to 3:30 PM)
                    if (now.date() == timestamp.date() and 
                        (now.hour < 15 or (now.hour == 15 and now.minute < 30)) and
                        (now.hour > 9 or (now.hour == 9 and now.minute >= 15))):
                        # If less than 5 minutes old during market hours
                        if (now - timestamp).total_seconds() < 300:
                            return cached_price
                
                # If weekend or after market hours, use cached price if less than 1 day old
                elif (datetime.now() - timestamp).total_seconds() < 86400:
                    return cached_price
            
            # Generate simulated price as last resort
            logger.warning(f"Using simulated price for {symbol} on {exchange}")
            
            # Get base price for symbol
            price_data = self._generate_simulated_price(symbol, exchange)
            
            # Save to market data cache file
            self._save_market_price(symbol, exchange, price_data)
            
            return price_data
        
        except Exception as e:
            logger.error(f"Critical error in get_market_price for {symbol} on {exchange}: {str(e)}")
            # Return minimal price data as last resort
            return {
                'ltp': 0, 
                'change_percent': 0, 
                'simulated': True,
                'error': True,
                'symbol': symbol,
                'exchange': exchange,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _generate_simulated_price(self, symbol, exchange):
        """
        Generate simulated price data for a symbol
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: Simulated price data
        """
        # Get base price for symbol
        if symbol in ["NIFTY", "BANKNIFTY", "SENSEX"]:
            base_price = 18000 if symbol == "NIFTY" else 40000 if symbol == "BANKNIFTY" else 60000
        else:
            base_price = 1000  # For individual stocks
        
        # Add some randomness
        price = base_price * (1 + random.uniform(-0.05, 0.05))
        change = random.uniform(-2.0, 2.0)
        
        return {
            'ltp': round(price, 2),
            'change': round(change, 2),
            'change_percent': round(change / price * 100, 2),
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'simulated': True
        }
    
    def _save_market_price(self, symbol, exchange, price_data):
        """
        Save market price to cache file with improved error handling
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            price_data (dict): Price data
            
        Returns:
            bool: Success status
        """
        try:
            # Load existing market data
            market_data = {}
            if os.path.exists(self.market_data_file):
                try:
                    with open(self.market_data_file, 'r') as f:
                        market_data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in market data file, creating new file")
                except Exception as e:
                    logger.error(f"Error reading market data file: {str(e)}")
            
            # Add timestamp if not present
            if 'timestamp' not in price_data:
                price_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to market data
            key = f"{symbol}_{exchange}"
            market_data[key] = price_data
            
            # Save to file with temporary file approach for safety
            temp_file = f"{self.market_data_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(market_data, f, indent=4)
            
            # If successful, replace the original file
            os.replace(temp_file, self.market_data_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving market price for {symbol} on {exchange}: {str(e)}")
            return False
    
    def _load_market_price(self, symbol, exchange):
        """
        Load market price from cache file with improved validation
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: Price data or None if not found or expired
        """
        try:
            if os.path.exists(self.market_data_file):
                try:
                    with open(self.market_data_file, 'r') as f:
                        market_data = json.load(f)
                    
                    key = f"{symbol}_{exchange}"
                    if key in market_data:
                        # Validate the price data structure
                        price_data = market_data[key]
                        if not self._is_valid_price_data(price_data):
                            logger.warning(f"Invalid price data format for {symbol} on {exchange}")
                            return None
                        
                        # Check if price is not too old (configurable expiration)
                        # For market prices, typically we want very recent data during market hours
                        if 'timestamp' in price_data:
                            timestamp = datetime.strptime(price_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                            now = datetime.now()
                            
                            # Different expiration rules for during/after market hours
                            # During market hours (9:15 AM to 3:30 PM), expire after 5 minutes
                            if (now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and \
                               (now.hour < 15 or (now.hour == 15 and now.minute < 30)):
                                # 5 minute expiration during market hours
                                if (now - timestamp).total_seconds() > 300:
                                    logger.info(f"Cached price expired (market hours) for {symbol} on {exchange}")
                                    return None
                            else:
                                # 12 hour expiration after market hours
                                if (now - timestamp).total_seconds() > 43200:
                                    logger.info(f"Cached price expired (after hours) for {symbol} on {exchange}")
                                    return None
                            
                            return price_data
                        else:
                            logger.warning(f"Missing timestamp in cached price data for {symbol} on {exchange}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in market data file")
                except Exception as e:
                    logger.error(f"Error reading market data file: {str(e)}")
            
            return None
        
        except Exception as e:
            logger.error(f"Error loading market price for {symbol} on {exchange}: {str(e)}")
            return None
    
    def _is_valid_price_data(self, price_data):
        """
        Validate the structure of price data
        
        Args:
            price_data (dict): Price data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Required fields
        required_fields = ['ltp', 'symbol', 'exchange', 'timestamp']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in price_data:
                return False
        
        # Check if ltp is a number
        if not isinstance(price_data['ltp'], (int, float)) or price_data['ltp'] < 0:
            return False
        
        # Try to parse the timestamp
        try:
            datetime.strptime(price_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        except:
            return False
        
        return True
    
    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df (pandas.DataFrame): Price dataframe to update
        """
        try:
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Make sure we have OHLCV columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
           # Simple Moving Averages
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        df['macd'] = df['ema12'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Returns
        df['daily_return'] = df['close'].pct_change()
        df['weekly_return'] = df['close'].pct_change(5)
        
        # Average True Range (ATR)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Clean up NaN values
        df = df.fillna(method='bfill')
        
        # Rolling statistics
        df['volatility'] = df['daily_return'].rolling(window=20).std()
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
