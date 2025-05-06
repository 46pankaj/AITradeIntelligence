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
            
            # Check if data is already cached
            if cache_key in self.cached_data:
                logger.info(f"Using cached data for {symbol} on {exchange}")
                return self.cached_data[cache_key]
            
            # Try to get data from the API
            if self.api is not None:
                logger.info(f"Fetching historical data for {symbol} on {exchange}")
                
                try:
                    # Convert timeframe to API format
                    if timeframe == "1 day":
                        api_timeframe = "1D"
                    elif timeframe == "1 hour":
                        api_timeframe = "1H"
                    elif timeframe == "30 minutes":
                        api_timeframe = "30M"
                    elif timeframe == "15 minutes":
                        api_timeframe = "15M"
                    elif timeframe == "5 minutes":
                        api_timeframe = "5M"
                    elif timeframe == "1 minute":
                        api_timeframe = "1M"
                    else:
                        api_timeframe = "1D"  # Default to daily
                    
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_back)
                    
                    # Format dates
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    
                    # Call API to get historical data
                    data = self.api.get_historical_data(exchange, symbol, api_timeframe, days_back)
                    
                    if data is not None and len(data) > 0:
                        # Convert to DataFrame
                        df = pd.DataFrame(data)
                        
                        # Cache the data
                        self.cached_data[cache_key] = df
                        
                        logger.info(f"Retrieved {len(df)} data points for {symbol} on {exchange}")
                        return df
                    
                except Exception as e:
                    logger.error(f"Error fetching historical data: {str(e)}")
            
            logger.warning(f"Falling back to simulated data for {symbol} on {exchange}")
            
            # Generate simulated data if API data is not available
            df = self._generate_simulated_data(symbol, exchange, timeframe, days_back)
            
            # Cache the simulated data
            self.cached_data[cache_key] = df
            
            return df
        
        except Exception as e:
            logger.error(f"Error in get_historical_data: {str(e)}")
            return None
    
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
            logger.error(f"Error getting market sentiment: {str(e)}")
            return None
    
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
            logger.error(f"Error getting open interest data: {str(e)}")
            return None
    
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
        except Exception as e:
            logger.error(f"Error loading strategies: {str(e)}")
            return {}
    
    def _save_strategies(self):
        """
        Save strategies to file
        """
        try:
            with open(self.strategies_file, 'w') as f:
                json.dump(self.strategies, f, indent=4)
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
            self._save_strategies()
            
            logger.info(f"Saved strategy: {strategy_id}")
            return strategy_id
        
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
            if 'id' not in strategy:
                logger.error("Cannot update strategy without ID")
                return False
                
            strategy_id = strategy['id']
            
            if strategy_id not in self.strategies:
                logger.warning(f"Strategy not found for update: {strategy_id}")
                return False
                
            # Update timestamp
            strategy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
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
            if strategy_id in self.strategies:
                del self.strategies[strategy_id]
                self._save_strategies()
                logger.info(f"Deleted strategy: {strategy_id}")
                return True
            else:
                logger.warning(f"Strategy not found: {strategy_id}")
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
            
            # Try to use API first
            if self.api is not None:
                try:
                    price_data = self.api.get_ltp(exchange, symbol)
                    
                    if price_data and 'ltp' in price_data:
                        # Save to market data cache file
                        self._save_market_price(symbol, exchange, price_data)
                        
                        return price_data
                except Exception as e:
                    logger.error(f"Error fetching market price: {str(e)}")
            
            # Try to load from cache file
            cached_price = self._load_market_price(symbol, exchange)
            if cached_price:
                logger.info(f"Using cached market price for {symbol} on {exchange}")
                return cached_price
            
            # Generate simulated price as last resort
            logger.warning(f"Using simulated price for {symbol} on {exchange}")
            
            # Get base price for symbol
            if symbol in ["NIFTY", "BANKNIFTY", "SENSEX"]:
                base_price = 18000 if symbol == "NIFTY" else 40000 if symbol == "BANKNIFTY" else 60000
            else:
                base_price = 1000  # For individual stocks
            
            # Add some randomness
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            change = random.uniform(-2.0, 2.0)
            
            price_data = {
                'ltp': round(price, 2),
                'change': round(change, 2),
                'change_percent': round(change / price * 100, 2),
                'symbol': symbol,
                'exchange': exchange,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'simulated': True
            }
            
            # Save to market data cache file
            self._save_market_price(symbol, exchange, price_data)
            
            return price_data
        
        except Exception as e:
            logger.error(f"Error in get_market_price: {str(e)}")
            return {'ltp': 0, 'change_percent': 0, 'simulated': True}
    
    def _save_market_price(self, symbol, exchange, price_data):
        """
        Save market price to cache file
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            price_data (dict): Price data
        """
        try:
            # Load existing market data
            market_data = {}
            if os.path.exists(self.market_data_file):
                try:
                    with open(self.market_data_file, 'r') as f:
                        market_data = json.load(f)
                except:
                    pass
            
            # Add timestamp if not present
            if 'timestamp' not in price_data:
                price_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to market data
            key = f"{symbol}_{exchange}"
            market_data[key] = price_data
            
            # Save to file
            with open(self.market_data_file, 'w') as f:
                json.dump(market_data, f, indent=4)
            
        except Exception as e:
            logger.error(f"Error saving market price: {str(e)}")
    
    def _load_market_price(self, symbol, exchange):
        """
        Load market price from cache file
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: Price data or None if not found
        """
        try:
            if os.path.exists(self.market_data_file):
                with open(self.market_data_file, 'r') as f:
                    market_data = json.load(f)
                
                key = f"{symbol}_{exchange}"
                if key in market_data:
                    # Check if price is not too old (max 1 day)
                    price_data = market_data[key]
                    if 'timestamp' in price_data:
                        timestamp = datetime.strptime(price_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                        if (datetime.now() - timestamp).days < 1:
                            return price_data
            
            return None
        
        except Exception as e:
            logger.error(f"Error loading market price: {str(e)}")
            return None
        
    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df (pandas.DataFrame): Price dataframe to update
        """
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Make sure we have OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
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