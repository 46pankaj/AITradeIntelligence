import logging
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import random
import uuid
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    """
    Complete DataManager class with all original methods and fixed market data functionality
    """
    def __init__(self, api=None):
        self.api = api
        self.cached_data = {}
        self.strategies_file = "data/strategies.json"
        self.market_data_file = "data/market_data.json"
        os.makedirs(os.path.dirname(self.strategies_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.market_data_file), exist_ok=True)
        self.strategies = self._load_strategies()

    # ===== FIXED MARKET DATA METHODS =====
    def get_historical_data(self, symbol, exchange, timeframe="1 day", days_back=60):
        """Fixed version with API retry logic and proper validation"""
        try:
            cache_key = f"{symbol}_{exchange}_{timeframe}_{days_back}"
            
            if cache_key in self.cached_data:
                return self.cached_data[cache_key]
                
            if self.api:
                for attempt in range(3):
                    try:
                        data = self._fetch_from_api(symbol, exchange, timeframe, days_back)
                        if data is not None:
                            df = self._process_raw_data(data)
                            self.cached_data[cache_key] = df
                            return df
                    except Exception as e:
                        if attempt == 2:
                            logger.error(f"API failed after 3 attempts: {e}")
                        time.sleep((attempt + 1) * 2)
            
            # Fallback to simulated data
            df = self._generate_simulated_data(symbol, exchange, timeframe, days_back)
            self.cached_data[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return None

    def _fetch_from_api(self, symbol, exchange, timeframe, days_back):
        """Helper for API data fetching"""
        timeframe_map = {
            "1 day": "1D", "1 hour": "1H", "30 minutes": "30M",
            "15 minutes": "15M", "5 minutes": "5M", "1 minute": "1M"
        }
        api_timeframe = timeframe_map.get(timeframe, "1D")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        return self.api.get_historical_data(
            exchange=exchange,
            symbol=symbol,
            timeframe=api_timeframe,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

    def _process_raw_data(self, data):
        """Convert and validate raw API data"""
        df = pd.DataFrame(data)
        
        # Validate columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            raise ValueError(f"Missing columns: {missing}")
        
        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Add indicators
        self._add_technical_indicators(df)
        return df

    def _add_technical_indicators(self, df):
        """Fixed indicator calculations with proper error handling"""
        try:
            # Validate input
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Expected pandas DataFrame")
                
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

            # Convert timestamp if needed
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # ===== INDICATOR CALCULATIONS =====
            # Moving Averages
            for period in [5, 20, 50]:
                df[f'sma{period}'] = df['close'].rolling(period, min_periods=1).mean()
                df[f'ema{period}'] = df['close'].ewm(span=period, min_periods=1, adjust=False).mean()

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14, min_periods=1).mean()
            avg_loss = loss.rolling(14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD (Fixed implementation)
            ema12 = df['close'].ewm(span=12, min_periods=1, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, min_periods=1, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
            df['bb_std'] = df['close'].rolling(20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])

            # Other Indicators
            df['daily_return'] = df['close'].pct_change()
            df['weekly_return'] = df['close'].pct_change(5)
            
            # ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14, min_periods=1).mean()
            
            # OBV
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Volatility/Momentum
            df['volatility'] = df['daily_return'].rolling(20, min_periods=1).std()
            df['momentum'] = df['close'] / df['close'].shift(5) - 1
            
            # Clean NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Indicator error: {e}")
            return df

    # ===== ORIGINAL METHODS (PRESERVED VERBATIM) =====
    def get_market_sentiment(self, symbol, exchange):
        """Original implementation without changes"""
        try:
            sentiment = {
                'bullish': random.uniform(0, 1),
                'bearish': random.uniform(0, 1),
                'neutral': random.uniform(0, 1),
            }
            total = sum(sentiment.values())
            for key in sentiment:
                sentiment[key] /= total
            return sentiment
        except Exception as e:
            logger.error(f"Sentiment error: {e}")
            return None

    def get_open_interest_data(self, symbol, exchange):
        """Original implementation without changes"""
        try:
            return {
                'call_oi': random.randint(10000, 1000000),
                'put_oi': random.randint(10000, 1000000),
                'call_oi_change': random.uniform(-0.1, 0.1),
                'put_oi_change': random.uniform(-0.1, 0.1),
                'pcr': random.uniform(0.5, 1.5)
            }
        except Exception as e:
            logger.error(f"OI data error: {e}")
            return None

    def _generate_simulated_data(self, symbol, exchange, timeframe, days_back):
        """Original implementation without changes"""
        np.random.seed(sum(ord(c) for c in symbol))
        
        timeframe_points = {
            "1 day": days_back,
            "1 hour": days_back * 8,
            "30 minutes": days_back * 16,
            "15 minutes": days_back * 32,
            "5 minutes": days_back * 96,
            "1 minute": days_back * 480
        }
        num_points = timeframe_points.get(timeframe, days_back)
        
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(num_points)]
        dates.reverse()
        
        base_price = 18000 if symbol == "NIFTY" else 40000 if symbol == "BANKNIFTY" else 60000 if symbol == "SENSEX" else 1000
        volatility = 0.01 if symbol in ["NIFTY", "BANKNIFTY", "SENSEX"] else 0.02
        
        prices = [base_price]
        for _ in range(1, num_points):
            prices.append(prices[-1] * (1 + np.random.normal(0.0001, volatility)))
        
        data = []
        for i in range(num_points):
            open_p = prices[i] * (1 + random.uniform(-0.005, 0.005))
            high_p = max(open_p, prices[i]) * (1 + random.uniform(0, 0.01))
            low_p = min(open_p, prices[i]) * (1 - random.uniform(0, 0.01))
            
            data.append({
                'timestamp': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(open_p, 2),
                'high': round(high_p, 2),
                'low': round(low_p, 2),
                'close': round(prices[i], 2),
                'volume': int(np.random.gamma(9.0, 1000))
            })
        
        df = pd.DataFrame(data)
        self._add_technical_indicators(df)
        return df

    def _load_strategies(self):
        """Original implementation without changes"""
        try:
            if os.path.exists(self.strategies_file):
                with open(self.strategies_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Load strategies error: {e}")
            return {}

    def _save_strategies(self):
        """Original implementation without changes"""
        try:
            with open(self.strategies_file, 'w') as f:
                json.dump(self.strategies, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Save strategies error: {e}")
            return False

    def save_strategy(self, strategy):
        """Original implementation without changes"""
        try:
            if 'id' not in strategy:
                strategy['id'] = str(uuid.uuid4())
            strategy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if 'created_at' not in strategy:
                strategy['created_at'] = strategy['updated_at']
            self.strategies[strategy['id']] = strategy
            self._save_strategies()
            return strategy['id']
        except Exception as e:
            logger.error(f"Save strategy error: {e}")
            return None

    def get_strategy(self, strategy_id):
        """Original implementation without changes"""
        try:
            return self.strategies.get(strategy_id)
        except Exception as e:
            logger.error(f"Get strategy error: {e}")
            return None

    def get_all_strategies(self):
        """Original implementation without changes"""
        try:
            return list(self.strategies.values())
        except Exception as e:
            logger.error(f"Get all strategies error: {e}")
            return []

    def load_all_strategies(self):
        """Original implementation without changes"""
        try:
            self.strategies = self._load_strategies()
            return list(self.strategies.values())
        except Exception as e:
            logger.error(f"Load all strategies error: {e}")
            return []

    def update_strategy(self, strategy):
        """Original implementation without changes"""
        try:
            if 'id' not in strategy:
                logger.error("Strategy ID missing")
                return False
                
            strategy_id = strategy['id']
            if strategy_id not in self.strategies:
                logger.error(f"Strategy not found: {strategy_id}")
                return False
                
            strategy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.strategies[strategy_id] = strategy
            return self._save_strategies()
            
        except Exception as e:
            logger.error(f"Update strategy error: {e}")
            return False

    def delete_strategy(self, strategy_id):
        """Original implementation without changes"""
        try:
            if strategy_id in self.strategies:
                del self.strategies[strategy_id]
                self._save_strategies()
                return True
            return False
        except Exception as e:
            logger.error(f"Delete strategy error: {e}")
            return False

    def get_market_price(self, symbol, exchange):
        """Original implementation without changes"""
        try:
            cache_key = f"{symbol}_{exchange}_ltp"
            
            if self.api:
                try:
                    price_data = self.api.get_ltp(exchange, symbol)
                    if price_data and 'ltp' in price_data:
                        self._save_market_price(symbol, exchange, price_data)
                        return price_data
                except Exception as e:
                    logger.error(f"API price error: {e}")
            
            cached_price = self._load_market_price(symbol, exchange)
            if cached_price:
                return cached_price
                
            base_price = 18000 if symbol == "NIFTY" else 40000 if symbol == "BANKNIFTY" else 60000 if symbol == "SENSEX" else 1000
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
            
            self._save_market_price(symbol, exchange, price_data)
            return price_data
            
        except Exception as e:
            logger.error(f"Market price error: {e}")
            return {'ltp': 0, 'change_percent': 0, 'simulated': True}

    def _save_market_price(self, symbol, exchange, price_data):
        """Original implementation without changes"""
        try:
            market_data = {}
            if os.path.exists(self.market_data_file):
                with open(self.market_data_file, 'r') as f:
                    market_data = json.load(f)
            
            if 'timestamp' not in price_data:
                price_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            market_data[f"{symbol}_{exchange}"] = price_data
            
            with open(self.market_data_file, 'w') as f:
                json.dump(market_data, f, indent=4)
        except Exception as e:
            logger.error(f"Save market price error: {e}")

    def _load_market_price(self, symbol, exchange):
        """Original implementation without changes"""
        try:
            if os.path.exists(self.market_data_file):
                with open(self.market_data_file, 'r') as f:
                    market_data = json.load(f)
                
                key = f"{symbol}_{exchange}"
                if key in market_data:
                    price_data = market_data[key]
                    if 'timestamp' in price_data:
                        timestamp = datetime.strptime(price_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                        if (datetime.now() - timestamp).days < 1:
                            return price_data
            return None
        except Exception as e:
            logger.error(f"Load market price error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    dm = DataManager()
    print(dm.get_historical_data("NIFTY", "NSE").tail())
    print(dm.get_market_sentiment("RELIANCE", "NSE"))
