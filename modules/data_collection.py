#!/usr/bin/env python
# Data Collection & Processing Module

import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataCollection")

class DataCollector:
    """Module for collecting and processing financial market data"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.historical_data = {}
        
        # Initialize Alpha Vantage client
        if self.alpha_vantage_key:
            self.alpha_vantage = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        else:
            logger.warning("Alpha Vantage API key not found")
            
    def fetch_stock_data(self, symbol, period="2y", interval="1d", source="yfinance"):
        """
        Fetch historical stock data from the specified source
        
        Args:
            symbol (str): Stock ticker symbol
            period (str): Time period to fetch (e.g., '1d', '5d', '1mo', '3mo', '1y', '2y', 'max')
            interval (str): Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            source (str): Data source ('yfinance', 'alpha_vantage', 'finnhub')
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        try:
            if source == "yfinance":
                logger.info(f"Fetching {symbol} data from Yahoo Finance for period {period} at {interval} interval")
                data = yf.download(symbol, period=period, interval=interval)
                
            elif source == "alpha_vantage":
                if not self.alpha_vantage_key:
                    raise ValueError("Alpha Vantage API key is required")
                
                if interval == "1d":
                    logger.info(f"Fetching {symbol} daily data from Alpha Vantage")
                    data, _ = self.alpha_vantage.get_daily(symbol=symbol, outputsize='full')
                elif interval == "1m":
                    logger.info(f"Fetching {symbol} intraday data from Alpha Vantage")
                    data, _ = self.alpha_vantage.get_intraday(symbol=symbol, interval='1min', outputsize='full')
                else:
                    raise ValueError(f"Unsupported interval {interval} for Alpha Vantage")
                    
            elif source == "finnhub":
                if not self.finnhub_key:
                    raise ValueError("Finnhub API key is required")
                
                # Calculate date range based on period
                end_date = datetime.now()
                if period == "1d":
                    start_date = end_date - timedelta(days=1)
                elif period == "1mo":
                    start_date = end_date - timedelta(days=30)
                elif period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif period == "2y":
                    start_date = end_date - timedelta(days=730)
                else:
                    start_date = end_date - timedelta(days=365)
                
                # Format dates for API
                start_timestamp = int(start_date.timestamp())
                end_timestamp = int(end_date.timestamp())
                
                # Set the resolution based on interval
                if interval == "1d":
                    resolution = "D"
                elif interval == "1h":
                    resolution = "60"
                elif interval == "1m":
                    resolution = "1"
                else:
                    resolution = "D"  # Default to daily
                    
                logger.info(f"Fetching {symbol} data from Finnhub with resolution {resolution}")
                url = f"https://finnhub.io/api/v1/stock/candle"
                params = {
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": start_timestamp,
                    "to": end_timestamp,
                    "token": self.finnhub_key
                }
                
                response = requests.get(url, params=params)
                result = response.json()
                
                if result.get("s") == "ok":
                    data = pd.DataFrame({
                        "Open": result["o"],
                        "High": result["h"],
                        "Low": result["l"],
                        "Close": result["c"],
                        "Volume": result["v"]
                    }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in result["t"]]))
                else:
                    raise ValueError(f"Failed to fetch data from Finnhub: {result}")
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
            # Store the data
            if not data.empty:
                self.historical_data[symbol] = data
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                return data
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def add_technical_indicators(self, data, indicators=None):
        """
        Add technical indicators to the price data
        
        Args:
            data (pandas.DataFrame): OHLCV price data
            indicators (list): List of indicator names to add
            
        Returns:
            pandas.DataFrame: Price data with indicators
        """
        if data is None or data.empty:
            logger.warning("Cannot add indicators to empty data")
            return data
            
        # Default indicators if none specified
        if indicators is None:
            indicators = ["sma", "ema", "rsi", "macd", "bbands", "atr", "obv"]
            
        try:
            df = data.copy()
            
            for indicator in indicators:
                if indicator == "sma":
                    # Add Simple Moving Averages
                    df["SMA_20"] = ta.sma(df["Close"], length=20)
                    df["SMA_50"] = ta.sma(df["Close"], length=50)
                    df["SMA_200"] = ta.sma(df["Close"], length=200)
                    
                elif indicator == "ema":
                    # Add Exponential Moving Averages
                    df["EMA_12"] = ta.ema(df["Close"], length=12)
                    df["EMA_26"] = ta.ema(df["Close"], length=26)
                    
                elif indicator == "rsi":
                    # Add Relative Strength Index
                    df["RSI_14"] = ta.rsi(df["Close"], length=14)
                    
                elif indicator == "macd":
                    # Add MACD
                    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
                    df = pd.concat([df, macd], axis=1)
                    
                elif indicator == "bbands":
                    # Add Bollinger Bands
                    bbands = ta.bbands(df["Close"], length=20, std=2)
                    df = pd.concat([df, bbands], axis=1)
                    
                elif indicator == "atr":
                    # Add Average True Range
                    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
                    
                elif indicator == "obv":
                    # Add On-Balance Volume
                    df["OBV"] = ta.obv(df["Close"], df["Volume"])
                    
                elif indicator == "stoch":
                    # Add Stochastic Oscillator
                    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3, smooth_k=3)
                    df = pd.concat([df, stoch], axis=1)
                    
                elif indicator == "adx":
                    # Add Average Directional Index
                    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
                    df = pd.concat([df, adx], axis=1)
                    
            logger.info(f"Added {len(indicators)} technical indicators to the data")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    def add_fundamental_data(self, symbol, price_data):
        """
        Add fundamental data to the price dataset
        
        Args:
            symbol (str): Stock ticker symbol
            price_data (pandas.DataFrame): Price data to merge with
            
        Returns:
            pandas.DataFrame: Enhanced dataset with fundamental metrics
        """
        try:
            if not self.alpha_vantage_key:
                logger.warning("Alpha Vantage API key required for fundamental data")
                return price_data
                
            logger.info(f"Fetching fundamental data for {symbol}")
            
            # Get company overview
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alpha_vantage_key}"
            r = requests.get(url)
            overview = r.json()
            
            if not overview or 'Symbol' not in overview:
                logger.warning(f"No fundamental data available for {symbol}")
                return price_data
                
            # Extract quarterly metrics
            fundamental_metrics = {
                'MarketCap': float(overview.get('MarketCapitalization', 0)),
                'PE': float(overview.get('PERatio', 0)),
                'PB': float(overview.get('PriceToBookRatio', 0)),
                'DividendYield': float(overview.get('DividendYield', 0)),
                'ROE': float(overview.get('ReturnOnEquityTTM', 0)),
                'ProfitMargin': float(overview.get('ProfitMargin', 0)),
                'Beta': float(overview.get('Beta', 0))
            }
            
            # Create a dataframe with fundamental data
            # We'll repeat the same values for all dates since this data doesn't change daily
            if not price_data.empty:
                for key, value in fundamental_metrics.items():
                    price_data[key] = value
                    
            logger.info(f"Added fundamental data for {symbol}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error adding fundamental data: {str(e)}")
            return price_data
            
    def add_market_indicators(self, data):
        """
        Add overall market indicators to the dataset
        
        Args:
            data (pandas.DataFrame): Price data
            
        Returns:
            pandas.DataFrame: Enhanced dataset with market indicators
        """
        try:
            # Fetch S&P 500 data as a market benchmark
            sp500 = self.fetch_stock_data("^GSPC", period="2y", interval="1d", source="yfinance")
            
            if sp500 is not None and not sp500.empty:
                # Calculate daily S&P 500 returns
                sp500['SP500_Return'] = sp500['Close'].pct_change()
                
                # Calculate volatility (20-day rolling standard deviation of returns)
                sp500['Market_Volatility'] = sp500['SP500_Return'].rolling(window=20).std()
                
                # Calculate market momentum (20-day rate of change)
                sp500['Market_Momentum'] = sp500['Close'].pct_change(periods=20)
                
                # Merge with the main data
                market_data = sp500[['SP500_Return', 'Market_Volatility', 'Market_Momentum']]
                enhanced_data = pd.merge(data, market_data, left_index=True, right_index=True, how='left')
                
                logger.info("Added market indicators to the dataset")
                return enhanced_data
            else:
                logger.warning("Could not fetch market data")
                return data
                
        except Exception as e:
            logger.error(f"Error adding market indicators: {str(e)}")
            return data
    
    def prepare_training_data(self, symbol, lookback_window=20, prediction_horizon=5, train_size=0.8):
        """
        Prepare data for training ML models
        
        Args:
            symbol (str): Stock ticker symbol
            lookback_window (int): Number of past days to use for prediction
            prediction_horizon (int): Number of days ahead to predict
            train_size (float): Proportion of data for training (0-1)
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler) for ML training
        """
        try:
            # Get data for the symbol
            if symbol not in self.historical_data:
                logger.info(f"Data for {symbol} not found in cache, fetching...")
                self.fetch_stock_data(symbol, period="2y", interval="1d")
                
            if symbol not in self.historical_data:
                raise ValueError(f"Could not fetch data for {symbol}")
                
            # Get the data and add indicators
            data = self.historical_data[symbol].copy()
            data = self.add_technical_indicators(data)
            data = self.add_market_indicators(data)
            
            # Forward-fill and backward-fill any missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Drop any remaining rows with NaN values
            data = data.dropna()
            
            # Feature columns (exclude Date)
            feature_columns = [col for col in data.columns 
                              if col not in ['Adj Close'] 
                              and 'Date' not in col]
            
            # Target variable - future price change percentage
            data['Target'] = data['Close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
            
            # Drop rows with NaN in target
            data = data.dropna()
            
            # Create sequences for time-series prediction
            X, y = [], []
            for i in range(lookback_window, len(data)):
                X.append(data[feature_columns].iloc[i-lookback_window:i].values)
                y.append(data['Target'].iloc[i])
                
            X = np.array(X)
            y = np.array(y)
            
            # Split into training and test sets
            split_idx = int(len(X) * train_size)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Prepared training data for {symbol}: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            
            return X_train, y_train, X_test, y_test, data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None, None, None, None
            
    def get_latest_features(self, symbol, lookback_window=20):
        """
        Get the latest features for prediction
        
        Args:
            symbol (str): Stock ticker symbol
            lookback_window (int): Number of past days to use for prediction
            
        Returns:
            numpy.ndarray: Features for the most recent time period
        """
        try:
            # Ensure we have the latest data
            self.fetch_stock_data(symbol, period="1mo", interval="1d")
            
            if symbol not in self.historical_data:
                raise ValueError(f"No data available for {symbol}")
                
            # Get the data and add indicators
            data = self.historical_data[symbol].copy()
            data = self.add_technical_indicators(data)
            data = self.add_market_indicators(data)
            
            # Forward-fill and backward-fill any missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Feature columns (exclude Date and Adj Close)
            feature_columns = [col for col in data.columns 
                              if col not in ['Adj Close'] 
                              and 'Date' not in col]
            
            # Get the latest lookback_window days of features
            latest_features = data[feature_columns].iloc[-lookback_window:].values
            
            # Reshape to match the model input format (1, lookback_window, n_features)
            latest_features = latest_features.reshape(1, lookback_window, len(feature_columns))
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error getting latest features: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    
    # Example 1: Fetch and display stock data
    symbol = "AAPL"
    data = collector.fetch_stock_data(symbol, period="1y", interval="1d")
    if data is not None:
        print(f"\nStock data for {symbol}:")
        print(data.tail())
        
        # Add technical indicators
        data_with_indicators = collector.add_technical_indicators(data)
        print(f"\nStock data with indicators for {symbol}:")
        print(data_with_indicators.tail())
        
    # Example 2: Prepare training data
    X_train, y_train, X_test, y_test, _ = collector.prepare_training_data(symbol)
    if X_train is not None:
        print(f"\nPrepared training data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")
        
    # Example 3: Get latest features for prediction
    latest_features = collector.get_latest_features(symbol)
    if latest_features is not None:
        print(f"\nLatest features shape for prediction: {latest_features.shape}")
