import pandas as pd
import numpy as np
import logging
from scipy.signal import find_peaks

class TechnicalAnalysis:
    """
    Class for performing technical analysis on market data
    """
    def __init__(self):
        """
        Initialize the technical analysis class
        """
        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def add_moving_average(self, df, period=50, column='close', ma_type='simple'):
        """
        Add moving average to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            period (int): Period for the moving average
            column (str): Column to calculate MA on
            ma_type (str): Type of moving average (simple, exponential, weighted)
            
        Returns:
            pandas.DataFrame: DataFrame with moving average added
        """
        ma_column = f'{ma_type.lower()}_ma_{period}'
        
        try:
            if ma_type.lower() == 'simple':
                df[ma_column] = df[column].rolling(window=period).mean()
            elif ma_type.lower() == 'exponential':
                df[ma_column] = df[column].ewm(span=period, adjust=False).mean()
            elif ma_type.lower() == 'weighted':
                weights = np.arange(1, period + 1)
                df[ma_column] = df[column].rolling(window=period).apply(
                    lambda x: np.sum(weights * x) / weights.sum(), raw=True
                )
            else:
                self.logger.warning(f"Unknown MA type: {ma_type}. Using simple MA.")
                df[ma_column] = df[column].rolling(window=period).mean()
                
            return df
        except Exception as e:
            self.logger.error(f"Error calculating moving average: {str(e)}")
            return df
    
    def add_rsi(self, df, period=14, column='close'):
        """
        Add Relative Strength Index (RSI) to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            period (int): Period for RSI
            column (str): Column to calculate RSI on
            
        Returns:
            pandas.DataFrame: DataFrame with RSI added
        """
        rsi_column = f'rsi_{period}'
        
        try:
            # Calculate price changes
            delta = df[column].diff()
            
            # Separate gains and losses
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and average loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df[rsi_column] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return df
    
    def add_macd(self, df, fast_period=12, slow_period=26, signal_period=9, column='close'):
        """
        Add Moving Average Convergence Divergence (MACD) to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            column (str): Column to calculate MACD on
            
        Returns:
            pandas.DataFrame: DataFrame with MACD added
        """
        try:
            # Calculate fast and slow EMAs
            ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            df['macd_line'] = ema_fast - ema_slow
            
            # Calculate signal line
            df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return df
    
    def add_bollinger_bands(self, df, period=20, std=2, column='close'):
        """
        Add Bollinger Bands to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            period (int): Period for the moving average
            std (int): Number of standard deviations
            column (str): Column to calculate Bollinger Bands on
            
        Returns:
            pandas.DataFrame: DataFrame with Bollinger Bands added
        """
        try:
            # Calculate middle band (SMA)
            df['bb_middle'] = df[column].rolling(window=period).mean()
            
            # Calculate standard deviation
            df['bb_std'] = df[column].rolling(window=period).std()
            
            # Calculate upper and lower bands
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std)
            
            # Calculate bandwidth and %B
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_b'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return df
    
    def add_stochastic_oscillator(self, df, k_period=14, d_period=3):
        """
        Add Stochastic Oscillator to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            k_period (int): Period for %K
            d_period (int): Period for %D
            
        Returns:
            pandas.DataFrame: DataFrame with Stochastic Oscillator added
        """
        try:
            # Calculate %K
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            
            # Calculate %D
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            return df
    
    def add_atr(self, df, period=14):
        """
        Add Average True Range (ATR) to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            period (int): Period for ATR
            
        Returns:
            pandas.DataFrame: DataFrame with ATR added
        """
        try:
            # Calculate true range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR
            df['atr'] = tr.rolling(window=period).mean()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return df
    
    def add_adx(self, df, period=14):
        """
        Add Average Directional Index (ADX) to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            period (int): Period for ADX
            
        Returns:
            pandas.DataFrame: DataFrame with ADX added
        """
        try:
            # Calculate directional movement
            df['up_move'] = df['high'].diff()
            df['down_move'] = df['low'].diff(-1).abs()
            
            # Calculate positive and negative directional movement
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # Calculate ATR
            df = self.add_atr(df, period)
            
            # Calculate positive and negative directional indicators
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
            
            # Calculate directional movement index
            df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
            
            # Calculate ADX
            df['adx'] = df['dx'].rolling(window=period).mean()
            
            # Clean up intermediate columns
            df.drop(['up_move', 'down_move', 'plus_dm', 'minus_dm'], axis=1, inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return df
    
    def find_support_resistance(self, df, window=10, threshold=0.01):
        """
        Find support and resistance levels
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            window (int): Window size for peak detection
            threshold (float): Minimum distance between levels (percentage)
            
        Returns:
            tuple: Lists of support and resistance levels
        """
        try:
            # Find peaks and troughs
            peaks, _ = find_peaks(df['high'].values, distance=window)
            troughs, _ = find_peaks(-df['low'].values, distance=window)
            
            # Get support and resistance levels
            resistance_levels = df['high'].iloc[peaks].values
            support_levels = df['low'].iloc[troughs].values
            
            # Group levels that are close to each other
            def group_levels(levels, threshold):
                if len(levels) == 0:
                    return []
                
                # Sort levels
                sorted_levels = np.sort(levels)
                
                # Group levels
                grouped_levels = []
                current_group = [sorted_levels[0]]
                
                for i in range(1, len(sorted_levels)):
                    # Calculate percentage difference
                    diff_percent = (sorted_levels[i] - current_group[0]) / current_group[0]
                    
                    if diff_percent < threshold:
                        # Add to current group
                        current_group.append(sorted_levels[i])
                    else:
                        # Add average of current group and start a new group
                        grouped_levels.append(np.mean(current_group))
                        current_group = [sorted_levels[i]]
                
                # Add the last group
                grouped_levels.append(np.mean(current_group))
                
                return grouped_levels
            
            support_levels = group_levels(support_levels, threshold)
            resistance_levels = group_levels(resistance_levels, threshold)
            
            return support_levels, resistance_levels
        except Exception as e:
            self.logger.error(f"Error finding support and resistance: {str(e)}")
            return [], []
    
    def detect_patterns(self, df):
        """
        Detect common chart patterns
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            
        Returns:
            dict: Dictionary of detected patterns
        """
        patterns = {}
        
        try:
            # Add required indicators
            df = self.add_bollinger_bands(df)
            
            # Check for breakouts
            last_close = df['close'].iloc[-1]
            last_bb_upper = df['bb_upper'].iloc[-1]
            last_bb_lower = df['bb_lower'].iloc[-1]
            
            if last_close > last_bb_upper:
                patterns['bollinger_breakout_up'] = True
            
            if last_close < last_bb_lower:
                patterns['bollinger_breakout_down'] = True
            
            # Check for trend
            df = self.add_moving_average(df, period=20)
            df = self.add_moving_average(df, period=50)
            
            ma_20 = df['simple_ma_20'].iloc[-1]
            ma_50 = df['simple_ma_50'].iloc[-1]
            
            if ma_20 > ma_50:
                patterns['uptrend'] = True
            
            if ma_20 < ma_50:
                patterns['downtrend'] = True
            
            # Check for crossovers
            if (df['simple_ma_20'].iloc[-2] < df['simple_ma_50'].iloc[-2]) and (ma_20 > ma_50):
                patterns['golden_cross'] = True
            
            if (df['simple_ma_20'].iloc[-2] > df['simple_ma_50'].iloc[-2]) and (ma_20 < ma_50):
                patterns['death_cross'] = True
            
            # Check for oversold/overbought conditions
            df = self.add_rsi(df)
            last_rsi = df['rsi_14'].iloc[-1]
            
            if last_rsi > 70:
                patterns['overbought'] = True
            
            if last_rsi < 30:
                patterns['oversold'] = True
            
            # Check for support/resistance tests
            support_levels, resistance_levels = self.find_support_resistance(df)
            
            # Check if price is near support or resistance
            for level in support_levels:
                if 0.99 * level < last_close < 1.01 * level:
                    patterns['at_support'] = True
                    break
            
            for level in resistance_levels:
                if 0.99 * level < last_close < 1.01 * level:
                    patterns['at_resistance'] = True
                    break
            
            return patterns
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return patterns
    
    def generate_signals(self, df):
        """
        Generate trading signals based on technical indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with market data and indicators
            
        Returns:
            dict: Dictionary with signals
        """
        signals = {
            'buy': False,
            'sell': False,
            'strength': 0,  # -100 to 100
            'indicators': {}
        }
        
        try:
            # Add required indicators if not already present
            if 'rsi_14' not in df.columns:
                df = self.add_rsi(df)
            
            if 'macd_line' not in df.columns or 'macd_signal' not in df.columns:
                df = self.add_macd(df)
            
            if 'simple_ma_20' not in df.columns:
                df = self.add_moving_average(df, period=20)
            
            if 'simple_ma_50' not in df.columns:
                df = self.add_moving_average(df, period=50)
            
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # RSI signals
            if last_row['rsi_14'] < 30:
                signals['indicators']['rsi'] = 'oversold'
                signals['strength'] += 20
            elif last_row['rsi_14'] > 70:
                signals['indicators']['rsi'] = 'overbought'
                signals['strength'] -= 20
            else:
                signals['indicators']['rsi'] = 'neutral'
            
            # MACD signals
            if last_row['macd_line'] > last_row['macd_signal'] and prev_row['macd_line'] <= prev_row['macd_signal']:
                signals['indicators']['macd'] = 'bullish_crossover'
                signals['strength'] += 30
            elif last_row['macd_line'] < last_row['macd_signal'] and prev_row['macd_line'] >= prev_row['macd_signal']:
                signals['indicators']['macd'] = 'bearish_crossover'
                signals['strength'] -= 30
            elif last_row['macd_line'] > last_row['macd_signal']:
                signals['indicators']['macd'] = 'bullish'
                signals['strength'] += 10
            elif last_row['macd_line'] < last_row['macd_signal']:
                signals['indicators']['macd'] = 'bearish'
                signals['strength'] -= 10
            else:
                signals['indicators']['macd'] = 'neutral'
            
            # Moving Average signals
            if last_row['simple_ma_20'] > last_row['simple_ma_50'] and prev_row['simple_ma_20'] <= prev_row['simple_ma_50']:
                signals['indicators']['moving_averages'] = 'golden_cross'
                signals['strength'] += 40
            elif last_row['simple_ma_20'] < last_row['simple_ma_50'] and prev_row['simple_ma_20'] >= prev_row['simple_ma_50']:
                signals['indicators']['moving_averages'] = 'death_cross'
                signals['strength'] -= 40
            elif last_row['simple_ma_20'] > last_row['simple_ma_50']:
                signals['indicators']['moving_averages'] = 'uptrend'
                signals['strength'] += 15
            elif last_row['simple_ma_20'] < last_row['simple_ma_50']:
                signals['indicators']['moving_averages'] = 'downtrend'
                signals['strength'] -= 15
            else:
                signals['indicators']['moving_averages'] = 'neutral'
            
            # Price action
            if last_row['close'] > last_row['simple_ma_20'] and prev_row['close'] <= prev_row['simple_ma_20']:
                signals['indicators']['price_action'] = 'breakout_up'
                signals['strength'] += 25
            elif last_row['close'] < last_row['simple_ma_20'] and prev_row['close'] >= prev_row['simple_ma_20']:
                signals['indicators']['price_action'] = 'breakdown'
                signals['strength'] -= 25
            else:
                signals['indicators']['price_action'] = 'neutral'
            
            # Generate final signals
            if signals['strength'] > 30:
                signals['buy'] = True
            elif signals['strength'] < -30:
                signals['sell'] = True
            
            # Limit strength to -100 to 100
            signals['strength'] = max(-100, min(100, signals['strength']))
            
            return signals
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return signals
    
    def analyze_data(self, df):
        """
        Perform comprehensive technical analysis on market data
        
        Args:
            df (pandas.DataFrame): DataFrame with market data
            
        Returns:
            dict: Dictionary with analysis results
        """
        try:
            # Add all indicators
            df = self.add_moving_average(df, period=20)
            df = self.add_moving_average(df, period=50)
            df = self.add_moving_average(df, period=200)
            df = self.add_rsi(df)
            df = self.add_macd(df)
            df = self.add_bollinger_bands(df)
            df = self.add_stochastic_oscillator(df)
            df = self.add_atr(df)
            df = self.add_adx(df)
            
            # Detect patterns
            patterns = self.detect_patterns(df)
            
            # Generate signals
            signals = self.generate_signals(df)
            
            # Find support and resistance
            support_levels, resistance_levels = self.find_support_resistance(df)
            
            # Compile analysis
            analysis = {
                'patterns': patterns,
                'signals': signals,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'indicators': {
                    'last_close': df['close'].iloc[-1],
                    'rsi': df['rsi_14'].iloc[-1],
                    'macd_line': df['macd_line'].iloc[-1],
                    'macd_signal': df['macd_signal'].iloc[-1],
                    'macd_histogram': df['macd_histogram'].iloc[-1],
                    'ma_20': df['simple_ma_20'].iloc[-1],
                    'ma_50': df['simple_ma_50'].iloc[-1],
                    'ma_200': df['simple_ma_200'].iloc[-1],
                    'bb_upper': df['bb_upper'].iloc[-1],
                    'bb_middle': df['bb_middle'].iloc[-1],
                    'bb_lower': df['bb_lower'].iloc[-1],
                    'stoch_k': df['stoch_k'].iloc[-1],
                    'stoch_d': df['stoch_d'].iloc[-1],
                    'atr': df['atr'].iloc[-1],
                    'adx': df['adx'].iloc[-1]
                }
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            return {
                'patterns': {},
                'signals': {'buy': False, 'sell': False, 'strength': 0, 'indicators': {}},
                'support_levels': [],
                'resistance_levels': [],
                'indicators': {}
            }
