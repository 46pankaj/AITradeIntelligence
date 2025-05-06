import logging
import numpy as np
import pandas as pd
from datetime import datetime
from utils.deep_learning_models import DeepLearningModels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIStrategyGenerator:
    """
    Class for generating trading strategies using AI models
    """
    def __init__(self, data_manager):
        """
        Initialize the AI strategy generator
        
        Args:
            data_manager: DataManager instance for fetching and storing data
        """
        self.data_manager = data_manager
        self.deep_learning_models = DeepLearningModels(data_manager)
        
    def generate_strategy(self, symbol, exchange, strategy_type="deep_learning", 
                          risk_level="Medium", timeframe="1 day", days_back=60):
        """
        Generate a trading strategy using AI models
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            strategy_type (str): Type of strategy (deep_learning, gan, ensemble)
            risk_level (str): Risk level (Low, Medium, High)
            timeframe (str): Candle timeframe (1 minute, 5 minutes, etc.)
            days_back (int): Number of days of historical data to use
            
        Returns:
            dict: AI-generated strategy
        """
        try:
            # Generate base strategy structure
            strategy = {
                'name': f"{strategy_type.upper()}_{symbol}_{exchange}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'symbol': symbol,
                'exchange': exchange,
                'type': strategy_type,
                'timeframe': timeframe,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'Active',
                'indicators': [],
                'entry_conditions': [],
                'exit_conditions': [],
                'take_profit': self._get_take_profit_for_risk(risk_level),
                'stop_loss': self._get_stop_loss_for_risk(risk_level),
                'risk_level': risk_level,
                'recommendations': [],
                'trades': [],
                'pnl': 0.0
            }
            
            # Generate strategy recommendation from deep learning models
            if strategy_type == "deep_learning":
                recommendation = self.deep_learning_models.generate_strategy_recommendation(
                    symbol, exchange, timeframe, days_back, "deep_learning"
                )
            elif strategy_type == "gan":
                recommendation = self.deep_learning_models.generate_strategy_recommendation(
                    symbol, exchange, timeframe, days_back, "gan"
                )
            elif strategy_type == "ensemble":
                recommendation = self.deep_learning_models.generate_strategy_recommendation(
                    symbol, exchange, timeframe, days_back, "ensemble"
                )
            else:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return None
            
            if recommendation is None:
                logger.error(f"Failed to generate recommendation for {symbol} on {exchange}")
                return None
            
            # Add indicators based on recommendation
            self._add_indicators_based_on_recommendation(strategy, recommendation)
            
            # Add entry conditions based on recommendation
            self._add_entry_conditions_based_on_recommendation(strategy, recommendation)
            
            # Add exit conditions based on recommendation
            self._add_exit_conditions_based_on_recommendation(strategy, recommendation)
            
            # Get current price
            current_price = self.data_manager.api.get_ltp(exchange, symbol)
            if current_price:
                current_price = current_price.get('ltp', 0)
            else:
                current_price = 0
                
            # Add the recommendation
            strategy['recommendations'].append({
                'action': recommendation['action'],
                'confidence': recommendation['confidence'],
                'reasoning': recommendation['reasoning'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': current_price
            })
            
            logger.info(f"Generated {strategy_type} strategy for {symbol} on {exchange}")
            
            return strategy
        
        except Exception as e:
            logger.error(f"Error generating AI strategy: {str(e)}")
            return None
    
# This method is not used in the current version, so we'll remove it for now
    
    def _get_take_profit_for_risk(self, risk_level):
        """
        Get take profit percentage based on risk level
        
        Args:
            risk_level (str): Risk level (Low, Medium, High)
            
        Returns:
            float: Take profit percentage
        """
        if risk_level == "Low":
            return 3.0
        elif risk_level == "Medium":
            return 5.0
        elif risk_level == "High":
            return 10.0
        else:
            return 5.0  # Default to Medium
    
    def _get_stop_loss_for_risk(self, risk_level):
        """
        Get stop loss percentage based on risk level
        
        Args:
            risk_level (str): Risk level (Low, Medium, High)
            
        Returns:
            float: Stop loss percentage
        """
        if risk_level == "Low":
            return 2.0
        elif risk_level == "Medium":
            return 3.0
        elif risk_level == "High":
            return 5.0
        else:
            return 3.0  # Default to Medium
    
    def _add_indicators_based_on_recommendation(self, strategy, recommendation):
        """
        Add indicators to strategy based on recommendation
        
        Args:
            strategy (dict): Strategy to update
            recommendation (dict): Model recommendation
        """
        # Add basic indicators for strategy
        strategy['indicators'].append({
            'type': 'moving_average',
            'params': {
                'period': 20,
                'type': 'SMA'
            }
        })
        
        strategy['indicators'].append({
            'type': 'moving_average',
            'params': {
                'period': 50,
                'type': 'SMA'
            }
        })
        
        strategy['indicators'].append({
            'type': 'rsi',
            'params': {
                'period': 14,
                'overbought': 70,
                'oversold': 30
            }
        })
        
        # Add model-specific indicators
        model_type = recommendation.get('model_type', '')
        
        if model_type == 'LSTM':
            # Add MACD for trend confirmation
            strategy['indicators'].append({
                'type': 'macd',
                'params': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                }
            })
        
        elif model_type == 'GAN':
            # Add Bollinger Bands for volatility
            strategy['indicators'].append({
                'type': 'bollinger_bands',
                'params': {
                    'period': 20,
                    'std_dev': 2
                }
            })
        
        elif model_type == 'Ensemble':
            # Add both MACD and Bollinger Bands
            strategy['indicators'].append({
                'type': 'macd',
                'params': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                }
            })
            
            strategy['indicators'].append({
                'type': 'bollinger_bands',
                'params': {
                    'period': 20,
                    'std_dev': 2
                }
            })
    
    def _add_entry_conditions_based_on_recommendation(self, strategy, recommendation):
        """
        Add entry conditions to strategy based on recommendation
        
        Args:
            strategy (dict): Strategy to update
            recommendation (dict): Model recommendation
        """
        action = recommendation['action']
        model_type = recommendation.get('model_type', '')
        
        # Basic entry conditions based on action
        if action == 'BUY':
            strategy['entry_conditions'].append(
                "Close > SMA(20) AND RSI(14) > 40"
            )
            
            if model_type == 'LSTM':
                strategy['entry_conditions'].append(
                    "MACD Line > Signal Line"
                )
            
            elif model_type == 'GAN':
                strategy['entry_conditions'].append(
                    "Close > Lower Bollinger Band(20, 2) AND RSI(14) < 50"
                )
            
            elif model_type == 'Ensemble' or model_type == 'ARIMA':
                strategy['entry_conditions'].append(
                    "MACD Line > Signal Line OR RSI(14) < 40"
                )
        
        elif action == 'SELL':
            strategy['entry_conditions'].append(
                "Close < SMA(20) AND RSI(14) < 60"
            )
            
            if model_type == 'LSTM':
                strategy['entry_conditions'].append(
                    "MACD Line < Signal Line"
                )
            
            elif model_type == 'GAN':
                strategy['entry_conditions'].append(
                    "Close < Upper Bollinger Band(20, 2) AND RSI(14) > 50"
                )
            
            elif model_type == 'Ensemble' or model_type == 'ARIMA':
                strategy['entry_conditions'].append(
                    "MACD Line < Signal Line OR RSI(14) > 60"
                )
        
        else:  # HOLD or unknown
            # Set more conservative entry conditions
            strategy['entry_conditions'].append(
                "RSI(14) < 30 AND Close > SMA(50)"  # Strong oversold condition for buy
            )
    
    def _add_exit_conditions_based_on_recommendation(self, strategy, recommendation):
        """
        Add exit conditions to strategy based on recommendation
        
        Args:
            strategy (dict): Strategy to update
            recommendation (dict): Model recommendation
        """
        action = recommendation['action']
        model_type = recommendation.get('model_type', '')
        
        # Basic exit conditions
        # Always include take profit and stop loss
        strategy['exit_conditions'].append(
            f"Profit >= {strategy['take_profit']}% OR Loss >= {strategy['stop_loss']}%"
        )
        
        # Add model-specific exit conditions
        if action == 'BUY':
            # Exit conditions for long positions
            strategy['exit_conditions'].append(
                "RSI(14) > 70"  # Overbought condition
            )
            
            if model_type == 'LSTM':
                strategy['exit_conditions'].append(
                    "MACD Line < Signal Line"
                )
            
            elif model_type == 'GAN':
                strategy['exit_conditions'].append(
                    "Close > Upper Bollinger Band(20, 2)"
                )
            
            elif model_type == 'Ensemble':
                strategy['exit_conditions'].append(
                    "RSI(14) > 70 OR MACD Line < Signal Line"
                )
        
        elif action == 'SELL':
            # Exit conditions for short positions
            strategy['exit_conditions'].append(
                "RSI(14) < 30"  # Oversold condition
            )
            
            if model_type == 'LSTM':
                strategy['exit_conditions'].append(
                    "MACD Line > Signal Line"
                )
            
            elif model_type == 'GAN':
                strategy['exit_conditions'].append(
                    "Close < Lower Bollinger Band(20, 2)"
                )
            
            elif model_type == 'Ensemble':
                strategy['exit_conditions'].append(
                    "RSI(14) < 30 OR MACD Line > Signal Line"
                )
        
        else:  # HOLD or unknown
            # Set general exit conditions
            strategy['exit_conditions'].append(
                "RSI(14) < 30 OR RSI(14) > 70"  # Extreme conditions
            )