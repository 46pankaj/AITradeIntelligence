import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import random

from utils.technical_analysis import TechnicalAnalysis
from utils.sentiment_analysis import SentimentAnalysis
from utils.oi_analysis import OIAnalysis

class StrategyGenerator:
    """
    Class for generating trading strategies based on various inputs
    """
    def __init__(self):
        """
        Initialize the strategy generator
        """
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalysis()
        self.oi_analyzer = OIAnalysis()
        
        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Define strategy templates
        self.strategy_templates = {
            'trend_following': {
                'name': 'Trend Following Strategy',
                'description': 'Follows the established market trend using moving averages and momentum indicators.',
                'indicators': ['moving_average', 'macd', 'adx'],
                'entry_conditions': ['ma_crossover', 'macd_confirmation'],
                'exit_conditions': ['ma_crossover_opposite', 'take_profit', 'stop_loss']
            },
            'mean_reversion': {
                'name': 'Mean Reversion Strategy',
                'description': 'Trades price reversion to the mean using oscillators and Bollinger Bands.',
                'indicators': ['bollinger_bands', 'rsi', 'stochastic'],
                'entry_conditions': ['oversold', 'overbought'],
                'exit_conditions': ['mean_reversion', 'take_profit', 'stop_loss']
            },
            'breakout': {
                'name': 'Breakout Strategy',
                'description': 'Identifies and trades breakouts from significant support/resistance levels.',
                'indicators': ['bollinger_bands', 'atr', 'volume'],
                'entry_conditions': ['price_breakout', 'volume_confirmation'],
                'exit_conditions': ['target_reached', 'take_profit', 'stop_loss']
            },
            'sentiment_based': {
                'name': 'Sentiment-Based Strategy',
                'description': 'Trades based on market sentiment analysis and news impact.',
                'indicators': ['sentiment', 'moving_average', 'volume'],
                'entry_conditions': ['sentiment_shift', 'price_confirmation'],
                'exit_conditions': ['sentiment_reversal', 'take_profit', 'stop_loss']
            },
            'oi_based': {
                'name': 'Open Interest Strategy',
                'description': 'Trades based on Open Interest analysis and option chain data.',
                'indicators': ['put_call_ratio', 'max_pain', 'oi_change'],
                'entry_conditions': ['pcr_extreme', 'oi_buildup'],
                'exit_conditions': ['oi_unwinding', 'take_profit', 'stop_loss']
            }
        }
    
    def generate_strategy(self, symbol, exchange, data, use_technical=True, 
                          use_sentiment=True, use_oi=True, risk_level="Moderate", 
                          take_profit=5.0, stop_loss=3.0, strategy_name=None):
        """
        Generate a trading strategy based on multiple analysis types
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            data (pandas.DataFrame): Historical price data
            use_technical (bool): Whether to use technical analysis
            use_sentiment (bool): Whether to use sentiment analysis
            use_oi (bool): Whether to use OI analysis
            risk_level (str): Risk level (Very Conservative, Conservative, Moderate, Aggressive, Very Aggressive)
            take_profit (float): Take profit percentage
            stop_loss (float): Stop loss percentage
            strategy_name (str): Custom strategy name
            
        Returns:
            dict: Generated strategy
        """
        try:
            # Initialize strategy structure
            strategy = {
                'name': strategy_name or f"{symbol} AI Strategy",
                'symbol': symbol,
                'exchange': exchange,
                'type': 'AI-Generated',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'risk_level': risk_level,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'status': 'Active',
                'pnl': 0.0,
                'parameters': {},
                'indicators': {},
                'signals': {},
                'entry_conditions': [],
                'exit_conditions': ['take_profit', 'stop_loss'],
                'recommendations': []
            }
            
            # Adjust risk parameters based on risk level
            self._adjust_risk_parameters(strategy)
            
            # Select strategy template based on analysis
            template = self._select_strategy_template(use_technical, use_sentiment, use_oi)
            
            # Apply template to strategy
            strategy['base_template'] = template['name']
            strategy['description'] = template['description']
            
            # Add technical analysis
            if use_technical and not data.empty:
                self._add_technical_analysis(strategy, data)
            
            # Add sentiment analysis
            if use_sentiment:
                self._add_sentiment_analysis(strategy, symbol)
            
            # Add OI analysis
            if use_oi:
                self._add_oi_analysis(strategy, symbol, exchange)
            
            # Generate entry and exit conditions
            self._generate_conditions(strategy, template)
            
            # Generate trade recommendations
            self._generate_recommendations(strategy)
            
            return strategy
        except Exception as e:
            self.logger.error(f"Error generating strategy: {str(e)}")
            return {
                'name': strategy_name or f"{symbol} Strategy",
                'symbol': symbol,
                'exchange': exchange,
                'type': 'AI-Generated',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'risk_level': risk_level,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'status': 'Error',
                'error': str(e)
            }
    
    def _adjust_risk_parameters(self, strategy):
        """
        Adjust strategy parameters based on risk level
        
        Args:
            strategy (dict): Strategy object to adjust
        """
        risk_level = strategy['risk_level']
        
        # Adjust take profit and stop loss based on risk level
        if risk_level == "Very Conservative":
            strategy['take_profit'] = min(strategy['take_profit'], 3.0)
            strategy['stop_loss'] = min(strategy['stop_loss'], 1.5)
        elif risk_level == "Conservative":
            strategy['take_profit'] = min(strategy['take_profit'], 4.0)
            strategy['stop_loss'] = min(strategy['stop_loss'], 2.0)
        elif risk_level == "Moderate":
            # Keep as is
            pass
        elif risk_level == "Aggressive":
            strategy['take_profit'] = max(strategy['take_profit'], 6.0)
            strategy['stop_loss'] = max(strategy['stop_loss'], 4.0)
        elif risk_level == "Very Aggressive":
            strategy['take_profit'] = max(strategy['take_profit'], 8.0)
            strategy['stop_loss'] = max(strategy['stop_loss'], 5.0)
    
    def _select_strategy_template(self, use_technical, use_sentiment, use_oi):
        """
        Select appropriate strategy template based on enabled analysis types
        
        Args:
            use_technical (bool): Whether technical analysis is enabled
            use_sentiment (bool): Whether sentiment analysis is enabled
            use_oi (bool): Whether OI analysis is enabled
            
        Returns:
            dict: Selected strategy template
        """
        # Determine weights for each template based on enabled analysis types
        weights = {}
        
        if use_technical:
            weights['trend_following'] = 0.3
            weights['mean_reversion'] = 0.3
            weights['breakout'] = 0.3
        else:
            weights['trend_following'] = 0
            weights['mean_reversion'] = 0
            weights['breakout'] = 0
        
        if use_sentiment:
            weights['sentiment_based'] = 0.3
        else:
            weights['sentiment_based'] = 0
        
        if use_oi:
            weights['oi_based'] = 0.3
        else:
            weights['oi_based'] = 0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        
        # Choose template based on weights
        templates = list(weights.keys())
        template_weights = [weights[t] for t in templates]
        
        # Default to trend following if no weights
        if sum(template_weights) == 0:
            chosen_template = self.strategy_templates['trend_following']
        else:
            chosen_template_key = random.choices(templates, weights=template_weights, k=1)[0]
            chosen_template = self.strategy_templates[chosen_template_key]
        
        return chosen_template
    
    def _add_technical_analysis(self, strategy, data):
        """
        Add technical analysis to the strategy
        
        Args:
            strategy (dict): Strategy object
            data (pandas.DataFrame): Historical price data
        """
        # Perform technical analysis
        analysis_results = self.technical_analyzer.analyze_data(data)
        
        # Add relevant indicators to strategy
        strategy['indicators']['technical'] = analysis_results['indicators']
        
        # Add signals
        strategy['signals']['technical'] = analysis_results['signals']
        
        # Add support and resistance levels
        strategy['parameters']['support_levels'] = analysis_results['support_levels']
        strategy['parameters']['resistance_levels'] = analysis_results['resistance_levels']
        
        # Add patterns
        strategy['parameters']['patterns'] = analysis_results['patterns']
    
    def _add_sentiment_analysis(self, strategy, symbol):
        """
        Add sentiment analysis to the strategy
        
        Args:
            strategy (dict): Strategy object
            symbol (str): Trading symbol
        """
        # Perform sentiment analysis
        sentiment_results = self.sentiment_analyzer.analyze_sentiment(symbol)
        
        # Add sentiment results to strategy
        strategy['indicators']['sentiment'] = {
            'overall_score': sentiment_results['overall_score'],
            'overall_label': sentiment_results['overall_label'],
            'news_sentiment': sentiment_results['news_sentiment'].get('sentiment_label', 'neutral'),
            'social_sentiment': sentiment_results['social_sentiment'].get('sentiment_label', 'neutral'),
            'market_sentiment': sentiment_results['market_sentiment'].get('sentiment_label', 'neutral')
        }
        
        # Add signals based on sentiment
        if sentiment_results['overall_label'] == 'bullish':
            strategy['signals']['sentiment'] = {'buy': True, 'sell': False, 'strength': 70}
        elif sentiment_results['overall_label'] == 'bearish':
            strategy['signals']['sentiment'] = {'buy': False, 'sell': True, 'strength': -70}
        else:
            strategy['signals']['sentiment'] = {'buy': False, 'sell': False, 'strength': 0}
    
    def _add_oi_analysis(self, strategy, symbol, exchange):
        """
        Add OI analysis to the strategy
        
        Args:
            strategy (dict): Strategy object
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
        """
        # We can't perform actual OI analysis without API access,
        # so we'll add placeholder data for demo purposes
        oi_results = {
            'put_call_ratio': 1.05,
            'max_pain': 18000,
            'support_levels': [17800, 17500, 17200],
            'resistance_levels': [18200, 18500, 18800],
            'oi_analysis': {
                'calls_oi_change': 1000,
                'puts_oi_change': -500,
                'sentiment': 'bullish'
            }
        }
        
        # Add OI results to strategy
        strategy['indicators']['oi'] = {
            'put_call_ratio': oi_results['put_call_ratio'],
            'max_pain': oi_results['max_pain'],
            'oi_sentiment': oi_results['oi_analysis']['sentiment']
        }
        
        # Add support and resistance levels from OI
        if 'support_levels' not in strategy['parameters']:
            strategy['parameters']['support_levels'] = []
        
        if 'resistance_levels' not in strategy['parameters']:
            strategy['parameters']['resistance_levels'] = []
        
        strategy['parameters']['support_levels'].extend(oi_results['support_levels'])
        strategy['parameters']['resistance_levels'].extend(oi_results['resistance_levels'])
        
        # Add signals based on OI
        if oi_results['oi_analysis']['sentiment'] == 'bullish':
            strategy['signals']['oi'] = {'buy': True, 'sell': False, 'strength': 60}
        elif oi_results['oi_analysis']['sentiment'] == 'bearish':
            strategy['signals']['oi'] = {'buy': False, 'sell': True, 'strength': -60}
        else:
            strategy['signals']['oi'] = {'buy': False, 'sell': False, 'strength': 0}
    
    def _generate_conditions(self, strategy, template):
        """
        Generate entry and exit conditions based on analysis
        
        Args:
            strategy (dict): Strategy object
            template (dict): Strategy template
        """
        # Add standard exit conditions
        if 'take_profit' not in strategy['exit_conditions']:
            strategy['exit_conditions'].append('take_profit')
        
        if 'stop_loss' not in strategy['exit_conditions']:
            strategy['exit_conditions'].append('stop_loss')
        
        # Generate entry conditions based on template and signals
        entry_conditions = []
        
        # Add technical conditions if available
        if 'technical' in strategy['signals']:
            tech_signal = strategy['signals']['technical']
            
            if tech_signal.get('buy', False):
                # Add specific technical conditions based on indicators
                for indicator, value in tech_signal.get('indicators', {}).items():
                    if indicator == 'rsi' and value == 'oversold':
                        entry_conditions.append('RSI oversold')
                    elif indicator == 'macd' and value in ['bullish_crossover', 'bullish']:
                        entry_conditions.append('MACD bullish crossover')
                    elif indicator == 'moving_averages' and value in ['golden_cross', 'uptrend']:
                        entry_conditions.append('Moving average golden cross')
                    elif indicator == 'price_action' and value == 'breakout_up':
                        entry_conditions.append('Price breakout')
            
            elif tech_signal.get('sell', False):
                # Add specific technical conditions based on indicators
                for indicator, value in tech_signal.get('indicators', {}).items():
                    if indicator == 'rsi' and value == 'overbought':
                        entry_conditions.append('RSI overbought')
                    elif indicator == 'macd' and value in ['bearish_crossover', 'bearish']:
                        entry_conditions.append('MACD bearish crossover')
                    elif indicator == 'moving_averages' and value in ['death_cross', 'downtrend']:
                        entry_conditions.append('Moving average death cross')
                    elif indicator == 'price_action' and value == 'breakdown':
                        entry_conditions.append('Price breakdown')
        
        # Add sentiment conditions if available
        if 'sentiment' in strategy['signals']:
            sentiment_signal = strategy['signals']['sentiment']
            
            if sentiment_signal.get('buy', False):
                entry_conditions.append('Bullish sentiment')
            elif sentiment_signal.get('sell', False):
                entry_conditions.append('Bearish sentiment')
        
        # Add OI conditions if available
        if 'oi' in strategy['signals']:
            oi_signal = strategy['signals']['oi']
            
            if oi_signal.get('buy', False):
                entry_conditions.append('Bullish OI')
            elif oi_signal.get('sell', False):
                entry_conditions.append('Bearish OI')
        
        # If no specific conditions were added, use template defaults
        if not entry_conditions and 'entry_conditions' in template:
            entry_conditions = template['entry_conditions']
        
        strategy['entry_conditions'] = entry_conditions
    
    def _generate_recommendations(self, strategy):
        """
        Generate trade recommendations based on signals
        
        Args:
            strategy (dict): Strategy object
        """
        # Initialize counts and strength
        buy_count = 0
        sell_count = 0
        total_strength = 0
        
        # Count signals and calculate overall strength
        for signal_type, signal in strategy['signals'].items():
            if signal.get('buy', False):
                buy_count += 1
                total_strength += signal.get('strength', 0)
            elif signal.get('sell', False):
                sell_count += 1
                total_strength += signal.get('strength', 0)
        
        # Generate recommendation based on signal counts and strength
        recommendation = {}
        
        if buy_count > sell_count:
            recommendation = {
                'action': 'BUY',
                'confidence': min(100, int(abs(total_strength))),
                'reasoning': f"Bullish signals from {buy_count} out of {len(strategy['signals'])} indicators.",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        elif sell_count > buy_count:
            recommendation = {
                'action': 'SELL',
                'confidence': min(100, int(abs(total_strength))),
                'reasoning': f"Bearish signals from {sell_count} out of {len(strategy['signals'])} indicators.",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            # If counts are equal, use strength to decide
            if total_strength > 20:
                recommendation = {
                    'action': 'BUY',
                    'confidence': min(100, int(abs(total_strength))),
                    'reasoning': f"Slightly bullish overall with strength score of {total_strength}.",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            elif total_strength < -20:
                recommendation = {
                    'action': 'SELL',
                    'confidence': min(100, int(abs(total_strength))),
                    'reasoning': f"Slightly bearish overall with strength score of {total_strength}.",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                recommendation = {
                    'action': 'HOLD',
                    'confidence': 50,
                    'reasoning': "Mixed signals with no clear direction.",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        strategy['recommendations'].append(recommendation)
