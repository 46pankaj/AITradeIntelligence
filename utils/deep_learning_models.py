import os
import logging
import random
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepLearningModels:
    """
    Mock class for implementing deep learning models for trading strategy generation.
    This simplified version doesn't require TensorFlow and is used for demonstration purposes.
    """
    def __init__(self, data_manager=None):
        """
        Initialize the deep learning models
        
        Args:
            data_manager: DataManager instance for fetching and storing data
        """
        self.data_manager = data_manager
        self.models_dir = "models"
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def generate_strategy_recommendation(self, symbol, exchange, timeframe="1 day", days_back=60, strategy_type="deep_learning"):
        """
        Generate a trading strategy recommendation using simulated deep learning models
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            timeframe (str): Candle timeframe (1 minute, 5 minutes, etc.)
            days_back (int): Number of days of historical data to use
            strategy_type (str): Type of strategy to generate (deep_learning, gan, ensemble)
            
        Returns:
            dict: Strategy recommendation with action, confidence, reasoning, and additional data
        """
        try:
            # Check if data manager is available
            if self.data_manager is None:
                logger.error("Data manager not provided for fetching historical data")
                return None
            
            # Fetch historical data
            df = self.data_manager.get_historical_data(symbol, exchange, timeframe, days_back)
            
            if df is None or len(df) < 30:  # Require at least 30 data points
                logger.error(f"Insufficient historical data for {symbol} on {exchange}")
                return {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasoning': "Insufficient historical data for analysis."
                }
            
            # Use appropriate model based on strategy type
            if strategy_type == "deep_learning":
                return self._generate_lstm_recommendation(df, symbol, exchange)
            elif strategy_type == "gan":
                return self._generate_gan_recommendation(df, symbol, exchange)
            elif strategy_type == "ensemble":
                return self._generate_ensemble_recommendation(df, symbol, exchange)
            else:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error generating strategy recommendation: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f"Error in strategy generation: {str(e)}"
            }
    
    def _generate_lstm_recommendation(self, df, symbol, exchange):
        """
        Generate a trading recommendation using a simulated LSTM model
        
        Args:
            df (pandas.DataFrame): Historical price data
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Recommendation with action, confidence, and reasoning
        """
        try:
            # Get recent price data
            recent_prices = df['close'].iloc[-30:].tolist()
            current_price = recent_prices[-1]
            
            # Calculate some basic stats from the price
            avg_price = sum(recent_prices) / len(recent_prices)
            price_change = (recent_prices[-1] / recent_prices[0] - 1) * 100
            
            # Calculate a trend score based on recent price movements
            trend_score = price_change * 2  # Scale factor
            
            # Determine action based on trend score
            action = 'HOLD'
            confidence = 50  # base confidence
            
            if trend_score > 5:  # Strong bullish signal
                action = 'BUY'
                confidence = min(90, 50 + int(trend_score))
            elif trend_score > 2:  # Moderate bullish signal
                action = 'BUY'
                confidence = min(80, 50 + int(trend_score * 2))
            elif trend_score < -5:  # Strong bearish signal
                action = 'SELL'
                confidence = min(90, 50 + int(abs(trend_score)))
            elif trend_score < -2:  # Moderate bearish signal
                action = 'SELL'
                confidence = min(80, 50 + int(abs(trend_score) * 2))
            
            # Calculate simulated predicted returns
            predicted_returns = []
            
            # Generate "predictions" for next 5 days
            base_drift = trend_score / 10  # Use trend as a base for predictions
            for i in range(5):
                # Add some randomness to the predictions, but maintain the trend
                daily_return = base_drift + random.uniform(-1.0, 1.0)
                predicted_returns.append(daily_return)
            
            avg_return = sum(predicted_returns) / len(predicted_returns)
            
            # Generate reasoning
            days = ["first", "second", "third", "fourth", "fifth"]
            price_projections = [f"{days[i]} day: {predicted_returns[i]:.2f}%" for i in range(len(days))]
            
            reasoning = f"LSTM model predicts an average return of {avg_return:.2f}% over the next 5 days. "
            reasoning += f"Projected returns by day: {', '.join(price_projections)}. "
            
            # Add analysis of recent price action
            if price_change > 0:
                reasoning += f"Recent price action shows an uptrend of {price_change:.2f}%. "
            else:
                reasoning += f"Recent price action shows a downtrend of {abs(price_change):.2f}%. "
            
            # Check for support/resistance levels
            if action == 'BUY':
                reasoning += "The model suggests bullish momentum in the near term."
            elif action == 'SELL':
                reasoning += "The model suggests bearish momentum in the near term."
            else:
                reasoning += "The model suggests sideways movement in the near term."
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'predicted_returns': predicted_returns,
                'prediction_horizon': len(predicted_returns),
                'model_type': 'LSTM'
            }
        
        except Exception as e:
            logger.error(f"Error generating LSTM recommendation: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f"Error in LSTM recommendation: {str(e)}"
            }
    
    def _generate_gan_recommendation(self, df, symbol, exchange):
        """
        Generate a trading recommendation using a simulated GAN model
        
        Args:
            df (pandas.DataFrame): Historical price data
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Recommendation with action, confidence, and reasoning
        """
        try:
            # Get recent price data
            recent_prices = df['close'].iloc[-30:].tolist()
            current_price = recent_prices[-1]
            
            # Generate multiple price scenarios
            num_scenarios = 20
            scenarios = []
            
            # Use recent price momentum to bias scenarios
            recent_momentum = (recent_prices[-1] / recent_prices[-10] - 1) * 100 if len(recent_prices) >= 10 else 0
            
            # Generate scenarios
            for _ in range(num_scenarios):
                # Base scenario on recent momentum with random variations
                scenario_bias = recent_momentum / 2
                scenario = []
                price = current_price
                
                for day in range(5):  # 5-day scenarios
                    # Daily return with some randomness, biased by momentum
                    daily_return = scenario_bias + random.uniform(-2.0, 2.0)
                    price = price * (1 + daily_return / 100)
                    scenario.append(price)
                
                scenarios.append(scenario)
            
            # Calculate bullish vs bearish scenarios
            bullish_count = sum(1 for scenario in scenarios if scenario[-1] > current_price * 1.01)
            bearish_count = sum(1 for scenario in scenarios if scenario[-1] < current_price * 0.99)
            neutral_count = num_scenarios - bullish_count - bearish_count
            
            # Calculate probabilities
            bullish_prob = bullish_count / num_scenarios
            bearish_prob = bearish_count / num_scenarios
            
            # Calculate average scenario
            avg_scenario = []
            for day in range(5):
                day_avg = sum(scenario[day] for scenario in scenarios) / num_scenarios
                avg_scenario.append(day_avg)
            
            # Determine action and confidence
            action = 'HOLD'
            confidence = 50
            
            if bullish_prob > 0.7:  # Strong bullish bias
                action = 'BUY'
                confidence = int(bullish_prob * 100)
            elif bullish_prob > 0.6:  # Moderate bullish bias
                action = 'BUY'
                confidence = int(bullish_prob * 90)
            elif bearish_prob > 0.7:  # Strong bearish bias
                action = 'SELL'
                confidence = int(bearish_prob * 100)
            elif bearish_prob > 0.6:  # Moderate bearish bias
                action = 'SELL'
                confidence = int(bearish_prob * 90)
            
            # Calculate average return
            avg_return = (avg_scenario[-1] / current_price - 1) * 100
            
            # Generate reasoning
            reasoning = f"GAN model generated {num_scenarios} potential market scenarios. "
            reasoning += f"{bullish_count} scenarios are bullish, {bearish_count} are bearish, and {neutral_count} are neutral. "
            reasoning += f"Average projected return is {avg_return:.2f}%. "
            
            if action == 'BUY':
                reasoning += f"With {bullish_prob:.1%} probability of bullish outcomes, a BUY signal is generated."
            elif action == 'SELL':
                reasoning += f"With {bearish_prob:.1%} probability of bearish outcomes, a SELL signal is generated."
            else:
                reasoning += "Market direction is uncertain, suggesting a HOLD position."
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'bullish_probability': bullish_prob,
                'bearish_probability': bearish_prob,
                'average_return': avg_return,
                'model_type': 'GAN'
            }
        
        except Exception as e:
            logger.error(f"Error generating GAN recommendation: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f"Error in GAN recommendation: {str(e)}"
            }
    
    def _generate_ensemble_recommendation(self, df, symbol, exchange):
        """
        Generate a trading recommendation using an ensemble of simulated models
        
        Args:
            df (pandas.DataFrame): Historical price data
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Recommendation with action, confidence, and reasoning
        """
        try:
            # Get recommendations from different models
            lstm_rec = self._generate_lstm_recommendation(df, symbol, exchange)
            gan_rec = self._generate_gan_recommendation(df, symbol, exchange)
            
            # Add statistical models for ensemble
            arima_rec = self._generate_arima_recommendation(df)
            
            # Calculate technical indicators
            tech_rec = self._generate_technical_recommendation(df)
            
            # Collect all recommendations
            recommendations = [
                (lstm_rec, 0.4),  # LSTM with 40% weight
                (gan_rec, 0.3),   # GAN with 30% weight
                (arima_rec, 0.2),  # ARIMA with 20% weight
                (tech_rec, 0.1)   # Technical with 10% weight
            ]
            
            # Filter out None recommendations
            valid_recs = [(rec, weight) for rec, weight in recommendations if rec is not None]
            
            if not valid_recs:
                return {
                    'action': 'HOLD',
                    'confidence': 0,
                    'reasoning': "All models failed to generate recommendations."
                }
            
            # Recalculate weights to ensure they sum to 1
            total_weight = sum(weight for _, weight in valid_recs)
            adj_recs = [(rec, weight / total_weight) for rec, weight in valid_recs]
            
            # Count weighted votes for each action
            action_votes = {
                'BUY': 0,
                'SELL': 0,
                'HOLD': 0
            }
            
            for rec, weight in adj_recs:
                action = rec['action']
                confidence = rec['confidence'] / 100  # Normalize to 0-1
                action_votes[action] += weight * confidence
            
            # Determine final action
            final_action = max(action_votes.keys(), key=lambda k: action_votes[k])
            final_confidence = int(action_votes[final_action] * 100)
            
            # Generate reasoning
            model_opinions = []
            for rec, weight in adj_recs:
                model_type = rec.get('model_type', 'Unknown')
                model_opinions.append(f"{model_type} recommends {rec['action']} with {rec['confidence']}% confidence")
            
            reasoning = "Ensemble recommendation based on multiple models:\n"
            reasoning += "- " + "\n- ".join(model_opinions) + "\n\n"
            
            if final_action == 'BUY':
                reasoning += f"The ensemble of models shows a strong bullish signal with {final_confidence}% confidence."
            elif final_action == 'SELL':
                reasoning += f"The ensemble of models shows a strong bearish signal with {final_confidence}% confidence."
            else:
                reasoning += f"The ensemble of models suggests a holding pattern with {final_confidence}% confidence."
            
            return {
                'action': final_action,
                'confidence': final_confidence,
                'reasoning': reasoning,
                'model_votes': action_votes,
                'model_type': 'Ensemble'
            }
        
        except Exception as e:
            logger.error(f"Error generating ensemble recommendation: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': f"Error in ensemble recommendation: {str(e)}"
            }
    
    def _generate_arima_recommendation(self, df):
        """
        Generate recommendation based on simulated ARIMA model
        
        Args:
            df (pandas.DataFrame): Historical price data
            
        Returns:
            dict: Recommendation with action, confidence, and reasoning
        """
        try:
            # Get recent price data
            recent_prices = df['close'].iloc[-30:].tolist()
            current_price = recent_prices[-1]
            
            # Analyze recent trend
            short_term_change = (recent_prices[-1] / recent_prices[-5] - 1) * 100 if len(recent_prices) >= 5 else 0
            medium_term_change = (recent_prices[-1] / recent_prices[-15] - 1) * 100 if len(recent_prices) >= 15 else 0
            long_term_change = (recent_prices[-1] / recent_prices[-30] - 1) * 100 if len(recent_prices) >= 30 else 0
            
            # Use weighted sum of changes to determine forecast direction
            trend = (0.5 * short_term_change + 0.3 * medium_term_change + 0.2 * long_term_change)
            
            # Generate "forecast" for next 5 days
            forecast = []
            
            for i in range(5):
                # Use trend as base, add some noise
                daily_change = trend / 5 + random.uniform(-0.8, 0.8)
                next_price = current_price * (1 + daily_change / 100)
                forecast.append(next_price)
            
            # Calculate predicted returns
            predicted_returns = [(price / current_price - 1) * 100 for price in forecast]
            avg_return = sum(predicted_returns) / len(predicted_returns)
            
            # Determine action
            action = 'HOLD'
            confidence = 50
            
            if avg_return > 1.5:
                action = 'BUY'
                confidence = min(80, 50 + int(avg_return * 8))
            elif avg_return < -1.5:
                action = 'SELL'
                confidence = min(80, 50 + int(abs(avg_return) * 8))
            
            # Generate reasoning
            reasoning = f"ARIMA model forecasts an average return of {avg_return:.2f}% over the next 5 days. "
            
            if action == 'BUY':
                reasoning += "Statistical analysis indicates a bullish trend."
            elif action == 'SELL':
                reasoning += "Statistical analysis indicates a bearish trend."
            else:
                reasoning += "Statistical analysis indicates a neutral trend."
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'predicted_returns': predicted_returns,
                'model_type': 'ARIMA'
            }
        
        except Exception as e:
            logger.warning(f"Error generating ARIMA recommendation: {str(e)}")
            return None
    
    def _generate_technical_recommendation(self, df):
        """
        Generate recommendation based on technical indicators
        
        Args:
            df (pandas.DataFrame): Historical price data
            
        Returns:
            dict: Recommendation with action, confidence, and reasoning
        """
        try:
            # Basic technical signals
            signals = []
            
            # Check if technical indicators are already in the dataframe
            has_indicators = all(col in df.columns for col in ['sma20', 'sma50', 'rsi'])
            
            if not has_indicators:
                # We'll just generate simulated signals based on price action
                recent_prices = df['close'].iloc[-30:].tolist()
                
                # Simple momentum
                momentum = (recent_prices[-1] / recent_prices[-10] - 1) * 100 if len(recent_prices) >= 10 else 0
                
                if momentum > 3:
                    signals.append(('BUY', 'Strong momentum to the upside', 75))
                elif momentum < -3:
                    signals.append(('SELL', 'Strong momentum to the downside', 75))
                
                # Price relative to short-term average
                short_avg = sum(recent_prices[-5:]) / 5 if len(recent_prices) >= 5 else recent_prices[-1]
                
                if recent_prices[-1] > short_avg * 1.02:
                    signals.append(('BUY', 'Price above short-term average', 65))
                elif recent_prices[-1] < short_avg * 0.98:
                    signals.append(('SELL', 'Price below short-term average', 65))
                
                # Reversal pattern detection (very simple)
                if len(recent_prices) >= 3:
                    if recent_prices[-3] > recent_prices[-2] and recent_prices[-1] > recent_prices[-2]:
                        signals.append(('BUY', 'Bullish reversal pattern', 70))
                    if recent_prices[-3] < recent_prices[-2] and recent_prices[-1] < recent_prices[-2]:
                        signals.append(('SELL', 'Bearish reversal pattern', 70))
            else:
                # Use the actual indicators in the dataframe
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2] if len(df) > 1 else last_row
                
                # Moving average crossover
                if 'sma20' in df.columns and 'sma50' in df.columns:
                    if last_row['sma20'] > last_row['sma50'] and prev_row['sma20'] <= prev_row['sma50']:
                        signals.append(('BUY', 'Golden Cross (SMA20 crossed above SMA50)', 75))
                    elif last_row['sma20'] < last_row['sma50'] and prev_row['sma20'] >= prev_row['sma50']:
                        signals.append(('SELL', 'Death Cross (SMA20 crossed below SMA50)', 75))
                
                # RSI signals
                if 'rsi' in df.columns:
                    if last_row['rsi'] < 30:
                        signals.append(('BUY', f'RSI oversold ({last_row["rsi"]:.1f})', 70))
                    elif last_row['rsi'] > 70:
                        signals.append(('SELL', f'RSI overbought ({last_row["rsi"]:.1f})', 70))
                
                # MACD signals
                if all(col in df.columns for col in ['macd', 'macd_signal']):
                    if last_row['macd'] > last_row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal']:
                        signals.append(('BUY', 'MACD crossed above signal line', 65))
                    elif last_row['macd'] < last_row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal']:
                        signals.append(('SELL', 'MACD crossed below signal line', 65))
                
                # Bollinger Band signals
                if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                    if last_row['close'] < last_row['bb_lower']:
                        signals.append(('BUY', 'Price below lower Bollinger Band', 60))
                    elif last_row['close'] > last_row['bb_upper']:
                        signals.append(('SELL', 'Price above upper Bollinger Band', 60))
            
            # Determine overall signal
            if not signals:
                return {
                    'action': 'HOLD',
                    'confidence': 50,
                    'reasoning': "No clear technical signals present.",
                    'model_type': 'Technical'
                }
            
            # Count signals by action
            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']
            
            action = 'HOLD'
            confidence = 50
            reasons = []
            
            if len(buy_signals) > len(sell_signals):
                action = 'BUY'
                confidence = max([s[2] for s in buy_signals]) if buy_signals else 50
                reasons = [s[1] for s in buy_signals]
            elif len(sell_signals) > len(buy_signals):
                action = 'SELL'
                confidence = max([s[2] for s in sell_signals]) if sell_signals else 50
                reasons = [s[1] for s in sell_signals]
            else:
                # Equal number of buy and sell signals
                all_confidences = [s[2] for s in signals]
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 50
                confidence = int(avg_confidence)
                reasons = [s[1] for s in signals]
                reasoning = "Mixed technical signals: " + "; ".join(reasons)
            
            reasoning = f"Technical analysis recommends {action} based on: " + "; ".join(reasons)
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'model_type': 'Technical'
            }
        
        except Exception as e:
            logger.warning(f"Error generating technical recommendation: {str(e)}")
            return None
    
    def _save_model_info(self, model_name, model_info):
        """
        Save model information to a JSON file
        
        Args:
            model_name (str): Name of the model
            model_info (dict): Model information
        """
        try:
            info_path = os.path.join(self.models_dir, f"{model_name}_info.json")
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            logger.info(f"Saved model info to {info_path}")
        
        except Exception as e:
            logger.error(f"Error saving model info: {str(e)}")
