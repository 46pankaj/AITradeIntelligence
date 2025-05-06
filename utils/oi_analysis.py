import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

class OIAnalysis:
    """
    Class for performing Open Interest (OI) analysis
    """
    def __init__(self):
        """
        Initialize the OI analysis class
        """
        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def get_option_chain(self, api, symbol, exchange="NSE"):
        """
        Get option chain data for a symbol
        
        Args:
            api: Angel One API instance
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: Option chain data or None if request failed
        """
        try:
            # In a real implementation, this would call the API to get the option chain
            # For now, we'll return None since this is implementation-specific to Angel One
            return None
        except Exception as e:
            self.logger.error(f"Error getting option chain: {str(e)}")
            return None
    
    def calculate_put_call_ratio(self, option_chain):
        """
        Calculate put-call ratio from option chain data
        
        Args:
            option_chain (dict): Option chain data
            
        Returns:
            float: Put-call ratio
        """
        try:
            if not option_chain or 'data' not in option_chain:
                return 1.0  # Neutral default
            
            puts_oi = 0
            calls_oi = 0
            
            # Extract put and call OI from option chain
            for strike in option_chain['data']:
                if 'PE' in strike and 'openInterest' in strike['PE']:
                    puts_oi += strike['PE']['openInterest']
                if 'CE' in strike and 'openInterest' in strike['CE']:
                    calls_oi += strike['CE']['openInterest']
            
            # Calculate put-call ratio
            if calls_oi > 0:
                pcr = puts_oi / calls_oi
            else:
                pcr = 1.0  # Default when no call OI
                
            return pcr
        except Exception as e:
            self.logger.error(f"Error calculating put-call ratio: {str(e)}")
            return 1.0
    
    def analyze_oi_change(self, option_chain, previous_option_chain):
        """
        Analyze changes in open interest
        
        Args:
            option_chain (dict): Current option chain data
            previous_option_chain (dict): Previous option chain data
            
        Returns:
            dict: OI change analysis
        """
        try:
            if (not option_chain or 'data' not in option_chain or 
                not previous_option_chain or 'data' not in previous_option_chain):
                return {
                    'calls_oi_change': 0,
                    'puts_oi_change': 0,
                    'max_calls_strike': 0,
                    'max_puts_strike': 0,
                    'sentiment': 'neutral'
                }
            
            # Initialize variables
            calls_oi_change = 0
            puts_oi_change = 0
            calls_oi_by_strike = {}
            puts_oi_by_strike = {}
            prev_calls_oi_by_strike = {}
            prev_puts_oi_by_strike = {}
            
            # Extract current OI by strike
            for strike in option_chain['data']:
                strike_price = strike.get('strikePrice', 0)
                if 'CE' in strike and 'openInterest' in strike['CE']:
                    calls_oi_by_strike[strike_price] = strike['CE']['openInterest']
                if 'PE' in strike and 'openInterest' in strike['PE']:
                    puts_oi_by_strike[strike_price] = strike['PE']['openInterest']
            
            # Extract previous OI by strike
            for strike in previous_option_chain['data']:
                strike_price = strike.get('strikePrice', 0)
                if 'CE' in strike and 'openInterest' in strike['CE']:
                    prev_calls_oi_by_strike[strike_price] = strike['CE']['openInterest']
                if 'PE' in strike and 'openInterest' in strike['PE']:
                    prev_puts_oi_by_strike[strike_price] = strike['PE']['openInterest']
            
            # Calculate total OI changes
            total_calls_oi = sum(calls_oi_by_strike.values())
            total_puts_oi = sum(puts_oi_by_strike.values())
            prev_total_calls_oi = sum(prev_calls_oi_by_strike.values())
            prev_total_puts_oi = sum(prev_puts_oi_by_strike.values())
            
            calls_oi_change = total_calls_oi - prev_total_calls_oi
            puts_oi_change = total_puts_oi - prev_total_puts_oi
            
            # Find strikes with maximum OI
            max_calls_strike = max(calls_oi_by_strike.items(), key=lambda x: x[1])[0] if calls_oi_by_strike else 0
            max_puts_strike = max(puts_oi_by_strike.items(), key=lambda x: x[1])[0] if puts_oi_by_strike else 0
            
            # Determine sentiment based on OI changes
            sentiment = 'neutral'
            if calls_oi_change > 0 and puts_oi_change < 0:
                sentiment = 'bullish'
            elif calls_oi_change < 0 and puts_oi_change > 0:
                sentiment = 'bearish'
            elif calls_oi_change > 0 and puts_oi_change > 0:
                if calls_oi_change > puts_oi_change:
                    sentiment = 'moderately_bullish'
                else:
                    sentiment = 'moderately_bearish'
            
            return {
                'calls_oi_change': calls_oi_change,
                'puts_oi_change': puts_oi_change,
                'max_calls_strike': max_calls_strike,
                'max_puts_strike': max_puts_strike,
                'sentiment': sentiment
            }
        except Exception as e:
            self.logger.error(f"Error analyzing OI change: {str(e)}")
            return {
                'calls_oi_change': 0,
                'puts_oi_change': 0,
                'max_calls_strike': 0,
                'max_puts_strike': 0,
                'sentiment': 'neutral'
            }
    
    def find_support_resistance_from_oi(self, option_chain):
        """
        Find support and resistance levels from option chain OI
        
        Args:
            option_chain (dict): Option chain data
            
        Returns:
            tuple: Support and resistance levels
        """
        try:
            if not option_chain or 'data' not in option_chain:
                return [], []
            
            # Extract OI data by strike
            calls_oi_by_strike = {}
            puts_oi_by_strike = {}
            
            for strike in option_chain['data']:
                strike_price = strike.get('strikePrice', 0)
                if 'CE' in strike and 'openInterest' in strike['CE']:
                    calls_oi_by_strike[strike_price] = strike['CE']['openInterest']
                if 'PE' in strike and 'openInterest' in strike['PE']:
                    puts_oi_by_strike[strike_price] = strike['PE']['openInterest']
            
            # Sort by OI
            sorted_calls = sorted(calls_oi_by_strike.items(), key=lambda x: x[1], reverse=True)
            sorted_puts = sorted(puts_oi_by_strike.items(), key=lambda x: x[1], reverse=True)
            
            # Get top resistance levels (call strikes with highest OI)
            resistance_levels = [strike for strike, oi in sorted_calls[:3]]
            
            # Get top support levels (put strikes with highest OI)
            support_levels = [strike for strike, oi in sorted_puts[:3]]
            
            return support_levels, resistance_levels
        except Exception as e:
            self.logger.error(f"Error finding support/resistance from OI: {str(e)}")
            return [], []
    
    def calculate_max_pain(self, option_chain):
        """
        Calculate max pain point from option chain
        
        Args:
            option_chain (dict): Option chain data
            
        Returns:
            float: Max pain point (strike price)
        """
        try:
            if not option_chain or 'data' not in option_chain:
                return 0
            
            # Extract OI data by strike
            calls_oi_by_strike = {}
            puts_oi_by_strike = {}
            all_strikes = set()
            
            for strike in option_chain['data']:
                strike_price = strike.get('strikePrice', 0)
                all_strikes.add(strike_price)
                
                if 'CE' in strike and 'openInterest' in strike['CE']:
                    calls_oi_by_strike[strike_price] = strike['CE']['openInterest']
                
                if 'PE' in strike and 'openInterest' in strike['PE']:
                    puts_oi_by_strike[strike_price] = strike['PE']['openInterest']
            
            # Calculate pain at each strike
            pain_by_strike = {}
            
            for test_strike in all_strikes:
                pain = 0
                
                # Calculate call options pain
                for strike, oi in calls_oi_by_strike.items():
                    if test_strike > strike:
                        pain += oi * (test_strike - strike)
                
                # Calculate put options pain
                for strike, oi in puts_oi_by_strike.items():
                    if test_strike < strike:
                        pain += oi * (strike - test_strike)
                
                pain_by_strike[test_strike] = pain
            
            # Find strike with minimum pain
            if pain_by_strike:
                max_pain = min(pain_by_strike.items(), key=lambda x: x[1])[0]
                return max_pain
            
            return 0
        except Exception as e:
            self.logger.error(f"Error calculating max pain: {str(e)}")
            return 0
    
    def analyze_oi(self, api, symbol, exchange="NSE"):
        """
        Perform comprehensive OI analysis
        
        Args:
            api: Angel One API instance
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            
        Returns:
            dict: OI analysis results
        """
        try:
            # Get current option chain
            option_chain = self.get_option_chain(api, symbol, exchange)
            
            # For demo purposes, we'll return a placeholder analysis
            # In a real implementation, you would analyze actual option chain data
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'put_call_ratio': 1.05,
                'max_pain': 18000,
                'support_levels': [17800, 17500, 17200],
                'resistance_levels': [18200, 18500, 18800],
                'oi_analysis': {
                    'calls_oi_change': 1000,
                    'puts_oi_change': -500,
                    'max_calls_strike': 18500,
                    'max_puts_strike': 17500,
                    'sentiment': 'bullish'
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing OI: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'put_call_ratio': 1.0,
                'max_pain': 0,
                'support_levels': [],
                'resistance_levels': [],
                'oi_analysis': {
                    'calls_oi_change': 0,
                    'puts_oi_change': 0,
                    'max_calls_strike': 0,
                    'max_puts_strike': 0,
                    'sentiment': 'neutral'
                }
            }
