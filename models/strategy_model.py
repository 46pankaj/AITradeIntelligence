import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging

class Strategy:
    """
    Class representing a trading strategy
    """
    def __init__(self, name, symbol, exchange, strategy_type="Manual", timeframe="1 day"):
        """
        Initialize a new strategy
        
        Args:
            name (str): Strategy name
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            strategy_type (str): Type of strategy (Manual, AI-Generated)
            timeframe (str): Trading timeframe
        """
        self.name = name
        self.symbol = symbol
        self.exchange = exchange
        self.type = strategy_type
        self.timeframe = timeframe
        self.indicators = {}
        self.entry_conditions = []
        self.exit_conditions = ["take_profit", "stop_loss"]
        self.take_profit = 5.0  # Default 5%
        self.stop_loss = 3.0  # Default 3%
        self.status = "Inactive"
        self.pnl = 0.0
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated = self.created_at
        self.recommendations = []
        self.trades = []
        
        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def add_indicator(self, indicator_type, params):
        """
        Add an indicator to the strategy
        
        Args:
            indicator_type (str): Type of indicator (e.g., moving_average, rsi)
            params (dict): Indicator parameters
        """
        self.indicators[indicator_type] = params
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_entry_condition(self, condition):
        """
        Add an entry condition to the strategy
        
        Args:
            condition (str): Entry condition
        """
        if condition not in self.entry_conditions:
            self.entry_conditions.append(condition)
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_exit_condition(self, condition):
        """
        Add an exit condition to the strategy
        
        Args:
            condition (str): Exit condition
        """
        if condition not in self.exit_conditions:
            self.exit_conditions.append(condition)
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_take_profit(self, percentage):
        """
        Set take profit percentage
        
        Args:
            percentage (float): Take profit percentage
        """
        self.take_profit = percentage
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_stop_loss(self, percentage):
        """
        Set stop loss percentage
        
        Args:
            percentage (float): Stop loss percentage
        """
        self.stop_loss = percentage
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def activate(self):
        """
        Activate the strategy
        """
        self.status = "Active"
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def deactivate(self):
        """
        Deactivate the strategy
        """
        self.status = "Inactive"
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_recommendation(self, action, confidence, reasoning):
        """
        Add a trade recommendation
        
        Args:
            action (str): Recommended action (BUY, SELL, HOLD)
            confidence (int): Confidence level (0-100)
            reasoning (str): Reasoning for the recommendation
        """
        recommendation = {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.recommendations.append(recommendation)
        self.last_updated = recommendation['timestamp']
    
    def record_trade(self, trade_type, quantity, price, status="Open"):
        """
        Record a trade executed by this strategy
        
        Args:
            trade_type (str): Type of trade (BUY, SELL)
            quantity (int): Trade quantity
            price (float): Trade price
            status (str): Trade status (Open, Closed)
        """
        trade = {
            'strategy': self.name,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'type': trade_type,
            'quantity': quantity,
            'price': price,
            'status': status,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.trades.append(trade)
        self.last_updated = trade['timestamp']
    
    def update_pnl(self, pnl):
        """
        Update strategy P&L
        
        Args:
            pnl (float): Profit and loss amount
        """
        self.pnl = pnl
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self):
        """
        Convert strategy to dictionary
        
        Returns:
            dict: Strategy as dictionary
        """
        return {
            'name': self.name,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'type': self.type,
            'timeframe': self.timeframe,
            'indicators': self.indicators,
            'entry_conditions': self.entry_conditions,
            'exit_conditions': self.exit_conditions,
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'status': self.status,
            'pnl': self.pnl,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'recommendations': self.recommendations,
            'trades': self.trades
        }
    
    def to_json(self):
        """
        Convert strategy to JSON string
        
        Returns:
            str: Strategy as JSON string
        """
        return json.dumps(self.to_dict(), indent=4)
    
    @classmethod
    def from_dict(cls, data):
        """
        Create strategy from dictionary
        
        Args:
            data (dict): Strategy data
            
        Returns:
            Strategy: Strategy object
        """
        strategy = cls(
            name=data['name'],
            symbol=data['symbol'],
            exchange=data['exchange'],
            strategy_type=data['type'],
            timeframe=data.get('timeframe', '1 day')
        )
        
        strategy.indicators = data.get('indicators', {})
        strategy.entry_conditions = data.get('entry_conditions', [])
        strategy.exit_conditions = data.get('exit_conditions', ["take_profit", "stop_loss"])
        strategy.take_profit = data.get('take_profit', 5.0)
        strategy.stop_loss = data.get('stop_loss', 3.0)
        strategy.status = data.get('status', 'Inactive')
        strategy.pnl = data.get('pnl', 0.0)
        strategy.created_at = data.get('created_at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        strategy.last_updated = data.get('last_updated', strategy.created_at)
        strategy.recommendations = data.get('recommendations', [])
        strategy.trades = data.get('trades', [])
        
        return strategy
    
    @classmethod
    def from_json(cls, json_str):
        """
        Create strategy from JSON string
        
        Args:
            json_str (str): Strategy as JSON string
            
        Returns:
            Strategy: Strategy object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
