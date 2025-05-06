import logging
from datetime import datetime

class RiskManager:
    """
    Class for managing trading risk
    """
    def __init__(self):
        """
        Initialize the risk manager
        """
        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize risk parameters
        self.risk_level = "Medium"
        self.max_capital_per_trade = 0.1  # 10% of total capital
        self.daily_loss_limit = 0.03  # 3% of total capital
        self.default_stop_loss = 0.03  # 3% from entry
        self.default_take_profit = 0.05  # 5% from entry
        
        # Initialize tracking variables
        self.daily_trades = {}
        self.daily_pnl = 0
        self.total_capital = 0
    
    def set_risk_level(self, risk_level):
        """
        Set risk level
        
        Args:
            risk_level (str): Risk level (Low, Medium, High)
        """
        self.risk_level = risk_level
        
        # Adjust other parameters based on risk level
        if risk_level == "Low":
            self.max_capital_per_trade = 0.05  # 5% of total capital
            self.daily_loss_limit = 0.02  # 2% of total capital
            self.default_stop_loss = 0.02  # 2% from entry
            self.default_take_profit = 0.03  # 3% from entry
        
        elif risk_level == "Medium":
            self.max_capital_per_trade = 0.1  # 10% of total capital
            self.daily_loss_limit = 0.03  # 3% of total capital
            self.default_stop_loss = 0.03  # 3% from entry
            self.default_take_profit = 0.05  # 5% from entry
        
        elif risk_level == "High":
            self.max_capital_per_trade = 0.2  # 20% of total capital
            self.daily_loss_limit = 0.05  # 5% of total capital
            self.default_stop_loss = 0.05  # 5% from entry
            self.default_take_profit = 0.08  # 8% from entry
        
        self.logger.info(f"Risk level set to {risk_level}")
    
    def set_total_capital(self, capital):
        """
        Set total trading capital
        
        Args:
            capital (float): Total capital
        """
        self.total_capital = capital
        self.logger.info(f"Total capital set to {capital}")
    
    def set_max_capital_per_trade(self, percentage):
        """
        Set maximum capital per trade
        
        Args:
            percentage (float): Percentage of total capital (0.0-1.0)
        """
        self.max_capital_per_trade = max(0.01, min(1.0, percentage))
        self.logger.info(f"Max capital per trade set to {percentage*100}%")
    
    def set_daily_loss_limit(self, percentage):
        """
        Set daily loss limit
        
        Args:
            percentage (float): Percentage of total capital (0.0-1.0)
        """
        self.daily_loss_limit = max(0.01, min(0.5, percentage))
        self.logger.info(f"Daily loss limit set to {percentage*100}%")
    
    def set_default_stop_loss(self, percentage):
        """
        Set default stop loss
        
        Args:
            percentage (float): Percentage from entry (0.0-1.0)
        """
        self.default_stop_loss = max(0.005, min(0.2, percentage))
        self.logger.info(f"Default stop loss set to {percentage*100}%")
    
    def set_default_take_profit(self, percentage):
        """
        Set default take profit
        
        Args:
            percentage (float): Percentage from entry (0.0-1.0)
        """
        self.default_take_profit = max(0.005, min(0.5, percentage))
        self.logger.info(f"Default take profit set to {percentage*100}%")
    
    def can_place_trade(self, symbol, quantity, price, trade_type):
        """
        Check if a trade can be placed based on risk parameters
        
        Args:
            symbol (str): Trading symbol
            quantity (int): Order quantity
            price (float): Entry price
            trade_type (str): BUY or SELL
            
        Returns:
            tuple: (can_trade, reason)
        """
        try:
            # Calculate trade value
            trade_value = quantity * price
            
            # Check if trade value exceeds max capital per trade
            if self.total_capital > 0:
                max_trade_value = self.total_capital * self.max_capital_per_trade
                
                if trade_value > max_trade_value:
                    return (False, f"Trade value ({trade_value}) exceeds max capital per trade ({max_trade_value})")
            
            # Check if daily loss limit is reached
            if self.daily_pnl < 0 and abs(self.daily_pnl) > self.total_capital * self.daily_loss_limit:
                return (False, f"Daily loss limit reached ({abs(self.daily_pnl)} > {self.total_capital * self.daily_loss_limit})")
            
            # Check trading hours (normally would be implemented)
            
            # All checks passed
            return (True, "")
        
        except Exception as e:
            self.logger.error(f"Error in can_place_trade: {str(e)}")
            return (False, str(e))
    
    def calculate_position_size(self, capital, price, risk_percentage=None):
        """
        Calculate appropriate position size based on risk
        
        Args:
            capital (float): Available capital
            price (float): Entry price
            risk_percentage (float): Risk percentage override (0.0-1.0)
            
        Returns:
            int: Recommended position size
        """
        try:
            # Use provided risk percentage or default based on risk level
            if risk_percentage is None:
                risk_percentage = self.max_capital_per_trade
            else:
                risk_percentage = max(0.01, min(1.0, risk_percentage))
            
            # Calculate position size
            max_capital = capital * risk_percentage
            position_size = int(max_capital / price)
            
            return max(1, position_size)
        
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 1
    
    def calculate_stop_loss(self, entry_price, trade_type, stop_percentage=None):
        """
        Calculate stop loss price
        
        Args:
            entry_price (float): Entry price
            trade_type (str): BUY or SELL
            stop_percentage (float): Stop loss percentage override (0.0-1.0)
            
        Returns:
            float: Stop loss price
        """
        try:
            # Use provided stop percentage or default
            if stop_percentage is None:
                stop_percentage = self.default_stop_loss
            else:
                stop_percentage = max(0.005, min(0.2, stop_percentage))
            
            # Calculate stop loss price
            if trade_type == "BUY":
                stop_loss_price = entry_price * (1 - stop_percentage)
            else:  # SELL
                stop_loss_price = entry_price * (1 + stop_percentage)
            
            return round(stop_loss_price, 2)
        
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price * 0.95 if trade_type == "BUY" else entry_price * 1.05
    
    def calculate_take_profit(self, entry_price, trade_type, profit_percentage=None):
        """
        Calculate take profit price
        
        Args:
            entry_price (float): Entry price
            trade_type (str): BUY or SELL
            profit_percentage (float): Take profit percentage override (0.0-1.0)
            
        Returns:
            float: Take profit price
        """
        try:
            # Use provided profit percentage or default
            if profit_percentage is None:
                profit_percentage = self.default_take_profit
            else:
                profit_percentage = max(0.005, min(0.5, profit_percentage))
            
            # Calculate take profit price
            if trade_type == "BUY":
                take_profit_price = entry_price * (1 + profit_percentage)
            else:  # SELL
                take_profit_price = entry_price * (1 - profit_percentage)
            
            return round(take_profit_price, 2)
        
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return entry_price * 1.05 if trade_type == "BUY" else entry_price * 0.95
    
    def get_risk_reward_ratio(self, entry_price, stop_loss_price, take_profit_price, trade_type):
        """
        Calculate risk-reward ratio
        
        Args:
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price
            take_profit_price (float): Take profit price
            trade_type (str): BUY or SELL
            
        Returns:
            float: Risk-reward ratio
        """
        try:
            if trade_type == "BUY":
                risk = entry_price - stop_loss_price
                reward = take_profit_price - entry_price
            else:  # SELL
                risk = stop_loss_price - entry_price
                reward = entry_price - take_profit_price
            
            if risk <= 0:
                self.logger.warning("Invalid risk value (<=0)")
                return 0
            
            return round(reward / risk, 2)
        
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {str(e)}")
            return 0
    
    def record_trade(self, symbol, quantity, entry_price, exit_price, trade_type):
        """
        Record a completed trade
        
        Args:
            symbol (str): Trading symbol
            quantity (int): Order quantity
            entry_price (float): Entry price
            exit_price (float): Exit price
            trade_type (str): BUY or SELL
            
        Returns:
            float: Trade P&L
        """
        try:
            # Calculate P&L
            if trade_type == "BUY":
                pnl = (exit_price - entry_price) * quantity
            else:  # SELL
                pnl = (entry_price - exit_price) * quantity
            
            # Record trade
            date_key = datetime.now().strftime("%Y-%m-%d")
            
            if date_key not in self.daily_trades:
                self.daily_trades[date_key] = []
            
            trade_record = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'trade_type': trade_type,
                'pnl': pnl,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.daily_trades[date_key].append(trade_record)
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            self.logger.info(f"Trade recorded for {symbol}: {pnl}")
            return pnl
        
        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")
            return 0
    
    def reset_daily_stats(self):
        """
        Reset daily statistics
        """
        self.daily_pnl = 0
        date_key = datetime.now().strftime("%Y-%m-%d")
        self.daily_trades[date_key] = []
        
        self.logger.info("Daily statistics reset")
    
    def get_daily_stats(self, date=None):
        """
        Get daily trading statistics
        
        Args:
            date (str): Date in format YYYY-MM-DD (default: today)
            
        Returns:
            dict: Daily statistics
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            if date not in self.daily_trades:
                return {
                    'date': date,
                    'trade_count': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'trades': []
                }
            
            trades = self.daily_trades[date]
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            return {
                'date': date,
                'trade_count': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'trades': trades
            }
        
        except Exception as e:
            self.logger.error(f"Error getting daily stats: {str(e)}")
            return {
                'date': date,
                'error': str(e),
                'trades': []
            }
