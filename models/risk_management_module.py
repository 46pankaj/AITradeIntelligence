import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("risk_management")

class RiskManager:
    """
    Risk Management Module for AI Trading Platform
    
    Handles various risk controls:
    - Position size limits
    - Stop-loss management
    - Drawdown protection
    - Volatility-based position sizing
    - Trading frequency limits
    - Correlation risk monitoring
    - Market condition filters
    """
    
    def __init__(self, config_path: str = "risk_config.json"):
        """
        Initialize the Risk Manager with configuration settings.
        
        Args:
            config_path: Path to the risk configuration file
        """
        self.config = self._load_config(config_path)
        self.trade_history = []
        self.daily_stats = {}
        self.current_positions = {}
        self.last_market_check = None
        self.market_status = "normal"  # Can be "normal", "volatile", "extreme"
        
        # Create directories for storing risk data if they don't exist
        os.makedirs("risk_data", exist_ok=True)
        
        logger.info("Risk Manager initialized with configuration")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load risk management configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Set defaults for missing config parameters
            defaults = {
                "max_position_size_percent": 5.0,
                "max_total_exposure_percent": 25.0,
                "stop_loss_percent": 2.0,
                "trailing_stop_percent": 1.5,
                "max_daily_loss_percent": 5.0,
                "max_drawdown_percent": 15.0,
                "max_trades_per_day": 20,
                "min_time_between_trades_seconds": 300,
                "volatility_scaling": True,
                "correlation_threshold": 0.7,
                "market_condition_filter": True,
                "vix_threshold": 30.0,
                "backtest_mode": False
            }
            
            # Apply defaults for any missing keys
            for key, value in defaults.items():
                if key not in config:
                    config[key] = value
                    
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default values.")
            return defaults
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file {config_path}. Using default values.")
            return defaults
    
    def update_config(self, new_config: Dict) -> None:
        """Update risk management configuration."""
        self.config.update(new_config)
        logger.info("Risk management configuration updated")
        
    def save_config(self, config_path: str = "risk_config.json") -> None:
        """Save current configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
    
    def check_trade(self, trade: Dict) -> Tuple[bool, str]:
        """
        Evaluate a proposed trade against all risk management rules.
        
        Args:
            trade: Dictionary containing trade details:
                  - symbol: Ticker symbol
                  - direction: "buy" or "sell"
                  - quantity: Number of shares/contracts
                  - price: Entry price
                  - account_value: Current total account value
        
        Returns:
            Tuple of (approved: bool, reason: str)
        """
        symbol = trade.get('symbol')
        direction = trade.get('direction')
        quantity = trade.get('quantity', 0)
        price = trade.get('price', 0)
        account_value = trade.get('account_value', 0)
        
        if not all([symbol, direction, quantity, price, account_value]):
            return False, "Missing required trade parameters"
        
        # Check position size limit
        position_value = quantity * price
        position_size_percent = (position_value / account_value) * 100
        
        if position_size_percent > self.config["max_position_size_percent"]:
            return False, f"Position size ({position_size_percent:.2f}%) exceeds maximum allowed ({self.config['max_position_size_percent']}%)"
        
        # Check total exposure
        total_exposure = self._calculate_total_exposure(account_value)
        new_total_exposure = total_exposure + position_size_percent
        
        if new_total_exposure > self.config["max_total_exposure_percent"]:
            return False, f"Total exposure ({new_total_exposure:.2f}%) would exceed maximum allowed ({self.config['max_total_exposure_percent']}%)"
        
        # Check trading frequency
        if not self._check_trading_frequency():
            return False, f"Trading frequency limit reached (max {self.config['max_trades_per_day']} trades per day)"
        
        # Check correlation risk
        if not self._check_correlation_risk(symbol):
            return False, f"Adding {symbol} would create excessive correlation risk"
        
        # Check market conditions if enabled
        if self.config["market_condition_filter"]:
            if not self._check_market_conditions():
                return False, f"Current market conditions do not meet trading criteria"
        
        # All checks passed
        logger.info(f"Trade approved: {symbol} {direction} {quantity} @ {price}")
        return True, "Trade approved"
    
    def _calculate_total_exposure(self, account_value: float) -> float:
        """Calculate current total exposure as percentage of account."""
        total_position_value = sum(pos.get('value', 0) for pos in self.current_positions.values())
        return (total_position_value / account_value) * 100 if account_value > 0 else 0
    
    def _check_trading_frequency(self) -> bool:
        """Check if trading frequency is within limits."""
        today = datetime.now().date()
        
        # Count trades for today
        today_trades = [t for t in self.trade_history 
                       if t.get('timestamp', datetime.now()).date() == today]
        
        if len(today_trades) >= self.config["max_trades_per_day"]:
            return False
        
        # Check time between trades
        if today_trades:
            last_trade_time = max(t.get('timestamp', datetime.min) for t in today_trades)
            time_since_last_trade = (datetime.now() - last_trade_time).total_seconds()
            
            if time_since_last_trade < self.config["min_time_between_trades_seconds"]:
                return False
                
        return True
    
    def _check_correlation_risk(self, new_symbol: str) -> bool:
        """
        Check if adding a new position would create excessive correlation risk.
        This is a simplified implementation - in production you'd use actual correlation data.
        """
        # This is a placeholder - in a real system, you would:
        # 1. Maintain correlation matrix of assets
        # 2. Check if new symbol correlates highly with existing positions
        
        # For demonstration purposes, assuming we have some correlation data
        if not hasattr(self, 'correlation_matrix'):
            self.correlation_matrix = {}  # Would be populated with real correlation data
        
        for symbol in self.current_positions:
            correlation = self.correlation_matrix.get((symbol, new_symbol), 0)
            if abs(correlation) > self.config["correlation_threshold"]:
                logger.warning(f"High correlation detected between {symbol} and {new_symbol}")
                return False
                
        return True
    
    def _check_market_conditions(self) -> bool:
        """
        Check if current market conditions are suitable for trading.
        Updates market status (normal, volatile, extreme) based on indicators.
        """
        # Refresh market check if needed (e.g., not more often than every 15 minutes)
        current_time = datetime.now()
        if (self.last_market_check is None or 
            (current_time - self.last_market_check).total_seconds() > 900):
            
            # In a real implementation, you would:
            # 1. Check VIX or other volatility indicators
            # 2. Monitor market breadth
            # 3. Check for significant index movements
            
            # Placeholder logic - would use actual market data API
            vix_value = self._get_current_vix()
            
            if vix_value > self.config["vix_threshold"] * 1.5:
                self.market_status = "extreme"
            elif vix_value > self.config["vix_threshold"]:
                self.market_status = "volatile"
            else:
                self.market_status = "normal"
                
            self.last_market_check = current_time
            
        # Trading rules based on market status
        if self.market_status == "extreme":
            return False  # No trading in extreme conditions
        elif self.market_status == "volatile":
            # In volatile markets, could implement special rules
            # For example, reduce position sizes by 50%
            pass
            
        return True
    
    def _get_current_vix(self) -> float:
        """
        Get current VIX value from market data.
        This is a placeholder - would call actual market data API.
        """
        # Placeholder - would use actual market data API
        return 20.0  # Example value
    
    def calculate_position_size(self, symbol: str, account_value: float, 
                               risk_per_trade_percent: float = None) -> int:
        """
        Calculate optimal position size based on risk parameters and volatility.
        
        Args:
            symbol: Trading symbol
            account_value: Current account value
            risk_per_trade_percent: Optional override for risk percentage
            
        Returns:
            quantity: Recommended quantity to trade
        """
        if risk_per_trade_percent is None:
            risk_per_trade_percent = self.config.get("risk_per_trade_percent", 1.0)
            
        # Calculate risk amount in currency
        risk_amount = account_value * (risk_per_trade_percent / 100)
        
        # Get current price and volatility metrics for the symbol
        price = self._get_current_price(symbol)
        atr = self._get_average_true_range(symbol)
        
        if not price or not atr:
            logger.warning(f"Missing price or ATR data for {symbol}")
            return 0
            
        # Calculate position size based on ATR (volatility)
        # If using a 2 ATR stop loss, for example:
        stop_distance = atr * 2
        
        # Position size = risk amount / stop distance
        quantity = int(risk_amount / stop_distance / price)
        
        # Apply volatility scaling if enabled
        if self.config["volatility_scaling"]:
            volatility_factor = self._calculate_volatility_factor(symbol)
            quantity = int(quantity * volatility_factor)
            
        # Ensure minimum of 1 share and maximum based on config
        quantity = max(1, quantity)
        
        # Additional check against max position size
        if (quantity * price / account_value * 100) > self.config["max_position_size_percent"]:
            # Limit quantity to max position size
            quantity = int(account_value * self.config["max_position_size_percent"] / 100 / price)
            
        return quantity
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        Placeholder - would call actual market data API.
        """
        # Placeholder - would use actual market data API
        return 100.0  # Example value
    
    def _get_average_true_range(self, symbol: str, period: int = 14) -> float:
        """
        Calculate Average True Range for volatility measurement.
        Placeholder - would call actual market data API or calculate from price data.
        """
        # Placeholder - would use actual market data API or calculate
        return 2.5  # Example value
    
    def _calculate_volatility_factor(self, symbol: str) -> float:
        """
        Calculate a scaling factor based on current volatility vs historical.
        Lower factor for higher volatility, higher for lower volatility.
        """
        # Placeholder - would calculate based on actual volatility data
        current_volatility = self._get_current_volatility(symbol)
        normal_volatility = self._get_historical_volatility(symbol)
        
        if normal_volatility == 0:
            return 1.0
            
        vol_ratio = normal_volatility / current_volatility if current_volatility > 0 else 1.0
        
        # Scale between 0.5 and 1.5
        return min(1.5, max(0.5, vol_ratio))
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility measure for a symbol."""
        # Placeholder - would use actual data
        return 15.0  # Example value - e.g., annualized volatility percentage
    
    def _get_historical_volatility(self, symbol: str) -> float:
        """Get normal/historical volatility for a symbol."""
        # Placeholder - would use actual data
        return 12.0  # Example value
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, direction: str) -> float:
        """
        Calculate appropriate stop loss level based on volatility and risk settings.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            direction: 'buy' or 'sell'
            
        Returns:
            stop_price: Calculated stop loss price
        """
        # Get ATR for this symbol
        atr = self._get_average_true_range(symbol)
        
        # Calculate stop distance (e.g., 2 ATR units)
        stop_atr_multiplier = self.config.get("stop_atr_multiplier", 2.0)
        stop_distance = atr * stop_atr_multiplier
        
        # Alternative: use percentage-based stop loss
        percentage_stop = entry_price * (self.config["stop_loss_percent"] / 100)
        
        # Use the larger of the two for safer stops
        effective_stop_distance = max(stop_distance, percentage_stop)
        
        # Calculate stop price based on direction
        if direction.lower() == "buy":
            stop_price = entry_price - effective_stop_distance
        else:  # sell/short
            stop_price = entry_price + effective_stop_distance
            
        return round(stop_price, 2)
    
    def update_trailing_stop(self, symbol: str, current_price: float, 
                             direction: str, current_stop: float) -> float:
        """
        Update trailing stop based on price movement and settings.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            direction: 'buy' or 'sell'
            current_stop: Current stop price
            
        Returns:
            new_stop_price: Updated stop loss price
        """
        trailing_percent = self.config["trailing_stop_percent"]
        
        if direction.lower() == "buy":
            # For long positions, trail upward
            trail_level = current_price * (1 - trailing_percent/100)
            new_stop = max(current_stop, trail_level)
        else:
            # For short positions, trail downward
            trail_level = current_price * (1 + trailing_percent/100)
            new_stop = min(current_stop, trail_level)
            
        return round(new_stop, 2)
    
    def check_drawdown_limits(self, account_value: float, 
                             peak_value: float) -> Tuple[bool, str]:
        """
        Check if drawdown exceeds specified limits.
        
        Args:
            account_value: Current account value
            peak_value: Peak account value
            
        Returns:
            Tuple of (within_limits: bool, message: str)
        """
        if peak_value == 0:
            return True, "No drawdown data available"
        
        current_drawdown = (peak_value - account_value) / peak_value * 100
        
        if current_drawdown > self.config["max_drawdown_percent"]:
            message = f"Maximum drawdown exceeded: {current_drawdown:.2f}% > {self.config['max_drawdown_percent']}%"
            logger.warning(message)
            return False, message
            
        return True, f"Current drawdown: {current_drawdown:.2f}%"
    
    def check_daily_loss_limit(self, account_value: float, 
                              day_start_value: float) -> Tuple[bool, str]:
        """
        Check if daily loss exceeds specified limits.
        
        Args:
            account_value: Current account value
            day_start_value: Account value at start of trading day
            
        Returns:
            Tuple of (within_limits: bool, message: str)
        """
        if day_start_value == 0:
            return True, "No daily data available"
        
        daily_change_percent = (account_value - day_start_value) / day_start_value * 100
        
        if daily_change_percent < -self.config["max_daily_loss_percent"]:
            message = f"Maximum daily loss exceeded: {daily_change_percent:.2f}% < -{self.config['max_daily_loss_percent']}%"
            logger.warning(message)
            return False, message
            
        return True, f"Current daily change: {daily_change_percent:.2f}%"
    
    def update_position(self, symbol: str, position_data: Dict) -> None:
        """Update information about current positions."""
        self.current_positions[symbol] = position_data
        logger.info(f"Position updated: {symbol}")
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking when closed."""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
            logger.info(f"Position removed from tracking: {symbol}")
    
    def record_trade(self, trade_data: Dict) -> None:
        """
        Record a trade for historical analysis and risk tracking.
        
        Args:
            trade_data: Dictionary with trade details
        """
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
            
        self.trade_history.append(trade_data)
        
        # Limit history size to prevent memory issues
        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-5000:]
            
        logger.info(f"Trade recorded: {trade_data.get('symbol')} {trade_data.get('direction')}")
    
    def save_trade_history(self, filename: str = "risk_data/trade_history.json") -> None:
        """Save trade history to file."""
        # Convert datetimes to strings for JSON serialization
        serializable_history = []
        for trade in self.trade_history:
            trade_copy = trade.copy()
            if isinstance(trade_copy.get('timestamp'), datetime):
                trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
            serializable_history.append(trade_copy)
            
        with open(filename, 'w') as f:
            json.dump(serializable_history, f, indent=4)
            
        logger.info(f"Trade history saved to {filename}")
    
    def load_trade_history(self, filename: str = "risk_data/trade_history.json") -> None:
        """Load trade history from file."""
        try:
            with open(filename, 'r') as f:
                serialized_history = json.load(f)
                
            # Convert string timestamps back to datetime objects
            self.trade_history = []
            for trade in serialized_history:
                if 'timestamp' in trade and isinstance(trade['timestamp'], str):
                    try:
                        trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                    except ValueError:
                        # If parsing fails, keep as string
                        pass
                self.trade_history.append(trade)
                
            logger.info(f"Loaded {len(self.trade_history)} trade records from {filename}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load trade history: {str(e)}")
            self.trade_history = []
    
    def generate_risk_report(self) -> Dict:
        """
        Generate a comprehensive risk report with current positions,
        exposure levels, and recent performance metrics.
        
        Returns:
            Dict containing risk metrics and statistics
        """
        now = datetime.now()
        today = now.date()
        
        # Calculate various metrics
        total_positions = len(self.current_positions)
        position_values = [p.get('value', 0) for p in self.current_positions.values()]
        total_exposure = sum(position_values)
        
        # Get today's trades
        today_trades = [t for t in self.trade_history 
                      if isinstance(t.get('timestamp'), datetime) and 
                         t['timestamp'].date() == today]
        
        # Get win/loss stats if available
        winning_trades = [t for t in self.trade_history if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('profit', 0) < 0]
        
        # Calculate P&L if data available
        pnl_today = sum(t.get('profit', 0) for t in today_trades)
        pnl_week = sum(t.get('profit', 0) for t in self.trade_history 
                     if isinstance(t.get('timestamp'), datetime) and 
                        (now - t['timestamp']).days <= 7)
        
        report = {
            "timestamp": now.isoformat(),
            "total_positions": total_positions,
            "current_exposure": total_exposure,
            "position_count": total_positions,
            "trades_today": len(today_trades),
            "pnl_today": pnl_today,
            "pnl_week": pnl_week,
            "win_rate": len(winning_trades) / (len(winning_trades) + len(losing_trades)) 
                       if (winning_trades or losing_trades) else 0,
            "market_status": self.market_status,
            "current_positions": list(self.current_positions.keys()),
            "risk_warnings": self._generate_risk_warnings(),
        }
        
        # Save report for historical tracking
        self._save_risk_report(report)
        
        return report
    
    def _generate_risk_warnings(self) -> List[str]:
        """Generate list of current risk warnings based on positions and metrics."""
        warnings = []
        
        # Check for concentration risk
        symbols = list(self.current_positions.keys())
        if len(symbols) > 0:
            max_position = max(self.current_positions.values(), 
                              key=lambda x: x.get('value', 0))
            max_symbol = max_position.get('symbol', 'unknown')
            max_value = max_position.get('value', 0)
            
            # Example warning for concentration
            if len(symbols) > 0 and max_value > 0:
                total_value = sum(p.get('value', 0) for p in self.current_positions.values())
                if total_value > 0:
                    concentration = max_value / total_value * 100
                    if concentration > 40:  # Over 40% in one position
                        warnings.append(f"High concentration risk: {max_symbol} is {concentration:.1f}% of portfolio")
        
        # Add other warning checks as needed
        return warnings
        
    def _save_risk_report(self, report: Dict, 
                        filename: str = "risk_data/latest_report.json") -> None:
        """Save risk report to file."""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)

    def shutdown(self) -> None:
        """Perform cleanup when shutting down risk manager."""
        try:
            # Save current state and history
            self.save_trade_history()
            self.save_config()
            
            logger.info("Risk Manager shutdown successfully")
        except Exception as e:
            logger.error(f"Error during Risk Manager shutdown: {str(e)}")


# Usage example
if __name__ == "__main__":
    # Create a sample configuration
    sample_config = {
        "max_position_size_percent": 3.0,
        "max_total_exposure_percent": 20.0,
        "stop_loss_percent": 2.0,
        "trailing_stop_percent": 1.0,
        "max_daily_loss_percent": 5.0,
        "max_drawdown_percent": 10.0,
        "max_trades_per_day": 15,
        "volatility_scaling": True
    }
    
    # Initialize risk manager
    risk_manager = RiskManager()
    risk_manager.update_config(sample_config)
    
    # Example trade
    sample_trade = {
        "symbol": "AAPL",
        "direction": "buy",
        "quantity": 10,
        "price": 150.0,
        "account_value": 100000.0
    }
    
    # Check if trade meets risk criteria
    approved, reason = risk_manager.check_trade(sample_trade)
    print(f"Trade approved: {approved}, Reason: {reason}")
    
    # Calculate stop loss
    stop_price = risk_manager.calculate_stop_loss("AAPL", 150.0, "buy")
    print(f"Recommended stop loss: ${stop_price:.2f}")
    
    # Calculate position size
    recommended_qty = risk_manager.calculate_position_size("AAPL", 100000.0)
    print(f"Recommended position size: {recommended_qty} shares")
    
    # Generate risk report
    report = risk_manager.generate_risk_report()
    print("Risk Report:", json.dumps(report, indent=2))
    
    # Cleanup
    risk_manager.shutdown()
