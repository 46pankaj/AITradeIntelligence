import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_strategy')

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    
    def update_current_price(self, price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        self.current_value = self.quantity * price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity

@dataclass
class Signal:
    symbol: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: Optional[str] = None
    source_model: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class Trade:
    symbol: str
    order_type: OrderType
    side: SignalType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    time_in_force: str = "DAY"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal_confidence: Optional[float] = None
    strategy_name: Optional[str] = None
    
    def to_order_dict(self) -> Dict:
        """Convert trade to order dictionary for broker API"""
        order = {
            "symbol": self.symbol,
            "orderType": self.order_type.value,
            "side": self.side.value,
            "quantity": self.quantity,
            "timeInForce": self.time_in_force
        }
        
        if self.price:
            order["price"] = self.price
            
        if self.stop_price:
            order["stopPrice"] = self.stop_price
            
        return order

class Strategy:
    """Base strategy class"""
    
    def __init__(self, name: str, symbols: List[str], params: Dict = None):
        self.name = name
        self.symbols = symbols
        self.params = params or {}
        self.positions = {}  # Current positions managed by this strategy
        logger.info(f"Initialized strategy: {name} for symbols: {symbols}")
        
    def generate_signals(self, predictions: Dict, market_data: Dict) -> List[Signal]:
        """Generate trading signals based on predictions and market data"""
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def convert_to_trades(self, signals: List[Signal], account_info: Dict) -> List[Trade]:
        """Convert signals to executable trades based on account state"""
        raise NotImplementedError("Subclasses must implement convert_to_trades method")
    
    def update_positions(self, positions: List[Position]):
        """Update the strategy's tracked positions"""
        self.positions = {pos.symbol: pos for pos in positions}
        
    def get_position_sizing(self, signal: Signal, account_info: Dict) -> float:
        """Calculate position size based on risk parameters"""
        raise NotImplementedError("Subclasses must implement get_position_sizing method")

class TrendFollowingStrategy(Strategy):
    """Trend following strategy that generates signals based on price momentum"""
    
    def __init__(self, symbols: List[str], params: Dict = None):
        default_params = {
            "trend_threshold": 0.02,  # 2% price change to confirm trend
            "confidence_threshold": 0.6,  # Minimum confidence to generate signal
            "position_sizing": "fixed_risk",  # 'fixed_risk', 'fixed_quantity', 'kelly'
            "risk_per_trade": 0.01,  # 1% of account per trade
            "max_positions": 5,  # Maximum number of simultaneous positions
            "stop_loss_pct": 0.05,  # 5% stop loss from entry
            "take_profit_pct": 0.1,  # 10% take profit from entry
        }
        
        _params = default_params.copy()
        if params:
            _params.update(params)
            
        super().__init__("TrendFollowing", symbols, _params)
    
    def generate_signals(self, predictions: Dict, market_data: Dict) -> List[Signal]:
        """Generate signals based on AI predictions and recent price action"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in predictions or symbol not in market_data:
                continue
                
            prediction = predictions[symbol]
            current_data = market_data[symbol]
            
            # Skip if confidence is below threshold
            if prediction["confidence"] < self.params["confidence_threshold"]:
                continue
                
            # Calculate recent price change to confirm trend
            recent_prices = current_data["close"][-10:]  # Last 10 periods
            price_change = (recent_prices[-1] / recent_prices[0]) - 1
            
            signal_type = SignalType.HOLD
            
            # Determine signal based on prediction and trend confirmation
            if (prediction["direction"] > 0 and 
                price_change > self.params["trend_threshold"]):
                signal_type = SignalType.BUY
            elif (prediction["direction"] < 0 and 
                  price_change < -self.params["trend_threshold"]):
                signal_type = SignalType.SELL
                
            if signal_type != SignalType.HOLD:
                current_price = current_data["close"][-1]
                
                # Calculate stop loss and take profit levels
                stop_loss = None
                take_profit = None
                
                if signal_type == SignalType.BUY:
                    stop_loss = current_price * (1 - self.params["stop_loss_pct"])
                    take_profit = current_price * (1 + self.params["take_profit_pct"])
                elif signal_type == SignalType.SELL:
                    stop_loss = current_price * (1 + self.params["stop_loss_pct"])
                    take_profit = current_price * (1 - self.params["take_profit_pct"])
                
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=prediction["confidence"],
                    timestamp=datetime.now(),
                    target_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    source_model=prediction.get("model_name"),
                    metadata={"price_change": price_change}
                ))
                
        return signals
    
    def get_position_sizing(self, signal: Signal, account_info: Dict) -> float:
        """Calculate position size based on risk parameters"""
        sizing_method = self.params["position_sizing"]
        current_price = signal.target_price
        
        if sizing_method == "fixed_quantity":
            # Fixed quantity per trade
            return self.params.get("fixed_quantity", 1)
            
        elif sizing_method == "fixed_risk":
            # Risk a fixed percentage of account per trade
            account_value = account_info["balance"]
            risk_amount = account_value * self.params["risk_per_trade"]
            
            # Calculate position size based on stop loss
            if signal.stop_loss and current_price:
                risk_per_share = abs(current_price - signal.stop_loss)
                if risk_per_share > 0:
                    return risk_amount / risk_per_share
                    
            # Fallback if stop loss isn't set
            return risk_amount / (current_price * self.params["stop_loss_pct"])
            
        elif sizing_method == "kelly":
            # Kelly criterion for position sizing
            win_rate = signal.confidence
            risk_reward = (signal.take_profit - current_price) / (current_price - signal.stop_loss) if signal.signal_type == SignalType.BUY else (current_price - signal.take_profit) / (signal.stop_loss - current_price)
            
            # Calculate Kelly percentage
            kelly_pct = win_rate - ((1 - win_rate) / risk_reward)
            kelly_pct = max(0, min(kelly_pct, 0.2))  # Cap at 20%
            
            account_value = account_info["balance"]
            return (account_value * kelly_pct) / current_price
            
        else:
            # Default to a small fixed quantity
            return 1
    
    def convert_to_trades(self, signals: List[Signal], account_info: Dict) -> List[Trade]:
        """Convert signals to executable trades based on account state"""
        trades = []
        active_positions = len(self.positions)
        
        # Sort signals by confidence
        sorted_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
        
        for signal in sorted_signals:
            # Skip if we already have a position in this symbol
            if signal.symbol in self.positions:
                continue
                
            # Skip if we've reached max positions
            if active_positions >= self.params["max_positions"]:
                break
                
            # Calculate position size
            quantity = self.get_position_sizing(signal, account_info)
            
            # Round to appropriate precision for the asset
            quantity = round(quantity, 2)  # Adjust precision as needed
            
            # Skip if quantity is too small
            if quantity <= 0:
                continue
                
            # Create trade object
            trade = Trade(
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                side=signal.signal_type,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_confidence=signal.confidence,
                strategy_name=self.name
            )
            
            trades.append(trade)
            active_positions += 1
            
        return trades

class MeanReversionStrategy(Strategy):
    """Mean reversion strategy that looks for price deviations from moving averages"""
    
    def __init__(self, symbols: List[str], params: Dict = None):
        default_params = {
            "lookback_periods": 20,
            "std_dev_threshold": 2.0,  # Standard deviations for entry
            "confidence_threshold": 0.65,
            "position_sizing": "percent_account",
            "account_percent": 0.05,  # 5% of account per position
            "max_positions": 5,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.05,
        }
        
        _params = default_params.copy()
        if params:
            _params.update(params)
            
        super().__init__("MeanReversion", symbols, _params)
    
    def generate_signals(self, predictions: Dict, market_data: Dict) -> List[Signal]:
        """Generate signals based on deviations from moving averages"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in predictions or symbol not in market_data:
                continue
                
            prediction = predictions[symbol]
            current_data = market_data[symbol]
            
            # Skip if confidence is below threshold
            if prediction["confidence"] < self.params["confidence_threshold"]:
                continue
                
            # Calculate moving average and standard deviation
            prices = np.array(current_data["close"][-self.params["lookback_periods"]:])
            moving_avg = np.mean(prices)
            std_dev = np.std(prices)
            
            current_price = prices[-1]
            z_score = (current_price - moving_avg) / std_dev if std_dev > 0 else 0
            
            signal_type = SignalType.HOLD
            
            # Determine signal based on z-score and prediction
            if (z_score < -self.params["std_dev_threshold"] and 
                prediction["direction"] > 0):
                # Price is below average and predicted to rise
                signal_type = SignalType.BUY
            elif (z_score > self.params["std_dev_threshold"] and 
                  prediction["direction"] < 0):
                # Price is above average and predicted to fall
                signal_type = SignalType.SELL
                
            if signal_type != SignalType.HOLD:
                # Calculate stop loss and take profit levels
                stop_loss = None
                take_profit = None
                
                if signal_type == SignalType.BUY:
                    stop_loss = current_price * (1 - self.params["stop_loss_pct"])
                    take_profit = current_price * (1 + self.params["take_profit_pct"])
                elif signal_type == SignalType.SELL:
                    stop_loss = current_price * (1 + self.params["stop_loss_pct"])
                    take_profit = current_price * (1 - self.params["take_profit_pct"])
                
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=prediction["confidence"],
                    timestamp=datetime.now(),
                    target_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    source_model=prediction.get("model_name"),
                    metadata={"z_score": z_score, "moving_avg": moving_avg}
                ))
                
        return signals
    
    def get_position_sizing(self, signal: Signal, account_info: Dict) -> float:
        """Calculate position size based on risk parameters"""
        current_price = signal.target_price
        
        if self.params["position_sizing"] == "percent_account":
            # Use a percentage of account value
            account_value = account_info["balance"]
            position_value = account_value * self.params["account_percent"]
            return position_value / current_price
        else:
            # Default to fixed risk approach
            account_value = account_info["balance"]
            risk_amount = account_value * 0.01  # 1% risk
            
            # Calculate position size based on stop loss
            if signal.stop_loss and current_price:
                risk_per_share = abs(current_price - signal.stop_loss)
                if risk_per_share > 0:
                    return risk_amount / risk_per_share
            
            # Fallback
            return (account_value * 0.01) / current_price
    
    def convert_to_trades(self, signals: List[Signal], account_info: Dict) -> List[Trade]:
        """Convert signals to executable trades based on account state"""
        trades = []
        active_positions = len(self.positions)
        
        # Sort signals by z-score (absolute value)
        sorted_signals = sorted(
            signals, 
            key=lambda x: abs(x.metadata.get("z_score", 0)) if x.metadata else 0, 
            reverse=True
        )
        
        for signal in sorted_signals:
            # Skip if we already have a position in this symbol
            if signal.symbol in self.positions:
                continue
                
            # Skip if we've reached max positions
            if active_positions >= self.params["max_positions"]:
                break
                
            # Calculate position size
            quantity = self.get_position_sizing(signal, account_info)
            
            # Round to appropriate precision for the asset
            quantity = round(quantity, 2)  # Adjust precision as needed
            
            # Skip if quantity is too small
            if quantity <= 0:
                continue
                
            # Create trade object
            trade = Trade(
                symbol=signal.symbol,
                order_type=OrderType.LIMIT,
                side=signal.signal_type,
                quantity=quantity,
                price=signal.target_price,  # Use limit order at current price
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_confidence=signal.confidence,
                strategy_name=self.name
            )
            
            trades.append(trade)
            active_positions += 1
            
        return trades

class StrategyManager:
    """Manages multiple trading strategies and combines their signals"""
    
    def __init__(self):
        self.strategies = {}
        self.position_manager = PositionManager()
        logger.info("Initialized Strategy Manager")
        
    def add_strategy(self, strategy: Strategy):
        """Add a strategy to the manager"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
        
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy from the manager"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
            
    def process_predictions(self, predictions: Dict, market_data: Dict, account_info: Dict) -> List[Trade]:
        """Process predictions and generate trades across all strategies"""
        all_signals = []
        
        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            strategy_symbols = set(strategy.symbols)
            
            # Filter predictions and market data for this strategy
            strategy_predictions = {s: p for s, p in predictions.items() if s in strategy_symbols}
            strategy_market_data = {s: d for s, d in market_data.items() if s in strategy_symbols}
            
            # Generate signals for this strategy
            signals = strategy.generate_signals(strategy_predictions, strategy_market_data)
            all_signals.extend(signals)
            logger.info(f"Strategy {strategy_name} generated {len(signals)} signals")
            
        # Update positions for all strategies
        current_positions = self.position_manager.get_positions()
        for strategy in self.strategies.values():
            strategy.update_positions([p for p in current_positions if p.symbol in strategy.symbols])
            
        # Process signals into trades for each strategy
        all_trades = []
        for strategy_name, strategy in self.strategies.items():
            # Filter signals for this strategy
            strategy_signals = [s for s in all_signals if s.symbol in strategy.symbols]
            
            # Convert signals to trades
            trades = strategy.convert_to_trades(strategy_signals, account_info)
            all_trades.extend(trades)
            
        logger.info(f"Generated {len(all_trades)} trades across all strategies")
        return all_trades
    
    def handle_trade_updates(self, filled_trades: List[Dict]):
        """Handle updates for executed trades"""
        self.position_manager.update_from_trades(filled_trades)
        
        # Update positions for all strategies
        current_positions = self.position_manager.get_positions()
        for strategy in self.strategies.values():
            strategy.update_positions([p for p in current_positions if p.symbol in strategy.symbols])

class PositionManager:
    """Manages current positions and their P&L"""
    
    def __init__(self):
        self.positions = {}  # Symbol -> Position
        
    def add_position(self, position: Position):
        """Add a new position"""
        self.positions[position.symbol] = position
        
    def update_position(self, symbol: str, quantity_change: float, price: float):
        """Update an existing position with a new trade"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.quantity + quantity_change <= 0:
                # Position closed
                del self.positions[symbol]
            else:
                # Update position
                new_quantity = pos.quantity + quantity_change
                # Calculate new average entry price for adds
                if quantity_change > 0:
                    pos.entry_price = ((pos.quantity * pos.entry_price) + 
                                      (quantity_change * price)) / new_quantity
                pos.quantity = new_quantity
                pos.update_current_price(price)
        elif quantity_change > 0:
            # New position
            self.add_position(Position(
                symbol=symbol,
                quantity=quantity_change,
                entry_price=price,
                entry_time=datetime.now(),
                current_price=price
            ))
            
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_current_price(price)
                
    def update_from_trades(self, trades: List[Dict]):
        """Update positions based on executed trades"""
        for trade in trades:
            symbol = trade["symbol"]
            quantity = trade["quantity"]
            price = trade["price"]
            
            if trade["side"] == "SELL":
                quantity = -quantity
                
            self.update_position(symbol, quantity, price)
                
    def get_positions(self) -> List[Position]:
        """Get all current positions"""
        return list(self.positions.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        return self.positions.get(symbol)
    
    def calculate_total_value(self) -> float:
        """Calculate total value of all positions"""
        return sum(pos.current_value for pos in self.positions.values() if pos.current_value is not None)
    
    def calculate_total_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values() if pos.unrealized_pnl is not None)


# Example usage

def create_sample_strategy_manager():
    """Create a sample strategy manager with some strategies"""
    manager = StrategyManager()
    
    # Add a trend following strategy
    trend_strategy = TrendFollowingStrategy(
        symbols=["RELIANCE", "INFY", "TCS", "HDFCBANK"],
        params={
            "trend_threshold": 0.015,
            "confidence_threshold": 0.7,
            "risk_per_trade": 0.01,
        }
    )
    manager.add_strategy(trend_strategy)
    
    # Add a mean reversion strategy
    mean_reversion = MeanReversionStrategy(
        symbols=["SBIN", "TATAMOTORS", "ITC", "HDFCBANK"],
        params={
            "std_dev_threshold": 2.2,
            "lookback_periods": 30,
            "account_percent": 0.03,
        }
    )
    manager.add_strategy(mean_reversion)
    
    return manager


if __name__ == "__main__":
    # Example simulation
    manager = create_sample_strategy_manager()
    
    # Sample prediction data
    predictions = {
        "RELIANCE": {"direction": 1, "confidence": 0.8, "model_name": "lstm_model"},
        "INFY": {"direction": -1, "confidence": 0.65, "model_name": "lstm_model"},
        "SBIN": {"direction": 1, "confidence": 0.75, "model_name": "lstm_model"},
        "HDFCBANK": {"direction": -1, "confidence": 0.9, "model_name": "lstm_model"},
    }
    
    # Sample market data
    market_data = {
        "RELIANCE": {
            "close": [2100, 2110, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2250]
        },
        "INFY": {
            "close": [1500, 1520, 1510, 1500, 1490, 1480, 1470, 1460, 1450, 1440]
        },
        "SBIN": {
            "close": [500, 510, 520, 530, 540, 550, 560, 570, 580, 590]
        },
        "HDFCBANK": {
            "close": [1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780]
        }
    }
    
    # Sample account info
    account_info = {
        "balance": 100000,
        "margin_available": 50000,
    }
    
    # Process predictions and generate trades
    trades = manager.process_predictions(predictions, market_data, account_info)
    
    # Print generated trades
    print(f"Generated {len(trades)} trades:")
    for trade in trades:
        print(f"{trade.side.value} {trade.quantity} {trade.symbol} @ {trade.price if trade.price else 'MARKET'}")
