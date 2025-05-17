"""
Order Management System (OMS) Module

This module is responsible for:
1. Executing trade orders based on signals from the Trading Strategy Module
2. Tracking open orders and positions
3. Managing order lifecycle (submission, modification, cancellation)
4. Maintaining order history and reporting
5. Handling order validation and error management
6. Interfacing with the Angel One Integration Layer
"""

import logging
import uuid
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import threading
import queue
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("oms.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OrderManagementSystem")

class OrderStatus(Enum):
    """Enum representing the possible states of an order."""
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class OrderType(Enum):
    """Enum representing the types of orders."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"


class OrderSide(Enum):
    """Enum representing buy or sell side."""
    BUY = "BUY"
    SELL = "SELL"


class Order:
    """
    Class representing a trade order.
    """
    def __init__(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.strategy_id = strategy_id
        self.status = OrderStatus.CREATED
        self.metadata = metadata or {}
        
        # Tracking fields
        self.creation_time = datetime.now()
        self.submission_time = None
        self.last_update_time = self.creation_time
        self.filled_quantity = 0.0
        self.average_fill_price = 0.0
        self.external_order_id = None  # ID assigned by the broker
        self.error_message = None
        self.execution_details = []  # List of individual fills
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary representation."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side.value if isinstance(self.side, OrderSide) else self.side,
            "order_type": self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "strategy_id": self.strategy_id,
            "status": self.status.value if isinstance(self.status, OrderStatus) else self.status,
            "creation_time": self.creation_time.isoformat(),
            "submission_time": self.submission_time.isoformat() if self.submission_time else None,
            "last_update_time": self.last_update_time.isoformat(),
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "external_order_id": self.external_order_id,
            "error_message": self.error_message,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        """Create an Order object from dictionary."""
        order = cls(
            symbol=data["symbol"],
            quantity=data["quantity"],
            side=OrderSide(data["side"]) if isinstance(data["side"], str) else data["side"],
            order_type=OrderType(data["order_type"]) if isinstance(data["order_type"], str) else data["order_type"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=data.get("time_in_force", "DAY"),
            strategy_id=data.get("strategy_id"),
            metadata=data.get("metadata", {})
        )
        
        # Update tracking fields
        order.order_id = data["order_id"]
        order.status = OrderStatus(data["status"]) if isinstance(data["status"], str) else data["status"]
        order.creation_time = datetime.fromisoformat(data["creation_time"])
        
        if data.get("submission_time"):
            order.submission_time = datetime.fromisoformat(data["submission_time"])
            
        order.last_update_time = datetime.fromisoformat(data["last_update_time"])
        order.filled_quantity = data.get("filled_quantity", 0.0)
        order.average_fill_price = data.get("average_fill_price", 0.0)
        order.external_order_id = data.get("external_order_id")
        order.error_message = data.get("error_message")
        
        return order
    
    def update_status(self, status: OrderStatus, error_message: Optional[str] = None):
        """Update the order status and tracking fields."""
        self.status = status
        self.last_update_time = datetime.now()
        if error_message:
            self.error_message = error_message
            logger.error(f"Order {self.order_id} status update: {status.value}, Error: {error_message}")
        else:
            logger.info(f"Order {self.order_id} status update: {status.value}")
    
    def update_fill(self, filled_quantity: float, fill_price: float, timestamp: Optional[datetime] = None):
        """Update order fill information when partial or full fills occur."""
        execution_time = timestamp or datetime.now()
        
        # Add this execution to the details
        execution_detail = {
            "timestamp": execution_time.isoformat(),
            "quantity": filled_quantity,
            "price": fill_price
        }
        self.execution_details.append(execution_detail)
        
        # Calculate new average price
        total_value = (self.average_fill_price * self.filled_quantity) + (filled_quantity * fill_price)
        self.filled_quantity += filled_quantity
        
        if self.filled_quantity > 0:  # Avoid division by zero
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update order status based on fill
        if abs(self.filled_quantity - self.quantity) < 1e-6:  # Handle floating point comparison
            self.update_status(OrderStatus.FILLED)
        else:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
            
        logger.info(f"Order {self.order_id} filled: {filled_quantity} @ {fill_price}, "
                   f"Total filled: {self.filled_quantity}/{self.quantity}")


class Position:
    """Class representing a trading position."""
    def __init__(self, symbol: str, quantity: float = 0.0, average_price: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity  # Positive for long positions, negative for short
        self.average_price = average_price
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.last_price = average_price if average_price > 0 else 0.0
        self.trades = []  # History of trades that contributed to this position
    
    def update_with_fill(self, quantity: float, price: float, order_id: str):
        """Update position with new fill information."""
        if quantity == 0:
            return
            
        # Track this trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "order_id": order_id,
            "quantity": quantity,
            "price": price
        }
        self.trades.append(trade)
        
        # Calculate PnL for closing positions
        if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            # Closing trade (fully or partially)
            closing_quantity = min(abs(self.quantity), abs(quantity))
            if self.quantity > 0:
                # Selling long position
                self.realized_pnl += closing_quantity * (price - self.average_price)
            else:
                # Buying back short position
                self.realized_pnl += closing_quantity * (self.average_price - price)
        
        # Update position
        old_quantity = self.quantity
        new_quantity = self.quantity + quantity
        
        # Calculate new average price for increasing positions
        if (old_quantity >= 0 and quantity > 0) or (old_quantity <= 0 and quantity < 0):
            # Position is increasing in the same direction
            self.average_price = (abs(old_quantity) * self.average_price + abs(quantity) * price) / abs(new_quantity)
            
        # If position changes direction, reset average price
        elif old_quantity * new_quantity < 0:
            # Position changed direction
            self.average_price = price
            
        self.quantity = new_quantity
        
        # Update last price and unrealized PnL
        self.update_market_price(price)
        
        logger.info(f"Position updated: {self.symbol}, Quantity: {self.quantity}, "
                   f"Avg Price: {self.average_price:.2f}, Realized PnL: {self.realized_pnl:.2f}")
    
    def update_market_price(self, price: float):
        """Update the last market price and recalculate unrealized PnL."""
        self.last_price = price
        
        if self.quantity != 0 and self.average_price > 0:
            if self.quantity > 0:
                # Long position
                self.unrealized_pnl = self.quantity * (price - self.average_price)
            else:
                # Short position
                self.unrealized_pnl = -self.quantity * (self.average_price - price)
        else:
            self.unrealized_pnl = 0
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary representation."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "last_price": self.last_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "market_value": self.quantity * self.last_price if self.last_price else 0,
            "trades": self.trades
        }


class BrokerAdapter:
    """
    Abstract base class for broker integration adapters.
    Concrete implementations will be created for specific brokers like Angel One.
    """
    def __init__(self):
        self.name = "BaseBrokerAdapter"
    
    async def submit_order(self, order: Order) -> Dict:
        """Submit an order to the broker."""
        raise NotImplementedError("Broker adapter must implement submit_order()")
    
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order with the broker."""
        raise NotImplementedError("Broker adapter must implement cancel_order()")
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get the current status of an order."""
        raise NotImplementedError("Broker adapter must implement get_order_status()")
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions from the broker."""
        raise NotImplementedError("Broker adapter must implement get_positions()")
    
    async def get_account_info(self) -> Dict:
        """Get account information from the broker."""
        raise NotImplementedError("Broker adapter must implement get_account_info()")


class MockBrokerAdapter(BrokerAdapter):
    """Mock broker adapter for testing and development."""
    def __init__(self, execution_delay: float = 0.5, fill_probability: float = 0.9):
        super().__init__()
        self.name = "MockBrokerAdapter"
        self.execution_delay = execution_delay  # Simulated execution delay in seconds
        self.fill_probability = fill_probability  # Probability of order being filled
        self.orders = {}  # Track mock orders
        self.positions = {}  # Track mock positions
        self.market_prices = {}  # Mock market prices
        logger.info("Mock broker adapter initialized")
    
    async def submit_order(self, order: Order) -> Dict:
        """Simulate submitting an order to the broker."""
        # Simulate network delay
        await asyncio.sleep(self.execution_delay)
        
        # Generate a broker-side order ID
        broker_order_id = f"MOCK-{uuid.uuid4()}"
        
        # Store the order
        self.orders[order.order_id] = {
            "broker_order_id": broker_order_id,
            "status": "PENDING",
            "order": order
        }
        
        # Simulate random market price if not available
        if order.symbol not in self.market_prices:
            base_price = order.price or 100.0  # Default to 100 if no price provided
            self.market_prices[order.symbol] = base_price * (1 + (random.random() - 0.5) * 0.01)
        
        logger.info(f"Mock order submitted: {order.order_id}, Broker ID: {broker_order_id}")
        
        # Schedule order execution
        asyncio.create_task(self._process_order(order.order_id))
        
        return {
            "success": True,
            "broker_order_id": broker_order_id,
            "message": "Order submitted successfully"
        }
    
    async def _process_order(self, order_id: str):
        """Simulate order processing and execution."""
        await asyncio.sleep(random.random() * self.execution_delay * 2)  # Random execution time
        
        if order_id not in self.orders:
            return
        
        order_info = self.orders[order_id]
        order = order_info["order"]
        
        # Decide if order gets filled
        if random.random() < self.fill_probability:
            # Determine execution price
            symbol = order.symbol
            market_price = self.market_prices.get(symbol, 100.0)
            
            execution_price = market_price
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT]:
                if order.side == OrderSide.BUY and order.price >= market_price:
                    execution_price = market_price
                elif order.side == OrderSide.SELL and order.price <= market_price:
                    execution_price = market_price
                else:
                    # Limit not met, don't execute
                    order_info["status"] = "PENDING"
                    return
            
            # Simulate partial fills with multiple executions
            remaining_qty = order.quantity
            filled_qty = 0
            
            # 50% chance of partial fill first
            if random.random() < 0.5 and remaining_qty > 1:
                partial_qty = remaining_qty * random.uniform(0.3, 0.7)
                partial_qty = round(partial_qty, 2)  # Round to 2 decimal places
                
                # Update order with partial fill
                await self._update_order_fill(order_id, partial_qty, execution_price)
                filled_qty += partial_qty
                remaining_qty -= partial_qty
                
                # Wait a bit before the next fill
                await asyncio.sleep(self.execution_delay)
            
            # Fill the rest if any remains
            if remaining_qty > 0:
                await self._update_order_fill(order_id, remaining_qty, execution_price * (1 + random.uniform(-0.001, 0.001)))
            
            # Update order status
            order_info["status"] = "FILLED"
            logger.info(f"Mock order {order_id} filled at {execution_price}")
            
            # Update market price slightly
            price_impact = 0.0002 * (1 if order.side == OrderSide.BUY else -1)
            self.market_prices[symbol] = market_price * (1 + price_impact)
        else:
            # Randomly reject some orders
            order_info["status"] = "REJECTED"
            logger.info(f"Mock order {order_id} rejected")
    
    async def _update_order_fill(self, order_id: str, quantity: float, price: float):
        """Update an order with fill information."""
        if order_id not in self.orders:
            return
            
        order_info = self.orders[order_id]
        order = order_info["order"]
        
        # Update position
        symbol = order.symbol
        position_qty = quantity
        if order.side == OrderSide.SELL:
            position_qty = -quantity
            
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "avg_price": 0
            }
        
        position = self.positions[symbol]
        old_qty = position["quantity"]
        new_qty = old_qty + position_qty
        
        # Update average price for positions
        if abs(new_qty) > 0:
            if (old_qty >= 0 and position_qty > 0) or (old_qty <= 0 and position_qty < 0):
                # Increasing position
                position["avg_price"] = (abs(old_qty) * position["avg_price"] + abs(position_qty) * price) / abs(new_qty)
            elif old_qty * new_qty < 0:
                # Position changed direction
                position["avg_price"] = price
        
        position["quantity"] = new_qty
        
        # Return fill details
        return {
            "order_id": order_id,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
    
    async def cancel_order(self, order_id: str) -> Dict:
        """Simulate canceling an order."""
        await asyncio.sleep(self.execution_delay * 0.5)  # Simulate network delay
        
        if order_id not in self.orders:
            return {
                "success": False,
                "message": "Order not found"
            }
            
        order_info = self.orders[order_id]
        if order_info["status"] in ["FILLED", "CANCELLED", "REJECTED"]:
            return {
                "success": False,
                "message": f"Cannot cancel order in status: {order_info['status']}"
            }
            
        # Cancel the order
        order_info["status"] = "CANCELLED"
        logger.info(f"Mock order {order_id} cancelled")
        
        return {
            "success": True,
            "message": "Order cancelled successfully"
        }
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get the current status of an order."""
        await asyncio.sleep(self.execution_delay * 0.2)  # Simulate network delay
        
        if order_id not in self.orders:
            return {
                "success": False,
                "message": "Order not found"
            }
            
        order_info = self.orders[order_id]
        
        return {
            "success": True,
            "order_id": order_id,
            "broker_order_id": order_info["broker_order_id"],
            "status": order_info["status"],
            "filled_quantity": order_info["order"].filled_quantity,
            "remaining_quantity": order_info["order"].quantity - order_info["order"].filled_quantity,
            "average_price": order_info["order"].average_fill_price
        }
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        await asyncio.sleep(self.execution_delay * 0.3)  # Simulate network delay
        
        positions_list = []
        for symbol, pos in self.positions.items():
            positions_list.append({
                "symbol": symbol,
                "quantity": pos["quantity"],
                "average_price": pos["avg_price"],
                "market_price": self.market_prices.get(symbol, pos["avg_price"]),
                "market_value": pos["quantity"] * self.market_prices.get(symbol, pos["avg_price"])
            })
            
        return positions_list
    
    async def get_account_info(self) -> Dict:
        """Get account information."""
        await asyncio.sleep(self.execution_delay * 0.3)  # Simulate network delay
        
        # Calculate equity from positions
        equity = 100000.0  # Starting capital
        for symbol, pos in self.positions.items():
            market_price = self.market_prices.get(symbol, pos["avg_price"])
            equity += pos["quantity"] * (market_price - pos["avg_price"])
        
        return {
            "account_id": "MOCK-ACCOUNT",
            "equity": equity,
            "buying_power": equity * 2,  # Simulate 2x margin
            "cash": equity - sum([
                pos["quantity"] * self.market_prices.get(symbol, pos["avg_price"])
                for symbol, pos in self.positions.items()
                if pos["quantity"] > 0
            ]),
            "day_trades_remaining": 3
        }


class OrderManagementSystem:
    """
    Main Order Management System responsible for handling orders, positions, and broker interactions.
    """
    def __init__(self, broker_adapter: BrokerAdapter):
        self.broker = broker_adapter
        self.orders = {}  # Dictionary of Order objects keyed by order_id
        self.positions = {}  # Dictionary of Position objects keyed by symbol
        self.order_queue = queue.Queue()  # Queue for order processing
        self.stop_flag = threading.Event()  # Flag to signal thread termination
        
        # Start order processing thread
        self.processor_thread = threading.Thread(target=self._order_processor_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        # Start position sync thread
        self.sync_thread = threading.Thread(target=self._position_sync_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        # Directory for order storage
        self.data_dir = "data/orders"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing orders and positions
        self._load_data()
        
        logger.info(f"Order Management System initialized with {self.broker.name}")
    
    def create_order(
        self,
        symbol: str,
        quantity: float,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Order:
        """
        Create a new order and add it to the system.
        """
        # Convert string enums to Enum types if needed
        if isinstance(side, str):
            side = OrderSide(side)
        if isinstance(order_type, str):
            order_type = OrderType(order_type)
            
        # Create the order
        order = Order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_id=strategy_id,
            metadata=metadata
        )
        
        # Add to our tracking
        self.orders[order.order_id] = order
        
        # Save the order
        self._save_order(order)
        
        logger.info(f"Order created: {order.order_id}, {symbol} {side.value} {quantity} @ {price}")
        
        return order
    
    def submit_order(self, order_id: str) -> bool:
        """
        Submit an order to the broker for execution.
        """
        if order_id not in self.orders:
            logger.error(f"Cannot submit unknown order: {order_id}")
            return False
            
        order = self.orders[order_id]
        
        # Check if order is in valid state for submission
        if order.status != OrderStatus.CREATED and order.status != OrderStatus.VALIDATED:
            logger.error(f"Cannot submit order {order_id} in status {order.status.value}")
            return False
            
        # Add to processing queue
        order.update_status(OrderStatus.VALIDATED)
        self.order_queue.put(("submit", order_id))
        
        logger.info(f"Order {order_id} queued for submission")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with the broker.
        """
        if order_id not in self.orders:
            logger.error(f"Cannot cancel unknown order: {order_id}")
            return False
            
        order = self.orders[order_id]
        
        # Check if order is in a state where it can be cancelled
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            logger.error(f"Cannot cancel order {order_id} in status {order.status.value}")
            return False
            
        # Add to processing queue
        self.order_queue.put(("cancel", order_id))
        
        logger.info(f"Order {order_id} queued for cancellation")
        return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self.orders.get(order_id)
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with specified status."""
        return [order for order in self.orders.values() if order.status == status]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a specific symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a specific strategy."""
        return [order for order in self.orders.values() if order.strategy_id == strategy_id]
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol. Creates a new one if it doesn't exist."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())
    
    async def sync_with_broker(self):
        """
        Synchronize positions and order statuses with the broker.
        """
        try:
            # Sync positions
            broker_positions = await self.broker.get_positions()
            for pos_data in broker_positions:
                symbol = pos_data["symbol"]
                quantity = pos_data["quantity"]
                avg_price = pos_data["average_price"]
                market_price = pos_data.get("market_price", avg_price)
                
                # Update our position
                position = self.get_position(symbol)
                
                # Only update quantity and avg price if different from broker
                if abs(position.quantity - quantity) > 1e-6 or abs(position.average_price - avg_price) > 1e-6:
                    logger.info(f"Updating position from broker: {symbol}, {quantity} @ {avg_price}")
                    position.quantity = quantity
                    position.average_price = avg_price
                
                # Always update market price
                position.update_market_price(market_price)
            
            # Sync open orders
            open_orders = self.get_orders_by_status(OrderStatus.SUBMITTED) + \
                         self.get_orders_by_status(OrderStatus.PARTIALLY_FILLED)
                         
            for order in open_orders:
                if not order.external_order_id:
                    continue
                    
                order_status = await self.broker.get_order_status(order.external_order_id)
                if not order_status.get("success", False):
                    continue
                    
                # Update order status
                broker_status = order_status["status"]
                filled_qty = order_status.get("filled_quantity", 0)
                avg_price = order_status.get("average_price", 0)
                
                # Update order based on broker status
                if broker_status == "FILLED" and order.status != OrderStatus.FILLED:
                    order.update_status(OrderStatus.FILLED)
                    
                    # Update fill if needed
                    remaining_qty = order.quantity - order.filled_quantity
                    if remaining_qty > 0:
                        order.update_fill(remaining_qty, avg_price)
                        
                        # Update position
                        self._update_position_from_fill(order, remaining_qty, avg_price)
                        
                elif broker_status == "PARTIALLY_FILLED" and filled_qty > order.filled_quantity:
                    # New partial fill
                    new_fill_qty = filled_qty - order.filled_quantity
                    order.update_fill(new_fill_qty, avg_price)
                    
                    # Update position
                    self._update_position_from_fill(order, new_fill_qty, avg_price)
                    
                elif broker_status == "CANCELLED" and order.status != OrderStatus.CANCELLED:
                    order.update_status(OrderStatus.CANCELLED)
                    
                elif broker_status == "REJECTED" and order.status != OrderStatus.REJECTED:
                    order.update_status(OrderStatus.REJECTED, 
                                       error_message=order_status.get("message", "Rejected by broker"))
            
            logger.debug("Synchronization with broker completed")
            
        except Exception as e:
            logger.error(f"Error during broker synchronization: {str(e)}")
            
    def _update_position_from_fill(self, order: Order, quantity: float, price: float):
        """Update position based on order fill information."""
        symbol = order.symbol
        position = self.get_position(symbol)
        
        # Adjust quantity based on order side
        position_quantity = quantity
        if order.side == OrderSide.SELL:
            position_quantity = -quantity
            
        # Update position
        position.update_with_fill(position_quantity, price, order.order_id)
        
        # Save position
        self._save_position(position)
        
    async def _process_order_submission(self, order_id: str):
        """Process an order submission to the broker."""
        if order_id not in self.orders:
            logger.error(f"Cannot process unknown order: {order_id}")
            return
            
        order = self.orders[order_id]
        
        try:
            # Mark as being submitted
            order.submission_time = datetime.now()
            order.update_status(OrderStatus.SUBMITTED)
            
            # Submit to broker
            result = await self.broker.submit_order(order)
            
            if result.get("success", False):
                # Store broker's order ID
                order.external_order_id = result.get("broker_order_id")
                logger.info(f"Order {order_id} submitted successfully, broker ID: {order.external_order_id}")
            else:
                # Handle submission failure
                error_msg = result.get("message", "Unknown submission error")
                order.update_status(OrderStatus.ERROR, error_message=error_msg)
                logger.error(f"Order {order_id} submission failed: {error_msg}")
        
        except Exception as e:
            order.update_status(OrderStatus.ERROR, error_message=f"Submission error: {str(e)}")
            logger.error(f"Exception during order {order_id} submission: {str(e)}")
        
        # Save order after processing
        self._save_order(order)
        
    async def _process_order_cancellation(self, order_id: str):
        """Process an order cancellation with the broker."""
        if order_id not in self.orders:
            logger.error(f"Cannot cancel unknown order: {order_id}")
            return
            
        order = self.orders[order_id]
        
        if not order.external_order_id:
            order.update_status(OrderStatus.ERROR, error_message="No broker order ID for cancellation")
            self._save_order(order)
            return
            
        try:
            # Cancel with broker
            result = await self.broker.cancel_order(order.external_order_id)
            
            if result.get("success", False):
                order.update_status(OrderStatus.CANCELLED)
                logger.info(f"Order {order_id} cancelled successfully")
            else:
                # Handle cancellation failure
                error_msg = result.get("message", "Unknown cancellation error")
                logger.warning(f"Order {order_id} cancellation failed: {error_msg}")
                
                # Check current status with broker
                status_result = await self.broker.get_order_status(order.external_order_id)
                if status_result.get("success", False):
                    # Update based on current status
                    broker_status = status_result.get("status")
                    if broker_status == "FILLED":
                        # Order was filled before we could cancel
                        logger.info(f"Order {order_id} was filled before cancellation")
                        
                        # Update fill
                        filled_qty = status_result.get("filled_quantity", order.quantity)
                        avg_price = status_result.get("average_price", 0)
                        
                        # Update order and position
                        order.update_fill(filled_qty - order.filled_quantity, avg_price)
                        self._update_position_from_fill(order, filled_qty - order.filled_quantity, avg_price)
                        
                    elif broker_status == "CANCELLED":
                        # Order was already cancelled
                        order.update_status(OrderStatus.CANCELLED)
                    else:
                        # Other status
                        order.update_status(OrderStatus.ERROR, 
                                           error_message=f"Cancellation failed, current status: {broker_status}")
        
        except Exception as e:
            order.update_status(OrderStatus.ERROR, error_message=f"Cancellation error: {str(e)}")
            logger.error(f"Exception during order {order_id} cancellation: {str(e)}")
        
        # Save order after processing
        self._save_order(order)
    
    def _order_processor_loop(self):
        """
        Background thread that processes the order queue.
        """
        logger.info("Order processor thread started")
        
        while not self.stop_flag.is_set():
            try:
                # Get an item with timeout to allow checking stop flag
                try:
                    action, order_id = self.order_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                logger.debug(f"Processing {action} for order {order_id}")
                
                # Process based on action
                if action == "submit":
                    # Use asyncio to run coroutine in thread
                    asyncio.run(self._process_order_submission(order_id))
                elif action == "cancel":
                    asyncio.run(self._process_order_cancellation(order_id))
                else:
                    logger.warning(f"Unknown order action: {action}")
                
                # Mark task as complete
                self.order_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in order processor thread: {str(e)}")
        
        logger.info("Order processor thread stopped")
    
    def _position_sync_loop(self):
        """
        Background thread that periodically syncs positions with the broker.
        """
        logger.info("Position sync thread started")
        
        sync_interval = 60  # seconds
        
        while not self.stop_flag.is_set():
            try:
                # Sync with broker
                asyncio.run(self.sync_with_broker())
            except Exception as e:
                logger.error(f"Error in position sync thread: {str(e)}")
                
            # Sleep for interval, checking stop flag periodically
            for _ in range(sync_interval):
                if self.stop_flag.is_set():
                    break
                time.sleep(1)
        
        logger.info("Position sync thread stopped")
    
    def _save_order(self, order: Order):
        """Save order to persistent storage."""
        try:
            order_data = order.to_dict()
            file_path = os.path.join(self.data_dir, f"order_{order.order_id}.json")
            
            with open(file_path, 'w') as f:
                json.dump(order_data, f, indent=2)
                
            logger.debug(f"Order {order.order_id} saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving order {order.order_id}: {str(e)}")
    
    def _save_position(self, position: Position):
        """Save position to persistent storage."""
        try:
            position_data = position.to_dict()
            file_path = os.path.join(self.data_dir, f"position_{position.symbol}.json")
            
            with open(file_path, 'w') as f:
                json.dump(position_data, f, indent=2)
                
            logger.debug(f"Position {position.symbol} saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving position {position.symbol}: {str(e)}")
    
    def _load_data(self):
        """Load orders and positions from persistent storage."""
        # Load orders
        order_pattern = os.path.join(self.data_dir, "order_*.json")
        for file_path in glob.glob(order_pattern):
            try:
                with open(file_path, 'r') as f:
                    order_data = json.load(f)
                    order = Order.from_dict(order_data)
                    self.orders[order.order_id] = order
                    logger.debug(f"Loaded order {order.order_id} from {file_path}")
            except Exception as e:
                logger.error(f"Error loading order from {file_path}: {str(e)}")
        
        # Load positions
        pos_pattern = os.path.join(self.data_dir, "position_*.json")
        for file_path in glob.glob(pos_pattern):
            try:
                with open(file_path, 'r') as f:
                    pos_data = json.load(f)
                    symbol = pos_data["symbol"]
                    position = Position(symbol, pos_data["quantity"], pos_data["average_price"])
                    position.realized_pnl = pos_data.get("realized_pnl", 0.0)
                    position.unrealized_pnl = pos_data.get("unrealized_pnl", 0.0)
                    position.last_price = pos_data.get("last_price", position.average_price)
                    position.trades = pos_data.get("trades", [])
                    self.positions[symbol] = position
                    logger.debug(f"Loaded position {symbol} from {file_path}")
            except Exception as e:
                logger.error(f"Error loading position from {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.orders)} orders and {len(self.positions)} positions")
    
    def shutdown(self):
        """Shut down the order management system cleanly."""
        logger.info("Shutting down Order Management System...")
        
        # Signal threads to stop
        self.stop_flag.set()
        
        # Wait for threads to complete
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5.0)
        if self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)
        
        # Save all data
        for order in self.orders.values():
            self._save_order(order)
        for position in self.positions.values():
            self._save_position(position)
            
        logger.info("Order Management System shutdown complete")
