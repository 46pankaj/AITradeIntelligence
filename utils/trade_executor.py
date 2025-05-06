import logging
from datetime import datetime
import json
import time
import threading

class TradeExecutor:
    """
    Class for executing trades based on strategies
    """
    def __init__(self, api):
        """
        Initialize the trade executor
        
        Args:
            api: Angel One API instance
        """
        self.api = api
        
        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize active strategy executions
        self.active_executions = {}
    
    def place_order(self, symbol, exchange, transaction_type, quantity, order_type, price=0, trigger_price=0, product_type="INTRADAY"):
        """
        Place an order with Angel One
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange (NSE, BSE)
            transaction_type (str): BUY or SELL
            quantity (int): Order quantity
            order_type (str): MARKET, LIMIT, SL, SL-M
            price (float): Order price (for LIMIT, SL orders)
            trigger_price (float): Trigger price (for SL, SL-M orders)
            product_type (str): DELIVERY, INTRADAY, MARGIN
            
        Returns:
            dict: Order response
        """
        try:
            self.logger.info(f"Placing {transaction_type} order for {quantity} {symbol} on {exchange}")
            
            # Call Angel One API to place the order
            order_response = self.api.place_order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                price=price,
                trigger_price=trigger_price,
                product_type=product_type
            )
            
            if order_response['status'] == 'success':
                self.logger.info(f"Order placed successfully. Order ID: {order_response['order_id']}")
            else:
                self.logger.error(f"Order placement failed: {order_response['message']}")
            
            return order_response
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error placing order: {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def execute_strategy(self, strategy, quantity, capital, enforce_strategy_rules=True):
        """
        Execute a trading strategy
        
        Args:
            strategy (dict): Strategy object
            quantity (int): Order quantity
            capital (float): Capital to allocate
            enforce_strategy_rules (bool): Whether to enforce strategy rules check
            
        Returns:
            dict: Execution response
        """
        try:
            strategy_id = strategy['name']
            self.logger.info(f"Executing strategy '{strategy_id}' for {strategy['symbol']}")
            
            # Verify strategy is active if enforcement is enabled
            if enforce_strategy_rules and strategy.get('status') != 'Active':
                return {"status": "error", "message": "Strategy is not active. Only active strategies can be executed."}
            
            # Check if we have recommendations
            if not strategy.get('recommendations', []):
                return {"status": "error", "message": "No recommendations in strategy"}
            
            # Get the latest recommendation
            recommendation = strategy['recommendations'][-1]
            action = recommendation['action']
            
            # Validate action
            if action not in ['BUY', 'SELL', 'HOLD']:
                return {"status": "error", "message": f"Invalid action: {action}"}
            
            # If HOLD, do nothing
            if action == 'HOLD':
                return {"status": "success", "message": "Strategy recommends HOLD, no action taken"}
            
            # Get symbol details
            symbol = strategy['symbol']
            exchange = strategy['exchange']
            
            # Get current price with retries
            max_retries = 3
            retry_count = 0
            price_data = None
            
            while retry_count < max_retries:
                price_data = self.api.get_ltp(exchange, symbol)
                if price_data:
                    break
                retry_count += 1
                self.logger.warning(f"Retry {retry_count}/{max_retries} - Failed to get price for {symbol}")
                time.sleep(2)  # Wait 2 seconds before retry
            
            if not price_data:
                error_msg = f"Failed to get current price after {max_retries} retries. Please check:\n1. API credentials\n2. Market hours\n3. Symbol validity"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg, "transaction_type": action}
            
            current_price = price_data['ltp']
            
            # Set transaction type based on strategy recommendation
            transaction_type = action
            
            # Calculate quantity based on capital if not provided
            if quantity <= 0 and capital > 0:
                quantity = int(capital / current_price)
            
            if quantity <= 0:
                return {"status": "error", "message": "Invalid quantity", "transaction_type": transaction_type}
            
            # Check if we have options recommendation
            options_data = recommendation.get('options')
            if options_data:
                # Modify symbol for options trading
                options_symbol = f"{symbol}{options_data['expiry']}{options_data['strike']}{options_data['type']}"
                
                # Place options order
                order_response = self.place_order(
                    symbol=options_symbol,
                    exchange=exchange,
                    transaction_type=action,
                    quantity=quantity,
                    order_type="MARKET",
                    product_type="INTRADAY"
                )
            else:
                # Place regular order
                order_response = self.place_order(
                    symbol=symbol,
                    exchange=exchange,
                    transaction_type=action,
                    quantity=quantity,
                    order_type="MARKET",
                    product_type="INTRADAY"
                )
            
            if order_response['status'] == 'success':
                # Store execution details for tracking
                self.active_executions[strategy_id] = {
                    'strategy': strategy,
                    'order_id': order_response['order_id'],
                    'entry_price': current_price,
                    'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'quantity': quantity,
                    'action': action,
                    'take_profit': strategy['take_profit'],
                    'stop_loss': strategy['stop_loss'],
                    'status': 'open'
                }
                
                # Start monitoring thread if this is an auto-trading strategy
                self._start_monitoring(strategy_id)
                
                return {
                    "status": "success", 
                    "message": f"Strategy execution initiated with {action} order",
                    "order_id": order_response['order_id'],
                    "transaction_type": action
                }
            else:
                # Ensure transaction_type is returned even in error case
                return {
                    "status": "error", 
                    "message": f"Failed to place order: {order_response['message']}",
                    "transaction_type": action  # Include the action/transaction_type in error response too
                }
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error executing strategy: {error_msg}")
            
            # Try to get transaction type from strategy recommendations
            transaction_type = "BUY"  # Default fallback
            try:
                if strategy.get('recommendations') and len(strategy.get('recommendations', [])) > 0:
                    last_recommendation = strategy['recommendations'][-1]
                    if last_recommendation.get('action'):
                        transaction_type = last_recommendation['action']
            except:
                self.logger.warning("Could not determine transaction type from strategy")
                
            return {
                "status": "error", 
                "message": error_msg,
                "transaction_type": transaction_type  # Include transaction type even in error cases
            }
    
    def _start_monitoring(self, strategy_id):
        """
        Start monitoring a strategy execution for take profit and stop loss
        
        Args:
            strategy_id (str): Strategy ID to monitor
        """
        # Start monitoring in a separate thread to avoid blocking
        thread = threading.Thread(target=self._monitor_execution, args=(strategy_id,))
        thread.daemon = True
        thread.start()
    
    def _monitor_execution(self, strategy_id):
        """
        Monitor a strategy execution for take profit and stop loss
        
        Args:
            strategy_id (str): Strategy ID to monitor
        """
        try:
            if strategy_id not in self.active_executions:
                self.logger.error(f"Strategy {strategy_id} not found in active executions")
                return
            
            execution = self.active_executions[strategy_id]
            strategy = execution['strategy']
            symbol = strategy['symbol']
            exchange = strategy['exchange']
            entry_price = execution['entry_price']
            quantity = execution['quantity']
            action = execution['action']
            take_profit = strategy['take_profit'] / 100  # Convert percentage to decimal
            stop_loss = strategy['stop_loss'] / 100  # Convert percentage to decimal
            
            self.logger.info(f"Started monitoring {strategy_id} with entry price {entry_price}")
            
            # Calculate target and stop prices
            if action == 'BUY':
                take_profit_price = entry_price * (1 + take_profit)
                stop_loss_price = entry_price * (1 - stop_loss)
            else:  # SELL
                take_profit_price = entry_price * (1 - take_profit)
                stop_loss_price = entry_price * (1 + stop_loss)
            
            # Monitor price until target or stop is hit
            monitoring = True
            check_interval = 5  # seconds
            
            while monitoring and execution['status'] == 'open':
                try:
                    # Get current price
                    price_data = self.api.get_ltp(exchange, symbol)
                    
                    if not price_data:
                        self.logger.warning(f"Failed to get price for {symbol}, retrying...")
                        time.sleep(check_interval)
                        continue
                    
                    current_price = price_data['ltp']
                    
                    # Check take profit and stop loss conditions
                    if action == 'BUY':
                        if current_price >= take_profit_price:
                            # Take profit hit, place sell order
                            self.logger.info(f"Take profit hit for {strategy_id} at {current_price}")
                            self._execute_exit(strategy_id, 'SELL', quantity, 'take_profit')
                            monitoring = False
                            
                        elif current_price <= stop_loss_price:
                            # Stop loss hit, place sell order
                            self.logger.info(f"Stop loss hit for {strategy_id} at {current_price}")
                            self._execute_exit(strategy_id, 'SELL', quantity, 'stop_loss')
                            monitoring = False
                    
                    else:  # SELL
                        if current_price <= take_profit_price:
                            # Take profit hit, place buy order
                            self.logger.info(f"Take profit hit for {strategy_id} at {current_price}")
                            self._execute_exit(strategy_id, 'BUY', quantity, 'take_profit')
                            monitoring = False
                            
                        elif current_price >= stop_loss_price:
                            # Stop loss hit, place buy order
                            self.logger.info(f"Stop loss hit for {strategy_id} at {current_price}")
                            self._execute_exit(strategy_id, 'BUY', quantity, 'stop_loss')
                            monitoring = False
                
                except Exception as e:
                    self.logger.error(f"Error during monitoring: {str(e)}")
                
                # Sleep before next check
                time.sleep(check_interval)
        
        except Exception as e:
            self.logger.error(f"Error in monitoring thread for {strategy_id}: {str(e)}")
    
    def _execute_exit(self, strategy_id, action, quantity, reason):
        """
        Execute an exit order
        
        Args:
            strategy_id (str): Strategy ID
            action (str): BUY or SELL
            quantity (int): Order quantity
            reason (str): Exit reason (take_profit, stop_loss)
        """
        try:
            if strategy_id not in self.active_executions:
                self.logger.error(f"Strategy {strategy_id} not found in active executions")
                return
            
            execution = self.active_executions[strategy_id]
            strategy = execution['strategy']
            symbol = strategy['symbol']
            exchange = strategy['exchange']
            
            # Place exit order
            order_response = self.place_order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=action,
                quantity=quantity,
                order_type="MARKET",
                product_type="INTRADAY"
            )
            
            if order_response['status'] == 'success':
                self.logger.info(f"Exit order placed for {strategy_id} due to {reason}")
                
                # Get current price for P&L calculation
                price_data = self.api.get_ltp(exchange, symbol)
                exit_price = price_data['ltp'] if price_data else 0
                
                # Update execution status
                execution['status'] = 'closed'
                execution['exit_price'] = exit_price
                execution['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                execution['exit_reason'] = reason
                
                # Calculate P&L
                if execution['action'] == 'BUY':
                    pnl = (exit_price - execution['entry_price']) * execution['quantity']
                else:  # SELL
                    pnl = (execution['entry_price'] - exit_price) * execution['quantity']
                
                execution['pnl'] = pnl
                
                # Update strategy P&L
                strategy['pnl'] = pnl
            else:
                self.logger.error(f"Failed to place exit order: {order_response['message']}")
        
        except Exception as e:
            self.logger.error(f"Error executing exit order: {str(e)}")
    
    def get_execution_status(self, strategy_id):
        """
        Get status of a strategy execution
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            dict: Execution status or None if not found
        """
        return self.active_executions.get(strategy_id)
    
    def get_all_executions(self):
        """
        Get all strategy executions
        
        Returns:
            dict: All executions
        """
        return self.active_executions
    
    def cancel_execution(self, strategy_id):
        """
        Cancel a strategy execution
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        try:
            if strategy_id not in self.active_executions:
                self.logger.error(f"Strategy {strategy_id} not found in active executions")
                return False
            
            execution = self.active_executions[strategy_id]
            
            if execution['status'] != 'open':
                self.logger.warning(f"Strategy {strategy_id} is already closed")
                return False
            
            # Get order details
            order_id = execution['order_id']
            
            # Cancel the order
            cancelled = self.api.cancel_order(order_id)
            
            if cancelled:
                self.logger.info(f"Execution for strategy {strategy_id} cancelled")
                execution['status'] = 'cancelled'
                return True
            else:
                self.logger.error(f"Failed to cancel execution for strategy {strategy_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error cancelling execution: {str(e)}")
            return False
