import logging
import time
import random
import threading
import uuid
from datetime import datetime, timedelta
import pyotp

from utils.ai_strategy_generator import AIStrategyGenerator
from utils.trade_executor import TradeExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoTrader:
    """
    Class for automated trading based on AI-generated strategies.
    This class automatically logs in, analyzes market conditions, generates appropriate strategies,
    and executes trades based on those strategies.
    """
    def __init__(self, api=None, data_manager=None):
        """
        Initialize the auto trader
        
        Args:
            api: API client for executing trades
            data_manager: DataManager instance for fetching and storing data
        """
        self.api = api
        self.data_manager = data_manager
        
        # Create these components lazily when needed, not during initialization
        self._ai_strategy_generator = None
        self._trade_executor = None
        
        # Flags to track state
        self.is_initialized = self.api is not None and self.data_manager is not None
        self.auto_trader_thread = None
        self.stop_event = threading.Event()
        
        # Trading configuration
        self.trading_interval = 15  # Minutes between trading cycles
        self.watched_symbols = ["INFY", "TCS", "HDFCBANK", "RELIANCE", "SBIN", "NIFTY"]  # Default symbols to watch
        self.trading_mode = "fully_automated"  # Default mode: "fully_automated" or "ai_strategy_only"
        
        # Advanced automation settings with default values
        self.auto_capital_allocation = True  # Auto adjust position size based on volatility and signal strength
        self.auto_profit_booking = True      # Auto book profits when take profit or stop loss is hit
        self.auto_hedging = False            # Auto hedge positions based on market conditions (advanced feature)
        self.auto_strategy_refresh = True    # Auto refresh strategies based on changing market conditions
        
        # Strategy generation settings
        self.strategy_generation_frequency = "Medium (3-5 per day)"  # How often to generate new strategies
        self.model_type = "Auto-Select Best Model"  # Which AI model type to use
        self.max_risk_per_trade = 5.0        # Maximum risk per trade as percentage of capital 
        
        # Track auto-generated strategies for cleanup
        self.auto_generated_strategies = []
        
        # Auto login thread to handle session management
        self.auto_login_thread = None
        self.auto_login_stop_event = threading.Event()
        
        # Session management
        self.last_login_time = None
        self.login_interval = 60  # minutes (session usually expires within 2 hours)
        
        # Symbol management
        self.symbol_weighting_strategy = "Equal Weight"  # How to prioritize symbols for analysis
        self.max_concurrent_symbols = 5  # Maximum symbols to analyze at once
        self.symbol_rotation_enabled = True  # Whether to automatically rotate through symbols
        self.symbol_rotation_interval = 15  # Minutes between symbol rotations
        
        # Activity tracking
        self.activity_logs = []  # Recent activity logs
        self.max_logs = 100  # Maximum number of logs to keep
        self.trading_start_time = None  # When trading was started
        self.today_generated_strategies = []  # Strategies generated today
        self.today_trades = []  # Trades executed today
        self.daily_metrics = {}  # Metrics for today's trading
        
        # Additional components
        self.scheduler = None  # Trading scheduler
        self.notification_manager = None  # Notification manager
        self.position_manager = None  # Position manager
        self.trading_journal = None  # Trading journal
        self.connection_monitor = None  # Connection monitor
        
        # Load any saved settings
        self._load_settings()
        
    def start_auto_trading(self, symbols=None, trading_interval=None, mode="fully_automated"):
        """
        Start the auto trader thread
        
        Args:
            symbols (list, optional): List of symbols to watch and trade
            trading_interval (int, optional): Interval between trading cycles in minutes
            mode (str): Trading mode - "fully_automated" or "ai_strategy_only"
            
        Returns:
            bool: Success status
        """
        self.trading_mode = mode  # Store the trading mode
        if not self.api or not self.data_manager:
            logger.error("API client or data manager not available")
            return False
            
        if self.auto_trader_thread and self.auto_trader_thread.is_alive():
            logger.warning("Auto trader is already running")
            return False
            
        # Update configuration if provided
        if symbols:
            self.watched_symbols = symbols
            
        if trading_interval:
            self.trading_interval = trading_interval
            
        # Clear stop event
        self.stop_event.clear()
        
        # Start auto login thread
        self.auto_login_thread = threading.Thread(target=self._auto_login_loop, daemon=True)
        self.auto_login_thread.start()
        
        # Start auto trader thread
        self.auto_trader_thread = threading.Thread(target=self._auto_trader_loop, daemon=True)
        self.auto_trader_thread.start()
        
        logger.info(f"Started auto trader for symbols: {', '.join(self.watched_symbols)}")
        return True
        
    def stop_auto_trading(self):
        """
        Stop the auto trader thread
        
        Returns:
            bool: Success status
        """
        if not self.auto_trader_thread or not self.auto_trader_thread.is_alive():
            logger.warning("Auto trader is not running")
            return False
            
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.auto_trader_thread:
            self.auto_trader_thread.join(timeout=10)
            
        if self.auto_login_thread:
            self.auto_login_thread.join(timeout=10)
            
        logger.info("Stopped auto trader")
        return True
        
    def is_auto_trading(self):
        """
        Check if auto trader is running
        
        Returns:
            bool: True if running, False otherwise
        """
        return self.auto_trader_thread is not None and self.auto_trader_thread.is_alive()
        
    def _auto_login_loop(self):
        """
        Automatic login loop to ensure API session stays active
        """
        logger.info("Starting auto login loop")
        
        while not self.stop_event.is_set():
            try:
                # Make sure API is initialized
                if self.api is None:
                    logger.error("API is not initialized, cannot perform auto login")
                    time.sleep(60)  # Check again after a minute
                    continue
                
                # Check if login is needed
                try:
                    is_authenticated = self.api.is_authenticated()
                except (AttributeError, Exception) as e:
                    logger.error(f"Error checking authentication status: {str(e)}")
                    is_authenticated = False
                
                needs_login = not is_authenticated
                
                # Also check time-based expiration
                if self.last_login_time and ((datetime.now() - self.last_login_time).total_seconds() / 60 >= self.login_interval):
                    needs_login = True
                
                if needs_login:
                    credentials = self._get_stored_credentials()
                    if not credentials:
                        logger.error("No credentials available for automatic login")
                        time.sleep(60)  # Check again after a minute
                        continue
                        
                    # Attempt login
                    logger.info("Attempting automatic login...")
                    login_successful = False
                    
                    # Get TOTP code if secret is available
                    totp_code = None
                    if credentials.get('totp_secret'):
                        try:
                            totp = pyotp.TOTP(credentials.get('totp_secret'))
                            totp_code = totp.now()
                            logger.info(f"Generated TOTP code: {totp_code}")
                        except Exception as e:
                            logger.error(f"Error generating TOTP code: {str(e)}")
                            
                    # Try to login
                    try:
                        login_successful = self.api.login(
                            api_key=credentials.get('api_key'),
                            client_id=credentials.get('username'),
                            client_password=credentials.get('password'),
                            totp_key=totp_code
                        )
                        
                        if login_successful:
                            self.last_login_time = datetime.now()
                            logger.info("Automatic login successful")
                        else:
                            logger.error("Automatic login failed")
                    except Exception as e:
                        logger.error(f"Error during automatic login: {str(e)}")
                
                # Check every 5 minutes for login status
                for _ in range(5):
                    if self.stop_event.is_set():
                        break
                    time.sleep(60)
                    
            except Exception as e:
                logger.error(f"Error in auto login loop: {str(e)}")
                time.sleep(60)  # Sleep on error
                
    def _auto_trader_loop(self):
        """
        Main loop for auto trading
        """
        logger.info("Starting auto trader loop")
        
        while not self.stop_event.is_set():
            try:
                # Check if API is initialized
                if self.api is None:
                    logger.error("API is not initialized, cannot perform auto trading")
                    time.sleep(60)  # Check again after a minute
                    continue
                
                # Check if market is open and API is authenticated
                market_open = self._is_market_open()
                
                # Safely check authentication
                try:
                    is_authenticated = self.api.is_authenticated()
                except (AttributeError, Exception) as e:
                    logger.error(f"Error checking authentication status: {str(e)}")
                    is_authenticated = False
                
                if not market_open or not is_authenticated:
                    logger.info("Market is closed or not authenticated. Waiting...")
                    time.sleep(300)  # Check again after 5 minutes
                    continue
                
                # Check profit targets for existing trades
                self._check_profit_targets()
                
                # Market analysis and strategy generation
                logger.info("Starting market analysis and strategy generation...")
                self._analyze_market_and_generate_strategies()
                
                # Execute trades for active strategies
                logger.info("Executing trades for active strategies...")
                self._execute_active_strategies()
                
                # Wait for next trading interval
                logger.info(f"Waiting for next trading cycle ({self.trading_interval} minutes)...")
                for _ in range(self.trading_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(60)
                    
            except Exception as e:
                logger.error(f"Error in auto trader loop: {str(e)}")
                time.sleep(300)  # Sleep on error
                
    def _analyze_market_and_generate_strategies(self):
        """
        Analyze market conditions and generate appropriate strategies
        """
        try:
            # Check if data_manager is available
            if self.data_manager is None:
                logger.error("Cannot analyze market: data_manager is None")
                return
                
            # Clean up old auto-generated strategies that are no longer active
            if self.auto_strategy_refresh:
                self._clean_old_strategies()
            
            # Get overall market sentiment
            market_sentiment = self._get_market_sentiment()
            logger.info(f"Overall market sentiment: {market_sentiment['sentiment']} (Score: {market_sentiment['score']:.2f})")
            
            # Determine which symbols to analyze based on market conditions
            if market_sentiment['sentiment'] == 'Bullish':
                risk_level = "Medium"
            elif market_sentiment['sentiment'] == 'Bearish':
                risk_level = "Medium"
            else:
                risk_level = "Low"
                
            # Get the AI strategy generator
            strategy_generator = self.get_ai_strategy_generator()
            if strategy_generator is None:
                logger.error("Cannot generate strategies: AI strategy generator is None")
                return
            
            # Determine how many strategies to generate based on frequency setting
            max_strategies_per_run = 1  # Default
            if self.strategy_generation_frequency == "Low (1-2 per day)":
                max_strategies_per_run = 1
            elif self.strategy_generation_frequency == "Medium (3-5 per day)":
                max_strategies_per_run = 2
            elif self.strategy_generation_frequency == "High (6-10 per day)":
                max_strategies_per_run = 4
            elif self.strategy_generation_frequency == "Very High (10+ per day)":
                max_strategies_per_run = 6
                
            # Count how many strategies we've generated this run
            strategies_generated = 0
            
            # Set the model type based on user preference
            selected_model_type = None
            if self.model_type != "Auto-Select Best Model":
                if "LSTM" in self.model_type:
                    selected_model_type = "deep_learning"
                elif "GAN" in self.model_type:
                    selected_model_type = "gan"
                elif "Ensemble" in self.model_type:
                    selected_model_type = "ensemble"
                elif "Reinforcement" in self.model_type:
                    selected_model_type = "reinforcement_learning"
            
            # For each symbol, generate a strategy if conditions are favorable
            for symbol in self.watched_symbols:
                # Stop if we've reached the maximum strategies for this run
                if strategies_generated >= max_strategies_per_run:
                    logger.info(f"Reached maximum of {max_strategies_per_run} strategies for this run")
                    break
                    
                try:
                    # Check if we already have an active strategy for this symbol
                    if self._has_active_strategy_for_symbol(symbol):
                        logger.info(f"Skipping {symbol} - already has an active strategy")
                        continue
                        
                    # Get symbol data and analyze
                    symbol_data = self._analyze_symbol(symbol)
                    if not symbol_data:
                        logger.warning(f"Could not get data for {symbol}")
                        continue
                        
                    # Determine if we should generate a strategy for this symbol
                    # Lower the threshold for higher frequency settings
                    min_strength = 0.6
                    if self.strategy_generation_frequency == "High (6-10 per day)":
                        min_strength = 0.5
                    elif self.strategy_generation_frequency == "Very High (10+ per day)":
                        min_strength = 0.4
                        
                    if symbol_data['strength'] < min_strength:
                        logger.info(f"Skipping {symbol} - signal strength too low ({symbol_data['strength']:.2f})")
                        continue
                        
                    # Determine strategy type based on model selection or auto-selection
                    if selected_model_type:
                        strategy_type = selected_model_type
                    else:
                        # Auto-select based on market conditions and symbol characteristics
                        if symbol_data['volatility'] > 0.7:
                            strategy_type = "gan"  # More volatile - use GAN model
                        elif market_sentiment['score'] > 0.7 or market_sentiment['score'] < 0.3:
                            strategy_type = "deep_learning"  # Strong market trend - use LSTM
                        else:
                            strategy_type = "ensemble"  # Uncertain - use ensemble approach
                        
                    # Generate the strategy
                    logger.info(f"Generating {strategy_type} strategy for {symbol} with {risk_level} risk...")
                    
                    try:
                        strategy = strategy_generator.generate_strategy(
                            symbol=symbol,
                            exchange="NSE",  # Default to NSE for Indian markets
                            strategy_type=strategy_type,
                            risk_level=risk_level,
                            timeframe="1 day",
                            days_back=60
                        )
                    except Exception as e:
                        logger.error(f"Error generating strategy for {symbol}: {str(e)}")
                        continue
                    
                    if strategy:
                        # Add a unique ID for this strategy
                        strategy['id'] = str(uuid.uuid4())
                        strategy['auto_generated'] = True
                        strategy['generation_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        strategy['market_sentiment'] = market_sentiment['sentiment']
                        strategy['status'] = 'Active'
                        
                        # Add automation settings to the strategy
                        strategy['auto_profit_booking'] = self.auto_profit_booking
                        strategy['auto_hedging'] = self.auto_hedging
                        strategy['auto_capital_allocation'] = self.auto_capital_allocation
                        strategy['max_risk_per_trade'] = self.max_risk_per_trade
                        
                        # Save the strategy
                        try:
                            if self.data_manager.save_strategy(strategy):
                                logger.info(f"Auto-generated strategy saved for {symbol}")
                                self.auto_generated_strategies.append(strategy['id'])
                                strategies_generated += 1
                            else:
                                logger.error(f"Failed to save auto-generated strategy for {symbol}")
                        except Exception as e:
                            logger.error(f"Error saving strategy for {symbol}: {str(e)}")
                    else:
                        logger.warning(f"Failed to generate strategy for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during market analysis and strategy generation: {str(e)}")
            
    def _execute_active_strategies(self):
        """
        Execute trades for active strategies based on the trading mode
        """
        try:
            # Check if data_manager is available
            if self.data_manager is None:
                logger.error("Cannot execute strategies: data_manager is None")
                return
                
            # Check trading mode - if it's "ai_strategy_only", we don't execute trades
            if hasattr(self, 'trading_mode') and self.trading_mode == "ai_strategy_only":
                logger.info("Running in AI Strategy Only mode - generating strategies but not executing trades")
                return
                
            # Load all active strategies
            try:
                all_strategies = self.data_manager.load_all_strategies()
            except Exception as e:
                logger.error(f"Error loading strategies: {str(e)}")
                return
                
            if not all_strategies:
                logger.info("No strategies loaded")
                return
                
            active_strategies = [s for s in all_strategies if s.get('status') == 'Active']
            
            if not active_strategies:
                logger.info("No active strategies found")
                return
                
            logger.info(f"Found {len(active_strategies)} active strategies")
            
            # Get the trade executor
            trade_executor = self.get_trade_executor()
            if trade_executor is None:
                logger.error("Cannot execute strategies: Trade executor is None")
                return
            
            # Execute each strategy
            for strategy in active_strategies:
                try:
                    strategy_id = strategy.get('id', strategy.get('name', ''))
                    symbol = strategy.get('symbol', '')
                    
                    # Check if we should execute this strategy now
                    if not self._should_execute_strategy(strategy):
                        logger.info(f"Skipping execution of strategy {strategy_id} for {symbol} - conditions not met")
                        continue
                        
                    # Determine appropriate position size based on risk level
                    quantity = self._calculate_position_size(strategy)
                    
                    # Get recommendation
                    recommendation = strategy.get('recommendations', [])[-1] if strategy.get('recommendations') else None
                    if not recommendation:
                        logger.warning(f"Strategy {strategy_id} has no recommendations")
                        continue
                        
                    action = recommendation.get('action')
                    if not action or action == 'HOLD':
                        logger.info(f"Strategy {strategy_id} recommends HOLD - no action needed")
                        continue
                        
                    # Execute the strategy (fully automated mode)
                    logger.info(f"Executing strategy {strategy_id} for {symbol} with action {action}...")
                    
                    try:
                        execution_result = trade_executor.execute_strategy(
                            strategy=strategy,
                            quantity=quantity,
                            capital=10000 if strategy.get('risk_level') == 'Low' else \
                                   25000 if strategy.get('risk_level') == 'Medium' else 50000
                        )
                    except Exception as e:
                        logger.error(f"Error during strategy execution: {str(e)}")
                        continue
                    
                    if not execution_result:
                        logger.error(f"Strategy {strategy_id} execution returned no result")
                        continue
                    
                    if execution_result.get('status') == 'success':
                        logger.info(f"Strategy {strategy_id} execution successful: {execution_result.get('message')}")
                        
                        # Update strategy in database
                        strategy['last_execution'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        strategy['last_execution_result'] = execution_result
                        
                        try:
                            self.data_manager.update_strategy(strategy)
                        except Exception as e:
                            logger.error(f"Error updating strategy after execution: {str(e)}")
                        
                        # Check if we need to monitor for profit booking
                        if execution_result.get('order_status') == 'COMPLETE':
                            self._setup_profit_monitoring(strategy, execution_result)
                    else:
                        logger.error(f"Strategy {strategy_id} execution failed: {execution_result.get('message')}")
                        
                except Exception as e:
                    logger.error(f"Error executing strategy: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in strategy execution: {str(e)}")
            
    def _check_profit_targets(self):
        """
        Check and execute profit booking for trades that have reached their targets
        """
        try:
            # Check if data_manager is available
            if self.data_manager is None:
                logger.error("Cannot check profit targets: data_manager is None")
                return
                
            # Get the trade executor
            trade_executor = self.get_trade_executor()
            if trade_executor is None:
                logger.error("Cannot check profit targets: Trade executor is None")
                return
                
            # Load all active strategies
            try:
                all_strategies = self.data_manager.load_all_strategies()
            except Exception as e:
                logger.error(f"Error loading strategies for profit check: {str(e)}")
                return
                
            if not all_strategies:
                return
                
            # Check each strategy for profit targets
            for strategy in all_strategies:
                # Skip strategies without profit monitoring
                profit_monitoring = strategy.get('profit_monitoring')
                if not profit_monitoring:
                    continue
                    
                # Check if auto profit booking is enabled for this strategy
                auto_profit_booking = profit_monitoring.get('auto_profit_booking', False)
                if not auto_profit_booking:
                    continue
                
                symbol = strategy.get('symbol', '')
                if not symbol:
                    continue
                
                # Get current price for the symbol
                try:
                    price_data = self.api.get_ltp("NSE", symbol)
                    if not price_data:
                        logger.warning(f"Could not get price data for {symbol}")
                        continue
                        
                    current_price = price_data.get('ltp', 0)
                    if current_price <= 0:
                        continue
                        
                    trade_type = profit_monitoring.get('trade_type', '')
                    trade_price = profit_monitoring.get('trade_price', 0)
                    take_profit_price = profit_monitoring.get('take_profit_price', 0)
                    stop_loss_price = profit_monitoring.get('stop_loss_price', 0)
                    quantity = profit_monitoring.get('quantity', 0)
                    
                    if not trade_type or trade_price <= 0 or not quantity:
                        continue
                    
                    # Calculate current profit/loss percentage
                    if trade_type == 'BUY':
                        pnl_percent = ((current_price - trade_price) / trade_price) * 100
                    else:  # SELL
                        pnl_percent = ((trade_price - current_price) / trade_price) * 100
                    
                    # Check if take profit or stop loss has been hit
                    exit_reason = None
                    exit_type = None
                    
                    # For BUY trades
                    if trade_type == 'BUY':
                        if current_price >= take_profit_price:
                            exit_reason = f"Take profit target of {profit_monitoring.get('take_profit_percent', 0):.2f}% reached"
                            exit_type = "SELL"
                        elif current_price <= stop_loss_price:
                            exit_reason = f"Stop loss of {profit_monitoring.get('stop_loss_percent', 0):.2f}% triggered"
                            exit_type = "SELL"
                    # For SELL trades
                    else:
                        if current_price <= take_profit_price:
                            exit_reason = f"Take profit target of {profit_monitoring.get('take_profit_percent', 0):.2f}% reached"
                            exit_type = "BUY"
                        elif current_price >= stop_loss_price:
                            exit_reason = f"Stop loss of {profit_monitoring.get('stop_loss_percent', 0):.2f}% triggered"
                            exit_type = "BUY"
                    
                    # Execute exit trade if target reached
                    if exit_reason and exit_type:
                        logger.info(f"Executing profit booking for {symbol}: {exit_reason} (current P&L: {pnl_percent:.2f}%)")
                        
                        # Prepare exit order
                        exit_params = {
                            'symbol': symbol,
                            'exchange': "NSE",
                            'quantity': quantity,
                            'trade_type': exit_type,
                            'order_type': 'MARKET',
                            'product_type': 'DELIVERY'  # Adjust as needed
                        }
                        
                        try:
                            # Execute exit order
                            exit_result = trade_executor.execute_trade(exit_params)
                            
                            if exit_result.get('status') == 'success':
                                logger.info(f"Successfully executed profit booking for {symbol}: {exit_result.get('message')}")
                                
                                # Update strategy status and remove profit monitoring
                                strategy['status'] = 'Closed'
                                strategy['exit_reason'] = exit_reason
                                strategy['exit_price'] = current_price
                                strategy['profit_loss'] = pnl_percent
                                strategy['profit_monitoring'] = None
                                
                                # Update the strategy in the database
                                try:
                                    self.data_manager.update_strategy(strategy)
                                    logger.info(f"Strategy updated after profit booking: {strategy.get('id')}")
                                except Exception as e:
                                    logger.error(f"Error updating strategy after profit booking: {str(e)}")
                            else:
                                logger.error(f"Profit booking execution failed for {symbol}: {exit_result.get('message')}")
                                
                        except Exception as e:
                            logger.error(f"Error executing profit booking for {symbol}: {str(e)}")
                    
                    # Update last check time
                    profit_monitoring['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    profit_monitoring['current_pnl'] = pnl_percent
                    
                    # Update the strategy in the database
                    try:
                        self.data_manager.update_strategy(strategy)
                    except Exception as e:
                        logger.error(f"Error updating strategy after profit check: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error checking profit targets for {symbol}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in profit target checking: {str(e)}")
            
    def _setup_profit_monitoring(self, strategy, execution_result):
        """
        Set up monitoring for profit booking
        
        Args:
            strategy (dict): Strategy that was executed
            execution_result (dict): Result of the execution
        """
        try:
            # Check if data_manager is available
            if self.data_manager is None:
                logger.error("Cannot set up profit monitoring: data_manager is None")
                return
            
            # Check if auto profit booking is enabled for this strategy
            auto_profit_booking = strategy.get('auto_profit_booking', self.auto_profit_booking)
            if not auto_profit_booking:
                logger.info(f"Auto profit booking is disabled for strategy {strategy.get('id', 'Unknown')}")
                return
                
            # Extract relevant information
            symbol = strategy.get('symbol', '')
            take_profit = strategy.get('take_profit', 5.0)  # Default to 5%
            stop_loss = strategy.get('stop_loss', 3.0)      # Default to 3%
            order_id = execution_result.get('order_id')
            trade_price = execution_result.get('trade_price', 0)
            trade_type = execution_result.get('trade_type', '')  # BUY or SELL
            
            if not symbol or not order_id or trade_price <= 0:
                logger.warning(f"Insufficient data for profit monitoring: {symbol}, {order_id}, {trade_price}")
                return
                
            logger.info(f"Setting up profit monitoring for {symbol} with {take_profit}% take profit and {stop_loss}% stop loss")
            
            # Get symbol data for dynamic adjustments
            symbol_data = self._analyze_symbol(symbol)
            volatility_adjusted_tp = take_profit
            volatility_adjusted_sl = stop_loss
            
            # Adjust take profit and stop loss based on volatility if we have data
            if symbol_data:
                volatility = symbol_data.get('volatility', 0.5)
                # For higher volatility, widen the take profit and stop loss bands
                if volatility > 0.6:
                    volatility_adjusted_tp = take_profit * (1 + (volatility - 0.5) * 2)  # Increase TP for higher volatility
                    volatility_adjusted_sl = stop_loss * (1 + (volatility - 0.5) * 1.5)  # Increase SL a bit less
                    logger.info(f"Adjusted take profit to {volatility_adjusted_tp:.2f}% and stop loss to {volatility_adjusted_sl:.2f}% based on volatility of {volatility:.2f}")
            
            # Calculate target prices for take profit and stop loss
            if trade_type == 'BUY':
                take_profit_price = trade_price * (1 + volatility_adjusted_tp / 100)
                stop_loss_price = trade_price * (1 - volatility_adjusted_sl / 100)
            else:  # SELL
                take_profit_price = trade_price * (1 - volatility_adjusted_tp / 100)
                stop_loss_price = trade_price * (1 + volatility_adjusted_sl / 100)
                
            # Store monitoring information in the strategy
            strategy['profit_monitoring'] = {
                'order_id': order_id,
                'trade_price': trade_price,
                'trade_type': trade_type,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_percent': volatility_adjusted_tp,
                'stop_loss_percent': volatility_adjusted_sl,
                'auto_profit_booking': auto_profit_booking,
                'quantity': execution_result.get('quantity', 0),
                'last_check': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update the strategy in the database
            try:
                self.data_manager.update_strategy(strategy)
            except Exception as e:
                logger.error(f"Error updating strategy with profit monitoring info: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error setting up profit monitoring: {str(e)}")
            
    def _load_settings(self):
        """
        Load settings from the configuration file
        """
        try:
            import os
            import json
            
            config_dir = "config"
            settings_file = os.path.join(config_dir, "auto_trading_settings.json")
            
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    
                # Update instance variables with loaded settings
                self.trading_interval = settings.get('trading_interval', self.trading_interval)
                self.watched_symbols = settings.get('watched_symbols', self.watched_symbols)
                
                # Advanced settings
                self.auto_capital_allocation = settings.get('auto_capital_allocation', self.auto_capital_allocation)
                self.auto_profit_booking = settings.get('auto_profit_booking', self.auto_profit_booking)
                self.auto_hedging = settings.get('auto_hedging', self.auto_hedging)
                self.auto_strategy_refresh = settings.get('auto_strategy_refresh', self.auto_strategy_refresh)
                self.strategy_generation_frequency = settings.get('strategy_generation_frequency', self.strategy_generation_frequency)
                self.model_type = settings.get('model_type', self.model_type)
                self.max_risk_per_trade = settings.get('max_risk_per_trade', self.max_risk_per_trade)
                
                logger.info("Auto trading settings loaded successfully")
        except Exception as e:
            logger.error(f"Error loading auto trading settings: {str(e)}")
    
    def _get_stored_credentials(self):
        """
        Get stored API credentials
        
        Returns:
            dict: API credentials or None if not available
        """
        try:
            # Use the connected scheduler instead of creating a new one
            if hasattr(self, 'scheduler') and self.scheduler:
                return self.scheduler.get_credentials()
            else:
                logger.error("No scheduler connected, cannot retrieve API credentials")
                return None
        except Exception as e:
            logger.error(f"Error retrieving API credentials: {str(e)}")
            return None
            
    def _is_market_open(self):
        """
        Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        # Get current day and time in IST
        now = datetime.now()
        current_day = now.strftime('%A').lower()
        current_time = now.strftime('%H:%M')
        
        # Indian markets are typically open Monday-Friday, 9:15 AM - 3:30 PM IST
        if current_day not in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            return False
            
        # Check if within trading hours
        return '09:15' <= current_time <= '15:30'
        
    def _get_market_sentiment(self):
        """
        Get overall market sentiment based on major indices
        
        Returns:
            dict: Market sentiment data
        """
        try:
            # Get NIFTY data as a proxy for market sentiment
            nifty_data = self._analyze_symbol('NIFTY')
            
            if not nifty_data:
                # Return neutral if can't get data
                return {
                    'sentiment': 'Neutral',
                    'score': 0.5,
                    'reasoning': "Could not retrieve market data"
                }
                
            # Determine sentiment based on signal direction and strength
            sentiment_score = nifty_data['direction'] * nifty_data['strength']
            
            if sentiment_score > 0.3:
                sentiment = 'Bullish'
            elif sentiment_score < -0.3:
                sentiment = 'Bearish'
            else:
                sentiment = 'Neutral'
                
            return {
                'sentiment': sentiment,
                'score': 0.5 + sentiment_score / 2,  # Convert to 0-1 scale
                'reasoning': nifty_data['analysis']
            }
            
        except Exception as e:
            logger.error(f"Error determining market sentiment: {str(e)}")
            return {
                'sentiment': 'Neutral',
                'score': 0.5,
                'reasoning': f"Error in analysis: {str(e)}"
            }
            
    def _analyze_symbol(self, symbol):
        """
        Analyze a specific symbol
        
        Args:
            symbol (str): Symbol to analyze
            
        Returns:
            dict: Analysis data or None if analysis failed
        """
        try:
            # Get historical data
            df = self.data_manager.get_historical_data(symbol, "NSE", "1 day", 60)
            
            if df is None or len(df) < 30:
                logger.warning(f"Insufficient historical data for {symbol}")
                return None
                
            # Calculate basic technical indicators
            # Moving Averages
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Get most recent data point
            latest = df.iloc[-1]
            
            # Determine trend direction
            trend_direction = 1 if latest['close'] > latest['SMA20'] else -1
            
            # Calculate RSI signal
            rsi_signal = 1 if latest['RSI'] < 30 else -1 if latest['RSI'] > 70 else 0
            
            # Calculate volatility (normalized)
            recent_volatility = df['close'].pct_change().rolling(window=20).std().iloc[-1]
            normalized_volatility = min(1.0, recent_volatility * 20)  # Scale for easier interpretation
            
            # Determine signal strength (0-1)
            signal_strength = abs(latest['close'] / latest['SMA50'] - 1) * 3  # Scale factor
            signal_strength = min(1.0, signal_strength)  # Cap at 1.0
            
            # Combine signals for overall direction (-1 to 1 scale)
            combined_direction = (trend_direction * 0.7 + rsi_signal * 0.3)
            
            analysis_text = f"{symbol} is "
            
            if combined_direction > 0.3:
                analysis_text += "showing bullish signals "
            elif combined_direction < -0.3:
                analysis_text += "showing bearish signals "
            else:
                analysis_text += "trading in a range "
                
            analysis_text += f"with {signal_strength:.0%} signal strength and {normalized_volatility:.0%} volatility."
            
            return {
                'symbol': symbol,
                'direction': combined_direction,
                'strength': signal_strength,
                'volatility': normalized_volatility,
                'analysis': analysis_text,
                'price': latest['close'],
                'rsi': latest['RSI']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
            return None
            
    def _has_active_strategy_for_symbol(self, symbol):
        """
        Check if we already have an active strategy for a given symbol
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            bool: True if active strategy exists, False otherwise
        """
        try:
            all_strategies = self.data_manager.load_all_strategies()
            for strategy in all_strategies:
                if strategy.get('symbol') == symbol and strategy.get('status') == 'Active':
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking for active strategy: {str(e)}")
            return False
            
    def _clean_old_strategies(self):
        """
        Clean up old auto-generated strategies that are no longer active or relevant
        """
        try:
            all_strategies = self.data_manager.load_all_strategies()
            
            for strategy in all_strategies:
                # Skip if not auto-generated
                if not strategy.get('auto_generated', False):
                    continue
                    
                # Check if strategy is expired (older than 3 days)
                generation_time = strategy.get('generation_time')
                if generation_time:
                    generation_dt = datetime.strptime(generation_time, '%Y-%m-%d %H:%M:%S')
                    if (datetime.now() - generation_dt).days >= 3:
                        logger.info(f"Deactivating expired auto-generated strategy for {strategy.get('symbol')}")
                        strategy['status'] = 'Inactive'
                        self.data_manager.update_strategy(strategy)
                        
                # Check if strategy has poor performance
                if strategy.get('pnl', 0) < -1000:
                    logger.info(f"Deactivating poorly performing strategy for {strategy.get('symbol')}")
                    strategy['status'] = 'Inactive'
                    self.data_manager.update_strategy(strategy)
                    
        except Exception as e:
            logger.error(f"Error cleaning old strategies: {str(e)}")
            
    def _should_execute_strategy(self, strategy):
        """
        Determine if a strategy should be executed now
        
        Args:
            strategy (dict): Strategy to check
            
        Returns:
            bool: True if strategy should be executed, False otherwise
        """
        try:
            # Check last execution time
            last_execution = strategy.get('last_execution')
            if last_execution:
                last_exec_dt = datetime.strptime(last_execution, '%Y-%m-%d %H:%M:%S')
                time_since_last = (datetime.now() - last_exec_dt).total_seconds() / 60
                
                # Don't execute more than once per hour
                if time_since_last < 60:
                    return False
                    
            # Check if market conditions match strategy conditions
            symbol = strategy.get('symbol')
            symbol_data = self._analyze_symbol(symbol)
            
            if not symbol_data:
                return False
                
            # Get latest recommendation
            recommendations = strategy.get('recommendations', [])
            if not recommendations:
                return False
                
            latest_rec = recommendations[-1]
            action = latest_rec.get('action')
            
            # Don't execute HOLD recommendations
            if action == 'HOLD':
                return False
                
            # Check auto settings in the strategy
            auto_strategy_refresh = strategy.get('auto_strategy_refresh', self.auto_strategy_refresh)
            
            # For BUY recommendation, check if current signal is bullish
            if action == 'BUY' and symbol_data['direction'] < 0:
                # Allow execution anyway if auto_strategy_refresh is disabled
                if auto_strategy_refresh:
                    logger.info(f"Skipping execution - BUY signal but bearish direction for {symbol}")
                    return False
                
            # For SELL recommendation, check if current signal is bearish
            if action == 'SELL' and symbol_data['direction'] > 0:
                # Allow execution anyway if auto_strategy_refresh is disabled
                if auto_strategy_refresh:
                    logger.info(f"Skipping execution - SELL signal but bullish direction for {symbol}")
                    return False
                
            # Check signal strength threshold - be more lenient when auto_strategy_refresh is off
            min_strength = 0.4 if auto_strategy_refresh else 0.3
            if symbol_data['strength'] < min_strength:
                logger.info(f"Skipping execution - signal strength too low ({symbol_data['strength']:.2f}) for {symbol}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking execution conditions: {str(e)}")
            return False
            
    def _calculate_position_size(self, strategy):
        """
        Calculate appropriate position size based on risk level and available capital
        
        Args:
            strategy (dict): Strategy to calculate position size for
            
        Returns:
            int: Position size (quantity)
        """
        try:
            risk_level = strategy.get('risk_level', 'Medium')
            symbol = strategy.get('symbol', '')
            
            # Get current price
            price_data = self.api.get_ltp("NSE", symbol)
            if not price_data:
                return 1  # Minimum quantity if price data not available
                
            current_price = price_data.get('ltp', 0)
            if current_price <= 0:
                return 1
            
            # Check if we should use auto capital allocation
            use_auto_allocation = strategy.get('auto_capital_allocation', self.auto_capital_allocation)
            max_risk_percent = strategy.get('max_risk_per_trade', self.max_risk_per_trade)
            
            # Fetch market data for volatility assessment
            symbol_data = self._analyze_symbol(symbol)
            volatility_factor = 1.0
            if symbol_data and use_auto_allocation:
                volatility = symbol_data.get('volatility', 0.5)
                signal_strength = symbol_data.get('strength', 0.5)
                
                # Adjust capital based on volatility (lower for higher volatility)
                volatility_factor = max(0.5, 1.0 - volatility)
                
                # Boost for strong signals
                if signal_strength > 0.7:
                    volatility_factor *= 1.2
            
            # Determine base capital based on risk level
            if risk_level == 'Low':
                base_capital = 10000
            elif risk_level == 'Medium':
                base_capital = 25000
            else:  # High
                base_capital = 50000
            
            # Apply auto-allocation adjustments if enabled
            if use_auto_allocation:
                # Adjust capital allocation based on volatility and other factors
                adjusted_capital = base_capital * volatility_factor
                
                # Apply risk management - limit to max_risk_percent of total capital
                # Assuming a total capital of 10x the base_capital for this risk level
                total_capital = base_capital * 10
                risk_limited_capital = total_capital * (max_risk_percent / 100)
                
                # Use the lower of the two capital amounts (risk-limited or volatility-adjusted)
                final_capital = min(adjusted_capital, risk_limited_capital)
                
                logger.info(f"Auto capital allocation for {symbol}: Base={base_capital}, "
                           f"Adjusted={adjusted_capital:.2f}, Risk-limited={risk_limited_capital:.2f}, "
                           f"Final={final_capital:.2f}")
            else:
                final_capital = base_capital
            
            # Calculate quantity
            quantity = int(final_capital / current_price)
            
            # Ensure minimum quantity
            return max(1, quantity)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1  # Return minimum quantity on error
            
    def get_ai_strategy_generator(self):
        """
        Get or create the AI strategy generator
        
        Returns:
            AIStrategyGenerator: AI strategy generator instance or None if data_manager is None
        """
        # Check if data_manager is available
        if self.data_manager is None:
            logger.error("Cannot create AIStrategyGenerator: data_manager is None")
            return None
            
        # Create the generator if it doesn't exist
        if self._ai_strategy_generator is None:
            try:
                from utils.ai_strategy_generator import AIStrategyGenerator
                self._ai_strategy_generator = AIStrategyGenerator(self.data_manager)
            except Exception as e:
                logger.error(f"Error creating AIStrategyGenerator: {str(e)}")
                return None
                
        return self._ai_strategy_generator
        
    def get_trade_executor(self):
        """
        Get or create the trade executor
        
        Returns:
            TradeExecutor: Trade executor instance or None if API is None
        """
        # Check if API is available
        if self.api is None:
            logger.error("Cannot create TradeExecutor: API is None")
            return None
            
        # Create the executor if it doesn't exist
        if self._trade_executor is None:
            try:
                from utils.trade_executor import TradeExecutor
                self._trade_executor = TradeExecutor(self.api)
            except Exception as e:
                logger.error(f"Error creating TradeExecutor: {str(e)}")
                return None
                
        return self._trade_executor
        
    def pre_market_preparation(self):
        """
        Prepare for the trading day before market open.
        This method is called by the scheduler during pre-market hours.
        """
        logger.info("Starting pre-market preparation")
        self._add_activity_log("Starting pre-market preparation")
        
        try:
            # Ensure we're logged in
            self._ensure_authenticated()
            
            # Fetch market data and prepare for the day
            self._add_activity_log("Fetching pre-market data and preparing analysis")
            
            # Prepare symbols to watch for the day
            self._prioritize_symbols()
            
            # Reset daily metrics
            self.today_generated_strategies = []
            self.today_trades = []
            self.daily_metrics = {
                "market_sentiment": "Neutral",
                "total_opportunities": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0
            }
            
            # Add pre-market updates to trading journal if available
            if hasattr(self, 'trading_journal') and self.trading_journal:
                self.trading_journal.add_market_analysis({
                    "type": "pre_market",
                    "sentiment": "Neutral",  # Default until analyzed
                    "notes": "Pre-market preparation complete"
                })
                
            # Send notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Pre-Market Preparation Complete",
                    message=f"The auto-trading system has completed pre-market preparations at {datetime.now().strftime('%H:%M:%S')}",
                    category="system_status",
                    importance=2
                )
                
            logger.info("Pre-market preparation completed successfully")
            self._add_activity_log("Pre-market preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during pre-market preparation: {str(e)}")
            self._add_activity_log(f"Error during pre-market preparation: {str(e)}")
            
            # Send error notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Pre-Market Preparation Error",
                    message=f"Error during pre-market preparation: {str(e)}",
                    category="system_status",
                    importance=4
                )
            
    def market_open_handler(self):
        """
        Handle market open event.
        This method is called by the scheduler when the market opens.
        """
        logger.info("Market open handler triggered")
        self._add_activity_log("Market has opened. Beginning trading operations.")
        
        try:
            # Ensure we're logged in
            self._ensure_authenticated()
            
            # Get initial market sentiment
            market_sentiment = self._get_market_sentiment()
            self.daily_metrics["market_sentiment"] = market_sentiment["sentiment"]
            
            # Log market open in trading journal if available
            if hasattr(self, 'trading_journal') and self.trading_journal:
                self.trading_journal.add_market_analysis({
                    "type": "market_open",
                    "sentiment": market_sentiment["sentiment"],
                    "score": market_sentiment["score"],
                    "notes": "Market open - starting trading operations"
                })
                
            # Send notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Market Open - Trading Begins",
                    message=f"The market is now open. Auto-trading system is active. Market sentiment: {market_sentiment['sentiment']}",
                    category="system_status",
                    importance=3
                )
                
            # Perform initial market analysis and strategy generation
            self._analyze_market_and_generate_strategies()
            
            logger.info("Market open handler completed successfully")
            self._add_activity_log("Market open handler completed successfully")
            
        except Exception as e:
            logger.error(f"Error during market open handling: {str(e)}")
            self._add_activity_log(f"Error during market open handling: {str(e)}")
            
            # Send error notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Market Open Handler Error",
                    message=f"Error during market open handling: {str(e)}",
                    category="system_status",
                    importance=4
                )
    
    def market_close_handler(self):
        """
        Handle market close event.
        This method is called by the scheduler when the market closes.
        """
        logger.info("Market close handler triggered")
        self._add_activity_log("Market has closed. Finalizing trading operations.")
        
        try:
            # Generate end-of-day summary
            summary = self._generate_trading_summary()
            
            # Close any open positions if appropriate
            if hasattr(self, 'position_manager') and self.position_manager and hasattr(self.position_manager, 'positions'):
                for position in self.position_manager.positions:
                    # Only close intraday positions automatically
                    if position.get('product_type') == 'INTRADAY':
                        self._add_activity_log(f"Closing intraday position for {position.get('symbol')}")
                        
                        # Get current price for closing
                        current_price = self._get_current_price(position.get('symbol'), position.get('exchange', 'NSE'))
                        
                        # Close position
                        self.position_manager.close_position(
                            position_id=position.get('id'),
                            exit_price=current_price,
                            exit_reason="Market Close - Auto Close"
                        )
            
            # Log market close in trading journal if available
            if hasattr(self, 'trading_journal') and self.trading_journal:
                self.trading_journal.add_market_analysis({
                    "type": "market_close",
                    "summary": summary,
                    "notes": "Market close - finalized trading operations"
                })
                
            # Send notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Market Close - Trading Summary",
                    message=f"The market has closed. Trading summary:\n\n"
                            f"Strategies Generated: {len(self.today_generated_strategies)}\n"
                            f"Trades Executed: {self.daily_metrics.get('total_trades', 0)}\n"
                            f"Total P&L: ₹{self.daily_metrics.get('total_pnl', 0):,.2f}\n"
                            f"Win Rate: {self._calculate_win_rate():.1f}%",
                    category="system_status",
                    importance=3
                )
                
            logger.info("Market close handler completed successfully")
            self._add_activity_log("Market close handler completed successfully")
            
        except Exception as e:
            logger.error(f"Error during market close handling: {str(e)}")
            self._add_activity_log(f"Error during market close handling: {str(e)}")
            
            # Send error notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Market Close Handler Error",
                    message=f"Error during market close handling: {str(e)}",
                    category="system_status",
                    importance=4
                )
    
    def trading_cycle_handler(self):
        """
        Handle regular trading cycle.
        This method is called by the scheduler at regular intervals during trading hours.
        """
        logger.info("Trading cycle handler triggered")
        self._add_activity_log("Starting regular trading cycle")
        
        try:
            # Ensure we're logged in
            self._ensure_authenticated()
            
            # Check profit targets for existing trades
            self._check_profit_targets()
            
            # Update market analysis
            if self.auto_strategy_refresh:
                self._add_activity_log("Refreshing market analysis")
                market_sentiment = self._get_market_sentiment()
                self.daily_metrics["market_sentiment"] = market_sentiment["sentiment"]
            
            # Analyze market and generate strategies
            self._add_activity_log("Analyzing market conditions and generating strategies")
            self._analyze_market_and_generate_strategies()
            
            # Execute trades if in fully automated mode
            if self.trading_mode == "fully_automated":
                self._add_activity_log("Executing trades for active strategies")
                self._execute_active_strategies()
            
            # Rotate symbols if enabled
            if hasattr(self, 'symbol_rotation_enabled') and self.symbol_rotation_enabled:
                self._rotate_symbols()
            
            logger.info("Trading cycle handler completed successfully")
            self._add_activity_log("Trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error during trading cycle: {str(e)}")
            self._add_activity_log(f"Error during trading cycle: {str(e)}")
            
            # Send error notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_notification(
                    subject="Trading Cycle Error",
                    message=f"Error during regular trading cycle: {str(e)}",
                    category="system_status",
                    importance=4
                )
    
    def _add_activity_log(self, message):
        """
        Add a message to the activity log
        
        Args:
            message (str): Log message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} - {message}"
        
        # Initialize activity logs if not existing
        if not hasattr(self, 'activity_logs'):
            self.activity_logs = []
            
        if not hasattr(self, 'max_logs'):
            self.max_logs = 100
        
        # Add to log
        self.activity_logs.append(log_entry)
        
        # Trim log if too large
        if len(self.activity_logs) > self.max_logs:
            self.activity_logs = self.activity_logs[-self.max_logs:]
    
    def _ensure_authenticated(self):
        """
        Ensure that the API is authenticated
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        if not self.api:
            logger.error("API client not available")
            return False
            
        try:
            is_authenticated = self.api.is_authenticated()
            
            if not is_authenticated:
                # Try to login
                credentials = self._get_stored_credentials()
                if not credentials:
                    logger.error("No credentials available for login")
                    return False
                    
                # Get TOTP code if secret is available
                totp_code = None
                if credentials.get('totp_secret'):
                    try:
                        totp = pyotp.TOTP(credentials.get('totp_secret'))
                        totp_code = totp.now()
                    except Exception as e:
                        logger.error(f"Error generating TOTP code: {str(e)}")
                        
                # Try to login
                login_successful = self.api.login(
                    api_key=credentials.get('api_key'),
                    client_id=credentials.get('username'),
                    client_password=credentials.get('password'),
                    totp_key=totp_code
                )
                
                if login_successful:
                    self.last_login_time = datetime.now()
                    logger.info("Login successful")
                    
                    # Send notification if available
                    if hasattr(self, 'notification_manager') and self.notification_manager:
                        self.notification_manager.send_authentication_notification(success=True)
                        
                    return True
                else:
                    logger.error("Login failed")
                    
                    # Send notification if available
                    if hasattr(self, 'notification_manager') and self.notification_manager:
                        self.notification_manager.send_authentication_notification(success=False, error_message="Login failed")
                        
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking authentication status: {str(e)}")
            
            # Send notification if available
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self.notification_manager.send_authentication_notification(success=False, error_message=str(e))
                
            return False
    
    def _prioritize_symbols(self):
        """
        Prioritize symbols for analysis based on the selected weighting strategy
        """
        if not self.watched_symbols:
            logger.warning("No symbols to prioritize")
            return
            
        # Check if we have the attribute
        if not hasattr(self, 'symbol_weighting_strategy'):
            self.symbol_weighting_strategy = "Equal Weight"
            
        logger.info(f"Prioritizing {len(self.watched_symbols)} symbols using strategy: {self.symbol_weighting_strategy}")
        
        # If using equal weight, nothing to do
        if self.symbol_weighting_strategy == "Equal Weight":
            return
            
        # Otherwise, prioritize based on strategy
        try:
            if self.symbol_weighting_strategy == "Volume Weighted":
                self._prioritize_by_volume()
            elif self.symbol_weighting_strategy == "Volatility Weighted":
                self._prioritize_by_volatility()
            elif self.symbol_weighting_strategy == "Smart (AI-based)":
                self._prioritize_by_ai()
        except Exception as e:
            logger.error(f"Error prioritizing symbols: {str(e)}")
    
    def _prioritize_by_volume(self):
        """
        Prioritize symbols by trading volume
        """
        logger.info("Prioritizing symbols by trading volume")
        
        # Get volume data for each symbol
        symbol_volumes = {}
        
        for symbol in self.watched_symbols:
            try:
                # Get historical data with volume
                hist_data = self._get_historical_data(symbol, "NSE", "1 day", 5)
                
                if hist_data is not None and "volume" in hist_data:
                    # Calculate average volume
                    avg_volume = hist_data["volume"].mean()
                    symbol_volumes[symbol] = avg_volume
            except Exception as e:
                logger.error(f"Error getting volume data for {symbol}: {str(e)}")
        
        # Sort symbols by volume (descending)
        if symbol_volumes:
            sorted_symbols = sorted(symbol_volumes.keys(), key=lambda s: symbol_volumes[s], reverse=True)
            self.watched_symbols = sorted_symbols
    
    def _prioritize_by_volatility(self):
        """
        Prioritize symbols by volatility
        """
        logger.info("Prioritizing symbols by volatility")
        
        # Get volatility data for each symbol
        symbol_volatility = {}
        
        for symbol in self.watched_symbols:
            try:
                # Get historical data
                hist_data = self._get_historical_data(symbol, "NSE", "1 day", 10)
                
                if hist_data is not None and "close" in hist_data:
                    # Calculate volatility (standard deviation of returns)
                    returns = hist_data["close"].pct_change().dropna()
                    volatility = returns.std()
                    symbol_volatility[symbol] = volatility
            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
        
        # Sort symbols by volatility (descending)
        if symbol_volatility:
            sorted_symbols = sorted(symbol_volatility.keys(), key=lambda s: symbol_volatility[s], reverse=True)
            self.watched_symbols = sorted_symbols
    
    def _prioritize_by_ai(self):
        """
        Prioritize symbols using AI-based approach
        """
        logger.info("Prioritizing symbols using AI-based approach")
        
        # This would typically use a more sophisticated AI model
        # For now, we'll use a simple combination of volume and price movement
        symbol_scores = {}
        
        for symbol in self.watched_symbols:
            try:
                # Get historical data
                hist_data = self._get_historical_data(symbol, "NSE", "1 day", 5)
                
                if hist_data is not None and "close" in hist_data and "volume" in hist_data:
                    # Calculate recent price change
                    price_change = (hist_data["close"].iloc[-1] / hist_data["close"].iloc[0] - 1) * 100
                    
                    # Calculate average volume
                    avg_volume = hist_data["volume"].mean()
                    
                    # Calculate volatility
                    returns = hist_data["close"].pct_change().dropna()
                    volatility = returns.std() * 100
                    
                    # Calculate score (combination of factors)
                    score = (abs(price_change) * 0.4) + (volatility * 0.3) + (avg_volume * 0.3)
                    symbol_scores[symbol] = score
            except Exception as e:
                logger.error(f"Error calculating AI score for {symbol}: {str(e)}")
        
        # Sort symbols by score (descending)
        if symbol_scores:
            sorted_symbols = sorted(symbol_scores.keys(), key=lambda s: symbol_scores[s], reverse=True)
            self.watched_symbols = sorted_symbols
    
    def _rotate_symbols(self):
        """
        Rotate through watched symbols
        """
        if not hasattr(self, 'max_concurrent_symbols'):
            self.max_concurrent_symbols = 5
            
        if not hasattr(self, 'symbol_rotation_enabled') or not self.symbol_rotation_enabled or len(self.watched_symbols) <= self.max_concurrent_symbols:
            return
            
        logger.info("Rotating symbols for analysis")
        
        # Move the first symbol to the end of the list
        self.watched_symbols = self.watched_symbols[1:] + [self.watched_symbols[0]]
    
    def _calculate_win_rate(self):
        """
        Calculate win rate from daily metrics
        
        Returns:
            float: Win rate as percentage
        """
        if not hasattr(self, 'daily_metrics'):
            self.daily_metrics = {
                "total_trades": 0,
                "winning_trades": 0
            }
            
        total_trades = self.daily_metrics.get('total_trades', 0)
        winning_trades = self.daily_metrics.get('winning_trades', 0)
        
        if total_trades > 0:
            return (winning_trades / total_trades) * 100
        else:
            return 0
    
    def _generate_trading_summary(self):
        """
        Generate a summary of today's trading
        
        Returns:
            dict: Trading summary
        """
        if not hasattr(self, 'daily_metrics'):
            self.daily_metrics = {
                "market_sentiment": "Neutral",
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0
            }
            
        if not hasattr(self, 'today_generated_strategies'):
            self.today_generated_strategies = []
            
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "market_sentiment": self.daily_metrics.get("market_sentiment", "Neutral"),
            "strategies_generated": len(self.today_generated_strategies),
            "trades_executed": self.daily_metrics.get("total_trades", 0),
            "winning_trades": self.daily_metrics.get("winning_trades", 0),
            "losing_trades": self.daily_metrics.get("losing_trades", 0),
            "total_pnl": self.daily_metrics.get("total_pnl", 0),
            "win_rate": self._calculate_win_rate(),
            "notes": "Auto-generated trading summary"
        }