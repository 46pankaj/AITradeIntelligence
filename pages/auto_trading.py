import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import pytz

from utils.scheduler import TradingScheduler
from utils.notification_manager import NotificationManager
from utils.position_manager import PositionManager
from utils.trading_journal import TradingJournal
from utils.connection_monitor import ConnectionMonitor
from utils.auto_trader import AutoTrader

def display_auto_trading():
    """Display auto trading page"""
    st.title("Auto Trading Configuration")
    
    # Initialize components if not in session state
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = TradingScheduler()
        
    if 'notification_manager' not in st.session_state:
        st.session_state.notification_manager = NotificationManager()
        
    if 'position_manager' not in st.session_state:
        st.session_state.position_manager = PositionManager()
        
    if 'trading_journal' not in st.session_state:
        st.session_state.trading_journal = TradingJournal()
        
    if 'connection_monitor' not in st.session_state:
        st.session_state.connection_monitor = ConnectionMonitor(api=st.session_state.get('api'))
    
    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = AutoTrader(
            api=st.session_state.get('api'),
            data_manager=st.session_state.get('data_manager')
        )
        
        # Connect components to the AutoTrader
        auto_trader = st.session_state.auto_trader
        auto_trader.scheduler = st.session_state.scheduler
        auto_trader.notification_manager = st.session_state.notification_manager
        auto_trader.position_manager = st.session_state.position_manager
        auto_trader.trading_journal = st.session_state.trading_journal
        auto_trader.connection_monitor = st.session_state.connection_monitor
    
    # Main navigation tabs
    tabs = st.tabs([
        "Status & Control", 
        "Trading Schedule", 
        "Symbol Selection", 
        "Position Management",
        "Notifications",
        "Connection Monitor",
        "Trading Journal"
    ])
    
    # Tab 1: Status & Control
    with tabs[0]:
        display_status_tab()
    
    # Tab 2: Trading Schedule  
    with tabs[1]:
        display_schedule_tab()
    
    # Tab 3: Symbol Selection
    with tabs[2]:
        display_symbol_tab()
    
    # Tab 4: Position Management
    with tabs[3]:
        display_position_tab()
    
    # Tab 5: Notifications
    with tabs[4]:
        display_notification_tab()
    
    # Tab 6: Connection Monitor
    with tabs[5]:
        display_connection_tab()
    
    # Tab 7: Trading Journal
    with tabs[6]:
        display_journal_tab()

def display_status_tab():
    """Display status and control tab"""
    st.header("Trading Status & Control")
    
    # Get components from session state
    auto_trader = st.session_state.auto_trader
    scheduler = st.session_state.scheduler
    connection_monitor = st.session_state.connection_monitor
    
    # Check login status
    is_logged_in = st.session_state.get('logged_in', False)
    
    # Display current status
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        # Check if auto trader is running
        is_active = auto_trader.is_auto_trading()
        
        if is_active:
            st.success("Automated Trading: **ACTIVE**")
            trading_mode = "AI Strategy Generation Only" if auto_trader.trading_mode == "ai_strategy_only" else "Fully Automated Trading"
            st.info(f"Trading Mode: **{trading_mode}**")
        else:
            st.warning("Automated Trading: **INACTIVE**")
        
        # Check scheduler status
        is_scheduler_active = scheduler.scheduler_thread is not None and scheduler.scheduler_thread.is_alive()
        if is_scheduler_active:
            st.success("Trading Schedule: **ACTIVE**")
        else:
            st.warning("Trading Schedule: **INACTIVE**")
            
        # Check connection monitor
        is_monitor_active = connection_monitor.is_monitoring
        connection_status = connection_monitor.connection_state.get("status", "Unknown")
        if is_monitor_active:
            if connection_status == "Connected":
                st.success("API Connection: **CONNECTED**")
            else:
                st.error("API Connection: **DISCONNECTED**")
        else:
            st.warning("Connection Monitoring: **INACTIVE**")
    
    with status_col2:
        if is_active:
            # Trading duration
            if hasattr(auto_trader, 'trading_start_time'):
                start_time = auto_trader.trading_start_time
                duration = datetime.now() - start_time
                st.info(f"Trading Active for: **{str(duration).split('.')[0]}**")
                
            # Show current symbols being monitored
            st.info(f"Monitoring **{len(auto_trader.watched_symbols)}** symbols")
            
            # Show today's generated strategies
            if hasattr(auto_trader, 'today_generated_strategies'):
                st.info(f"Strategies Generated Today: **{len(auto_trader.today_generated_strategies)}**")
            
            # Show today's trades
            if hasattr(auto_trader, 'today_trades'):
                st.info(f"Trades Executed Today: **{len(auto_trader.today_trades)}**")
        
        else:
            # Trading mode selection with detailed explanations
            trading_mode = st.radio(
                "Trading Mode",
                options=[
                    "AI Strategy Generation Only", 
                    "Fully Automated Trading (AI + Execution)"
                ],
                index=0,
                help="Choose between AI strategy generation only (manual execution) or fully automated trading (AI generates and executes)"
            )
            st.session_state.trading_mode = "ai_strategy_only" if trading_mode == "AI Strategy Generation Only" else "fully_automated"
    
    # Start/Stop buttons
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        if is_active:
            if st.button("Stop Auto Trading", type="primary", use_container_width=True):
                if auto_trader.stop_auto_trading():
                    if connection_monitor.is_monitoring:
                        connection_monitor.stop_monitoring()
                    if scheduler.scheduler_thread and scheduler.scheduler_thread.is_alive():
                        scheduler.stop_scheduler()
                    st.success("Automated trading stopped successfully!")
                    st.rerun()
                else:
                    st.error("Failed to stop automated trading.")
        else:
            # Check requirements before allowing start
            start_disabled = not (is_logged_in and st.session_state.get('api'))
            
            if start_disabled:
                st.button("Start Auto Trading", disabled=True, type="primary", use_container_width=True)
                if not is_logged_in:
                    st.error("Please log in first to enable auto trading.")
            else:
                if st.button("Start Auto Trading", type="primary", use_container_width=True):
                    # Get saved settings
                    symbols = auto_trader.watched_symbols
                    trading_interval = auto_trader.trading_interval
                    trading_mode = st.session_state.get('trading_mode', 'ai_strategy_only')
                    
                    # Start connection monitor if not already running
                    if not connection_monitor.is_monitoring:
                        connection_monitor.start_monitoring()
                        
                    # Start scheduler if not already running
                    if not scheduler.scheduler_thread or not scheduler.scheduler_thread.is_alive():
                        # Set up callbacks
                        scheduler.set_callbacks(
                            pre_market=auto_trader.pre_market_preparation,
                            market_open=auto_trader.market_open_handler,
                            market_close=auto_trader.market_close_handler,
                            trading_cycle=auto_trader.trading_cycle_handler
                        )
                        
                        # Start scheduler
                        scheduler.start_scheduler()
                    
                    # Start auto trading
                    if auto_trader.start_auto_trading(
                        symbols=symbols,
                        trading_interval=trading_interval,
                        mode=trading_mode
                    ):
                        st.success(f"Auto trading started successfully in {trading_mode} mode!")
                        st.rerun()
                    else:
                        st.error("Failed to start auto trading. Please check the logs for details.")
    
    with control_col2:
        if is_active:
            if st.button("Pause Auto Trading", use_container_width=True):
                if hasattr(auto_trader, 'pause_auto_trading') and callable(auto_trader.pause_auto_trading):
                    if auto_trader.pause_auto_trading():
                        st.success("Automated trading paused!")
                        st.rerun()
                    else:
                        st.error("Failed to pause automated trading.")
                else:
                    st.error("Pause functionality not available in current version.")
        else:
            if st.button("Test Connection", use_container_width=True):
                if st.session_state.get('api'):
                    try:
                        is_connected = st.session_state.api.is_authenticated()
                        if is_connected:
                            st.success("API connection successful!")
                        else:
                            st.error("API connection failed. Please check your credentials.")
                    except Exception as e:
                        st.error(f"Error testing connection: {str(e)}")
                else:
                    st.error("API not initialized. Please login first.")
    
    # Recent activity log
    st.subheader("Recent Activity Log")
    st.info("Activity logs show recent events from the automated trading system.")
    
    # Show log entries (simulated for now - in real implementation these would come from the auto trader)
    if is_active and hasattr(auto_trader, 'activity_logs'):
        for log in auto_trader.activity_logs:
            st.text(log)
    else:
        # Simulated logs for UI prototyping
        log_entries = [
            f"{(datetime.now() - timedelta(minutes=i*5)).strftime('%H:%M:%S')} - {action}" 
            for i, action in enumerate([
                "System initialized and ready",
                "Monitoring markets for entry signals",
                "Connection status: healthy (ping: 127ms)",
                "Waiting for next trading cycle",
                "Checking profit targets for existing positions",
                "No positions currently active"
            ])
        ]
        
        # Display logs in reverse chronological order (newest first)
        for log in log_entries:
            st.text(log)

def display_schedule_tab():
    """Display trading schedule configuration tab"""
    st.header("Trading Schedule Configuration")
    
    # Get scheduler from session state
    scheduler = st.session_state.scheduler
    
    # Trading days selection
    st.subheader("Trading Days")
    
    # Convert scheduler's days (numbers) to boolean list for checkboxes
    current_days = scheduler.trading_days
    days_selected = [0 in current_days, 1 in current_days, 2 in current_days, 
                     3 in current_days, 4 in current_days, 5 in current_days, 
                     6 in current_days]
    
    # Create 7 columns, one for each day
    day_cols = st.columns(7)
    days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    updated_days = []
    
    # Create a checkbox for each day
    for i, day in enumerate(days_of_week):
        with day_cols[i]:
            if st.checkbox(day, value=days_selected[i], key=f"day_{i}"):
                updated_days.append(i)
    
    # Trading hours
    st.subheader("Trading Hours")
    
    # Get current values
    current_start = scheduler.trading_start_time
    current_end = scheduler.trading_end_time
    current_start_hour = current_start.hour
    current_start_minute = current_start.minute
    current_end_hour = current_end.hour
    current_end_minute = current_end.minute
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Market Open Time")
        start_hour = st.slider("Hour", 0, 23, current_start_hour, key="start_hour")
        start_minute = st.slider("Minute", 0, 59, current_start_minute, step=5, key="start_minute")
        st.write(f"Market opens at: **{start_hour:02d}:{start_minute:02d}**")
    
    with col2:
        st.write("Market Close Time")
        end_hour = st.slider("Hour", 0, 23, current_end_hour, key="end_hour")
        end_minute = st.slider("Minute", 0, 59, current_end_minute, step=5, key="end_minute")
        st.write(f"Market closes at: **{end_hour:02d}:{end_minute:02d}**")
    
    # Pre-market time
    st.subheader("Pre-Market Preparation")
    pre_market_minutes = st.slider("Minutes before market open", 
                                 15, 120, scheduler.pre_market_minutes, 
                                 step=15, key="pre_market_minutes")
    
    st.info(f"The system will wake up **{pre_market_minutes} minutes** before market open to prepare strategies and check market conditions.")
    
    # Holiday calendar
    st.subheader("Market Holidays")
    
    # Display current holidays
    if scheduler.holidays:
        holiday_list = ", ".join([str(h) for h in scheduler.holidays])
        st.info(f"Current holidays: {holiday_list}")
    else:
        st.info("No market holidays configured.")
    
    # Add holiday
    new_holiday = st.date_input("Add market holiday", min_value=datetime.now().date())
    if st.button("Add Holiday"):
        date_str = new_holiday.strftime("%Y-%m-%d")
        scheduler.add_holiday(date_str)
        st.success(f"Added {date_str} to market holidays")
        st.rerun()
    
    # Clear holidays
    if st.button("Clear All Holidays"):
        scheduler.clear_holidays()
        st.success("Cleared all market holidays")
        st.rerun()
    
    # Save settings
    if st.button("Save Schedule Settings", type="primary"):
        # Update scheduler settings
        scheduler.set_trading_days(updated_days)
        scheduler.set_trading_hours(start_hour, start_minute, end_hour, end_minute)
        scheduler.set_pre_market_time(pre_market_minutes)
        
        st.success("Trading schedule settings saved successfully!")
        
        # If auto trader is active, update its settings as well
        auto_trader = st.session_state.auto_trader
        if auto_trader.is_auto_trading():
            auto_trader.trading_interval = (end_hour * 60 + end_minute - (start_hour * 60 + start_minute)) // 10
            st.info(f"Updated auto trader trading interval to {auto_trader.trading_interval} minutes")

def display_symbol_tab():
    """Display symbol selection and configuration tab"""
    st.header("Trading Symbols Configuration")
    
    # Get auto trader from session state
    auto_trader = st.session_state.auto_trader
    
    # Symbol watchlist
    st.subheader("Symbol Watchlist")
    
    # Create tabs for different markets
    market_tabs = st.tabs(["Indices", "Large Cap", "Mid Cap", "Small Cap", "Custom"])
    
    # Tab 1: Indices
    with market_tabs[0]:
        st.write("Major Indian Indices")
        indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "NIFTYIT"]
        
        # Create columns for checkboxes
        idx_cols = st.columns(3)
        selected_indices = []
        
        # Check which indices are already selected
        current_symbols = auto_trader.watched_symbols if hasattr(auto_trader, 'watched_symbols') else []
        
        for i, idx in enumerate(indices):
            with idx_cols[i % 3]:
                if st.checkbox(idx, value=idx in current_symbols, key=f"idx_{idx}"):
                    selected_indices.append(idx)
    
    # Tab 2: Large Cap
    with market_tabs[1]:
        st.write("Top Large Cap Stocks")
        large_caps = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC", 
                     "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK"]
        
        # Create columns for checkboxes
        large_cap_cols = st.columns(3)
        selected_large_caps = []
        
        for i, stock in enumerate(large_caps):
            with large_cap_cols[i % 3]:
                if st.checkbox(stock, value=stock in current_symbols, key=f"large_{stock}"):
                    selected_large_caps.append(stock)
    
    # Tab 3: Mid Cap
    with market_tabs[2]:
        st.write("Top Mid Cap Stocks")
        mid_caps = ["FEDERALBNK", "ABFRL", "LICHSGFIN", "RECLTD", "IBULHSGFIN", 
                   "PFC", "BANKINDIA", "CANBK", "INDIANB", "NMDC", "SAIL"]
        
        # Create columns for checkboxes
        mid_cap_cols = st.columns(3)
        selected_mid_caps = []
        
        for i, stock in enumerate(mid_caps):
            with mid_cap_cols[i % 3]:
                if st.checkbox(stock, value=stock in current_symbols, key=f"mid_{stock}"):
                    selected_mid_caps.append(stock)
    
    # Tab 4: Small Cap
    with market_tabs[3]:
        st.write("Top Small Cap Stocks")
        small_caps = ["IRFC", "SUZLON", "TRIDENT", "JPPOWER", "RPOWER", 
                     "IDEA", "YESBANK", "UJJIVANSFB", "IRB", "GMRINFRA"]
        
        # Create columns for checkboxes
        small_cap_cols = st.columns(3)
        selected_small_caps = []
        
        for i, stock in enumerate(small_caps):
            with small_cap_cols[i % 3]:
                if st.checkbox(stock, value=stock in current_symbols, key=f"small_{stock}"):
                    selected_small_caps.append(stock)
    
    # Tab 5: Custom Symbols
    with market_tabs[4]:
        st.write("Add Custom Symbols")
        custom_symbols = st.session_state.get('custom_symbols', '')
        
        custom_input = st.text_area(
            "Enter custom symbols (comma-separated)",
            value=custom_symbols,
            height=100,
            help="Enter comma-separated stock symbols to add to watchlist (e.g., TATAMOTORS, WIPRO, ADANIPORTS)"
        )
        
        # Parse custom symbols
        selected_customs = []
        if custom_input:
            selected_customs = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
            st.session_state.custom_symbols = custom_input
    
    # Combine all selected symbols
    all_selected = list(set(selected_indices + selected_large_caps + selected_mid_caps + selected_small_caps + selected_customs))
    
    if all_selected:
        st.success(f"Selected {len(all_selected)} symbols: {', '.join(all_selected)}")
    else:
        st.warning("No symbols selected. Please select at least one symbol for auto trading.")
    
    # Advanced symbol settings
    with st.expander("Advanced Symbol Settings"):
        st.subheader("Symbol Prioritization")
        st.write("Configure how the system prioritizes symbols for analysis and trading")
        
        # Symbol weighting strategy
        weighting_strategy = st.selectbox(
            "Symbol Prioritization Strategy",
            options=["Equal Weight", "Volume Weighted", "Volatility Weighted", "Smart (AI-based)"],
            index=0,
            help="Determines how the system allocates time and resources to different symbols"
        )
        
        # Maximum symbols to analyze at once
        max_symbols = st.slider(
            "Maximum Symbols to Analyze Concurrently",
            min_value=1,
            max_value=20,
            value=5,
            help="Limits how many symbols the system analyzes at the same time (higher values use more resources)"
        )
        
        # Symbol rotation interval
        symbol_rotation = st.checkbox(
            "Enable Symbol Rotation",
            value=True,
            help="Automatically rotate through symbols to ensure all are analyzed"
        )
        
        if symbol_rotation:
            rotation_interval = st.slider(
                "Symbol Rotation Interval (minutes)",
                min_value=5,
                max_value=60,
                value=15,
                step=5,
                help="How often to switch to new symbols for analysis"
            )
    
    # Save button
    if st.button("Save Symbol Settings", type="primary"):
        # Update auto trader watched symbols
        auto_trader.watched_symbols = all_selected
        
        # Save advanced settings if available in auto_trader
        if hasattr(auto_trader, 'symbol_weighting_strategy'):
            auto_trader.symbol_weighting_strategy = weighting_strategy
            
        if hasattr(auto_trader, 'max_concurrent_symbols'):
            auto_trader.max_concurrent_symbols = max_symbols
            
        if hasattr(auto_trader, 'symbol_rotation_enabled'):
            auto_trader.symbol_rotation_enabled = symbol_rotation
            auto_trader.symbol_rotation_interval = rotation_interval if symbol_rotation else 0
        
        st.success("Symbol settings saved successfully!")

def display_position_tab():
    """Display position management configuration tab"""
    st.header("Position & Risk Management")
    
    # Get position manager from session state
    position_manager = st.session_state.position_manager
    
    # Capital allocation
    st.subheader("Capital Allocation")
    
    # Trading capital
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Trading Capital (₹)",
            min_value=10000,
            max_value=10000000,
            value=int(position_manager.initial_capital),
            step=10000,
            format="%d",
            help="Total capital available for trading"
        )
    
    with col2:
        # Show available capital (read-only)
        st.metric(
            "Available Trading Capital",
            f"₹{position_manager.available_capital:,.2f}",
            delta=f"{((position_manager.available_capital / position_manager.initial_capital) - 1) * 100:.1f}%" if position_manager.initial_capital > 0 else "0%"
        )
    
    # Risk management
    st.subheader("Risk Parameters")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        max_risk_per_trade = st.slider(
            "Maximum Risk Per Trade (%)",
            min_value=0.5,
            max_value=10.0,
            value=position_manager.max_risk_per_trade,
            step=0.5,
            help="Maximum percentage of capital to risk on a single trade"
        )
        
        max_portfolio_risk = st.slider(
            "Maximum Portfolio Risk (%)",
            min_value=5.0,
            max_value=50.0,
            value=position_manager.max_portfolio_risk,
            step=5.0,
            help="Maximum percentage of capital at risk across all positions"
        )
    
    with risk_col2:
        max_risk_per_symbol = st.slider(
            "Maximum Risk Per Symbol (%)",
            min_value=1.0,
            max_value=20.0,
            value=position_manager.max_risk_per_symbol,
            step=1.0,
            help="Maximum percentage of capital to risk on a single symbol"
        )
    
    # Position sizing
    st.subheader("Position Sizing")
    
    # Position sizing mode
    sizing_mode = st.selectbox(
        "Position Sizing Strategy",
        options=["Fixed Size", "Volatility-Based", "Signal Strength-Based"],
        index=["fixed", "volatility", "signal_strength"].index(position_manager.position_size_mode) if position_manager.position_size_mode in ["fixed", "volatility", "signal_strength"] else 0,
        help="Method used to determine position size for each trade"
    )
    
    # Map selected mode to internal value
    mode_map = {
        "Fixed Size": "fixed",
        "Volatility-Based": "volatility",
        "Signal Strength-Based": "signal_strength"
    }
    selected_mode = mode_map[sizing_mode]
    
    # Show relevant settings based on mode
    if selected_mode == "fixed":
        default_position_size = st.number_input(
            "Default Position Size (₹)",
            min_value=5000,
            max_value=1000000,
            value=int(position_manager.default_position_size),
            step=5000,
            format="%d",
            help="Fixed position size for all trades"
        )
    elif selected_mode == "volatility":
        st.info("Volatility-based position sizing adjusts trade size based on market volatility.")
        st.write("Position Size Multipliers:")
        
        vol_col1, vol_col2 = st.columns(2)
        
        with vol_col1:
            very_low_mult = st.number_input("Very Low Volatility", 0.5, 2.0, position_manager.volatility_multipliers.get("Very Low", 1.5), 0.1)
            low_mult = st.number_input("Low Volatility", 0.5, 2.0, position_manager.volatility_multipliers.get("Low", 1.2), 0.1)
            med_mult = st.number_input("Medium Volatility", 0.5, 2.0, position_manager.volatility_multipliers.get("Medium", 1.0), 0.1)
        
        with vol_col2:
            high_mult = st.number_input("High Volatility", 0.5, 2.0, position_manager.volatility_multipliers.get("High", 0.8), 0.1)
            very_high_mult = st.number_input("Very High Volatility", 0.5, 2.0, position_manager.volatility_multipliers.get("Very High", 0.6), 0.1)
        
        # Update multipliers
        volatility_multipliers = {
            "Very Low": very_low_mult,
            "Low": low_mult,
            "Medium": med_mult,
            "High": high_mult,
            "Very High": very_high_mult
        }
    else:  # signal_strength
        st.info("Signal strength-based position sizing adjusts trade size based on confidence in the signal.")
        st.write("Position Size Multipliers:")
        
        sig_col1, sig_col2 = st.columns(2)
        
        with sig_col1:
            very_weak_mult = st.number_input("Very Weak Signal", 0.1, 2.0, position_manager.signal_multipliers.get("Very Weak", 0.5), 0.1)
            weak_mult = st.number_input("Weak Signal", 0.1, 2.0, position_manager.signal_multipliers.get("Weak", 0.8), 0.1)
            mod_mult = st.number_input("Moderate Signal", 0.5, 2.0, position_manager.signal_multipliers.get("Moderate", 1.0), 0.1)
        
        with sig_col2:
            strong_mult = st.number_input("Strong Signal", 0.5, 2.0, position_manager.signal_multipliers.get("Strong", 1.2), 0.1)
            very_strong_mult = st.number_input("Very Strong Signal", 0.5, 2.0, position_manager.signal_multipliers.get("Very Strong", 1.5), 0.1)
        
        # Update multipliers
        signal_multipliers = {
            "Very Weak": very_weak_mult,
            "Weak": weak_mult,
            "Moderate": mod_mult,
            "Strong": strong_mult,
            "Very Strong": very_strong_mult
        }
    
    # Diversification settings
    st.subheader("Diversification Limits")
    
    div_col1, div_col2 = st.columns(2)
    
    with div_col1:
        max_positions = st.number_input(
            "Maximum Open Positions",
            min_value=1,
            max_value=20,
            value=position_manager.max_positions,
            step=1,
            help="Maximum number of concurrent open positions"
        )
        
        max_positions_per_sector = st.number_input(
            "Maximum Positions Per Sector",
            min_value=1,
            max_value=10,
            value=position_manager.max_positions_per_sector,
            step=1,
            help="Maximum number of positions in a single market sector"
        )
    
    with div_col2:
        max_exposure_per_sector = st.slider(
            "Maximum Sector Exposure (%)",
            min_value=10.0,
            max_value=100.0,
            value=position_manager.max_exposure_per_sector,
            step=5.0,
            help="Maximum percentage of capital allocated to a single sector"
        )
    
    # Save settings
    if st.button("Save Position Settings", type="primary"):
        # Update position manager settings
        position_manager.set_capital(initial_capital)
        position_manager.set_risk_parameters(max_risk_per_trade, max_risk_per_symbol, max_portfolio_risk)
        
        if selected_mode == "fixed":
            position_manager.set_position_sizing(selected_mode, default_position_size)
        elif selected_mode == "volatility":
            position_manager.set_position_sizing(selected_mode)
            position_manager.volatility_multipliers = volatility_multipliers
        else:  # signal_strength
            position_manager.set_position_sizing(selected_mode)
            position_manager.signal_multipliers = signal_multipliers
        
        position_manager.set_diversification_limits(max_positions, max_positions_per_sector, max_exposure_per_sector)
        
        st.success("Position management settings saved successfully!")

def display_notification_tab():
    """Display notification settings tab"""
    st.header("Notification Settings")
    
    # Get notification manager from session state
    notification_manager = st.session_state.notification_manager
    
    # Create tabs for different notification methods
    notification_tabs = st.tabs(["Email Notifications", "SMS Notifications", "Notification Preferences"])
    
    # Tab 1: Email Notifications
    with notification_tabs[0]:
        st.subheader("Email Notification Settings")
        
        # Enable/disable email notifications
        email_enabled = st.toggle(
            "Enable Email Notifications",
            value=notification_manager.email_enabled,
            help="Send notifications via email"
        )
        
        if email_enabled:
            email_col1, email_col2 = st.columns(2)
            
            with email_col1:
                email_from = st.text_input(
                    "From Email",
                    value=notification_manager.email_from,
                    help="The email address to send notifications from"
                )
                
                email_password = st.text_input(
                    "Email Password / App Password",
                    type="password",
                    help="Email account password or app-specific password"
                )
            
            with email_col2:
                email_smtp_server = st.text_input(
                    "SMTP Server",
                    value=notification_manager.email_smtp_server,
                    help="Email server address (e.g., smtp.gmail.com)"
                )
                
                email_smtp_port = st.number_input(
                    "SMTP Port",
                    min_value=1,
                    max_value=65535,
                    value=notification_manager.email_smtp_port,
                    help="Email server port (typically 587 for TLS)"
                )
            
            # Email recipients
            st.subheader("Email Recipients")
            current_recipients = ", ".join(notification_manager.email_recipients)
            email_recipients = st.text_area(
                "Email Recipients (comma-separated)",
                value=current_recipients,
                help="List of email addresses to receive notifications"
            )
            
            # Parse recipients
            recipient_list = [email.strip() for email in email_recipients.split(",") if email.strip()]
            
            # Test email button
            if st.button("Test Email"):
                try:
                    if not email_from or not email_password:
                        st.error("Email sender address and password are required")
                    elif not recipient_list:
                        st.error("At least one recipient email is required")
                    else:
                        # Temporarily configure email for test
                        notification_manager.configure_email(
                            enabled=True,
                            from_email=email_from,
                            password=email_password,
                            recipients=recipient_list,
                            smtp_server=email_smtp_server,
                            smtp_port=email_smtp_port
                        )
                        
                        # Start notification worker if not running
                        if not notification_manager.worker_thread or not notification_manager.worker_thread.is_alive():
                            notification_manager.start_notification_worker()
                        
                        # Send test notification
                        notification_manager.send_notification(
                            subject="Test Notification",
                            message="This is a test notification from TradeAI. If you received this, email notifications are working correctly.",
                            category="general",
                            importance=1
                        )
                        
                        st.success("Test email sent! Please check your inbox.")
                except Exception as e:
                    st.error(f"Error sending test email: {str(e)}")
            
            # Save email settings button
            if st.button("Save Email Settings"):
                # Save configuration to notification manager
                notification_manager.configure_email(
                    enabled=email_enabled,
                    from_email=email_from,
                    password=email_password if email_password else notification_manager.email_password,
                    recipients=recipient_list,
                    smtp_server=email_smtp_server,
                    smtp_port=email_smtp_port
                )
                
                st.success("Email notification settings saved!")
    
    # Tab 2: SMS Notifications
    with notification_tabs[1]:
        st.subheader("SMS Notification Settings")
        
        # Check if Twilio credentials are set
        twilio_account_sid = notification_manager.twilio_account_sid
        twilio_auth_token = notification_manager.twilio_auth_token
        twilio_phone_number = notification_manager.twilio_phone_number
        
        if not twilio_account_sid or not twilio_auth_token or not twilio_phone_number:
            st.warning("Twilio credentials are not configured. Please add the required secrets to enable SMS notifications.")
            st.info("Required secrets: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
            
            if st.button("Configure Twilio Integration"):
                st.session_state['ask_for_secrets'] = True
        
        # Enable/disable SMS notifications
        sms_enabled = st.toggle(
            "Enable SMS Notifications",
            value=notification_manager.sms_enabled,
            disabled=not (twilio_account_sid and twilio_auth_token and twilio_phone_number),
            help="Send notifications via SMS"
        )
        
        if sms_enabled and twilio_account_sid and twilio_auth_token and twilio_phone_number:
            # SMS recipients
            st.subheader("SMS Recipients")
            current_recipients = ", ".join(notification_manager.sms_recipients)
            sms_recipients = st.text_area(
                "Phone Numbers (comma-separated)",
                value=current_recipients,
                help="List of phone numbers to receive SMS notifications (include country code, e.g., +91XXXXXXXXXX)"
            )
            
            # Parse recipients
            recipient_list = [phone.strip() for phone in sms_recipients.split(",") if phone.strip()]
            
            # Test SMS button
            if st.button("Test SMS"):
                try:
                    if not recipient_list:
                        st.error("At least one recipient phone number is required")
                    else:
                        # Temporarily configure SMS for test
                        notification_manager.configure_sms(
                            enabled=True,
                            recipients=recipient_list
                        )
                        
                        # Start notification worker if not running
                        if not notification_manager.worker_thread or not notification_manager.worker_thread.is_alive():
                            notification_manager.start_notification_worker()
                        
                        # Send test notification
                        notification_manager.send_notification(
                            subject="Test SMS",
                            message="This is a test SMS from TradeAI. If you received this, SMS notifications are working correctly.",
                            category="general",
                            importance=5  # High importance to ensure it sends as SMS
                        )
                        
                        st.success("Test SMS sent! Please check your phone.")
                except Exception as e:
                    st.error(f"Error sending test SMS: {str(e)}")
            
            # Save SMS settings button
            if st.button("Save SMS Settings"):
                # Save configuration to notification manager
                notification_manager.configure_sms(
                    enabled=sms_enabled,
                    recipients=recipient_list
                )
                
                st.success("SMS notification settings saved!")
    
    # Tab 3: Notification Preferences
    with notification_tabs[2]:
        st.subheader("Notification Preferences")
        
        # Importance thresholds
        st.write("Set the minimum importance level for each notification method")
        
        threshold_col1, threshold_col2 = st.columns(2)
        
        with threshold_col1:
            email_threshold = st.slider(
                "Email Notification Threshold",
                min_value=1,
                max_value=5,
                value=notification_manager.email_importance_threshold,
                help="Minimum importance level for email notifications (1-5)"
            )
        
        with threshold_col2:
            sms_threshold = st.slider(
                "SMS Notification Threshold",
                min_value=1,
                max_value=5,
                value=notification_manager.sms_importance_threshold,
                help="Minimum importance level for SMS notifications (1-5)"
            )
        
        # Notification categories
        st.subheader("Notification Categories")
        st.write("Configure importance levels for different notification categories")
        
        # Get current categories and their importance levels
        categories = notification_manager.categories
        
        # Create two columns of category importance sliders
        cat_col1, cat_col2 = st.columns(2)
        
        updated_categories = {}
        cat_keys = list(categories.keys())
        half = len(cat_keys) // 2 + len(cat_keys) % 2
        
        with cat_col1:
            for cat in cat_keys[:half]:
                importance = st.slider(
                    f"{cat.replace('_', ' ').title()}",
                    min_value=1,
                    max_value=5,
                    value=categories[cat],
                    help=f"Importance level for {cat} notifications (1-5)"
                )
                updated_categories[cat] = importance
        
        with cat_col2:
            for cat in cat_keys[half:]:
                importance = st.slider(
                    f"{cat.replace('_', ' ').title()}",
                    min_value=1,
                    max_value=5,
                    value=categories[cat],
                    help=f"Importance level for {cat} notifications (1-5)"
                )
                updated_categories[cat] = importance
        
        # Save preferences button
        if st.button("Save Notification Preferences"):
            # Update importance thresholds
            notification_manager.set_importance_thresholds(email_threshold, sms_threshold)
            
            # Update category importance levels
            notification_manager.categories = updated_categories
            
            # Restart notification worker if needed
            if notification_manager.worker_thread and notification_manager.worker_thread.is_alive():
                notification_manager.stop_notification_worker()
                notification_manager.start_notification_worker()
            elif notification_manager.email_enabled or notification_manager.sms_enabled:
                notification_manager.start_notification_worker()
            
            st.success("Notification preferences saved!")

def display_connection_tab():
    """Display connection monitor tab"""
    st.header("Connection Health Monitor")
    
    # Get connection monitor from session state
    connection_monitor = st.session_state.connection_monitor
    
    # Update API connection if needed
    if st.session_state.get('api') and connection_monitor.api is None:
        connection_monitor.set_api(st.session_state.api)
    
    # Connection status
    st.subheader("Connection Status")
    
    # Get current connection state
    connection_state = connection_monitor.get_connection_state()
    status = connection_state.get("status", "Unknown")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if status == "Connected":
            st.success("Status: **CONNECTED**")
        elif status == "Disconnected":
            st.error("Status: **DISCONNECTED**")
        else:
            st.warning("Status: **UNKNOWN**")
        
        # Show uptime percentage
        uptime = connection_state.get("uptime_percent", 0)
        st.metric("Connection Uptime", f"{uptime:.1f}%")
    
    with status_col2:
        # Show last check time
        last_check = connection_state.get("last_check", "Never")
        st.info(f"Last Check: {last_check}")
        
        # Show average response time
        avg_response = connection_state.get("average_response_time", 0)
        st.metric("Avg Response Time", f"{avg_response*1000:.0f}ms")
    
    # Connection monitor controls
    st.subheader("Connection Monitor Controls")
    
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        is_monitoring = connection_monitor.is_monitoring
        
        if is_monitoring:
            if st.button("Stop Monitoring", use_container_width=True):
                if connection_monitor.stop_monitoring():
                    st.success("Connection monitoring stopped")
                    st.rerun()
                else:
                    st.error("Failed to stop connection monitoring")
        else:
            api_available = st.session_state.get('api') is not None
            
            if st.button("Start Monitoring", disabled=not api_available, use_container_width=True):
                if api_available:
                    if connection_monitor.start_monitoring():
                        st.success("Connection monitoring started")
                        st.rerun()
                    else:
                        st.error("Failed to start connection monitoring")
                else:
                    st.error("API not available. Please login first.")
    
    with control_col2:
        if st.button("Check Connection Now", disabled=not st.session_state.get('api'), use_container_width=True):
            if connection_monitor.check_connection():
                st.success("Connection check successful!")
                st.rerun()
            else:
                st.error("Connection check failed")
                st.rerun()
    
    # Monitor configuration
    st.subheader("Monitor Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        check_interval = st.slider(
            "Health Check Interval (seconds)",
            min_value=10,
            max_value=300,
            value=connection_monitor.health_check_interval,
            step=10,
            help="How often to check the API connection"
        )
        
        auto_recovery = st.checkbox(
            "Enable Auto Recovery",
            value=connection_monitor.auto_recovery,
            help="Automatically attempt to recover the connection when it fails"
        )
    
    with config_col2:
        max_failures = st.slider(
            "Max Consecutive Failures",
            min_value=1,
            max_value=10,
            value=connection_monitor.max_consecutive_failures,
            help="Number of consecutive failures before attempting recovery"
        )
    
    # Save configuration button
    if st.button("Save Monitor Configuration"):
        connection_monitor.configure(
            health_check_interval=check_interval,
            max_consecutive_failures=max_failures,
            auto_recovery=auto_recovery
        )
        
        st.success("Connection monitor configuration saved!")
    
    # Connection history
    st.subheader("Connection History")
    
    # Get connection history
    history = connection_monitor.get_connection_history(limit=20)
    
    if history:
        # Convert to DataFrame for display
        history_df = pd.DataFrame(history)
        
        # Format response time and add status indicator
        if "response_time" in history_df.columns:
            history_df["response_time"] = history_df["response_time"].apply(lambda x: f"{x*1000:.0f}ms")
        
        if "status" in history_df.columns:
            history_df["status"] = history_df["status"].apply(
                lambda x: "🟢 " + x if x == "Connected" else "🔴 " + x
            )
        
        # Display the history
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No connection history available yet.")

def display_journal_tab():
    """Display trading journal tab"""
    st.header("Trading Journal")
    
    # Get trading journal from session state
    trading_journal = st.session_state.trading_journal
    
    # Create tabs for different journal sections
    journal_tabs = st.tabs([
        "Trade History", 
        "Strategy History", 
        "Performance Analytics",
        "Journal Settings"
    ])
    
    # Tab 1: Trade History
    with journal_tabs[0]:
        st.subheader("Trade History")
        
        # Filter options
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Symbol filter
            symbol_filter = st.text_input(
                "Filter by Symbol",
                value="",
                help="Enter a symbol to filter trades"
            )
        
        with filter_col2:
            # Date range filter
            date_from = st.date_input(
                "From Date",
                value=datetime.now().date() - timedelta(days=30),
                help="Start date for filtering trades"
            )
        
        with filter_col3:
            date_to = st.date_input(
                "To Date",
                value=datetime.now().date(),
                help="End date for filtering trades"
            )
        
        # Get trades with filters
        trades = trading_journal.get_trades(
            symbol=symbol_filter if symbol_filter else None,
            date_from=date_from.strftime("%Y-%m-%d"),
            date_to=date_to.strftime("%Y-%m-%d")
        )
        
        if trades:
            # Convert to DataFrame for display
            trades_df = pd.DataFrame(trades)
            
            # Format values for display
            if "price" in trades_df.columns:
                trades_df["price"] = trades_df["price"].apply(lambda x: f"₹{x:,.2f}")
            
            if "pnl" in trades_df.columns:
                trades_df["pnl"] = trades_df["pnl"].apply(lambda x: f"₹{x:,.2f}")
            
            # Display trades
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades found with the current filters.")
        
        # Add note for selected trade
        with st.expander("Add Note to Trade"):
            trade_id = st.text_input("Trade ID", help="Enter the ID of the trade to add a note to")
            note_text = st.text_area("Note", help="Enter your note about this trade")
            
            if st.button("Add Note"):
                if trade_id and note_text:
                    # Find the trade
                    trade = None
                    for t in trades:
                        if t.get("id") == trade_id:
                            trade = t
                            break
                    
                    if trade:
                        # Add note
                        note_id = trading_journal.add_note(
                            note_text=note_text,
                            tags=["trade"],
                            related_ids=[trade_id]
                        )
                        
                        st.success(f"Note added successfully (ID: {note_id})")
                    else:
                        st.error("Trade not found with the specified ID")
                else:
                    st.error("Both Trade ID and Note are required")
    
    # Tab 2: Strategy History
    with journal_tabs[1]:
        st.subheader("Strategy History")
        
        # Filter options
        strat_filter_col1, strat_filter_col2, strat_filter_col3 = st.columns(3)
        
        with strat_filter_col1:
            # Symbol filter
            strat_symbol_filter = st.text_input(
                "Filter by Symbol",
                value="",
                key="strat_symbol_filter",
                help="Enter a symbol to filter strategies"
            )
        
        with strat_filter_col2:
            # Date range filter
            strat_date_from = st.date_input(
                "From Date",
                value=datetime.now().date() - timedelta(days=30),
                key="strat_date_from",
                help="Start date for filtering strategies"
            )
        
        with strat_filter_col3:
            strat_date_to = st.date_input(
                "To Date",
                value=datetime.now().date(),
                key="strat_date_to",
                help="End date for filtering strategies"
            )
        
        # Get strategies with filters
        strategies = trading_journal.get_strategies(
            symbol=strat_symbol_filter if strat_symbol_filter else None,
            date_from=strat_date_from.strftime("%Y-%m-%d"),
            date_to=strat_date_to.strftime("%Y-%m-%d")
        )
        
        if strategies:
            # Convert to DataFrame for display
            strategies_df = pd.DataFrame(strategies)
            
            # Display strategies
            st.dataframe(strategies_df, use_container_width=True)
        else:
            st.info("No strategies found with the current filters.")
    
    # Tab 3: Performance Analytics
    with journal_tabs[2]:
        st.subheader("Performance Analytics")
        
        # Date range for analytics
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            analytics_date_from = st.date_input(
                "From Date",
                value=datetime.now().date() - timedelta(days=30),
                key="analytics_date_from"
            )
        
        with analytics_col2:
            analytics_date_to = st.date_input(
                "To Date",
                value=datetime.now().date(),
                key="analytics_date_to"
            )
        
        # Optional symbol filter
        analytics_symbol = st.text_input(
            "Filter by Symbol (optional)",
            value="",
            key="analytics_symbol"
        )
        
        # Generate report button
        if st.button("Generate Performance Report"):
            # Get trade statistics
            stats = trading_journal.get_trade_statistics(
                date_from=analytics_date_from.strftime("%Y-%m-%d"),
                date_to=analytics_date_to.strftime("%Y-%m-%d"),
                symbol=analytics_symbol if analytics_symbol else None
            )
            
            # Display statistics
            st.subheader("Trade Statistics")
            
            # Key metrics in 4 columns
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Trades", stats["total_trades"])
                st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
            
            with metric_col2:
                st.metric("Winning Trades", stats["winning_trades"])
                st.metric("Losing Trades", stats["losing_trades"])
            
            with metric_col3:
                st.metric("Total P&L", f"₹{stats['total_pnl']:,.2f}")
                st.metric("Average P&L", f"₹{stats['average_pnl']:,.2f}")
            
            with metric_col4:
                st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
                st.metric("Avg Hold Time", stats["average_hold_time"])
            
            # Additional metrics
            st.subheader("Detailed Metrics")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.metric("Largest Win", f"₹{stats['largest_win']:,.2f}")
                st.metric("Average Win", f"₹{stats['average_win']:,.2f}")
            
            with detail_col2:
                st.metric("Largest Loss", f"₹{stats['largest_loss']:,.2f}")
                st.metric("Average Loss", f"₹{stats['average_loss']:,.2f}")
            
            # Generate daily P&L chart
            st.subheader("Daily P&L Chart")
            
            # Get data for chart (this would be implemented with real data from TradingJournal)
            # For now, we'll use simulated data for the UI
            dates = [
                (analytics_date_from + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range((analytics_date_to - analytics_date_from).days + 1)
            ]
            
            # Simulated P&L data
            # In a real implementation, this would come from the trading journal
            # trading_journal.get_daily_pnl(date_from=analytics_date_from.strftime("%Y-%m-%d"), 
            #                               date_to=analytics_date_to.strftime("%Y-%m-%d"),
            #                               symbol=analytics_symbol if analytics_symbol else None)
            
            if len(dates) > 0:
                st.line_chart(
                    data=pd.DataFrame({
                        "Date": dates,
                        "P&L": [0] * len(dates)  # Placeholder for real data
                    }).set_index("Date"),
                    use_container_width=True
                )
                
                st.info("Note: This chart shows simulated data. In the production version, it will display actual P&L data from your trades.")
    
    # Tab 4: Journal Settings
    with journal_tabs[3]:
        st.subheader("Journal Settings")
        
        # Export journal data
        st.write("Export journal data for backup or analysis")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            export_path = st.text_input(
                "Export Directory",
                value="data/journal_export",
                help="Directory path to save exported data"
            )
        
        with export_col2:
            if st.button("Export Journal Data"):
                if trading_journal.export_to_csv(export_path):
                    st.success(f"Journal data exported successfully to {export_path}")
                else:
                    st.error("Failed to export journal data")
        
        # Journal retention settings
        st.subheader("Journal Retention Settings")
        
        retention_col1, retention_col2 = st.columns(2)
        
        with retention_col1:
            max_entries = st.number_input(
                "Maximum Entries per Category",
                min_value=100,
                max_value=10000,
                value=trading_journal.max_entries,
                step=100,
                help="Maximum number of journal entries to keep in each category"
            )
        
        # Clear journal data
        st.subheader("Clear Journal Data")
        st.warning("This will permanently delete journal entries. Use with caution!")
        
        if st.button("Clear All Journal Data"):
            # Show confirmation dialog
            st.error("Are you sure you want to clear all journal data?")
            
            confirm_col1, confirm_col2 = st.columns(2)
            
            with confirm_col1:
                if st.button("Yes, Clear Data"):
                    # Clear each category
                    trading_journal.trade_entries = []
                    trading_journal.strategy_entries = []
                    trading_journal.market_entries = []
                    trading_journal.notes = []
                    trading_journal._save_data()
                    
                    st.success("Journal data cleared successfully")
                    st.rerun()
            
            with confirm_col2:
                if st.button("No, Cancel"):
                    st.info("Operation cancelled")
                    st.rerun()

def run():
    """Run the auto trading page"""
    display_auto_trading()

if __name__ == "__main__":
    run()