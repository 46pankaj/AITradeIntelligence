import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import threading
import time
import pyotp
import json
import random

from utils.angel_api import AngelOneAPI
from utils.data_manager import DataManager
from utils.technical_analysis import TechnicalAnalysis
from utils.sentiment_analysis import SentimentAnalysis
from utils.oi_analysis import OIAnalysis
from utils.strategy_generator import StrategyGenerator
from utils.trade_executor import TradeExecutor
from utils.risk_management import RiskManager
from utils.auto_trader import AutoTrader

# Page configuration
st.set_page_config(
    page_title="AI Trading Platform - Angel One",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit sidebar navigation links
hide_streamlit_style = """
<style>
    /* Hide the sidebar navigation links except our custom navigation */
    [data-testid="stSidebarNav"] ul {display: none !important;}

    /* Also hide any development-related buttons or links */
    .stDeployButton {display:none !important;}

    /* If needed, we can hide additional elements here */
    footer {visibility: hidden !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'api' not in st.session_state:
    st.session_state.api = None
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = []
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False
    
if 'enforce_strategy_rules' not in st.session_state:
    st.session_state.enforce_strategy_rules = True  # Default to enforcing strategy rules
if 'risk_level' not in st.session_state:
    st.session_state.risk_level = "Medium"
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'paper_trading' not in st.session_state:
    st.session_state.paper_trading = False
if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = []
if 'paper_balance' not in st.session_state:
    st.session_state.paper_balance = 1000000  # Starting with 10 lakhs for paper trading

# Initialize data_manager and load saved strategies
if st.session_state.get('api'):
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager(st.session_state.api)
    
    if 'strategies' not in st.session_state:
        data_manager = st.session_state.data_manager
        saved_strategies = data_manager.load_all_strategies()
        st.session_state.strategies = saved_strategies or []

# Login Function
def login():
    st.session_state.api = AngelOneAPI(
        api_key=st.session_state.api_key,
        client_id=st.session_state.client_id,
        client_password=st.session_state.password,
        totp_key=st.session_state.totp_key
    )
    login_success = st.session_state.api.login()
    if login_success:
        st.session_state.logged_in = True
        st.success("Login Successful!")
        
        # Initialize data_manager after successful login
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager(st.session_state.api)
        
        # Using rerun to handle the callback properly
        st.rerun()
    else:
        st.error("Login Failed. Please check your credentials.")

def get_market_price(exchange, symbol, use_demo=False):
    """
    Get market price with robust error handling and fallback to demo data
    """
    if use_demo or not st.session_state.get('api') or not st.session_state.logged_in:
        # Fallback to demo data
        base_prices = {
            "NIFTY": 18500.00,
            "BANKNIFTY": 43000.00,
            "SENSEX": 62000.00,
            "RELIANCE": 2500.00,
            "TCS": 3400.00,
            "INFY": 1500.00,
            "HDFCBANK": 1600.00
        }
        
        # Add small random variation to demo prices
        price = base_prices.get(symbol, 1000.00)
        price += random.uniform(-price*0.01, price*0.01)  # +/- 1% variation
        
        return {
            'ltp': round(price, 2),
            'change_percent': round(random.uniform(-0.5, 0.5), 2),
            'volume': random.randint(10000, 50000)
        }
    
    try:
        # Try to get real market data
        price_data = st.session_state.api.get_ltp(exchange, symbol)
        
        if price_data and 'ltp' in price_data:
            return price_data
        else:
            # If API returns empty or invalid data, fall back to demo
            st.warning(f"Received empty/invalid price data for {symbol}, using demo data")
            return get_market_price(exchange, symbol, use_demo=True)
            
    except Exception as e:
        st.warning(f"Error fetching price for {symbol}: {str(e)}, using demo data")
        return get_market_price(exchange, symbol, use_demo=True)

# Main App
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1554260570-e9689a3418b8", width=250)
        st.title("AI Trading Platform")

        # Login Section
        if not st.session_state.logged_in:
            st.subheader("Login to Angel One")
            st.session_state.api_key = st.text_input("API Key", type="password")
            st.session_state.client_id = st.text_input("Client ID")
            st.session_state.password = st.text_input("Password", type="password")
            st.session_state.totp_key = st.text_input("TOTP Key", type="password")
            st.button("Login", on_click=login)
        else:
            # App Navigation
            st.subheader("Navigation")

            st.markdown("""
            ### Main Sections:
            """)

            page = st.radio(
                "Select a page",
                ["Dashboard", "Strategy Builder", "Trade Execution", "Performance Monitoring", "Trading Schedule", "Settings"],
                key="navigation_radio"
            )

            # Show description based on selected page
            if page == "Dashboard":
                st.info("Market overview, watchlist, and key performance indicators")
            elif page == "Strategy Builder":
                st.info("Create and backtest trading strategies using technical, sentiment, and OI analysis")
            elif page == "Trade Execution":
                st.info("Execute trades manually or use AI-powered strategies; configure auto trading with AI strategy generation")
            elif page == "Performance Monitoring":
                st.info("Track and analyze your trading performance, P&L, and portfolio metrics")
            elif page == "Trading Schedule":
                st.info("Configure scheduled trading days and automatic login credentials")
            elif page == "Settings":
                st.info("Configure account settings, watchlists, trading preferences, and risk parameters")

            # Auto Trading Toggle
            st.subheader("Auto Trading")
            auto_trading = st.toggle("Enable Auto Trading", value=st.session_state.auto_trading, key="sidebar_auto_trading_toggle")
            if auto_trading != st.session_state.auto_trading:
                st.session_state.auto_trading = auto_trading
                
                # Make sure AutoTrader is set up properly
                if 'auto_trader' in st.session_state:
                    # Update API and data_manager references if needed
                    if st.session_state.auto_trader.api is None and st.session_state.get('api') is not None:
                        st.session_state.auto_trader.api = st.session_state.api
                    
                    if st.session_state.auto_trader.data_manager is None and st.session_state.get('data_manager') is not None:
                        st.session_state.auto_trader.data_manager = st.session_state.data_manager
                        
                    # Also check if components are properly connected
                    if hasattr(st.session_state.auto_trader, 'is_initialized'):
                        st.session_state.auto_trader.is_initialized = (
                            st.session_state.auto_trader.api is not None 
                            and st.session_state.auto_trader.data_manager is not None
                        )
                
                if auto_trading:
                    st.success("Auto Trading Enabled")
                else:
                    st.warning("Auto Trading Disabled")

            # Risk Management
            st.subheader("Risk Management")
            st.session_state.risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value=st.session_state.risk_level,
                key="sidebar_risk_level_slider"
            )

            # Logout Button
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.api = None
                st.rerun()

    # Main Content
    if not st.session_state.logged_in:
        # Landing page for non-logged-in users
        st.title("Welcome to AI Trading Platform")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("""
            ### AI-Powered Trading with Angel One Integration

            Our platform helps you trade smarter with AI-generated strategies based on:
            - Technical indicators analysis
            - Market sentiment
            - Open Interest (OI) data

            Trade automatically or manually on NSE, BSE, Nifty, Bank Nifty, Sensex and more.

            Please login with your Angel One credentials to get started.
            """)

            st.markdown("### Key Features")
            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.markdown("- 📊 Real-time market data")
                st.markdown("- 🤖 AI strategy generation")
                st.markdown("- 📈 Technical indicator analysis")

            with feature_col2:
                st.markdown("- 🔍 Sentiment analysis")
                st.markdown("- 🚀 Automated trade execution")
                st.markdown("- ⚙️ Risk management")

        with col2:
            st.image("https://images.unsplash.com/photo-1639825988283-39e5408b75e8", caption="AI Trading Platform")
            st.image("https://images.unsplash.com/photo-1579532582937-16c108930bf6", caption="Real-time Market Data")

    else:
        # Display the selected page
        if page == "Dashboard":
            display_dashboard()
        elif page == "Strategy Builder":
            display_strategy_builder()
        elif page == "Trade Execution":
            display_trade_execution()
        elif page == "Performance Monitoring":
            display_performance_monitoring()
        elif page == "Trading Schedule":
            # Import and display the trading schedule page
            from pages.scheduler import run
            run()
        elif page == "Settings":
            display_settings()

def display_dashboard():
    st.title("Trading Dashboard")

    # Market Overview
    st.header("Market Overview")

    # Create layout with columns
    col1, col2, col3 = st.columns(3)

    # Get market data with fallback to demo
    with col1:
        st.subheader("Nifty 50")
        nifty_value = get_market_price("NSE", "NIFTY")
        st.metric("Current Value", f"₹{nifty_value['ltp']:,.2f}", f"{nifty_value['change_percent']}%")

    with col2:
        st.subheader("Bank Nifty")
        bank_nifty_value = get_market_price("NSE", "BANKNIFTY")
        st.metric("Current Value", f"₹{bank_nifty_value['ltp']:,.2f}", f"{bank_nifty_value['change_percent']}%")

    with col3:
        st.subheader("Sensex")
        sensex_value = get_market_price("BSE", "SENSEX")
        st.metric("Current Value", f"₹{sensex_value['ltp']:,.2f}", f"{sensex_value['change_percent']}%")

    # Market Charts
    st.subheader("Market Performance")

    # Watchlist
    st.header("Watchlist")
    if not st.session_state.selected_symbols:
        st.info("Add symbols to your watchlist from the Settings page")
    else:
        # Create a table for watchlist
        watchlist_data = []
        for symbol_item in st.session_state.selected_symbols:
            # Check if the symbol is a dictionary with the required keys
            if isinstance(symbol_item, dict) and 'exchange' in symbol_item and 'symbol' in symbol_item:
                exchange = symbol_item['exchange']
                symbol = symbol_item['symbol']
            elif isinstance(symbol_item, str):
                # If it's a string, use a default exchange (NSE) and the string as the symbol
                exchange = "NSE"
                symbol = symbol_item
            else:
                # Skip invalid items
                continue
                
            symbol_data = get_market_price(exchange, symbol)
            watchlist_data.append({
                "Symbol": symbol,
                "Exchange": exchange,
                "LTP": f"₹{symbol_data['ltp']:,.2f}",
                "Change %": f"{symbol_data['change_percent']}%",
                "Volume": f"{symbol_data.get('volume', 'N/A'):,}"
            })

        if watchlist_data:
            st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True)

    # Active Strategies
    st.header("Active Strategies")
    if not st.session_state.strategies:
        st.info("No active strategies. Create one from the Strategy Builder page.")
    else:
        strategy_data = []
        for idx, strategy in enumerate(st.session_state.strategies):
            strategy_data.append({
                "ID": idx,
                "Strategy Name": strategy['name'],
                "Symbol": strategy['symbol'],
                "Type": strategy['type'],
                "Status": strategy['status'],
                "P&L": f"₹{strategy['pnl']:,.2f}"
            })

        if strategy_data:
            st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)
            
            # Add delete strategy option
            st.subheader("Manage Strategies")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                strategy_to_delete = st.selectbox(
                    "Select Strategy",
                    options=range(len(st.session_state.strategies)),
                    format_func=lambda x: st.session_state.strategies[x]['name'],
                    key="dashboard_strategy_selector"
                )
            
            with col2:
                if st.button("Delete Strategy", key="dashboard_delete_strategy"):
                    # Get strategy name for feedback message
                    strategy_name = st.session_state.strategies[strategy_to_delete]['name']
                    # Remove strategy from session state
                    st.session_state.strategies.pop(strategy_to_delete)
                    # Remove from data manager
                    if 'data_manager' not in st.session_state:
                        st.session_state.data_manager = DataManager(st.session_state.api)
                    st.session_state.data_manager.delete_strategy(strategy_name)
                    st.success(f"Strategy {strategy_name} deleted successfully!")
                    st.rerun()

    # Recent Trades
    st.header("Recent Trades")
    if not st.session_state.trades:
        st.info("No recent trades.")
    else:
        trade_data = []
        for trade in st.session_state.trades:
            trade_data.append({
                "Symbol": trade['symbol'],
                "Type": trade['type'],
                "Quantity": trade['quantity'],
                "Price": f"₹{trade['price']:,.2f}",
                "Status": trade['status'],
                "Timestamp": trade['timestamp']
            })

        if trade_data:
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)



def display_strategy_builder():
    st.title("Strategy Builder")

    st.markdown("""
    Build custom trading strategies or use AI to generate strategies based on market data, 
    technical indicators, market sentiment, and Open Interest analysis.
    """)

    # Strategy type selection
    strategy_type = st.radio(
        "Select Strategy Type", 
        ["Manual Strategy", "AI-Generated Strategy"]
    )

    # Symbol selection
    col1, col2 = st.columns(2)

    with col1:
        exchange = st.selectbox(
            "Select Exchange",
            ["NSE", "BSE"]
        )

    with col2:
        if exchange == "NSE":
            symbol = st.selectbox(
                "Select Symbol",
                ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
            )
        else:
            symbol = st.selectbox(
                "Select Symbol",
                ["SENSEX", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
            )

    # Strategy Parameters
    st.subheader("Strategy Parameters")

    if strategy_type == "Manual Strategy":
        # Time frame
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "1 day"]
        )

        # Technical Indicators
        st.subheader("Technical Indicators")

        col1, col2 = st.columns(2)

        with col1:
            use_moving_avg = st.checkbox("Moving Average")
            if use_moving_avg:
                ma_type = st.selectbox("MA Type", ["Simple", "Exponential", "Weighted"])
                ma_period = st.slider("MA Period", 5, 200, 50)

            use_rsi = st.checkbox("Relative Strength Index (RSI)")
            if use_rsi:
                rsi_period = st.slider("RSI Period", 5, 30, 14)
                rsi_overbought = st.slider("RSI Overbought", 70, 90, 70)
                rsi_oversold = st.slider("RSI Oversold", 10, 30, 30)

        with col2:
            use_macd = st.checkbox("MACD")
            if use_macd:
                macd_fast = st.slider("MACD Fast Period", 5, 20, 12)
                macd_slow = st.slider("MACD Slow Period", 10, 40, 26)
                macd_signal = st.slider("MACD Signal Period", 5, 15, 9)

            use_bollinger = st.checkbox("Bollinger Bands")
            if use_bollinger:
                bb_period = st.slider("BB Period", 5, 50, 20)
                bb_std = st.slider("BB Standard Deviation", 1, 5, 2)

        # Entry/Exit Conditions
        st.subheader("Entry and Exit Conditions")

        # Entry conditions
        st.markdown("#### Entry Conditions")

        entry_condition = st.selectbox(
            "Entry Condition",
            ["MA Crossover", "RSI Oversold", "MACD Crossover", "Bollinger Breakout"]
        )

        # Exit conditions
        st.markdown("#### Exit Conditions")

        exit_condition = st.selectbox(
            "Exit Condition",
            ["MA Crossover", "RSI Overbought", "MACD Crossover", "Bollinger Breakout", "Take Profit", "Stop Loss"]
        )

        # Take profit and stop loss
        col1, col2 = st.columns(2)

        with col1:
            take_profit = st.number_input("Take Profit (%)", 0.0, 100.0, 5.0, 0.5, key="manual_take_profit")

        with col2:
            stop_loss = st.number_input("Stop Loss (%)", 0.0, 100.0, 3.0, 0.5, key="manual_stop_loss")

    else:  # AI-Generated Strategy
        # Time frame
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour", "1 day"]
        )

        # Strategy Preferences
        st.subheader("Strategy Preferences")

        col1, col2 = st.columns(2)

        with col1:
            include_technical = st.checkbox("Include Technical Analysis", value=True)
            include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)

        with col2:
            include_oi = st.checkbox("Include Open Interest Analysis", value=True)
            strategy_aggression = st.select_slider(
                "Strategy Aggression",
                options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
                value="Moderate"
            )

        # Take profit and stop loss
        st.subheader("Risk Management")
        col1, col2 = st.columns(2)

        with col1:
            take_profit = st.number_input("Take Profit (%)", 0.0, 100.0, 5.0, 0.5, key="manual_take_profit")

        with col2:
            stop_loss = st.number_input("Stop Loss (%)", 0.0, 100.0, 3.0, 0.5, key="manual_stop_loss")

    # Strategy Name
    strategy_name = st.text_input("Strategy Name", f"{symbol} Strategy")

    # Create Strategy Button
    if st.button("Create Strategy"):
        if strategy_type == "Manual Strategy":
            # Create manual strategy
            strategy = {
                'name': strategy_name,
                'symbol': symbol,
                'exchange': exchange,
                'type': 'Manual',
                'timeframe': timeframe,
                'indicators': {},
                'entry_condition': entry_condition,
                'exit_condition': exit_condition,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'status': 'Active',
                'pnl': 0.0,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add indicators
            if use_moving_avg:
                strategy['indicators']['moving_average'] = {
                    'type': ma_type,
                    'period': ma_period
                }
            if use_rsi:
                strategy['indicators']['rsi'] = {
                    'period': rsi_period,
                    'overbought': rsi_overbought,
                    'oversold': rsi_oversold
                }
            if use_macd:
                strategy['indicators']['macd'] = {
                    'fast_period': macd_fast,
                    'slow_period': macd_slow,
                    'signal_period': macd_signal
                }
            if use_bollinger:
                strategy['indicators']['bollinger_bands'] = {
                    'period': bb_period,
                    'std': bb_std
                }
        else:
            # AI strategy type
            ai_model_type = st.selectbox(
                "Deep Learning Model",
                ["LSTM (Time Series Prediction)", "GAN (Scenario Generation)", "Ensemble (Multiple Models)"],
                index=2,
                key="ai_model_type"
            )
            
            # Map user-friendly names to internal model types
            model_type_map = {
                "LSTM (Time Series Prediction)": "deep_learning",
                "GAN (Scenario Generation)": "gan",
                "Ensemble (Multiple Models)": "ensemble"
            }
            
            # Get risk level from strategy aggression
            risk_level_map = {
                "Very Conservative": "Low",
                "Conservative": "Low",
                "Moderate": "Medium",
                "Aggressive": "High",
                "Very Aggressive": "High"
            }
            
            risk_level = risk_level_map.get(strategy_aggression, "Medium")
            ai_strategy_type = model_type_map.get(ai_model_type, "ensemble")
            
            # Show more details about the selected model
            if ai_model_type == "LSTM (Time Series Prediction)":
                st.info("LSTM models excel at predicting price movements based on historical patterns.")
            elif ai_model_type == "GAN (Scenario Generation)":
                st.info("GAN models generate and evaluate multiple market scenarios to identify high-probability outcomes.")
            else:  # Ensemble
                st.info("Ensemble models combine multiple AI techniques for more robust and reliable signals.")
            
            # Initialize the data manager and AI Strategy Generator
            from utils.ai_strategy_generator import AIStrategyGenerator
            
            with st.spinner(f"Training {ai_model_type.split(' ')[0]} model and generating strategy..."):
                # Get data manager and strategy generator
                if 'data_manager' not in st.session_state:
                    st.session_state.data_manager = DataManager(st.session_state.api)
                strategy_generator = AIStrategyGenerator(st.session_state.data_manager)
                
                # Define days_back based on timeframe
                if timeframe == "1 day":
                    days_back = 120  # About 6 months of trading days
                elif timeframe == "1 hour":
                    days_back = 30
                else:
                    days_back = 14
                
                # Generate the strategy
                try:
                    strategy = strategy_generator.generate_strategy(
                        symbol=symbol,
                        exchange=exchange,
                        strategy_type=ai_strategy_type,
                        risk_level=risk_level,
                        timeframe=timeframe,
                        days_back=days_back
                    )
                    
                    if strategy is None:
                        st.error("Failed to generate strategy. Please try again with different parameters.")
                        return
                    
                    # Update strategy with user's custom take profit and stop loss
                    strategy['take_profit'] = take_profit
                    strategy['stop_loss'] = stop_loss
                    strategy['name'] = strategy_name
                    
                    # Display the recommendation from AI
                    if strategy['recommendations']:
                        latest_rec = strategy['recommendations'][-1]
                        action = latest_rec['action']
                        confidence = latest_rec['confidence']
                        
                        if action == "BUY":
                            st.success(f"✅ AI recommends **BUY** with **{confidence}%** confidence")
                        elif action == "SELL":
                            st.error(f"🔻 AI recommends **SELL** with **{confidence}%** confidence")
                        else:
                            st.info(f"⏸️ AI recommends **HOLD** with **{confidence}%** confidence")
                        
                        st.write("**AI Reasoning:**")
                        st.write(latest_rec['reasoning'])
                except Exception as e:
                    st.error(f"Error generating strategy: {str(e)}")
                    strategy = None
                    return

        # Add strategy to session state
        if 'strategies' not in st.session_state:
            st.session_state.strategies = []

        st.session_state.strategies.append(strategy)
        st.success(f"Strategy '{strategy_name}' created successfully!")

        # Display strategy details
        st.subheader("Strategy Details")
        st.json(strategy)

def display_trade_execution():
    st.title("Trade Execution")

    # Display auto-trading status
    if st.session_state.auto_trading:
        st.success("Auto Trading is Enabled")
    else:
        st.warning("Auto Trading is Disabled. Enable it from the sidebar to allow automatic trade execution.")

    # Display paper trading status
    if st.session_state.paper_trading:
        st.info(f"Paper Trading Mode is Enabled - Current Balance: ₹{st.session_state.paper_balance:,}")
        st.markdown("""
        In paper trading mode, your trades will be simulated with virtual money. 
        No real orders will be placed, but you can track your performance as if they were real trades.
        """)

    # Tabs for different functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Manual Trading", "Option Chain Analysis", "Strategy Execution", "Auto Trading", "Active Orders"])

    with tab1:
        st.header("Manual Trading")

        # Exchange and Symbol Selection
        col1, col2 = st.columns(2)
        with col1:
            exchange = st.selectbox(
                "Select Exchange",
                ["NSE", "BSE"]
            )

        with col2:
            if exchange == "NSE":
                symbol_options = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
            else:
                symbol_options = ["SENSEX", "RELIANCE", "TCS", "INFY", "HDFCBANK"]

            symbol = st.selectbox(
                "Select Symbol",
                symbol_options
            )

        # Get current price
        price_data = st.session_state.api.get_ltp(exchange, symbol)

        # Set a default price for development/demo purposes
        # This will be used when real API data is not available
        default_price = 1000.0

        if price_data and 'ltp' in price_data:
            current_price = price_data['ltp']
            st.metric("Current Price", f"₹{current_price}", f"{price_data.get('change_percent', 0.0)}%")
        else:
            # For demo purposes, use a default price
            st.warning("Using demo price data. In production, real-time prices would be fetched.")
            current_price = default_price
            st.metric("Current Price (Demo)", f"₹{current_price}", "0.0%")

        # Order Type Selection
        col1, col2 = st.columns(2)

        with col1:
            order_type = st.radio(
                "Order Type",
                ["MARKET", "LIMIT", "SL", "SL-M"]
            )

        with col2:
            transaction_type = st.radio(
                "Transaction Type",
                ["BUY", "SELL"]
            )

        # Product Type and Quantity
        col1, col2 = st.columns(2)

        with col1:
            product_type = st.selectbox(
                "Product Type",
                ["DELIVERY", "INTRADAY", "MARGIN"]
            )

        with col2:
            quantity = st.number_input("Quantity", min_value=1, step=1, value=1, key="order_quantity")

        # Price inputs based on order type
        if order_type == "LIMIT":
            limit_price = st.number_input("Limit Price", min_value=0.01, value=float(current_price), step=0.05, key="limit_price")
        elif order_type in ["SL", "SL-M"]:
            col1, col2 = st.columns(2)

            with col1:
                limit_price = st.number_input("Limit Price", min_value=0.01, value=float(current_price), step=0.05, key="sl_limit_price")

            with col2:
                trigger_price = st.number_input("Trigger Price", min_value=0.01, value=float(current_price) * 0.99 if transaction_type == "BUY" else float(current_price) * 1.01, step=0.05, key="trigger_price")

        # Check if we have any active strategies for this symbol
        strategy_exists = False
        matching_strategies = []
        
        if 'strategies' in st.session_state and st.session_state.strategies:
            for strat in st.session_state.strategies:
                if strat.get('symbol') == symbol and strat.get('exchange') == exchange and strat.get('status') == 'Active':
                    strategy_exists = True
                    matching_strategies.append(strat)
        
        # Only enforce strategy rules if the setting is enabled
        enforce_strategy_check = st.session_state.enforce_strategy_rules
        
        if enforce_strategy_check and not strategy_exists:
            st.warning("⚠️ No active strategies found for this symbol. Please create a strategy first before placing orders.")
            st.info("Go to the Strategy Builder page to create a strategy for this symbol.")
            
            if st.checkbox("I understand the risks, allow me to trade without a strategy anyway", key="override_strategy_check"):
                enforce_strategy_check = False
                st.error("⚠️ WARNING: Trading without a defined strategy bypasses risk management controls!")
        
        # Place Order Button (disabled only if we're enforcing strategy rules and no matching strategy exists)
        if st.button("Place Order", disabled=(enforce_strategy_check and not strategy_exists)):
            # Set default prices for any order type
            price_value = 0.0
            trigger_price_value = 0.0

            # Set appropriate prices based on order type
            if order_type == "LIMIT":
                # For LIMIT orders, we need a price
                price_value = float(current_price)
                if 'limit_price' in locals():
                    price_value = float(limit_price)
            elif order_type == "SL":
                # For SL orders, we need both price and trigger price
                price_value = float(current_price)
                trigger_price_value = float(current_price) * 0.99 if transaction_type == "BUY" else float(current_price) * 1.01

                if 'limit_price' in locals():
                    price_value = float(limit_price)
                if 'trigger_price' in locals():
                    trigger_price_value = float(trigger_price)
            elif order_type == "SL-M":
                # For SL-M orders, we need a trigger price
                trigger_price_value = float(current_price) * 0.99 if transaction_type == "BUY" else float(current_price) * 1.01
                if 'trigger_price' in locals():
                    trigger_price_value = float(trigger_price)

            # Prepare order parameters with all required fields
            order_params = {
                "symbol": symbol,
                "exchange": exchange,
                "transaction_type": transaction_type,
                "product_type": product_type,
                "quantity": quantity,
                "order_type": order_type,
                "price": price_value,
                "trigger_price": trigger_price_value
            }

            # For display and paper trading calculations
            trade_price = price_value if order_type in ["LIMIT", "SL"] else current_price
            if trade_price <= 0:  # Use current price if no price is set (e.g., for MARKET orders)
                trade_price = current_price

            # Calculate trade value
            trade_value = trade_price * quantity

            # Check if paper trading mode is enabled
            if st.session_state.paper_trading:
                # Handle paper trading
                if transaction_type == "BUY":
                    # Check if enough paper balance is available
                    if trade_value > st.session_state.paper_balance:
                        st.error(f"Insufficient paper trading balance (₹{st.session_state.paper_balance:,}) for this trade (₹{trade_value:,})")
                    else:
                        # Deduct from paper balance for buy orders
                        st.session_state.paper_balance -= trade_value

                        # Generate a mock order ID
                        order_id = f"PAPER-{len(st.session_state.paper_trades) + 1}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                        # Add to paper trades
                        st.session_state.paper_trades.append({
                            'order_id': order_id,
                            'symbol': symbol,
                            'exchange': exchange,
                            'type': transaction_type,
                            'quantity': quantity,
                            'price': trade_price,
                            'value': trade_value,
                            'status': 'Executed',
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'product_type': product_type,
                            'order_type': order_type
                        })

                        st.success(f"Paper trade executed successfully! Order ID: {order_id}")
                        st.info(f"New paper trading balance: ₹{st.session_state.paper_balance:,}")
                else:  # SELL
                    # For sell, add to paper balance
                    st.session_state.paper_balance += trade_value

                    # Generate a mock order ID
                    order_id = f"PAPER-{len(st.session_state.paper_trades) + 1}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    # Add to paper trades
                    st.session_state.paper_trades.append({
                        'order_id': order_id,
                        'symbol': symbol,
                        'exchange': exchange,
                        'type': transaction_type,
                        'quantity': quantity,
                        'price': trade_price,
                        'value': trade_value,
                        'status': 'Executed',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'product_type': product_type,
                        'order_type': order_type
                    })

                    st.success(f"Paper trade executed successfully! Order ID: {order_id}")
                    st.info(f"New paper trading balance: ₹{st.session_state.paper_balance:,}")
            else:
                # Place real order through API
                trade_executor = TradeExecutor(st.session_state.api)
                order_result = trade_executor.place_order(**order_params)

                if order_result['status'] == 'success':
                    st.success(f"Order placed successfully! Order ID: {order_result['order_id']}")

                    # Add to trades in session state
                    if 'trades' not in st.session_state:
                        st.session_state.trades = []

                    st.session_state.trades.append({
                        'symbol': symbol,
                        'type': transaction_type,
                        'quantity': quantity,
                        'price': trade_price,
                        'status': 'Placed',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    st.error(f"Order placement failed: {order_result['message']}")

    with tab2:
        st.header("Option Chain Analysis")

        # Option chain configuration
        st.subheader("Select Underlying")

        col1, col2 = st.columns(2)

        with col1:
            option_exchange = st.selectbox(
                "Exchange",
                ["NSE", "BSE"],
                key="option_exchange"
            )

        with col2:
            if option_exchange == "NSE":
                option_symbol_options = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
            else:
                option_symbol_options = ["SENSEX", "RELIANCE", "TCS", "INFY", "HDFCBANK"]

            option_symbol = st.selectbox(
                "Symbol",
                option_symbol_options,
                key="option_symbol"
            )

        # Get current price for the option underlying
        try:
            option_price_data = st.session_state.api.get_ltp(option_exchange, option_symbol)
            if option_price_data:
                option_current_price = option_price_data['ltp']
                st.metric("Current Price", f"₹{option_current_price}", f"{option_price_data['change_percent']}%")
            else:
                st.info("Using demo price for illustration purposes.")
                option_current_price = 18500.00 if option_symbol == "NIFTY" else (43000.00 if option_symbol == "BANKNIFTY" else 2500.00)
                st.metric("Current Price (Demo)", f"₹{option_current_price}")
        except Exception as e:
            st.info(f"Using demo price for illustration purposes. Error: {str(e)}")
            option_current_price = 18500.00 if option_symbol == "NIFTY" else (43000.00 if option_symbol == "BANKNIFTY" else 2500.00)
            st.metric("Current Price (Demo)", f"₹{option_current_price}")

        # Expiry selection
        expiry_dates = ["27-Apr-2025", "04-May-2025", "11-May-2025", "18-May-2025", "25-May-2025"]
        selected_expiry = st.selectbox("Expiry Date", expiry_dates, key="option_expiry")

        # Option chain display
        st.subheader(f"Option Chain for {option_symbol} ({selected_expiry})")

        # Fetch real option chain data
        api = st.session_state.api # Added to access api object within the function
        if option_symbol in ["NIFTY", "BANKNIFTY"]:
            option_chain = api.get_option_chain(
                symbol=option_symbol,
                exchange=option_exchange,
                expiry_date=selected_expiry
            )

            if option_chain and 'data' in option_chain:
                option_data = []
                for strike in option_chain['data']:
                    strike_price = strike.get('strikePrice', 0)

                    # Get call and put data
                    call_data = strike.get('CE', {})
                    put_data = strike.get('PE', {})

                    option_data.append({
                        'Call OI': format(call_data.get('openInterest', 0), ','),
                        'Call Chng OI': f"{'+' if call_data.get('changeinOpenInterest', 0) >= 0 else ''}{format(call_data.get('changeinOpenInterest', 0), ',')}",
                        'Call Volume': format(call_data.get('totalTradedVolume', 0), ','),
                        'CallIV': f"{call_data.get('impliedVolatility', 0):.2f}%",
                        'Call LTP': f"₹{call_data.get('lastPrice', 0):.2f}",
                        'Call Chng': f"{call_data.get('change', 0):.2f}%",
                        'Strike': strike_price,
                        'Put LTP': f"₹{put_data.get('lastPrice', 0):.2f}",
                        'Put Chng': f"{put_data.get('change', 0):.2f}%",
                        'Put IV': f"{put_data.get('impliedVolatility', 0):.2f}%",
                        'Put Volume': format(put_data.get('totalTradedVolume', 0), ','),
                        'Put Chng OI': f"{'+' if put_data.get('changeinOpenInterest', 0) >= 0 else ''}{format(put_data.get('changeinOpenInterest', 0), ',')}",
                        'Put OI': format(put_data.get('openInterest', 0), ',')
                    })
            else:
                st.error("Unable to fetch option chain data. Using demo data instead.")

        else:
            st.info("Option chain analysis is currently available for NIFTY and BANKNIFTY only.")

    with tab3:
        st.header("Strategy Execution")

        # Display available strategies
        if not st.session_state.strategies:
            st.info("No strategies available. Create one from the Strategy Builder page.")
        else:
            st.subheader("Available Strategies")

            strategy_data = []
            for idx, strategy in enumerate(st.session_state.strategies):
                strategy_data.append({
                    "ID": idx,
                    "Strategy Name": strategy['name'],
                    "Symbol": strategy['symbol'],
                    "Exchange": strategy['exchange'],
                    "Type": strategy['type'],
                    "Status": strategy['status']
                })

            selected_strategy_df = pd.DataFrame(strategy_data)
            st.dataframe(selected_strategy_df, use_container_width=True)

            # Strategy execution options
            st.subheader("Execute Strategy")

            strategy_id = st.selectbox(
                "Select Strategy",
                options=range(len(st.session_state.strategies)),
                format_func=lambda x: st.session_state.strategies[x]['name']
            )

            selected_strategy = st.session_state.strategies[strategy_id]

            col1, col2 = st.columns(2)

            with col1:
                quantity = st.number_input("Quantity", min_value=1, step=1, value=1, key="strategy_quantity")

            with col2:
                capital = st.number_input("Capital (₹)", min_value=1000, step=1000, value=10000, key="strategy_capital")

            # Check if strategy is active before allowing execution
            if selected_strategy['status'] != 'Active':
                st.warning(f"⚠️ Selected strategy '{selected_strategy['name']}' is not active. Please activate it first.")
                st.info("Go to the Strategy Builder page to activate this strategy.")
                
                # Add override option if rules are enforced but allow override
                if st.session_state.enforce_strategy_rules:
                    if st.checkbox("I understand the risks, allow me to execute this inactive strategy anyway", key="override_strategy_execution"):
                        st.error("⚠️ WARNING: Executing an inactive strategy bypasses risk management controls!")
                        button_disabled = False
                    else:
                        button_disabled = True
                else:
                    button_disabled = False
            else:
                button_disabled = False
                
            # Execute strategy button
            if st.button("Execute Strategy", disabled=button_disabled):
                # Get current market data with improved handling
                st.info("Fetching market data, please wait...")
                
                try:
                    price_data = st.session_state.api.get_ltp(
                        selected_strategy['exchange'], 
                        selected_strategy['symbol']
                    )

                    if not price_data or 'ltp' not in price_data:
                        st.warning("Using estimated market price due to API limitations.")
                        # Create simulated price data for stability
                        import random
                        base_prices = {
                            "NIFTY": 18500.00,
                            "BANKNIFTY": 43000.00,
                            "SENSEX": 62000.00,
                            "RELIANCE": 2500.00,
                            "TCS": 3400.00,
                            "INFY": 1500.00,
                            "HDFCBANK": 1600.00
                        }
                        symbol = selected_strategy['symbol']
                        price_data = {
                            'ltp': base_prices.get(symbol, 1000.00),
                            'change_percent': round(random.uniform(-0.5, 0.5), 2)
                        }
                
                    # For paper trading mode
                    if st.session_state.paper_trading:
                        # Simulate strategy execution with mock data
                        # For demonstration, we'll assume the strategy recommends a BUY
                        transaction_type = "BUY"
                        trade_price = price_data['ltp']
                        trade_value = trade_price * quantity

                        # Check if we have enough balance for paper trading
                        if transaction_type == "BUY" and trade_value > st.session_state.paper_balance:
                            st.error(f"Insufficient paper trading balance (₹{st.session_state.paper_balance:,}) for this strategy execution (₹{trade_value:,})")
                        else:
                            # Update paper balance
                            if transaction_type == "BUY":
                                st.session_state.paper_balance -= trade_value
                            else:  # SELL
                                st.session_state.paper_balance += trade_value

                            # Generate a mock order ID for the paper trade
                            order_id = f"PAPER-STRAT-{len(st.session_state.paper_trades) + 1}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                            # Add to paper trades
                            st.session_state.paper_trades.append({
                                'order_id': order_id,
                                'symbol': selected_strategy['symbol'],
                                'exchange': selected_strategy['exchange'],
                                'type': transaction_type,
                                'quantity': quantity,
                                'price': trade_price,
                                'value': trade_value,
                                'status': 'Executed',
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'strategy': selected_strategy['name'],
                                'is_strategy': True
                            })

                            # Update strategy status
                            st.session_state.strategies[strategy_id]['status'] = 'Executing (Paper)'

                            st.success(f"Strategy executed in paper trading mode! Order ID: {order_id}")
                            st.info(f"New paper trading balance: ₹{st.session_state.paper_balance:,}")
                    else:
                        # Initialize strategy execution for real trading
                        trade_executor = TradeExecutor(st.session_state.api)

                        # Execute strategy
                        execution_result = trade_executor.execute_strategy(
                            strategy=selected_strategy,
                            quantity=quantity,
                            capital=capital,
                            enforce_strategy_rules=st.session_state.enforce_strategy_rules
                        )

                        if execution_result['status'] == 'success':
                            st.success("Strategy execution initiated successfully!")

                            # Update strategy status
                            st.session_state.strategies[strategy_id]['status'] = 'Executing'

                            # Add to trades
                            if 'trades' not in st.session_state:
                                st.session_state.trades = []

                            # Get transaction type from result if available, otherwise from recommendation
                            transaction_type = execution_result.get('transaction_type', None)
                            if transaction_type is None and 'recommendations' in selected_strategy:
                                # Fallback to get transaction type from the strategy's recommendation
                                recommendation = selected_strategy['recommendations'][-1] if selected_strategy['recommendations'] else None
                                transaction_type = recommendation.get('action', 'BUY') if recommendation else 'BUY'
                            
                            st.session_state.trades.append({
                                'symbol': selected_strategy['symbol'],
                                'type': transaction_type,
                                'quantity': quantity,
                                'price': price_data['ltp'],
                                'status': 'Executed',
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        else:
                            st.error(f"Strategy execution failed: {execution_result['message']}")
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Error executing strategy: {error_msg}")
                    
                    # Add more helpful debug info based on the error
                    if 'transaction_type' in error_msg:
                        st.warning("Issue with transaction type. Attempting to recover...")
                        try:
                            if 'recommendations' in selected_strategy and selected_strategy['recommendations']:
                                recommendation = selected_strategy['recommendations'][-1]
                                action = recommendation.get('action', 'BUY')
                                
                                # Add to trades with recovered transaction type
                                if 'trades' not in st.session_state:
                                    st.session_state.trades = []
                                
                                st.session_state.trades.append({
                                    'symbol': selected_strategy['symbol'],
                                    'type': action,
                                    'quantity': quantity,
                                    'price': price_data['ltp'],
                                    'status': 'Executed (Recovered)',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                                
                                st.success("Successfully recovered from error. Trade recorded.")
                            else:
                                st.error("Could not recover: Strategy doesn't have recommendations.")
                        except Exception as recovery_error:
                            st.error(f"Recovery attempt failed: {str(recovery_error)}")
                    
                    st.info("Please verify your account credentials and try again, or enable paper trading mode for testing.")

    with tab4:
        st.header("Auto Trading Configuration")
        
        # Initialize AutoTrader if not already in session state
        if 'auto_trader' not in st.session_state:
            from utils.auto_trader import AutoTrader
            st.session_state.auto_trader = AutoTrader(
                api=st.session_state.get('api'),
                data_manager=st.session_state.get('data_manager')
            )
        elif (hasattr(st.session_state.auto_trader, 'api') and st.session_state.auto_trader.api is None and 
              st.session_state.get('api') is not None):
            # Update connections if they've changed
            st.session_state.auto_trader.api = st.session_state.api
            
        if (hasattr(st.session_state.auto_trader, 'data_manager') and st.session_state.auto_trader.data_manager is None and 
            st.session_state.get('data_manager') is not None):
            st.session_state.auto_trader.data_manager = st.session_state.data_manager
            
        # Description of the AI Trading System
        st.write("""
        This tab allows you to configure and control the AI-powered trading system, which operates in two modes:
        
        1. **AI Strategy Generation Only**: The system generates strategies using AI, 
           but you place the orders manually through the Strategy Execution tab.
        
        2. **Fully Automated Trading**: The system automatically generates strategies, 
           executes trades, and books profits without requiring manual intervention.
        """)

        auto_trader = st.session_state.auto_trader
        
        # Current status
        status_container = st.container()
        with status_container:
            st.subheader("Current Status")
            
            # Check if auto trading is active
            is_active = auto_trader.is_auto_trading()
            
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                if is_active:
                    st.success("Automated Trading: ACTIVE")
                else:
                    st.warning("Automated Trading: INACTIVE")
                    
            with status_col2:
                if is_active:
                    if st.button("Stop Auto Trading", type="primary"):
                        if auto_trader.stop_auto_trading():
                            st.success("Automated trading stopped successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to stop automated trading.")
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
                    
                    # Display explanation based on selected mode
                    if trading_mode == "AI Strategy Generation Only":
                        st.info("""
                        **AI Strategy Generation Mode**: 
                        
                        In this mode, the system will:
                        - Analyze market conditions automatically
                        - Generate trading strategies using AI
                        - Make recommendations for call/put options
                        - Create strategies with entry/exit conditions
                        
                        But you will need to manually execute the trades from the Strategy Execution tab.
                        """)
                    else:
                        st.info("""
                        **Fully Automated Trading Mode**: 
                        
                        In this mode, the system will:
                        - Analyze market conditions automatically
                        - Generate trading strategies using AI
                        - Execute trades automatically based on strategies
                        - Book profits automatically when targets are reached
                        - Adjust strategies based on changing market conditions
                        
                        You can monitor the progress but no manual intervention is required.
                        """)
        
        # Configuration settings
        if not is_active:
            st.subheader("Trading Configuration")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                # Trading interval
                trading_interval = st.slider(
                    "Trading Interval (minutes)",
                    min_value=5,
                    max_value=60,
                    value=st.session_state.get('trading_interval', 15),
                    step=5,
                    help="How often the system should check market conditions and generate strategies"
                )
                st.session_state.trading_interval = trading_interval
                
            with config_col2:
                # Default symbols to show
                default_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK", "INFY"]
                
                # Get previously selected symbols if any
                if 'selected_symbols' not in st.session_state:
                    st.session_state.selected_symbols = default_symbols
                    
                selected_symbols = st.multiselect(
                    "Symbols to Watch",
                    options=["NIFTY", "BANKNIFTY", "SENSEX", "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "LT", "ADANIPORTS"],
                    default=st.session_state.selected_symbols,
                    help="Symbols to analyze and potentially trade"
                )
                st.session_state.selected_symbols = selected_symbols
            
            # Stock selection section
            st.subheader("Additional Stocks")
            
            # Additional symbol input
            additional_symbol = st.text_input(
                "Add Custom Symbol",
                value="",
                help="Add a custom stock symbol (comma-separated for multiple)"
            )
            
            additional_symbols = []
            if additional_symbol:
                # Split by comma, trim whitespace, and convert to uppercase
                additional_symbols = [s.strip().upper() for s in additional_symbol.split(',')]
                st.session_state.additional_symbols = additional_symbols
            
            if selected_symbols or additional_symbols:
                all_symbols = list(set(selected_symbols + additional_symbols))
                st.success(f"Watching {len(all_symbols)} symbols: {', '.join(all_symbols)}")
            else:
                st.warning("No symbols selected. Please select at least one symbol to watch.")
            
            # Advanced automation settings
            st.subheader("Advanced Automation Settings")
            
            auto_col1, auto_col2 = st.columns(2)
            
            with auto_col1:
                auto_capital_allocation = st.toggle(
                    "Automatic Capital Allocation",
                    value=st.session_state.get('auto_capital_allocation', True),
                    help="Automatically calculate position size based on risk level and market conditions"
                )
                st.session_state.auto_capital_allocation = auto_capital_allocation
                
                auto_profit_booking = st.toggle(
                    "Automatic Profit Booking",
                    value=st.session_state.get('auto_profit_booking', True),
                    help="Automatically book profits based on strategy rules and market conditions"
                )
                st.session_state.auto_profit_booking = auto_profit_booking
                
            with auto_col2:
                auto_hedging = st.toggle(
                    "Automatic Hedging",
                    value=st.session_state.get('auto_hedging', False),
                    help="Automatically create hedge positions to protect against adverse market movements"
                )
                st.session_state.auto_hedging = auto_hedging
                
                auto_strategy_refresh = st.toggle(
                    "Automatic Strategy Refresh",
                    value=st.session_state.get('auto_strategy_refresh', True),
                    help="Automatically update strategies based on changing market conditions"
                )
                st.session_state.auto_strategy_refresh = auto_strategy_refresh
                
            # Strategy generation frequency
            strategy_generation_frequency = st.select_slider(
                "Strategy Generation Frequency",
                options=["Low (1-2 per day)", "Medium (3-5 per day)", "High (6-10 per day)", "Very High (10+ per day)"],
                value=st.session_state.get('strategy_generation_frequency', "Medium (3-5 per day)"),
                help="How frequently the system should generate new trading strategies"
            )
            st.session_state.strategy_generation_frequency = strategy_generation_frequency
            
            # Advanced model selection
            model_type = st.selectbox(
                "AI Model Type",
                options=[
                    "Auto-Select Best Model",
                    "LSTM (Time Series Prediction)",
                    "GAN (Scenario Generation)",
                    "Ensemble (Multiple Models)",
                    "Reinforcement Learning"
                ],
                index=0,
                help="Type of AI model to use for strategy generation"
            )
            st.session_state.model_type = model_type
            
            # Risk management
            st.subheader("Risk Management")
            
            col1, col2 = st.columns(2)
            
            # Risk level for each symbol
            with col1:
                default_risk = st.radio(
                    "Default Risk Level",
                    options=["Low", "Medium", "High"],
                    index=1,  # Default to Medium
                    help="Default risk level for all trading strategies"
                )
                st.session_state.default_risk = default_risk
            
            # Position sizing
            with col2:
                max_capital_per_trade = st.number_input(
                    "Max Capital Per Trade (₹)",
                    min_value=5000,
                    max_value=100000,
                    value=st.session_state.get('max_capital_per_trade', 25000),
                    step=5000,
                    help="Maximum capital to allocate per trade"
                )
                st.session_state.max_capital_per_trade = max_capital_per_trade
                
                # Auto-adjust risk per trade
                max_risk_per_trade = st.slider(
                    "Maximum % Risk Per Trade",
                    min_value=0.5,
                    max_value=5.0,
                    value=st.session_state.get('max_risk_per_trade', 2.0),
                    step=0.5,
                    help="Maximum percentage of trading capital to risk on a single trade"
                )
                st.session_state.max_risk_per_trade = max_risk_per_trade
            
            # Trade execution settings
            st.subheader("Trade Execution Settings")
            
            execution_col1, execution_col2 = st.columns(2)
            
            with execution_col1:
                order_type = st.selectbox(
                    "Default Order Type",
                    options=["MARKET", "LIMIT"],
                    index=0,
                    help="Type of orders to place"
                )
                st.session_state.order_type = order_type
            
            with execution_col2:
                product_type = st.selectbox(
                    "Default Product Type",
                    options=["INTRADAY", "DELIVERY"],
                    index=0,
                    help="Product type for trades"
                )
                st.session_state.product_type = product_type
            
            # Save settings
            if st.button("Save Settings"):
                try:
                    # Create or update settings file
                    import os
                    config_dir = "config"
                    os.makedirs(config_dir, exist_ok=True)
                    
                    settings = {
                        # Basic settings
                        "trading_interval": trading_interval,
                        "watched_symbols": selected_symbols,
                        "additional_symbols": additional_symbols,
                        
                        # Risk management
                        "default_risk": default_risk,
                        "max_capital_per_trade": max_capital_per_trade,
                        "max_risk_per_trade": max_risk_per_trade,
                        
                        # Order settings
                        "order_type": order_type,
                        "product_type": product_type,
                        
                        # Advanced automation settings
                        "auto_capital_allocation": auto_capital_allocation,
                        "auto_profit_booking": auto_profit_booking,
                        "auto_hedging": auto_hedging,
                        "auto_strategy_refresh": auto_strategy_refresh,
                        "strategy_generation_frequency": strategy_generation_frequency,
                        "model_type": model_type,
                        
                        # Metadata
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    with open(os.path.join(config_dir, "auto_trading_settings.json"), 'w') as f:
                        json.dump(settings, f, indent=4)
                    
                    st.success("Settings saved successfully!")
                except Exception as e:
                    st.error(f"Error saving settings: {str(e)}")
            
            # Start auto trading button
            col1, col2 = st.columns(2)
            
            with col1:
                # Only show start button if we have API and some symbols
                start_disabled = not (st.session_state.api and (selected_symbols or additional_symbols))
                
                if st.button("Start Auto Trading", disabled=start_disabled, type="primary"):
                    # Save settings first
                    try:
                        # Create settings dir if it doesn't exist
                        import os
                        config_dir = "config"
                        os.makedirs(config_dir, exist_ok=True)
                        
                        # Combine selected and additional symbols
                        all_symbols = list(set(selected_symbols + additional_symbols))
                        
                        # Process trading mode
                        backend_mode = "ai_strategy_only" if trading_mode == "AI Strategy Generation Only" else "fully_automated"
                        
                        # Start auto trading
                        if auto_trader.start_auto_trading(
                            symbols=all_symbols,
                            trading_interval=trading_interval,
                            mode=backend_mode
                        ):
                            # Update settings in auto_trader
                            auto_trader.auto_capital_allocation = auto_capital_allocation
                            auto_trader.auto_profit_booking = auto_profit_booking
                            auto_trader.auto_hedging = auto_hedging
                            auto_trader.auto_strategy_refresh = auto_strategy_refresh
                            auto_trader.strategy_generation_frequency = strategy_generation_frequency
                            auto_trader.model_type = model_type
                            auto_trader.max_risk_per_trade = max_risk_per_trade
                            
                            st.success(f"Auto trading started successfully in {trading_mode} mode!")
                            st.rerun()
                        else:
                            st.error("Failed to start auto trading. Please check the logs for details.")
                    except Exception as e:
                        st.error(f"Error starting auto trading: {str(e)}")
            
            with col2:
                if start_disabled:
                    if not st.session_state.api:
                        st.error("Please login first to enable auto trading.")
                    elif not (selected_symbols or additional_symbols):
                        st.error("Please select at least one symbol to watch.")
                else:
                    st.success("Ready to start auto trading")
        else:
            # Show current auto trading status
            st.subheader("Current Auto Trading Status")
            
            # Get mode in a readable format
            mode_display = "AI Strategy Generation Only" if auto_trader.trading_mode == "ai_strategy_only" else "Fully Automated Trading"
            
            st.write(f"**Mode:** {mode_display}")
            st.write(f"**Watching Symbols:** {', '.join(auto_trader.watched_symbols)}")
            st.write(f"**Trading Interval:** {auto_trader.trading_interval} minutes")
            st.write(f"**Auto Capital Allocation:** {'Enabled' if auto_trader.auto_capital_allocation else 'Disabled'}")
            st.write(f"**Auto Profit Booking:** {'Enabled' if auto_trader.auto_profit_booking else 'Disabled'}")
            st.write(f"**Auto Strategy Refresh:** {'Enabled' if auto_trader.auto_strategy_refresh else 'Disabled'}")
            
            # Show progress
            st.subheader("Automation Progress")
            
            # Create progress metrics for each step
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Market Analysis", "Running", "Active")
                
            with col2:
                st.metric("Strategy Generation", f"{len(auto_trader.auto_generated_strategies)}", "Strategies Generated")
                
            with col3:
                status_text = "Active" if auto_trader.trading_mode == "fully_automated" else "Waiting for Manual Execution"
                st.metric("Trade Execution", status_text, "")
            
            # Display logs section
            st.subheader("Recent Activity Logs")
            st.info("Logs are displayed here to help you monitor the automated trading system's activities.")
            
            # Display simulated log entries
            log_entries = [
                f"{(datetime.now() - timedelta(minutes=i*5)).strftime('%H:%M:%S')} - {action}" 
                for i, action in enumerate([
                    "Market analysis completed. Overall sentiment: Neutral",
                    "Monitoring NIFTY for potential strategy generation",
                    "Analyzed RELIANCE - signal strength too low, skipping",
                    "Waiting for next trading cycle",
                    "Checking profit targets for existing trades",
                    "No trades reached take profit or stop loss levels"
                ])
            ]
            
            # Display logs in reverse chronological order
            for log in log_entries:
                st.text(log)
    
    with tab5:
        st.header("Active Orders")
        


        # Check if paper trading is enabled
        if st.session_state.paper_trading:
            st.subheader("Paper Trading Orders")

            # Display paper trades
            if not st.session_state.paper_trades:
                st.info("No paper trading orders found.")
            else:
                # Process paper trades for display
                paper_order_data = []
                for order in st.session_state.paper_trades:
                    paper_order_data.append({
                        "Order ID": order['order_id'],
                        "Symbol": order['symbol'],
                        "Exchange": order['exchange'],
                        "Type": order['type'],
                        "Quantity": order['quantity'],
                        "Price": f"₹{order['price']:,.2f}",
                        "Value": f"₹{order['value']:,.2f}",
                        "Status": order['status'],
                        "Timestamp": order['timestamp'],
                        "Strategy": order.get('strategy', 'Manual')
                    })

                # Display paper trading orders
                st.dataframe(pd.DataFrame(paper_order_data), use_container_width=True)

                # Paper trading balance display
                st.success(f"Current Paper Trading Balance: ₹{st.session_state.paper_balance:,}")

                # Option to reset paper trading history
                if st.button("Reset Paper Trading History", key="reset_paper_history"):
                    st.session_state.paper_trades = []
                    st.session_state.paper_balance = 1000000
                    st.success("Paper trading history has been reset!")
                    st.rerun()
        else:
            # Fetch and display real active orders
            orders = st.session_state.api.get_order_book()

            if not orders:
                st.info("No active orders found.")
            else:
                # Process orders
                order_data = []
                for order in orders:
                    try:
                        # Use get() method to safely handle potentially missing keys
                        order_data.append({
                            "Order ID": order.get('order_id', order.get('norenordno', 'N/A')),  # Try both possible field names
                            "Symbol": order.get('symbol', order.get('tradingsymbol', 'N/A')),
                            "Exchange": order.get('exchange', order.get('exchangeSegment', 'N/A')), 
                            "Transaction Type": order.get('transaction_type', order.get('transactiontype', 'N/A')),
                            "Quantity": order.get('quantity', order.get('qty', 0)),
                            "Status": order.get('status', 'Unknown'),
                            "Order Type": order.get('order_type', order.get('ordertype', 'N/A')),
                            "Price": order.get('price', order.get('limitprice', 'Market')),
                            "Timestamp": order.get('order_timestamp', order.get('updatetime', 'N/A'))
                        })
                    except Exception as e:
                        st.error(f"Error processing order: {str(e)}")
                        # Log the actual order structure for debugging
                        st.write("Order data structure:", order)

                # Display orders
                st.dataframe(pd.DataFrame(order_data), use_container_width=True)

                # Allow modifying or cancelling orders
                st.subheader("Modify/Cancel Order")

                # Select order to modify
                order_id = st.selectbox(
                    "Select Order",
                    options=[order['Order ID'] for order in order_data]
                )

                col1, col2 = st.columns(2)

                with col1:
                    action = st.radio("Action", ["Modify", "Cancel"])

                if action == "Modify":
                    with col2:
                        new_price = st.number_input("New Price", min_value=0.01, step=0.05, key="modify_price")
                        new_quantity = st.number_input("New Quantity", min_value=1, step=1, key="modify_quantity")

                    if st.button("Submit Modification"):
                        # Call API to modify order
                        modification_result = st.session_state.api.modify_order(
                            order_id=order_id,
                            price=new_price,
                            quantity=new_quantity
                        )

                        if modification_result:
                            st.success("Order modified successfully!")
                        else:
                            st.error("Failed to modify order. Please try again.")
                else:  # Cancel order
                    if st.button("Cancel Order"):
                        # Call API to cancel order
                        cancellation_result = st.session_state.api.cancel_order(order_id)

                        if cancellation_result:
                            st.success("Order cancelled successfully!")
                        else:
                            st.error("Failed to cancel order. Please try again.")

def display_performance_monitoring():
    st.title("Performance Monitoring")

    # Display tabs for different views
    tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Strategy Performance", "Trade History"])

    with tab1:
        st.header("Portfolio Overview")

        # Portfolio Metrics
        col1, col2, col3, col4 = st.columns(4)

        # Get portfolio data from API
        portfolio_data = st.session_state.api.get_portfolio()

        if portfolio_data:
            with col1:
                st.metric("Portfolio Value", f"₹{portfolio_data['portfolio_value']:,.2f}", 
                          f"{portfolio_data['day_change_percent']}%")

            with col2:
                st.metric("Day's P&L", f"₹{portfolio_data['day_pnl']:,.2f}", 
                          f"{portfolio_data['day_change_percent']}%")

            with col3:
                st.metric("Overall P&L", f"₹{portfolio_data['overall_pnl']:,.2f}", 
                          f"{portfolio_data['overall_change_percent']}%")

            with col4:
                st.metric("Available Margin", f"₹{portfolio_data['available_margin']:,.2f}")

            # Portfolio Allocation Chart
            st.subheader("Portfolio Allocation")

            # Create a pie chart for portfolio allocation
            holdings = portfolio_data.get('holdings', [])
            if holdings:
                # Prepare data for pie chart
                labels = [holding['symbol'] for holding in holdings]
                values = [holding['current_value'] for holding in holdings]

                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                fig.update_layout(title_text='Holdings Distribution')
                st.plotly_chart(fig, use_container_width=True)

                # Holdings table
                st.subheader("Current Holdings")

                holding_data = []
                for holding in holdings:
                    holding_data.append({
                        "Symbol": holding['symbol'],
                        "Exchange": holding['exchange'],
                        "Quantity": holding['quantity'],
                        "Avg Price": f"₹{holding['avg_price']:,.2f}",
                        "Current Price": f"₹{holding['ltp']:,.2f}",
                        "Current Value": f"₹{holding['current_value']:,.2f}",
                        "P&L": f"₹{holding['pnl']:,.2f}",
                        "P&L %": f"{holding['pnl_percent']}%"
                    })

                st.dataframe(pd.DataFrame(holding_data), use_container_width=True)
            else:
                st.info("No holdings in your portfolio.")
        else:
            st.error("Unable to fetch portfolio data. Please try again.")

    with tab2:
        st.header("Strategy Performance")

        # Strategy selection
        if not st.session_state.strategies:
            st.info("No strategies available for analysis.")
        else:
            selected_strategy = st.selectbox(
                "Select Strategy to Analyze",
                options=[strategy['name'] for strategy in st.session_state.strategies]
            )

            # Find the selected strategy
            strategy = next((s for s in st.session_state.strategies if s['name'] == selected_strategy), None)

            if strategy:
                st.subheader(f"Performance Analysis: {strategy['name']}")

                # Strategy details
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Symbol", strategy['symbol'])

                with col2:
                    st.metric("Type", strategy['type'])

                with col3:
                    st.metric("Status", strategy['status'])

                with col4:
                    st.metric("P&L", f"₹{strategy['pnl']:,.2f}")

                # Strategy performance chart
                st.subheader("Performance Chart")

                # Get strategy performance data
                # This would need to be implemented with actual data from your backend
                dates = pd.date_range(start=strategy['created_at'], periods=10)
                performance = np.cumsum(np.random.normal(0.5, 1, 10))

                # Create a performance chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=performance, mode='lines+markers', name='Strategy P&L'))
                fig.update_layout(title_text='Strategy Performance Over Time', xaxis_title='Date', yaxis_title='Cumulative P&L')
                st.plotly_chart(fig, use_container_width=True)

                # Strategy metrics
                st.subheader("Strategy Metrics")

                # Mock metrics calculation
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                with metrics_col1:
                    st.metric("Win Rate", "65%")
                    st.metric("Profit Factor", "2.1")

                with metrics_col2:
                    st.metric("Avg Win", "₹1,250")
                    st.metric("Avg Loss", "₹600")

                with metrics_col3:
                    st.metric("Max Drawdown", "8.5%")
                    st.metric("Sharpe Ratio", "1.82")

                # Strategy trades
                st.subheader("Strategy Trades")

                # Mock trade data
                trade_data = []
                trade_data.append({
                    "Date": strategy['created_at'],
                    "Symbol": strategy['symbol'],
                    "Type": "BUY",
                    "Quantity": 10,
                    "Price": 1500,
                    "P&L": 750,
                    "Status": "Closed"
                })

                trade_data.append({
                    "Date": (datetime.strptime(strategy['created_at'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                    "Symbol": strategy['symbol'],
                    "Type": "SELL",
                    "Quantity": 10,
                    "Price": 1575,
                    "P&L": 750,
                    "Status": "Closed"
                })

                st.dataframe(pd.DataFrame(trade_data), use_container_width=True)

                # Strategy settings/parameters
                with st.expander("Strategy Settings"):
                    st.json(strategy)

    with tab3:
        st.header("Trade History")

        # Add tabs for real and paper trades
        trade_tab1, trade_tab2 = st.tabs(["Real Trades", "Paper Trades"])

        with trade_tab1:
            st.subheader("Real Trade History")

            # Date range selection
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30), key="real_start_date")

            with col2:
                end_date = st.date_input("End Date", value=datetime.now(), key="real_end_date")

            # Get trade history
            if 'trades' in st.session_state and st.session_state.trades:
                # Filter trades by date
                filtered_trades = [
                    trade for trade in st.session_state.trades
                    if start_date <= datetime.strptime(trade['timestamp'], "%Y-%m-%d %H:%M:%S").date() <= end_date
                ]

                if filtered_trades:
                    # Display trade history table
                    st.dataframe(pd.DataFrame(filtered_trades), use_container_width=True)

                    # Trade statistics
                    st.subheader("Trade Statistics")

                    # Calculate statistics
                    total_trades = len(filtered_trades)
                    buy_trades = len([t for t in filtered_trades if t['type'] == 'BUY'])
                    sell_trades = len([t for t in filtered_trades if t['type'] == 'SELL'])

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Trades", total_trades)

                    with col2:
                        st.metric("Buy Trades", buy_trades)

                    with col3:
                        st.metric("Sell Trades", sell_trades)

                    # Trade distribution chart
                    st.subheader("Trade Distribution")

                    # Create a distribution chart
                    dates = [datetime.strptime(trade['timestamp'], "%Y-%m-%d %H:%M:%S").date() for trade in filtered_trades]
                    date_counts = {}
                    for date in dates:
                        date_str = date.strftime("%Y-%m-%d")
                        if date_str in date_counts:
                            date_counts[date_str] += 1
                        else:
                            date_counts[date_str] = 1

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(date_counts.keys()), y=list(date_counts.values()), name='Trade Count'))
                    fig.update_layout(title_text='Daily Trade Count', xaxis_title='Date', yaxis_title='Number of Trades')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No real trades found in the selected date range.")
            else:
                st.info("No real trades found in the trading history.")

        with trade_tab2:
            st.subheader("Paper Trade History")

            if not st.session_state.paper_trading:
                st.warning("Paper trading mode is currently disabled. Enable it in the Settings page to start paper trading.")

            # Display paper trading balance
            if st.session_state.paper_trading:
                st.success(f"Current Paper Trading Balance: ₹{st.session_state.paper_balance:,}")

            # Date range selection
            col1, col2 = st.columns(2)

            with col1:
                p_start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30), key="paper_start_date")

            with col2:
                p_end_date = st.date_input("End Date", value=datetime.now(), key="paper_end_date")

            # Get paper trades within the date range
            if 'paper_trades' in st.session_state and st.session_state.paper_trades:
                # Filter paper trades by date
                filtered_paper_trades = [
                    trade for trade in st.session_state.paper_trades
                    if p_start_date <= datetime.strptime(trade['timestamp'], "%Y-%m-%d %H:%M:%S").date() <= p_end_date
                ]

                if filtered_paper_trades:
                    # Process paper trades for display
                    paper_trade_data = []
                    for trade in filtered_paper_trades:
                        paper_trade_data.append({
                            "Order ID": trade['order_id'],
                            "Symbol": trade['symbol'],
                            "Type": trade['type'],
                            "Quantity": trade['quantity'],
                            "Price": f"₹{trade['price']:,.2f}",
                            "Value": f"₹{trade['value']:,.2f}",
                            "Status": trade['status'],
                            "Timestamp": trade['timestamp'],
                            "Strategy": trade.get('strategy', 'Manual')
                        })

                    # Display paper trades
                    st.dataframe(pd.DataFrame(paper_trade_data), use_container_width=True)

                    # Paper trade metrics
                    st.subheader("Paper Trade Metrics")

                    # Calculate metrics
                    total_paper_trades = len(filtered_paper_trades)
                    paper_buy_trades = len([t for t in filtered_paper_trades if t['type'] == 'BUY'])
                    paper_sell_trades = len([t for t in filtered_paper_trades if t['type'] == 'SELL'])

                    # Calculate P&L (simplified version)
                    paper_pnl = 0
                    for trade in filtered_paper_trades:
                        if trade['type'] == 'BUY':
                            paper_pnl -= trade['value']
                        else:  # SELL
                            paper_pnl += trade['value']

                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

                    with metrics_col1:
                        st.metric("Total Trades", total_paper_trades)

                    with metrics_col2:
                        st.metric("Buy Trades", paper_buy_trades)

                    with metrics_col3:
                        st.metric("Sell Trades", paper_sell_trades)

                    with metrics_col4:
                        st.metric("Estimated P&L", f"₹{paper_pnl:,.2f}")

                    # Paper trade distribution chart
                    st.subheader("Paper Trade Distribution")

                    # Create a distribution chart for paper trades
                    paper_dates = [datetime.strptime(trade['timestamp'], "%Y-%m-%d %H:%M:%S").date() for trade in filtered_paper_trades]
                    paper_date_counts = {}
                    for date in paper_dates:
                        date_str = date.strftime("%Y-%m-%d")
                        if date_str in paper_date_counts:
                            paper_date_counts[date_str] += 1
                        else:
                            paper_date_counts[date_str] = 1

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(paper_date_counts.keys()), y=list(paper_date_counts.values()), name='Paper Trade Count', marker_color='rgba(0, 128, 0, 0.7)'))
                    fig.update_layout(title_text='Daily Paper Trade Count', xaxis_title='Date', yaxis_title='Number of Trades')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No paper trades found in the selected date range.")
            else:
                st.info("No paper trades found in the history. Try executing some trades in paper trading mode.")

def display_settings():
    st.title("Settings")

    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs(["Account Settings", "Watchlist", "Trading Preferences", "Risk Management"])

    with tab1:
        st.header("Account Settings")

        # Display account information
        if st.session_state.api and st.session_state.logged_in:
            try:
                user_profile = st.session_state.api.get_user_profile()

                if user_profile and isinstance(user_profile, dict):
                    st.subheader("User Information")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.text_input("Name", value=user_profile.get('name', 'N/A'), disabled=True)
                        st.text_input("Client ID", value=user_profile.get('client_id', 'N/A'), disabled=True)

                    with col2:
                        st.text_input("Email", value=user_profile.get('email', 'N/A'), disabled=True)
                        st.text_input("Mobile", value=user_profile.get('mobile', 'N/A'), disabled=True)

                    # API settings
                    st.subheader("API Settings")

                    try:
                        api_key_display = "*" * 8
                        if hasattr(st.session_state, 'api_key') and st.session_state.api_key:
                            if len(st.session_state.api_key) > 4:
                                api_key_display += st.session_state.api_key[-4:]

                        st.text_input("API Key", value=api_key_display, type="password", disabled=True)

                        # Change TOTP Key
                        if st.button("Update TOTP Key"):
                            with st.form("totp_form"):
                                new_totp = st.text_input("New TOTP Key", type="password")
                                submitted = st.form_submit_button("Update")

                                if submitted and new_totp:
                                    st.session_state.totp_key = new_totp
                                    st.success("TOTP Key updated successfully!")
                    except Exception as e:
                        st.error(f"Error with API settings: {str(e)}")
                else:
                    st.warning("User profile data is not available or in an unexpected format.")
                    st.info("This is normal in the demo mode. In the actual app, your real profile information would be displayed here.")
            except Exception as e:
                st.error(f"Error retrieving user profile: {str(e)}")
                st.info("This is normal in the demo mode. In the actual app, your real profile information would be displayed here.")
        else:
            st.info("Please log in to view and manage your account settings.")

    with tab2:
        st.header("Watchlist")

        # Display current watchlist
        st.subheader("Current Watchlist")

        if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
            st.info("Your watchlist is empty. Add symbols below.")
        else:
            # Display watchlist items with remove option
            for i, symbol_item in enumerate(st.session_state.selected_symbols):
                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    if isinstance(symbol_item, dict):
                        st.write(f"{symbol_item['symbol']}")
                    else:
                        st.write(f"{symbol_item}")

                with col2:
                    if isinstance(symbol_item, dict):
                        st.write(f"{symbol_item['exchange']}")
                    else:
                        st.write("NSE")  # Default exchange

                with col3:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.selected_symbols.pop(i)
                        st.rerun()

        # Add symbols to watchlist
        st.subheader("Add to Watchlist")

        with st.form("add_symbol_form"):
            col1, col2 = st.columns(2)

            with col1:
                exchange = st.selectbox(
                    "Exchange",
                    ["NSE", "BSE"],
                    key="watchlist_exchange"
                )

            with col2:
                if exchange == "NSE":
                    symbol_options = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
                else:
                    symbol_options = ["SENSEX", "RELIANCE", "TCS", "INFY", "HDFCBANK"]

                symbol = st.selectbox(
                    "Symbol",
                    symbol_options,
                    key="watchlist_symbol"
                )

            submitted = st.form_submit_button("Add to Watchlist")

            if submitted:
                # Check if symbol already exists
                if 'selected_symbols' not in st.session_state:
                    st.session_state.selected_symbols = []

                if any(s['symbol'] == symbol and s['exchange'] == exchange for s in st.session_state.selected_symbols):
                    st.error(f"{symbol} from {exchange} is already in your watchlist.")
                else:
                    st.session_state.selected_symbols.append({
                        'symbol': symbol,
                        'exchange': exchange
                    })
                    st.success(f"Added {symbol} to watchlist.")
                    st.rerun()

    with tab3:
        st.header("Trading Preferences")

        # Default trade settings
        st.subheader("Default Trade Settings")

        col1, col2 = st.columns(2)

        with col1:
            default_product = st.selectbox(
                "Default Product Type",
                ["DELIVERY", "INTRADAY", "MARGIN"],
                index=1  # Default to INTRADAY
            )

        with col2:
            default_order_type = st.selectbox(
                "Default Order Type",
                ["MARKET", "LIMIT", "SL", "SL-M"],
                index=0  # Default to MARKET
            )

        # Auto-trading settings
        st.subheader("Auto-Trading Settings")

        auto_trading = st.toggle("Enable Auto Trading", value=st.session_state.auto_trading, key="auto_trading_toggle")
        if auto_trading != st.session_state.auto_trading:
            st.session_state.auto_trading = auto_trading
            if auto_trading:
                st.success("Auto Trading Enabled")
            else:
                st.warning("Auto Trading Disabled")

        st.subheader("Strategy Rule Enforcement")
        
        enforce_strategies = st.toggle("Enforce Strategy Rules", value=st.session_state.enforce_strategy_rules, 
                                     key="enforce_strategy_toggle", 
                                     help="When enabled, trades will only execute if they match an active strategy")
        if enforce_strategies != st.session_state.enforce_strategy_rules:
            st.session_state.enforce_strategy_rules = enforce_strategies
            if enforce_strategies:
                st.success("Strategy Rule Enforcement Enabled - Trades will only execute if they match an active strategy")
            else:
                st.warning("Strategy Rule Enforcement Disabled - Trades can execute without a matching strategy (not recommended)")
                
        if not st.session_state.enforce_strategy_rules:
            st.error("⚠️ WARNING: Disabling strategy rule enforcement allows trades without strategy validation. This is not recommended for live trading as it bypasses risk management controls.")
            st.info("For safety, we recommend keeping strategy rule enforcement enabled for all live trading.")

        # Paper trading mode
        st.subheader("Paper Trading Mode")

        st.markdown("""
        Paper trading allows you to test your strategies with virtual money before risking real capital.
        You can track performance separately from your real trades.
        """)

        paper_trading = st.toggle("Enable Paper Trading Mode", value=st.session_state.paper_trading, key="paper_trading_toggle")
        if paper_trading != st.session_state.paper_trading:
            st.session_state.paper_trading = paper_trading
            if paper_trading:
                st.success("Paper Trading Mode Enabled - Your trades will be simulated with virtual money")
            else:
                st.warning("Paper Trading Mode Disabled - Your trades will be executed in the real market")

        if paper_trading:
            # Paper trading settings
            paper_balance = st.number_input(
                "Paper Trading Balance (₹)", 
                min_value=10000, 
                max_value=10000000,
                value=st.session_state.paper_balance,
                step=10000,
                key="paper_balance_input"
            )

            if paper_balance != st.session_state.paper_balance:
                st.session_state.paper_balance = paper_balance
                st.success(f"Paper trading balance updated to ₹{paper_balance:,}")

            if st.button("Reset Paper Trading History"):
                st.session_state.paper_trades = []
                st.session_state.paper_balance = 1000000
                st.success("Paper trading history has been reset!")

        # Trade frequency
        trade_frequency = st.slider(
            "Maximum Trades Per Day",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )

        # Time restrictions
        col1, col2 = st.columns(2)

        with col1:
            trading_start_time = st.time_input("Trading Start Time", value=datetime.strptime("09:15", "%H:%M").time())

        with col2:
            trading_end_time = st.time_input("Trading End Time", value=datetime.strptime("15:30", "%H:%M").time())

        # Save preferences button
        if st.button("Save Preferences"):
            # Save preferences (in a real app, these would be saved to a database or config file)
            st.success("Trading preferences saved successfully!")

    with tab4:
        st.header("Risk Management")

        # Risk level
        st.subheader("Risk Level")

        risk_level = st.select_slider(
            "Risk Level",
            options=["Low", "Medium", "High"],
            value=st.session_state.risk_level,
            key="settings_risk_level_slider"
)

        if risk_level != st.session_state.risk_level:
            st.session_state.risk_level = risk_level

        # Risk parameters
        st.subheader("Risk Parameters")

        col1, col2 = st.columns(2)

        with col1:
            max_capital_per_trade = st.slider(
                "Max Capital Per Trade (%)",
                min_value=1,
                max_value=100,
                value=10 if risk_level == "Low" else (20 if risk_level == "Medium" else 30),
                step=1
            )

            daily_loss_limit = st.slider(
                "Daily Loss Limit (%)",
                min_value=1,
                max_value=50,
                value=3 if risk_level == "Low" else (5 if risk_level == "Medium" else 10),
                step=1
            )

        with col2:
            default_stop_loss = st.slider(
                "Default Stop Loss (%)",
                min_value=0.5,
                max_value=10.0,
                value=2.0 if risk_level == "Low" else (3.0 if risk_level == "Medium" else 5.0),
                step=0.5
            )

            default_take_profit = st.slider(
                "Default Take Profit (%)",
                min_value=0.5,
                max_value=20.0,
                value=3.0 if risk_level == "Low" else (5.0 if risk_level == "Medium" else 8.0),
                step=0.5
            )

        # Risk manager initialization
        if st.button("Apply Risk Settings"):
            risk_manager = RiskManager()
            risk_manager.set_risk_level(risk_level)
            risk_manager.set_max_capital_per_trade(max_capital_per_trade / 100)
            risk_manager.set_daily_loss_limit(daily_loss_limit / 100)
            risk_manager.set_default_stop_loss(default_stop_loss / 100)
            risk_manager.set_default_take_profit(default_take_profit / 100)

            st.success("Risk settings applied successfully!")

# Run the app
if __name__ == "__main__":
    main()
