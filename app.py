import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional

# Import our modules
from trading_strategy_module import StrategyManager, TrendFollowingStrategy, MeanReversionStrategy
from ai_prediction_engine import PredictionEngine, ModelFactory
from data_collection import DataCollector
from angel_one_integration import AngelOneIntegrationLayer, OrderParams
from order_management_system import OrderManagementSystem, Order, OrderSide, OrderType, OrderStatus
from risk_management_module import RiskManager

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('streamlit_app')

# Set page config
st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.strategy_manager = None
    st.session_state.prediction_engine = None
    st.session_state.data_collector = None
    st.session_state.broker_integration = None
    st.session_state.order_manager = None
    st.session_state.risk_manager = None
    st.session_state.selected_symbols = []
    st.session_state.account_info = {
        "balance": 1000000,
        "margin_available": 500000,
        "equity": 1000000
    }
    st.session_state.market_data = {}
    st.session_state.predictions = {}
    st.session_state.trades = []
    st.session_state.positions = []
    st.session_state.last_update = None

def initialize_system():
    """Initialize all system components"""
    try:
        # Initialize components
        st.session_state.data_collector = DataCollector()
        st.session_state.prediction_engine = PredictionEngine(model_dir=MODEL_DIR)
        st.session_state.strategy_manager = StrategyManager()
        st.session_state.risk_manager = RiskManager()
        
        # Initialize broker integration (using mock for demo)
        st.session_state.broker_integration = AngelOneIntegrationLayer()
        
        # Initialize order management system
        st.session_state.order_manager = OrderManagementSystem(
            st.session_state.broker_integration.api
        )
        
        # Add sample strategies
        st.session_state.strategy_manager.add_strategy(
            TrendFollowingStrategy(
                symbols=["RELIANCE", "INFY", "TCS", "HDFCBANK"],
                params={
                    "trend_threshold": 0.015,
                    "confidence_threshold": 0.7,
                    "risk_per_trade": 0.01,
                }
            )
        )
        
        st.session_state.strategy_manager.add_strategy(
            MeanReversionStrategy(
                symbols=["SBIN", "TATAMOTORS", "ITC", "HDFCBANK"],
                params={
                    "std_dev_threshold": 2.2,
                    "lookback_periods": 30,
                    "account_percent": 0.03,
                }
            )
        )
        
        st.session_state.initialized = True
        st.success("System initialized successfully!")
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")

def fetch_market_data(symbols: List[str], period: str = "1mo", interval: str = "1d"):
    """Fetch market data for selected symbols"""
    try:
        with st.spinner(f"Fetching market data for {len(symbols)} symbols..."):
            for symbol in symbols:
                data = st.session_state.data_collector.fetch_stock_data(
                    symbol, period=period, interval=interval
                )
                if data is not None:
                    st.session_state.market_data[symbol] = data
                    logger.info(f"Fetched data for {symbol}")
                else:
                    logger.warning(f"Failed to fetch data for {symbol}")
                    
        st.session_state.last_update = datetime.now()
        st.success("Market data updated!")
        
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        logger.error(f"Market data error: {str(e)}")

def train_models(symbols: List[str], model_type: str = "lstm"):
    """Train prediction models for selected symbols"""
    try:
        with st.spinner(f"Training {model_type} models..."):
            for symbol in symbols:
                if symbol not in st.session_state.market_data:
                    continue
                    
                # Prepare training data
                X_train, y_train, X_test, y_test, _ = st.session_state.data_collector.prepare_training_data(symbol)
                
                if X_train is None:
                    continue
                    
                # Train model
                model, history, metrics = st.session_state.prediction_engine.train_model(
                    symbol, X_train, y_train, X_test, y_test, model_type=model_type
                )
                
                if model is not None:
                    st.session_state.prediction_engine.save_model(symbol, model_type)
                    logger.info(f"Trained {model_type} model for {symbol}")
                else:
                    logger.warning(f"Failed to train model for {symbol}")
                    
        st.success("Model training completed!")
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        logger.error(f"Training error: {str(e)}")

def generate_predictions(symbols: List[str]):
    """Generate predictions for selected symbols"""
    try:
        st.session_state.predictions = {}
        
        with st.spinner("Generating predictions..."):
            for symbol in symbols:
                if symbol not in st.session_state.market_data:
                    continue
                    
                # Get latest features
                features = st.session_state.data_collector.get_latest_features(symbol)
                if features is None:
                    continue
                    
                # Make prediction
                prediction = st.session_state.prediction_engine.predict(symbol, features)
                
                if prediction is not None:
                    # Simple direction prediction (1 for up, -1 for down)
                    direction = 1 if prediction > 0 else -1
                    confidence = abs(prediction) / 10  # Simple confidence measure
                    
                    st.session_state.predictions[symbol] = {
                        "direction": direction,
                        "confidence": min(max(confidence, 0), 1),  # Clamp to 0-1
                        "model_name": "lstm_model",
                        "predicted_change": prediction
                    }
                    logger.info(f"Generated prediction for {symbol}")
                else:
                    logger.warning(f"Failed to generate prediction for {symbol}")
                    
        st.success("Predictions generated!")
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")

def generate_trades():
    """Generate trades based on predictions and strategies"""
    try:
        if not st.session_state.predictions:
            st.warning("No predictions available. Generate predictions first.")
            return
            
        with st.spinner("Generating trades..."):
            # Process predictions and generate trades
            trades = st.session_state.strategy_manager.process_predictions(
                st.session_state.predictions,
                st.session_state.market_data,
                st.session_state.account_info
            )
            
            st.session_state.trades = trades
            logger.info(f"Generated {len(trades)} trades")
            
        st.success(f"Generated {len(trades)} trades!")
        
    except Exception as e:
        st.error(f"Error generating trades: {str(e)}")
        logger.error(f"Trade generation error: {str(e)}")

def execute_trades():
    """Execute generated trades through the order management system"""
    try:
        if not st.session_state.trades:
            st.warning("No trades available. Generate trades first.")
            return
            
        with st.spinner("Executing trades..."):
            executed_trades = []
            
            for trade in st.session_state.trades:
                # Convert Trade object to OrderParams
                order_params = OrderParams(
                    symbol=trade.symbol + "-EQ",  # Add exchange suffix
                    quantity=int(trade.quantity),
                    side=trade.side.value,
                    order_type=trade.order_type.value,
                    product_type="INTRADAY",
                    price=trade.price,
                    stop_loss=trade.stop_loss,
                    take_profit=trade.take_profit
                )
                
                # Execute trade
                result = st.session_state.broker_integration.execute_trade(order_params)
                
                if result.get("status"):
                    executed_trades.append({
                        "symbol": trade.symbol,
                        "side": trade.side.value,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "status": "EXECUTED",
                        "order_id": result.get("order_id")
                    })
                    logger.info(f"Executed trade for {trade.symbol}")
                else:
                    executed_trades.append({
                        "symbol": trade.symbol,
                        "side": trade.side.value,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "status": "FAILED",
                        "error": result.get("message")
                    })
                    logger.warning(f"Failed to execute trade for {trade.symbol}")
                    
            st.session_state.trades = executed_trades
            st.success(f"Executed {len([t for t in executed_trades if t['status'] == 'EXECUTED'])} trades!")
            
            # Update positions
            update_positions()
            
    except Exception as e:
        st.error(f"Error executing trades: {str(e)}")
        logger.error(f"Trade execution error: {str(e)}")

def update_positions():
    """Update current positions from broker"""
    try:
        with st.spinner("Updating positions..."):
            # Get positions from broker
            positions = st.session_state.broker_integration.get_portfolio_positions()
            
            if positions.get("status"):
                st.session_state.positions = positions.get("data", [])
                logger.info("Updated positions from broker")
            else:
                logger.warning("Failed to get positions from broker")
                
    except Exception as e:
        st.error(f"Error updating positions: {str(e)}")
        logger.error(f"Position update error: {str(e)}")

def display_market_data(symbol: str):
    """Display market data for a specific symbol"""
    if symbol not in st.session_state.market_data:
        return
        
    data = st.session_state.market_data[symbol]
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    )])
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recent data
    st.subheader("Recent Market Data")
    st.dataframe(data.tail(10))

def display_predictions():
    """Display generated predictions"""
    if not st.session_state.predictions:
        st.warning("No predictions available")
        return
        
    predictions_df = pd.DataFrame.from_dict(
        st.session_state.predictions, 
        orient='index',
        columns=['direction', 'confidence', 'model_name', 'predicted_change']
    )
    predictions_df.index.name = 'symbol'
    predictions_df.reset_index(inplace=True)
    
    st.subheader("Current Predictions")
    st.dataframe(predictions_df)
    
    # Visualize predictions
    fig = go.Figure()
    
    for symbol, pred in st.session_state.predictions.items():
        fig.add_trace(go.Bar(
            x=[symbol],
            y=[pred['predicted_change']],
            name=symbol,
            text=[f"Confidence: {pred['confidence']:.2f}"],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Predicted Price Changes",
        xaxis_title="Symbol",
        yaxis_title="Predicted Change (%)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_trades():
    """Display generated and executed trades"""
    if not st.session_state.trades:
        st.warning("No trades available")
        return
        
    trades_df = pd.DataFrame(st.session_state.trades)
    st.subheader("Generated/Executed Trades")
    st.dataframe(trades_df)

def display_positions():
    """Display current positions"""
    if not st.session_state.positions:
        st.warning("No positions available")
        return
        
    positions_df = pd.DataFrame(st.session_state.positions)
    st.subheader("Current Positions")
    st.dataframe(positions_df)
    
    # Calculate portfolio allocation
    if not positions_df.empty and 'market_value' in positions_df.columns:
        total_value = positions_df['market_value'].sum()
        if total_value > 0:
            positions_df['allocation'] = positions_df['market_value'] / total_value * 100
            
            fig = go.Figure(go.Pie(
                labels=positions_df['symbol'],
                values=positions_df['market_value'],
                textinfo='label+percent',
                hoverinfo='value+text'
            ))
            
            fig.update_layout(
                title="Portfolio Allocation",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_risk_report():
    """Display risk management report"""
    if not st.session_state.risk_manager:
        st.warning("Risk manager not initialized")
        return
        
    report = st.session_state.risk_manager.generate_risk_report()
    
    st.subheader("Risk Management Report")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Positions", report.get("total_positions", 0))
    col2.metric("Current Exposure", f"{report.get('current_exposure', 0):.2f}")
    col3.metric("Market Status", report.get("market_status", "unknown").capitalize())
    
    # P&L metrics
    st.write("### Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Today's P&L", f"{report.get('pnl_today', 0):.2f}")
    col2.metric("Weekly P&L", f"{report.get('pnl_week', 0):.2f}")
    
    # Risk warnings
    if report.get("risk_warnings"):
        st.warning("### Risk Warnings")
        for warning in report["risk_warnings"]:
            st.write(f"⚠️ {warning}")

def main():
    """Main Streamlit application"""
    st.title("📈 AI Trading Platform")
    st.markdown("""
        This platform integrates AI predictions with automated trading strategies,
        risk management, and order execution.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        # Symbol selection
        st.session_state.selected_symbols = st.multiselect(
            "Select Symbols",
            ["RELIANCE", "INFY", "TCS", "HDFCBANK", "SBIN", "TATAMOTORS", "ITC"],
            default=["RELIANCE", "INFY", "TCS"]
        )
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["lstm", "random_forest", "xgboost", "ensemble"],
            index=0
        )
        
        # Initialize button
        if st.button("Initialize System"):
            initialize_system()
            
        st.markdown("---")
        st.header("Data & Predictions")
        
        if st.button("Fetch Market Data"):
            if st.session_state.selected_symbols:
                fetch_market_data(st.session_state.selected_symbols)
            else:
                st.warning("Please select symbols first")
                
        if st.button("Train Models"):
            if st.session_state.selected_symbols and st.session_state.initialized:
                train_models(st.session_state.selected_symbols, model_type)
            else:
                st.warning("Please initialize system and select symbols first")
                
        if st.button("Generate Predictions"):
            if st.session_state.selected_symbols and st.session_state.initialized:
                generate_predictions(st.session_state.selected_symbols)
            else:
                st.warning("Please initialize system and select symbols first")
                
        st.markdown("---")
        st.header("Trading")
        
        if st.button("Generate Trades"):
            if st.session_state.initialized and st.session_state.predictions:
                generate_trades()
            else:
                st.warning("Please generate predictions first")
                
        if st.button("Execute Trades"):
            if st.session_state.initialized and st.session_state.trades:
                execute_trades()
            else:
                st.warning("Please generate trades first")
                
        if st.button("Update Positions"):
            if st.session_state.initialized:
                update_positions()
            else:
                st.warning("Please initialize system first")
                
        st.markdown("---")
        st.write(f"Last update: {st.session_state.last_update or 'Never'}")

    # Main content area
    if not st.session_state.initialized:
        st.warning("Please initialize the system from the sidebar")
        return
        
    # Display system status
    st.subheader("System Status")
    col1, col2, col3 = st.columns(3)
    col1.metric("Initialized", "✅" if st.session_state.initialized else "❌")
    col2.metric("Symbols Loaded", len(st.session_state.market_data))
    col3.metric("Active Strategies", len(st.session_state.strategy_manager.strategies))
    
    # Tab layout for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Data", "Predictions", "Trades", "Positions", "Risk"
    ])
    
    with tab1:
        if st.session_state.selected_symbols:
            symbol = st.selectbox(
                "Select Symbol to View",
                st.session_state.selected_symbols
            )
            display_market_data(symbol)
        else:
            st.warning("No symbols selected")
            
    with tab2:
        display_predictions()
        
    with tab3:
        display_trades()
        
    with tab4:
        display_positions()
        
    with tab5:
        display_risk_report()

if __name__ == "__main__":
    main()
