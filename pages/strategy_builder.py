import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from utils.angel_api import AngelOneAPI
from utils.technical_analysis import TechnicalAnalysis
from utils.strategy_generator import StrategyGenerator

def run():
    st.title("Strategy Builder")
    
    if not st.session_state.get("logged_in", False):
        st.warning("Please login to use the Strategy Builder.")
        return
    
    # Get API instance
    api = st.session_state.api
    
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
        

    # Strategy Status
    strategy = {}  # Initialize strategy dictionary
    strategy_status = st.toggle("Strategy Active", value=True, key="strategy_status_toggle")
    strategy['status'] = 'Active' if strategy_status else 'Inactive'

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
            take_profit = st.number_input("Take Profit (%)", 0.0, 100.0, 5.0, 0.5)
        
        with col2:
            stop_loss = st.number_input("Stop Loss (%)", 0.0, 100.0, 3.0, 0.5)
        
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
            take_profit = st.number_input("Take Profit (%)", 0.0, 100.0, 5.0, 0.5)
        
        with col2:
            stop_loss = st.number_input("Stop Loss (%)", 0.0, 100.0, 3.0, 0.5)
    
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
            # Generate AI strategy
            strategy_generator = StrategyGenerator()
            
            # Get historical data
            data = api.get_historical_data(exchange, symbol, timeframe, 100)
            
            # Generate strategy
            if data is not None:
                strategy = strategy_generator.generate_strategy(
                    symbol=symbol,
                    exchange=exchange,
                    data=data,
                    use_technical=include_technical,
                    use_sentiment=include_sentiment,
                    use_oi=include_oi,
                    risk_level=strategy_aggression,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    strategy_name=strategy_name
                )
                
                st.json(strategy)
            else:
                st.error("Failed to fetch historical data. Please try again.")
                return
        
        # Add strategy to session state
        if 'strategies' not in st.session_state:
            st.session_state.strategies = []
        
        # Save strategy to disk
        data_manager = DataManager(api)
        if data_manager.save_strategy(strategy):
            st.session_state.strategies.append(strategy)
            st.success(f"Strategy '{strategy_name}' created successfully!")
        else:
            st.error("Failed to save strategy. Please try again.")
        
        # Display strategy details
        st.subheader("Strategy Details")
        st.json(strategy)
        
        # If it's an AI-generated strategy with technical analysis, show charts
        if strategy_type == "AI-Generated Strategy" and include_technical and data is not None:
            st.subheader("Technical Analysis Visualization")
            
            # Initialize technical analyzer
            technical_analyzer = TechnicalAnalysis()
            
            # Add indicators to data
            if 'moving_average' in strategy.get('indicators', {}).get('technical', {}):
                data = technical_analyzer.add_moving_average(data, period=20)
                data = technical_analyzer.add_moving_average(data, period=50)
            
            if 'rsi' in strategy.get('indicators', {}).get('technical', {}):
                data = technical_analyzer.add_rsi(data)
            
            if 'macd' in strategy.get('indicators', {}).get('technical', {}):
                data = technical_analyzer.add_macd(data)
            
            if 'bollinger_bands' in strategy.get('indicators', {}).get('technical', {}):
                data = technical_analyzer.add_bollinger_bands(data)
            
            # Create price chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ))
            
            # Add moving averages if available
            if 'simple_ma_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['simple_ma_20'],
                    name="MA 20",
                    line=dict(color='blue')
                ))
            
            if 'simple_ma_50' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['simple_ma_50'],
                    name="MA 50",
                    line=dict(color='red')
                ))
            
            # Add Bollinger Bands if available
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['bb_upper'],
                    name="BB Upper",
                    line=dict(color='green', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['bb_lower'],
                    name="BB Lower",
                    line=dict(color='green', dash='dash')
                ))
            
            fig.update_layout(title=f"{symbol} Price Chart with Indicators",
                             xaxis_title="Date",
                             yaxis_title="Price",
                             xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create RSI chart if available
            if 'rsi_14' in data.columns:
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['rsi_14'],
                    name="RSI 14",
                    line=dict(color='purple')
                ))
                
                # Add overbought/oversold lines
                fig_rsi.add_shape(
                    type="line",
                    x0=data.index[0],
                    y0=70,
                    x1=data.index[-1],
                    y1=70,
                    line=dict(color="red", dash="dash")
                )
                
                fig_rsi.add_shape(
                    type="line",
                    x0=data.index[0],
                    y0=30,
                    x1=data.index[-1],
                    y1=30,
                    line=dict(color="green", dash="dash")
                )
                
                fig_rsi.update_layout(title="RSI Indicator",
                                     xaxis_title="Date",
                                     yaxis_title="RSI Value",
                                     yaxis=dict(range=[0, 100]))
                
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Create MACD chart if available
            if 'macd_line' in data.columns and 'macd_signal' in data.columns:
                fig_macd = go.Figure()
                
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['macd_line'],
                    name="MACD Line",
                    line=dict(color='blue')
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['macd_signal'],
                    name="Signal Line",
                    line=dict(color='red')
                ))
                
                # Add histogram
                if 'macd_histogram' in data.columns:
                    fig_macd.add_trace(go.Bar(
                        x=data.index,
                        y=data['macd_histogram'],
                        name="Histogram",
                        marker_color='green'
                    ))
                
                fig_macd.update_layout(title="MACD Indicator",
                                      xaxis_title="Date",
                                      yaxis_title="MACD Value")
                
                st.plotly_chart(fig_macd, use_container_width=True)
