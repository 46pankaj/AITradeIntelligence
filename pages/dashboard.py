import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time # Added for time.sleep()

from utils.angel_api import AngelOneAPI

def run():
    st.title("Dashboard")

    if not st.session_state.get("logged_in", False):
        st.warning("Please login to view the dashboard.")
        return

    # Get API instance
    api = st.session_state.api

    # Create layout
    col1, col2, col3 = st.columns(3)

    # Market Overview
    st.header("Market Overview")

    # Create layout with columns
    col1, col2, col3 = st.columns(3)

    # Function to format price values
    def format_price(value):
        if isinstance(value, (int, float)):
            return f"₹{value:,.2f}"
        return "N/A"

    # Function to format percentage changes
    def format_change(value):
        if isinstance(value, (int, float)):
            return f"{value:+.2f}%"
        return "0.00%"

    # Auto-refresh container for real-time updates
    with st.empty():
        while True:
            try:
                # Force re-login before fetching data
                if not st.session_state.api.login():
                    st.error("Failed to authenticate with Angel One API")
                    time.sleep(5)
                    continue
                    
                # Get real-time market data with proper headers
                st.session_state.api.headers["Authorization"] = f"Bearer {st.session_state.api.jwt_token}"
                nifty_value = st.session_state.api.get_ltp("NSE", "NIFTY")
                bank_nifty_value = st.session_state.api.get_ltp("NSE", "BANKNIFTY") 
                sensex_value = st.session_state.api.get_ltp("BSE", "SENSEX")

                # Validate data before displaying
                if not any([nifty_value, bank_nifty_value, sensex_value]):
                    st.warning("Unable to fetch real-time prices. Retrying...")
                    time.sleep(5)
                    continue

            with col1:
                st.subheader("Nifty 50")
                if nifty_value:
                    st.metric(
                        "Live Price",
                        format_price(nifty_value.get('ltp')),
                        format_change(nifty_value.get('change_percent')),
                        delta_color="normal"
                    )

            with col2:
                st.subheader("Bank Nifty")
                if bank_nifty_value:
                    st.metric(
                        "Live Price",
                        format_price(bank_nifty_value.get('ltp')),
                        format_change(bank_nifty_value.get('change_percent')),
                        delta_color="normal"
                    )

            with col3:
                st.subheader("Sensex")
                if sensex_value:
                    st.metric(
                        "Live Price",
                        format_price(sensex_value.get('ltp')),
                        format_change(sensex_value.get('change_percent')),
                        delta_color="normal"
                    )

            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")
                time.sleep(5)
                continue
                
            # Add a short delay before next update
            time.sleep(5)  # Update every 5 seconds to avoid rate limits

    # Market Charts
    st.subheader("Market Performance")

    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["Nifty 50", "Bank Nifty", "Sensex"])

    with tab1:
        # Get Nifty data
        nifty_data = api.get_historical_data("NSE", "NIFTY", "1 day", 30)
        if nifty_data is not None and not nifty_data.empty:
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=nifty_data.index,
                open=nifty_data['open'],
                high=nifty_data['high'],
                low=nifty_data['low'],
                close=nifty_data['close']
            )])

            fig.update_layout(title='Nifty 50 (Daily)',
                             xaxis_title='Date',
                             yaxis_title='Price')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for Nifty 50.")

    with tab2:
        # Get Bank Nifty data
        banknifty_data = api.get_historical_data("NSE", "BANKNIFTY", "1 day", 30)
        if banknifty_data is not None and not banknifty_data.empty:
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=banknifty_data.index,
                open=banknifty_data['open'],
                high=banknifty_data['high'],
                low=banknifty_data['low'],
                close=banknifty_data['close']
            )])

            fig.update_layout(title='Bank Nifty (Daily)',
                             xaxis_title='Date',
                             yaxis_title='Price')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for Bank Nifty.")

    with tab3:
        # Get Sensex data
        sensex_data = api.get_historical_data("BSE", "SENSEX", "1 day", 30)
        if sensex_data is not None and not sensex_data.empty:
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=sensex_data.index,
                open=sensex_data['open'],
                high=sensex_data['high'],
                low=sensex_data['low'],
                close=sensex_data['close']
            )])

            fig.update_layout(title='Sensex (Daily)',
                             xaxis_title='Date',
                             yaxis_title='Price')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for Sensex.")

    # Watchlist
    st.header("Watchlist")

    # Get symbols from session state
    if not st.session_state.get("selected_symbols", []):
        st.info("Add symbols to your watchlist from the Settings page")
    else:
        # Create a table for watchlist
        watchlist_data = []
        for symbol in st.session_state.selected_symbols:
            symbol_data = api.get_ltp(symbol['exchange'], symbol['symbol'])
            if symbol_data:
                watchlist_data.append({
                    "Symbol": symbol['symbol'],
                    "Exchange": symbol['exchange'],
                    "LTP": symbol_data['ltp'],
                    "Change %": symbol_data['change_percent'],
                    "Volume": symbol_data.get('volume', 'N/A')
                })

        if watchlist_data:
            st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True)

    # Active Strategies
    st.header("Active Strategies")
    if not st.session_state.get("strategies", []):
        st.info("No active strategies. Create one from the Strategy Builder page.")
    else:
        strategy_data = []
        for strategy in st.session_state.strategies:
            strategy_data.append({
                "Strategy Name": strategy['name'],
                "Symbol": strategy['symbol'],
                "Type": strategy['type'],
                "Status": strategy['status'],
                "P&L": strategy['pnl']
            })

        if strategy_data:
            st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)

    # Recent Trades
    st.header("Recent Trades")
    if not st.session_state.get("trades", []):
        st.info("No recent trades.")
    else:
        trade_data = []
        for trade in st.session_state.trades:
            trade_data.append({
                "Symbol": trade['symbol'],
                "Type": trade['type'],
                "Quantity": trade['quantity'],
                "Price": trade['price'],
                "Status": trade['status'],
                "Timestamp": trade['timestamp']
            })

        if trade_data:
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)