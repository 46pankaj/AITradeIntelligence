import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils.angel_api import AngelOneAPI

def run():
    st.title("Performance Monitoring")
    
    if not st.session_state.get("logged_in", False):
        st.warning("Please login to view performance monitoring.")
        return
    
    # Get API instance
    api = st.session_state.api
    
    # Display tabs for different views
    tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Strategy Performance", "Trade History"])
    
    with tab1:
        st.header("Portfolio Overview")
        
        # Portfolio Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get portfolio data from API
        portfolio_data = api.get_portfolio()
        
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
        if not st.session_state.get("strategies", []):
            st.info("No strategies available for analysis.")
        else:
            selected_strategy = st.selectbox(
                "Select Strategy to Analyze",
                options=[strategy['name'] for strategy in st.session_state.strategies],
                key="strategy_analysis_selector"
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
                dates = pd.date_range(start=strategy.get('created_at', datetime.now().strftime("%Y-%m-%d %H:%M:%S")), periods=10)
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
                    "Date": strategy.get('created_at', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "Symbol": strategy['symbol'],
                    "Type": "BUY",
                    "Quantity": 10,
                    "Price": 1500,
                    "P&L": 750,
                    "Status": "Closed"
                })
                
                trade_data.append({
                    "Date": (datetime.strptime(strategy.get('created_at', datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "%Y-%m-%d %H:%M:%S") + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
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
                
                # Delete strategy button
                if st.button("Delete Strategy", key="performance_delete_strategy"):
                    from utils.data_manager import DataManager
                    # Get strategy name for feedback message
                    strategy_name = strategy['name']
                    # Find the index of the strategy in the session state list
                    strategy_index = next((i for i, s in enumerate(st.session_state.strategies) if s['name'] == strategy_name), None)
                    if strategy_index is not None:
                        # Remove strategy from session state
                        st.session_state.strategies.pop(strategy_index)
                        # Remove from data manager
                        data_manager = DataManager(api)
                        data_manager.delete_strategy(strategy_name)
                        st.success(f"Strategy {strategy_name} deleted successfully!")
                        st.rerun()
    
    with tab3:
        st.header("Trade History")
        
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30), key="trade_start_date")
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now(), key="trade_end_date")
        
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
                st.info("No trades found for the selected date range.")
        else:
            st.info("No trade history available.")
