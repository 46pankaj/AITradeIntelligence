import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
from datetime import datetime

from utils.angel_api import AngelOneAPI
from utils.trade_executor import TradeExecutor
from utils.data_manager import DataManager

def run():
    st.title("Manual Trade Execution with AI Strategies")
    
    st.write("""
    This page allows you to manually execute trades based on AI-generated strategies.
    You can choose between completely manual trading or executing trades based on the AI-generated strategies.
    """)
    
    if not st.session_state.get("logged_in", False):
        st.warning("Please login to execute trades.")
        return
    
    # Get API instance
    api = st.session_state.api
    
    # Get data manager
    data_manager = st.session_state.data_manager if "data_manager" in st.session_state else None
    
    # Auto trading status check
    if 'auto_trader' in st.session_state:
        auto_trader = st.session_state.auto_trader
        is_auto_trading = auto_trader.is_auto_trading()
        
        if is_auto_trading:
            auto_mode = "AI Strategy Generation" if auto_trader.trading_mode == "ai_strategy_only" else "Fully Automated Trading"
            st.success(f"🟢 Auto Trading is active in {auto_mode} mode")
        else:
            st.info("ℹ️ Auto Trading is inactive. You can enable it from the 'Auto Trading' tab below to generate AI strategies automatically.")
    
    # Create tabs for different trading interfaces
    tab1, tab2, tab3, tab4 = st.tabs(["Manual Trading", "AI Strategy Execution", "Auto Trading", "Active Orders"])
    
    # Tab 1: Manual Trading
    with tab1:
        st.header("Manual Trade Entry")
        
        # Symbol input
        symbol = st.text_input("Symbol", value="NIFTY", key="manual_symbol")
        
        # Exchange selection
        exchange = st.selectbox("Exchange", ["NSE", "BSE"], key="manual_exchange")
        
        # Contract type
        product_type = st.selectbox("Product Type", ["DELIVERY", "INTRADAY", "MARGIN"], key="manual_product_type")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction type
            transaction_type = st.selectbox("Transaction Type", ["BUY", "SELL"], key="manual_transaction_type")
            
            # Order type
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], key="manual_order_type")
            
            # Get current price if available
            if api and symbol and exchange:
                try:
                    price_data = api.get_ltp(exchange, symbol)
                    if price_data and "ltp" in price_data:
                        current_price = price_data["ltp"]
                        st.write(f"Current Price: ₹{current_price:,.2f}")
                    else:
                        current_price = None
                except Exception as e:
                    st.error(f"Error getting current price: {str(e)}")
                    current_price = None
            else:
                current_price = None
            
        with col2:
            # Quantity input
            quantity = st.number_input("Quantity", min_value=1, value=1, key="manual_quantity")
            
            # Price inputs based on order type
            if order_type == "LIMIT" or order_type == "SL":
                price = st.number_input("Price (₹)", min_value=0.05, step=0.05, 
                                        value=current_price if current_price else 0.0,
                                        key="manual_price")
            else:
                price = 0
                
            if order_type == "SL" or order_type == "SL-M":
                trigger_price = st.number_input("Trigger Price (₹)", min_value=0.05, step=0.05,
                                               value=current_price*0.99 if current_price else 0.0,
                                               key="manual_trigger_price")
            else:
                trigger_price = 0
        
        # Risk management
        with st.expander("Risk Management", expanded=False):
            take_profit = st.number_input("Take Profit %", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            stop_loss = st.number_input("Stop Loss %", min_value=0.0, max_value=100.0, value=3.0, step=0.5)
            
            st.info("""
            Setting Take Profit and Stop Loss values will create additional orders to automatically
            book profits or limit losses once your position reaches the specified price points.
            """)
        
        # Execute trade button
        if st.button("Execute Trade", type="primary", key="manual_execute"):
            if api and symbol and exchange and order_type and quantity > 0:
                # Create trade executor if needed
                trade_executor = TradeExecutor(api)
                
                try:
                    trade_result = trade_executor.execute_trade(
                        symbol=symbol,
                        exchange=exchange,
                        transaction_type=transaction_type,
                        quantity=quantity,
                        order_type=order_type,
                        price=price,
                        trigger_price=trigger_price,
                        product_type=product_type,
                        take_profit=take_profit,
                        stop_loss=stop_loss
                    )
                    
                    if trade_result.get("success", False):
                        st.success(f"Trade executed successfully! Order ID: {trade_result.get('order_id', 'N/A')}")
                        
                        # Save order in session state
                        if "orders" not in st.session_state:
                            st.session_state.orders = []
                            
                        st.session_state.orders.append({
                            "order_id": trade_result.get("order_id", "N/A"),
                            "symbol": symbol,
                            "exchange": exchange,
                            "transaction_type": transaction_type,
                            "quantity": quantity,
                            "price": price if order_type == "LIMIT" else "MARKET",
                            "order_type": order_type,
                            "product_type": product_type,
                            "status": "OPEN",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        st.error(f"Trade execution failed: {trade_result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error executing trade: {str(e)}")
            else:
                st.error("Please ensure all trade details are filled correctly.")
                
    # Tab 2: AI Strategy Execution
    with tab2:
        st.header("Execute AI-Generated Strategies")
        
        # Check if there are any strategies
        if 'strategies' not in st.session_state or not st.session_state.strategies:
            st.warning("No AI strategies found. Please generate strategies first using the Strategy Builder.")
            
            st.write("""
            To generate AI strategies:
            1. Go to the **Strategy Builder** page to create strategies manually, or
            2. Enable **AI Strategy Generation** from the Auto Trading tab
            """)
        else:
            # Filter for active strategies
            active_strategies = [s for s in st.session_state.strategies if s.get('status', '') == 'Active']
            
            if not active_strategies:
                st.warning("No active strategies found. Please activate strategies in the Strategy Builder.")
            else:
                st.write(f"Found {len(active_strategies)} active AI-generated strategies.")
                
                # Display strategies
                for i, strategy in enumerate(active_strategies):
                    with st.container():
                        st.subheader(f"{strategy.get('name', 'Unnamed Strategy')}")
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Symbol**: {strategy.get('symbol', '')}")
                            st.write(f"**Exchange**: {strategy.get('exchange', '')}")
                            st.write(f"**Type**: {strategy.get('type', 'Manual')}")
                            
                            # Show indicators
                            indicators = strategy.get('indicators', [])
                            if indicators:
                                st.write("**Indicators:**")
                                for ind in indicators:
                                    st.write(f"- {ind.get('type', '')}: {ind.get('params', {})}")
                        
                        with col2:
                            # Entry/exit conditions
                            entry_conditions = strategy.get('entry_conditions', [])
                            if entry_conditions:
                                st.write("**Entry:**")
                                for cond in entry_conditions:
                                    st.write(f"- {cond}")
                            
                            exit_conditions = strategy.get('exit_conditions', [])
                            if exit_conditions:
                                st.write("**Exit:**")
                                for cond in exit_conditions:
                                    st.write(f"- {cond}")
                        
                        with col3:
                            # Risk parameters
                            st.write(f"**Take Profit**: {strategy.get('take_profit', 0)}%")
                            st.write(f"**Stop Loss**: {strategy.get('stop_loss', 0)}%")
                            
                            # Recent recommendations
                            recommendations = strategy.get('recommendations', [])
                            if recommendations:
                                latest = recommendations[-1]
                                action = latest.get('action', 'HOLD')
                                confidence = latest.get('confidence', 0)
                                
                                if action == 'BUY':
                                    st.success(f"**Recommendation**: {action} (Confidence: {confidence}%)")
                                elif action == 'SELL':
                                    st.error(f"**Recommendation**: {action} (Confidence: {confidence}%)")
                                else:
                                    st.info(f"**Recommendation**: {action} (Confidence: {confidence}%)")
                        
                        # Execute strategy button
                        if st.button(f"Execute Strategy", key=f"execute_strategy_{i}"):
                            if api:
                                # Create trade executor if needed
                                trade_executor = TradeExecutor(api)
                                
                                try:
                                    # Get latest recommendation
                                    if recommendations:
                                        latest = recommendations[-1]
                                        action = latest.get('action', 'HOLD')
                                        
                                        if action == 'BUY' or action == 'SELL':
                                            # Execute trade based on strategy
                                            execution_result = trade_executor.execute_strategy(strategy)
                                            
                                            if execution_result.get("success", False):
                                                st.success(f"Strategy executed successfully! Order ID: {execution_result.get('order_id', 'N/A')}")
                                                
                                                # Update strategy with executed trade
                                                if data_manager:
                                                    strategy['trades'] = strategy.get('trades', [])
                                                    strategy['trades'].append({
                                                        "order_id": execution_result.get("order_id", "N/A"),
                                                        "transaction_type": action,
                                                        "quantity": execution_result.get("quantity", 0),
                                                        "price": execution_result.get("price", 0),
                                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        "status": "OPEN"
                                                    })
                                                    
                                                    # Update in session state
                                                    st.session_state.strategies[i] = strategy
                                                    
                                                    # Update in database
                                                    data_manager.update_strategy(strategy)
                                            else:
                                                st.error(f"Strategy execution failed: {execution_result.get('message', 'Unknown error')}")
                                        else:
                                            st.warning(f"No actionable recommendation (current: {action}). Strategy requires BUY or SELL signal.")
                                    else:
                                        st.warning("No recommendations available for this strategy.")
                                except Exception as e:
                                    st.error(f"Error executing strategy: {str(e)}")
                            else:
                                st.error("API not initialized. Please login first.")
                                
                        st.divider()
    
    # Tab 3: Auto Trading
    with tab3:
        # Use the comprehensive Auto Trading page implementation
        from pages.auto_trading import display_auto_trading
        display_auto_trading()
    
    # Tab 4: Active Orders
    with tab4:
        st.header("Active Orders & Positions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Refresh button
            refresh_button = st.button("Refresh Orders", key="refresh_orders")
        
        with col2:
            # Auto-refresh toggle
            auto_refresh = st.toggle("Auto-refresh", value=True, key="auto_refresh_orders")
            if auto_refresh:
                st.write("Auto-refreshing every 60 seconds")
                
        # Orders section
        st.subheader("Open Orders")
        
        # Fetch orders
        orders = []
        if api:
            try:
                # Get orders from the API
                api_orders = api.get_order_book()
                
                if api_orders:
                    # Convert to format for display
                    for order in api_orders:
                        orders.append({
                            "order_id": order.get("order_id", "N/A"),
                            "symbol": order.get("symbol", ""),
                            "exchange": order.get("exchange", ""),
                            "transaction_type": order.get("transaction_type", ""),
                            "quantity": order.get("quantity", 0),
                            "price": order.get("price", 0),
                            "order_type": order.get("order_type", ""),
                            "product_type": order.get("product_type", ""),
                            "status": order.get("status", ""),
                            "timestamp": order.get("order_timestamp", "")
                        })
            except Exception as e:
                st.error(f"Error fetching orders: {str(e)}")
                
                # Use orders from session state if available
                if "orders" in st.session_state:
                    orders = st.session_state.orders
        
        # Display orders
        if orders:
            # Convert to DataFrame for display
            orders_df = pd.DataFrame(orders)
            st.dataframe(orders_df, hide_index=True)
            
            # Add button to cancel selected orders
            selected_order = st.selectbox("Select Order to Cancel", options=[o.get("order_id", "N/A") for o in orders])
            
            if st.button("Cancel Order", key="cancel_order"):
                if api and selected_order:
                    try:
                        cancel_result = api.cancel_order(selected_order)
                        
                        if cancel_result:
                            st.success(f"Order {selected_order} cancelled successfully!")
                            
                            # Update order status in session state
                            if "orders" in st.session_state:
                                for order in st.session_state.orders:
                                    if order.get("order_id") == selected_order:
                                        order["status"] = "CANCELLED"
                        else:
                            st.error(f"Failed to cancel order {selected_order}")
                    except Exception as e:
                        st.error(f"Error cancelling order: {str(e)}")
        else:
            st.info("No open orders found.")
            
        # Positions section
        st.subheader("Open Positions")
        
        # Fetch positions
        positions = []
        if api:
            try:
                # Get positions from the API
                api_positions = api.get_positions()
                
                if api_positions:
                    # Convert to format for display
                    for pos in api_positions:
                        positions.append({
                            "symbol": pos.get("symbol", ""),
                            "exchange": pos.get("exchange", ""),
                            "product_type": pos.get("product_type", ""),
                            "quantity": pos.get("quantity", 0),
                            "average_price": pos.get("average_price", 0),
                            "ltp": pos.get("ltp", 0),
                            "pnl": pos.get("pnl", 0),
                            "pnl_percent": pos.get("pnl_percent", 0)
                        })
            except Exception as e:
                st.error(f"Error fetching positions: {str(e)}")
        
        # Display positions
        if positions:
            # Convert to DataFrame for display
            positions_df = pd.DataFrame(positions)
            
            # Add PnL percent column
            if "pnl_percent" not in positions_df:
                positions_df["pnl_percent"] = positions_df.apply(
                    lambda row: (row["ltp"] / row["average_price"] - 1) * 100 if row["average_price"] > 0 else 0,
                    axis=1
                )
            
            # Style the dataframe
            styled_df = positions_df.style.apply(
                lambda row: ["background-color: #E6F9E8" if val > 0 else "background-color: #F9E6E6" if val < 0 else "" for val in row],
                subset=["pnl", "pnl_percent"]
            )
            
            st.dataframe(styled_df, hide_index=True)
            
            # Add button to close selected position
            if positions:
                selected_position = st.selectbox("Select Position to Close", options=[f"{p.get('symbol')} ({p.get('exchange')})" for p in positions])
                
                if st.button("Close Position", key="close_position"):
                    if api and selected_position:
                        symbol = selected_position.split(" ")[0]
                        exchange = selected_position.split("(")[1].split(")")[0]
                        
                        # Find position
                        for pos in positions:
                            if pos.get("symbol") == symbol and pos.get("exchange") == exchange:
                                try:
                                    # Create trade executor if needed
                                    trade_executor = TradeExecutor(api)
                                    
                                    # Determine transaction type (opposite of position)
                                    transaction_type = "SELL" if pos.get("quantity", 0) > 0 else "BUY"
                                    
                                    # Execute closing trade
                                    close_result = trade_executor.execute_trade(
                                        symbol=symbol,
                                        exchange=exchange,
                                        transaction_type=transaction_type,
                                        quantity=abs(pos.get("quantity", 0)),
                                        order_type="MARKET",
                                        price=0,
                                        trigger_price=0,
                                        product_type=pos.get("product_type", "INTRADAY")
                                    )
                                    
                                    if close_result.get("success", False):
                                        st.success(f"Position closed successfully! Order ID: {close_result.get('order_id', 'N/A')}")
                                    else:
                                        st.error(f"Failed to close position: {close_result.get('message', 'Unknown error')}")
                                except Exception as e:
                                    st.error(f"Error closing position: {str(e)}")
                                break
        else:
            st.info("No open positions found.")
            
if __name__ == "__main__":
    run()