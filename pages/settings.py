import streamlit as st
import pandas as pd
from datetime import datetime

from utils.angel_api import AngelOneAPI
from utils.risk_management import RiskManager

def run():
    st.title("Settings")
    
    if not st.session_state.get("logged_in", False):
        st.warning("Please login to access settings.")
        return
    
    # Get API instance
    api = st.session_state.api
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs(["Account Settings", "Watchlist", "Trading Preferences", "Risk Management"])
    
    with tab1:
        st.header("Account Settings")
        
        # Display account information
        if st.session_state.api:
            user_profile = api.get_user_profile()
            
            if user_profile:
                st.subheader("User Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_input("Name", value=user_profile['name'], disabled=True)
                    st.text_input("Client ID", value=user_profile['client_id'], disabled=True)
                
                with col2:
                    st.text_input("Email", value=user_profile['email'], disabled=True)
                    st.text_input("Mobile", value=user_profile['mobile'], disabled=True)
                
                # API settings
                st.subheader("API Settings")
                
                api_key = st.text_input("API Key", value="*" * 8 + st.session_state.api_key[-4:], type="password", disabled=True)
                
                # Change TOTP Key
                if st.button("Update TOTP Key"):
                    with st.form("totp_form"):
                        new_totp = st.text_input("New TOTP Key", type="password")
                        submitted = st.form_submit_button("Update")
                        
                        if submitted and new_totp:
                            st.session_state.totp_key = new_totp
                            st.success("TOTP Key updated successfully!")
            else:
                st.error("Unable to fetch user profile.")
    
    with tab2:
        st.header("Watchlist")
        
        # Display current watchlist
        st.subheader("Current Watchlist")
        
        if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
            st.info("Your watchlist is empty. Add symbols below.")
        else:
            # Display watchlist items with remove option
            for i, symbol in enumerate(st.session_state.selected_symbols):
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"{symbol['symbol']}")
                
                with col2:
                    st.write(f"{symbol['exchange']}")
                
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
        
        auto_trading = st.toggle("Enable Auto Trading", value=st.session_state.get("auto_trading", False))
        if auto_trading != st.session_state.get("auto_trading", False):
            st.session_state.auto_trading = auto_trading
            if auto_trading:
                st.success("Auto Trading Enabled")
            else:
                st.warning("Auto Trading Disabled")
        
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
            # Save preferences to session state
            st.session_state.default_product = default_product
            st.session_state.default_order_type = default_order_type
            st.session_state.trade_frequency = trade_frequency
            st.session_state.trading_start_time = trading_start_time.strftime("%H:%M")
            st.session_state.trading_end_time = trading_end_time.strftime("%H:%M")
            
            st.success("Trading preferences saved successfully!")
    
    with tab4:
        st.header("Risk Management")
        
        # Risk level
        st.subheader("Risk Level")
        
        risk_level = st.select_slider(
            "Risk Level",
            options=["Low", "Medium", "High"],
            value=st.session_state.get("risk_level", "Medium")
        )
        
        if risk_level != st.session_state.get("risk_level", "Medium"):
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
            
            # Save to session state
            st.session_state.max_capital_per_trade = max_capital_per_trade
            st.session_state.daily_loss_limit = daily_loss_limit
            st.session_state.default_stop_loss = default_stop_loss
            st.session_state.default_take_profit = default_take_profit
            
            st.success("Risk settings applied successfully!")
