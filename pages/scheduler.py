import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from utils.scheduler import TradingScheduler

logger = logging.getLogger(__name__)

def display_scheduler():
    st.title("Trading Schedule")

    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = TradingScheduler()

    scheduler = st.session_state.scheduler

    # Trading Days Selection
    st.subheader("Trading Days")

    days = {
        'monday': 'Monday',
        'tuesday': 'Tuesday', 
        'wednesday': 'Wednesday',
        'thursday': 'Thursday',
        'friday': 'Friday'
    }

    current_schedule = scheduler.get_schedule()

    col1, col2, col3 = st.columns(3)

    for i, (day_key, day_name) in enumerate(days.items()):
        with [col1, col2, col3][i % 3]:
            enabled = st.checkbox(
                day_name,
                value=current_schedule.get(day_key, {}).get('enabled', False),
                key=f"day_{day_key}"
            )
            current_schedule[day_key] = current_schedule.get(day_key, {})
            current_schedule[day_key]['enabled'] = enabled

    # Trading Hours
    st.subheader("Trading Hours")

    col1, col2 = st.columns(2)

    with col1:
        start_time = st.time_input(
            "Start Time",
            value=datetime.strptime('09:15', '%H:%M').time(),
            key="trading_start_time"
        )

    with col2:
        end_time = st.time_input(
            "End Time",
            value=datetime.strptime('15:30', '%H:%M').time(),
            key="trading_end_time"
        )

    if st.button("Save Schedule"):
        for day_key in days:
            scheduler.update_schedule(
                day=day_key,
                enabled=current_schedule[day_key]['enabled'],
                start_time=start_time.strftime('%H:%M'),
                end_time=end_time.strftime('%H:%M')
            )
        st.success("Trading schedule updated successfully!")

    # API Credentials configuration
    st.subheader("API Credentials for Automatic Login")

    # Check if credentials are already stored
    credentials = scheduler.get_credentials()

    if credentials:
        st.info("API credentials are already stored. Fill the form below to update them.")

    # Credential form
    with st.form("credential_form"):
        username = st.text_input("API Username", value=credentials.get('username', '') if credentials else '')
        password = st.text_input("API Password", type="password")
        totp_secret = st.text_input("TOTP Secret (if applicable)", type="password",
                                   value=credentials.get('totp_secret', '') if credentials else '')

        submitted = st.form_submit_button("Save Credentials")

        if submitted:
            if username and password:
                if scheduler.store_credentials(username, password, totp_secret if totp_secret else None):
                    st.success("API credentials saved successfully")
                else:
                    st.error("Failed to save API credentials")
            else:
                st.error("Username and password are required")

    # Strategy selection
    st.subheader("Strategies for Automatic Trading")

    # Check for strategies in session state directly
    if 'strategies' in st.session_state:
        strategies = st.session_state.strategies
        active_strategies = [s for s in strategies if s.get('status') == 'Active']

        if not active_strategies:
            st.warning("No active strategies found. Please create and activate strategies first.")
        else:
            # Display selected strategies
            selected_strategy_ids = current_schedule.get('strategies', [])
            selected_strategies = [s for s in active_strategies if s.get('id', s.get('name', '')) in selected_strategy_ids]

            if selected_strategies:
                st.write("Selected strategies for automatic trading:")

                strategy_df = pd.DataFrame([{
                    'Strategy Name': s.get('name', 'Unnamed Strategy'),
                    'Symbol': s.get('symbol', ''),
                    'Exchange': s.get('exchange', ''),
                    'Type': s.get('type', 'Manual')
                } for s in selected_strategies])

                st.dataframe(strategy_df, hide_index=True)

            # Strategy selection
            st.write("Select strategies for automatic trading:")

            for strategy in active_strategies:
                # Use name as ID if actual ID is not available
                strategy_id = strategy.get('id', strategy.get('name', ''))
                strategy_name = strategy.get('name', 'Unnamed Strategy')
                strategy_symbol = strategy.get('symbol', '')

                is_selected = strategy_id in selected_strategy_ids

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**{strategy_name}** ({strategy_symbol})")

                with col2:
                    if is_selected:
                        if st.button("Remove", key=f"remove_{strategy_id}"):
                            scheduler.remove_strategy_from_schedule(strategy_id)
                            st.success(f"Removed {strategy_name} from automatic trading")
                            st.rerun()
                    else:
                        if st.button("Add", key=f"add_{strategy_id}"):
                            scheduler.add_strategy_to_schedule(strategy_id)
                            st.success(f"Added {strategy_name} to automatic trading")
                            st.rerun()

                st.divider()
    else:
        st.warning("No strategies created yet. Go to the Strategy Builder page to create strategies.")


def run():
    display_scheduler()

if __name__ == "__main__":
    run()