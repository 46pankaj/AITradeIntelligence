# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import datetime
from controller import run_full_pipeline

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📈 AI-Based Trading Dashboard")

# Sidebar Controls
st.sidebar.header("Settings")
live_mode = st.sidebar.toggle("Live Trading Mode", value=False)
symbols = st.sidebar.multiselect("Select Symbols", ["RELIANCE-EQ", "INFY-EQ", "TCS-EQ", "HDFCBANK-EQ"], default=["RELIANCE-EQ"])
risk_percent = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Trading Pipeline")
st.sidebar.markdown("---")
st.sidebar.subheader("Execution Info")
st.sidebar.markdown(f"**Mode:** {'Live' if live_mode else 'Paper'}")

# Execute
if run_button:
    st.subheader("🔄 Execution Log")
    with st.spinner("Running trading pipeline..."):
        result = run_full_pipeline(symbols=symbols, live=live_mode, risk=risk_percent)
    st.success("Pipeline completed.")

    st.subheader("📋 Trade Decisions")
    if result:
        st.dataframe(pd.DataFrame(result))
    else:
        st.info("No trades were triggered.")

# Simulated Current Positions
st.subheader("📊 Current Positions (Simulated)")
positions_data = pd.DataFrame({
    "Symbol": ["RELIANCE-EQ", "TCS-EQ"],
    "Quantity": [10, 5],
    "Avg Price": [2500.0, 3500.0],
    "Last Price": [2525.0, 3550.0],
    "PnL": [250.0, 250.0]
})
st.dataframe(positions_data)

# Simulated Recent Signals
st.subheader("🔔 Recent AI Signals")
signals_data = pd.DataFrame({
    "Symbol": ["INFY-EQ", "HDFCBANK-EQ"],
    "Signal": ["BUY", "SELL"],
    "Confidence": [0.81, 0.73],
    "Time": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 2
})
st.dataframe(signals_data)

st.markdown("---")
st.caption("AI Trading System | Powered by Streamlit")
