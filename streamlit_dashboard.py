import pandas as pd
import streamlit as st

from backtest_engine import compute_sharpe_ratio, simulate_strategy

st.set_page_config(layout="wide")
st.title("📈 ES Futures Strategy Dashboard")

with st.sidebar:
    st.header("Strategy Config")
    rsi_thresh = st.slider("RSI Threshold", 50, 90, 60)
    mult = st.slider("Return Multiplier", 0.5, 2.0, 1.0)
    sl = st.slider("Stop Loss", 0.01, 0.1, 0.05)
    tp = st.slider("Take Profit", 0.01, 0.2, 0.1)

    if st.button("Run Strategy"):
        df = pd.read_csv("es_fut_combined.csv")
        df["Time"] = pd.to_datetime(df["Time"])

        pnl = simulate_strategy(df, rsi_thresh, mult, sl, tp)
        sharpe = compute_sharpe_ratio(pnl)

        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.line_chart(pd.Series(pnl).cumsum(), height=300)
