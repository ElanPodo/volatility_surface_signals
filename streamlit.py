import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
from src.rv_estimators import close_to_close_rv

project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Volatility Surface Signals", layout="wide")

st.title("Volatility Surface Signals")
st.markdown(
    "Realized volatility analysis for equity index options. "
    "Enter a ticker and date range to compute rolling close-to-close RV."
)

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="SPY").upper()
    start_date = st.date_input("Start date", value=pd.Timestamp("2020-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp("2025-12-31"))
    window = st.slider("RV window (trading days)", min_value=5, max_value=63, value=13, step=1)

@st.cache_data
def fetch_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns.name = None
    return data

with st.spinner(f"Fetching {ticker} prices..."):
    prices = fetch_prices(ticker, start_date, end_date)

if prices.empty:
    st.error(f"No data returned for {ticker}. Check the ticker and date range.")
    st.stop()

rv = close_to_close_rv(prices["Close"], window=window)

col1, col2, col3 = st.columns(3)
col1.metric("Latest RV", f"{rv.iloc[-1] * 100:.1f}%")
col2.metric("Mean RV", f"{rv.mean() * 100:.1f}%")
col3.metric("Max RV", f"{rv.max() * 100:.1f}%")

st.subheader(f"{ticker} {window}-Day Close-to-Close Realized Volatility")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(rv.index, rv * 100, linewidth=1)
ax.set_xlabel("Date")
ax.set_ylabel("Realized Volatility (%)")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

with st.expander("Show price data"):
    st.dataframe(prices.tail(20))