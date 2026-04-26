import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
from src.rv_estimators import close_to_close_rv, parkinson_rv


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
    window = st.slider("RV window (trading days)", min_value=5, max_value=63, value=21, step=1)

    available_estimators = ["Close-to-Close", "Parkinson"]
    plotted_estimators = st.multiselect(
        "Estimators to plot",
        options=available_estimators,
        default=available_estimators,
    )
    stats_estimator = st.selectbox(
        "Stats estimator",
        options=available_estimators,
    )

@st.cache_data
def fetch_prices(ticker, start, end):
    parquet_path = Path(__file__).parent / "data" / f"{ticker.lower()}_prices.parquet"
    data = pd.read_parquet(parquet_path)
    data = data.loc[(data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
    return data

with st.spinner(f"Fetching {ticker} prices..."):
    prices = fetch_prices(ticker, start_date, end_date)

if prices.empty:
    st.error(f"No data returned for {ticker}. Check the ticker and date range.")
    st.stop()

estimators = {
    "Close-to-Close": close_to_close_rv(prices["Close"], window=window),
    "Parkinson": parkinson_rv(prices["High"], prices["Low"], window=window),
}

selected_rv = estimators[stats_estimator].dropna()

st.subheader(f"{stats_estimator} stats")
col1, col2, col3 = st.columns(3)
col1.metric("Latest RV", f"{selected_rv.iloc[-1] * 100:.1f}%")
col2.metric("Mean RV", f"{selected_rv.mean() * 100:.1f}%")
col3.metric("Max RV", f"{selected_rv.max() * 100:.1f}%")

st.subheader(f"{ticker} {window}-Day Realized Volatility")

if not plotted_estimators:
    st.info("Select at least one estimator to plot.")
else:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in plotted_estimators:
        series = estimators[name]
        ax.plot(series.index, series * 100, linewidth=1, label=f"{name} RV")
    ax.set_xlabel("Date")
    ax.set_ylabel("Realized Volatility (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with st.expander("Show price data"):
    st.dataframe(prices.tail(20))