import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
from src.rv_estimators import close_to_close_rv


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

@st.cache_data
def fetch_prices(ticker, start, end):
    import hashlib
    parquet_path = Path(__file__).parent / "data" / f"{ticker.lower()}_prices.parquet"
    
    with open(parquet_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    data = pd.read_parquet(parquet_path)
    
    st.write("=" * 50)
    st.write("STREAMLIT")
    st.write("=" * 50)
    st.write("File path:", str(parquet_path))
    st.write("File hash:", file_hash)
    st.write("Pandas version:", pd.__version__)
    st.write("Shape:", data.shape)
    st.write("Columns:", data.columns.tolist())
    st.write("Index name:", data.index.name)
    st.write("Index dtype:", str(data.index.dtype))
    st.write("First 3 rows:", data.head(3))
    st.write("Close mean:", data["Close"].mean())
    
    if pd.Timestamp("2024-12-31") in data.index:
        st.write("Close at 2024-12-31:", data.loc["2024-12-31", "Close"])
    else:
        st.write("2024-12-31 not in index")
    
    data = data.loc[(data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
    return data

with st.spinner(f"Fetching {ticker} prices..."):
    prices = fetch_prices(ticker, start_date, end_date)

if prices.empty:
    st.error(f"No data returned for {ticker}. Check the ticker and date range.")
    st.stop()

rv = close_to_close_rv(prices["Close"], window=window)

st.write("STREAMLIT RV")
st.write("Window used:", window)
st.write("RV shape:", rv.shape)
st.write("RV mean:", rv.mean())
st.write("RV max:", rv.max())
if pd.Timestamp("2024-12-31") in rv.index:
    st.write("RV at 2024-12-31:", rv.loc["2024-12-31"])
if pd.Timestamp("2020-03-20") in rv.index:
    st.write("RV at 2020-03-20:", rv.loc["2020-03-20"])

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