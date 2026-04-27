import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.rv_estimators import close_to_close_rv, parkinson_rv, garman_klass_rv, yang_zhang_rv, parkinson_total_rv, garman_klass_total_rv


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

    available_estimators_rv = ["Close-to-Close", "Parkinson", "Garman-Klass", "Yang-Zhang"]
    plotted_estimators_rv = st.multiselect(
        "RV estimators to plot",
        options=available_estimators_rv,
        default=available_estimators_rv,
    )

    available_estimators_frv = ["Close-to-Close", "Yang-Zhang", "YZ Parkinson", "YZ Garman-Klass"]
    plotted_estimators_frv = st.multiselect(
        "Forward RV estimators to plot",
        options=available_estimators_frv,
        default=available_estimators_frv,
    )

    available_vrp_estimator = ["Close-to-Close", 'Yang-Zhang', 'Yang-Zhang Adjusted Parkinson', 'Yang-Zhang Adjusted Garman-Klass']


@st.cache_data
def fetch_prices(ticker, start, end):
    parquet_path = Path(__file__).parent / "data" / f"{ticker.lower()}_prices.parquet"
    data = pd.read_parquet(parquet_path)
    data = data.loc[(data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
    return data


@st.cache_data
def fetch_vix(start, end):
    parquet_path = Path(__file__).parent / "data" / "vix_prices.parquet"
    data = pd.read_parquet(parquet_path)
    data.index.name = "date"
    data = data.loc[(data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
    return data


with st.spinner(f"Fetching {ticker} prices..."):
    prices = fetch_prices(ticker, start_date, end_date)

if prices.empty:
    st.error(f"No data returned for {ticker}. Check the ticker and date range.")
    st.stop()

estimators_tab1 = {
    "Close-to-Close": close_to_close_rv(prices["Close"], window=window),
    "Parkinson": parkinson_rv(prices["High"], prices["Low"], window=window),
    "Garman-Klass": garman_klass_rv(prices["High"], prices["Low"], prices["Open"], prices["Close"], window=window),
    "Yang-Zhang": yang_zhang_rv(prices["High"], prices["Low"], prices["Open"], prices["Close"], window=window),
}

estimators_tab2 = {
    "Close-to-Close": close_to_close_rv(prices["Close"], window=window),
    "Yang-Zhang": yang_zhang_rv(prices["High"], prices["Low"], prices["Open"], prices["Close"], window=window),
    "Yang-Zhang Adjusted Garman-Klass": garman_klass_total_rv(prices["High"], prices["Low"], prices["Open"], prices["Close"], window=window),
    "Yang-Zhang Adjusted Parkinson": parkinson_total_rv(prices["High"], prices["Low"], prices["Open"], prices["Close"], window=window),
}

tab1, tab2 = st.tabs(["RV Estimators", "Forward RV vs VIX"])

with tab1:
    stats_estimator = st.selectbox(
        "Stats estimator",
        options=available_estimators_rv,
        index=available_estimators_rv.index("Yang-Zhang"),
        help="Drives the statistics for each estimator present in the plot below."
    )

    selected_rv = estimators_tab1[stats_estimator].dropna()

    st.subheader(f"{stats_estimator} stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest RV", f"{selected_rv.iloc[-1] * 100:.1f}%")
    col2.metric("Mean RV", f"{selected_rv.mean() * 100:.1f}%")
    col3.metric("Max RV", f"{selected_rv.max() * 100:.1f}%")

    st.subheader(f"{ticker} {window}-Day Realized Volatility")

    if not plotted_estimators_rv:
        st.info("Select at least one estimator to plot.")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in plotted_estimators_rv:
            series = estimators_tab1[name]
            ax.plot(series.index, series * 100, linewidth=1, label=f"{name} RV")
        ax.set_xlabel("Date")
        ax.set_ylabel("Realized Volatility (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with st.expander("Show price data"):
        st.dataframe(prices.tail(20))

with tab2:
    st.subheader("Forward Realized Volatility vs VIX")
    st.markdown(
        f"VIX(t) is the market's IV forecast for [t, t+30 calendar days]. "
        f"Forward RV(t) is the RV that actually realized over [t, t+{window} trading days]. "
        f"The spread VIX − forward RV is the realized variance risk premium."
    )

    if ticker != "SPY":
        st.warning(
            f"Forward RV vs VIX comparison is only meaningful for SPY (VIX is SPX-implied vol). "
            f"Current ticker: {ticker}. Switch to SPY in the sidebar to view this chart."
        )
    elif not plotted_estimators_frv:
        st.info("Select at least one estimator in the sidebar to plot.")
    else:
        with st.spinner("Fetching VIX..."):
            vix = fetch_vix(start_date, end_date)

        vix_adjusted = vix["Close"]
        vrp_estimator = st.selectbox("VRP estimator",
        options=available_vrp_estimator,
        index=available_vrp_estimator.index("Yang-Zhang"),
        help="Drives the VRP metrics and spread subplot below.")
        forward_rv_stats = estimators_tab2[vrp_estimator].shift(-window) * 100
        vrp = (vix_adjusted - forward_rv_stats).dropna()

        st.markdown(f"**Variance risk premium stats (using {vrp_estimator})**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean VRP", f"{vrp.mean():.2f} vol pts")
        m2.metric("% days VRP > 0", f"{(vrp > 0).mean() * 100:.1f}%")
        m3.metric("Worst day", f"{vrp.min():.2f}")
        m4.metric("Corr(VIX, fwd RV)", f"{vix_adjusted.corr(forward_rv_stats):.2f}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(vix_adjusted.index, vix_adjusted, linewidth=1.2, label="VIX)", color="black")
        for name in available_vrp_estimator:
            forward_rv = estimators_tab2[name].shift(-window) * 100
            ax.plot(forward_rv.index, forward_rv, linewidth=1, alpha=0.8, label=f"Forward {name} RV")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"**VRP spread: VIX − Forward {vrp_estimator} RV**")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.fill_between(vrp.index, vrp, 0, where=(vrp >= 0), alpha=0.4, color="green", label="VIX > RV (vol overpriced)")
        ax2.fill_between(vrp.index, vrp, 0, where=(vrp < 0), alpha=0.4, color="red", label="VIX < RV (vol underpriced)")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.axhline(vrp.mean(), color="blue", linewidth=0.8, linestyle="--", label=f"Mean ({vrp.mean():.2f})")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("VRP (vol pts)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

        st.caption(
            f"Note: the last {window} trading days have NaN forward RV (the future hasn't realized yet), "
            f"so RV lines end before the VIX line."
        )