import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.rv_estimators import (close_to_close_rv, parkinson_rv, garman_klass_rv, yang_zhang_rv, parkinson_total_rv, 
                               garman_klass_total_rv, har_rv)


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
def fetch_optionsdx(start, end):
    parquet_path = Path(__file__).parent / "data" / "spy_atm_iv_daily.parquet"
    data = pd.read_parquet(parquet_path)
    data.index.name = "date"
    data = data.loc[(data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
    return data

@st.cache_data
def fit_har_and_align(ticker, start, end):
    if ticker != "SPY":
        return None, None
    
    sp = fetch_prices(ticker, start, end)
    dx = fetch_optionsdx(start, end)

    iv_daily = dx['iv']
    
    model, har = har_rv(sp)
    har['HAR vol'] = np.sqrt(har['RV Forecast(t+1)'])
    
    plot_df = pd.DataFrame({
        'HAR Forecast Vol': har['HAR vol'],
        'Implied Vol': iv_daily,
    }).dropna()
    plot_df['IV_minus_RV'] = plot_df['Implied Vol'] - plot_df['HAR Forecast Vol']
    
    return plot_df, str(model.summary())

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

tab1, tab2, tab3 = st.tabs(["RV Estimators", "Forward RV vs VIX", "HAR-RV vs IV"])

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
        with st.spinner("Fetching Options Chain..."):
            dx = fetch_optionsdx(start_date, end_date)

        atm_30d = dx[(dx['moneyness'].between(0.98, 1.02)) & (dx['dte'].between(20, 40))]
        iv_daily = atm_30d.groupby('date')['vol'].mean()
        vrp_estimator = st.selectbox("VRP estimator",
        options=available_vrp_estimator,
        index=available_vrp_estimator.index("Yang-Zhang"),
        help="Drives the VRP metrics and spread subplot below.")
        forward_rv_stats = estimators_tab2[vrp_estimator].shift(-window) * 100
        vrp = (iv_daily - forward_rv_stats).dropna()

        st.markdown(f"**Variance risk premium stats (using {vrp_estimator})**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean VRP", f"{vrp.mean():.2f} vol pts")
        m2.metric("% days VRP > 0", f"{(vrp > 0).mean() * 100:.1f}%")
        m3.metric("Worst day", f"{vrp.min():.2f}")
        m4.metric("Corr(VIX, fwd RV)", f"{iv_daily.corr(forward_rv_stats):.2f}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(iv_daily.index, iv_daily, linewidth=1.2, label="VIX)", color="black")
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

with tab3:
    st.subheader("HAR-RV Forecast vs Implied Volatility")
    st.markdown(
        "HAR-RV produces a one-day-ahead forecast of realized volatility from a heterogeneous "
        "autoregression on daily, weekly (5d), and monthly (22d) RV components. The spread "
        "between ATM ~30-day implied volatility and the HAR forecast is the variance risk "
        "premium signal that drives the backtest."
    )

    if ticker != "SPY":
        st.warning(
            f"HAR-RV vs IV requires SPY option chain data. Current ticker: {ticker}. "
            "Switch to SPY in the sidebar."
        )
    else:
        with st.spinner("Fitting HAR-RV and aligning IV..."):
            plot_df, model_summary = fit_har_and_align(ticker, start_date, end_date)

        if plot_df is None or plot_df.empty:
            st.error("No aligned data available for the selected range.")
        else:
            # Summary stats
            st.markdown("**Variance risk premium stats**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean spread", f"{plot_df['IV_minus_RV'].mean():.3f}")
            c2.metric("Median spread", f"{plot_df['IV_minus_RV'].median():.3f}")
            c3.metric("% days IV > RV", f"{(plot_df['IV_minus_RV'] > 0).mean() * 100:.1f}%")
            c4.metric("Corr(IV, HAR)", f"{plot_df['Implied Vol'].corr(plot_df['HAR Forecast Vol']):.2f}")

            # HAR vs IV
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(plot_df.index, plot_df['HAR Forecast Vol'] * 100, linewidth=1.2, label='HAR-RV Forecast')
            ax.plot(plot_df.index, plot_df['Implied Vol'] * 100, linewidth=1.2, alpha=0.85, label='Implied Vol (ATM ~30d)')
            ax.set_xlabel("Date")
            ax.set_ylabel("Annualized Volatility (%)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # Spread
            st.markdown("**VRP spread: IV − HAR Forecast**")
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            spread = plot_df['IV_minus_RV'] * 100
            ax2.fill_between(spread.index, spread, 0, where=(spread >= 0),
                             alpha=0.4, color='green', label='IV > HAR (vol rich)')
            ax2.fill_between(spread.index, spread, 0, where=(spread < 0),
                             alpha=0.4, color='red', label='IV < HAR (vol cheap)')
            ax2.axhline(0, color='black', linewidth=0.8)
            ax2.axhline(spread.mean(), color='blue', linewidth=0.8, linestyle='--',
                        label=f'Mean ({spread.mean():.2f})')
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Spread (vol pts)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='lower right')
            st.pyplot(fig2)

            with st.expander("HAR-RV model summary"):
                st.code(model_summary)

            with st.expander("Data hygiene notes"):
                st.markdown(
                    "- IV for **2021-11-10 to 2021-11-24** was recomputed via Black-Scholes "
                    "Newton-Raphson from option mid prices because the recorded IVs were corrupted.\n"
                    "- ATM IV aggregated as the mean of contracts with moneyness ∈ [0.98, 1.02] "
                    "and DTE ∈ [20, 40] per date.\n"
                    "- HAR-RV daily input uses Rogers-Satchell + squared overnight return per day; "
                    "weekly = 5d rolling mean, monthly = 22d rolling mean.\n"
                    "- HAR fit with HAC (Newey-West, 22 lags) standard errors."
                )