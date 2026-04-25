# Volatility Surface Signals

Research and backtesting framework for implied vs. realized volatility signals on SPY options.

## Motivation

The variance risk premium (VRP) — the tendency of implied volatility to exceed subsequently realized volatility — is one of the more robust empirical regularities in options markets. This project investigates VRP signal construction and systematic harvesting via delta-hedged short volatility positions.

## Structure

- `src/` — core library code (data loading, RV estimators, Black-Scholes, backtest loop)
- `notebooks/` — exploratory analysis
- `tests/` — unit tests for pricing and estimator functions
- `data/` — local parquet files (gitignored)

## Data

Options chain data sourced from the [post-no-preference/options](https://www.dolthub.com/repositories/post-no-preference/options) Dolthub repository. Underlying price data from `yfinance`.

## Status

In development. See commit history for progress.

## References

- Sinclair, *Volatility Trading* (2013)
- Hull, *Options, Futures, and Other Derivatives* (11th ed.)
- Natenberg, *Option Volatility and Pricing*