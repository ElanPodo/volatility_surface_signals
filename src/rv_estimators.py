import numpy as np
import pandas as pd


def close_to_close_rv(prices, window=21, annualize=True, trading_days=252):
    log_returns = np.log(prices / prices.shift(1))
    rv = log_returns.rolling(window=window).std()
    if annualize:
        rv = rv * np.sqrt(trading_days)
    return rv

def parkinson_rv(high_price, low_price, window=21, annualize=True, trading_days=252):
    log_diff_square = np.square(np.log(high_price / low_price))
    constant = 1 / (4 * window * np.log(2))
    rv = np.sqrt(constant * log_diff_square.rolling(window=window).sum())
    if annualize:
        rv = rv * np.sqrt(trading_days)
    return rv

