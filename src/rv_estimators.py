import numpy as np
import pandas as pd


def close_to_close_rv(prices, window=21, annualize=True, trading_days=150):
    log_returns = np.log(prices / prices.shift(1))
    rv = log_returns.rolling(window=window).std()
    if annualize:
        rv = rv * np.sqrt(trading_days)
    return rv