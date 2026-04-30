import numpy as np
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm


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

def garman_klass_rv(high_price, low_price, open, close, window=21, annualize=True, trading_days=252):
    hl_term = 0.5 * np.square(np.log(high_price / low_price))
    co_term = (2 * np.log(2) - 1) * np.square(np.log(close / open))
    daily_var = hl_term - co_term
    rv = np.sqrt(daily_var.rolling(window=window).mean())
    if annualize:
        rv = rv * np.sqrt(trading_days)
    return rv

def daily_rv(high, low, open, close):
    overnight = np.log(open / close.shift(1)) ** 2
    rogers_satchell = (np.log(high / close) * np.log(high / open) + np.log(low / close) * np.log(low / open))
    return overnight + rogers_satchell

def yang_zhang_rv(high_price, low_price, open, close, window=21, annualize=True, trading_days=252):
    open_to_close = np.log(close / open)
    close_to_open = np.log(open / close.shift(1))
    high_to_open = np.log(high_price / open)
    high_to_close = np.log(high_price / close)
    low_to_open = np.log(low_price / open)
    low_to_close = np.log(low_price / close)

    sum_rs = (high_to_open * high_to_close) + (low_to_open * low_to_close)

    var_open = close_to_open.rolling(window=window).var(ddof=1)
    var_close = open_to_close.rolling(window=window).var(ddof=1)
    var_rs = sum_rs.rolling(window=window).mean()

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))

    rv = np.sqrt(var_open + (k * var_close) + ((1-k) * var_rs))
    if annualize:
        rv = rv * np.sqrt(trading_days)
    return rv

def overnight_variance(open, close, window=21, annualize=True, trading_days=252):
    co_returns = np.log(open / close.shift(1))
    var = co_returns.rolling(window).var(ddof=1)
    if annualize:
        var = var * trading_days
    return var

def parkinson_total_rv(high_price, low_price, open, close, window=21):
    park = parkinson_rv(high_price, low_price, window=window, annualize=False)
    overnight = overnight_variance(open, close, window=window, annualize=False)
    total_var = park**2 + overnight
    return np.sqrt(total_var * 252)

def garman_klass_total_rv(high_price, low_price, open, close, window=21):
    gk = garman_klass_rv(high_price, low_price, open, close, window=window, annualize=False)
    overnight = overnight_variance(open, close, window=window, annualize=False)
    total_var = gk**2 + overnight
    return np.sqrt(total_var * 252)

def har_rv(data, annualize=True, trading_days=252):
    rv_1 = daily_rv(data['High'], data['Low'], data['Open'], data['Close'])
    if annualize:
        rv_1 = rv_1 * trading_days

    df = pd.DataFrame({"RV Day(t)": rv_1, "RV Week": rv_1.rolling(5).mean(), "RV Month": rv_1.rolling(22).mean()}, index=data.index)
    df['RV Target'] = df['RV Day(t)'].shift(-1)
    df =  df.dropna()

    X = sm.add_constant(df[['RV Day(t)', 'RV Week', 'RV Month']])
    y = df['RV Target']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':22})
    df['RV Forecast(t+1)'] = model.predict(sm.add_constant(df[['RV Day(t)', 'RV Week', 'RV Month']]))
    return model, df

def bs_price(S, K, T, r, q, sigma, cp_flag):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if cp_flag == 'C':
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bs_vega(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)

def implied_vol_nr(price, S, K, T, r, q, cp_flag, 
                   sigma0=0.20, tol=1e-6, max_iter=100):
    # Sanity: price must be inside no-arb bounds
    if cp_flag == 'C':
        intrinsic = max(S*np.exp(-q*T) - K*np.exp(-r*T), 0)
        upper = S*np.exp(-q*T)
    else:
        intrinsic = max(K*np.exp(-r*T) - S*np.exp(-q*T), 0)
        upper = K*np.exp(-r*T)
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return np.nan
    
    sigma = sigma0
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, q, sigma, cp_flag) - price
        if abs(diff) < tol:
            return sigma
        v = bs_vega(S, K, T, r, q, sigma)
        if v < 1e-10:  # vega too small, NR will diverge
            break
        sigma -= diff / v
        if sigma <= 0 or sigma > 5:  # out of reasonable range
            break
    
    # Bisection fallback for cases where NR fails (deep ITM/OTM, near expiry)
    lo, hi = 1e-5, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, r, q, mid, cp_flag) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return mid
    return np.nan

r = 0.0015
q = 0.013

def recompute_row(row):
    return implied_vol_nr(
        price=row['mid'],
        S=row['Close'],
        K=row['strike'],
        T=row['dte'] / 365.0,
        r=r,
        q=q,
        cp_flag=row['call_put'],
    )




