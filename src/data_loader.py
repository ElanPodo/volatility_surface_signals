import os
import pandas as pd
import yfinance as yf

def fetch_dolthub_options(chain_csv: str = "data/spy_option_chain.csv",
    chain_parquet: str = "data/spy_option_chain.parquet",
    vol_csv: str = "data/spy_volatility_history.csv",
    vol_parquet: str = "data/spy_volatility_history.parquet", 
    force_refresh: bool = False):
    if os.path.exists(chain_parquet) and not force_refresh:
        oc = pd.read_parquet(chain_parquet)
    else:
        oc = pd.read_csv(chain_csv, parse_dates=["date", "expiration"])
        oc.to_parquet(chain_parquet)

    if os.path.exists(vol_parquet) and not force_refresh:
        vh = pd.read_parquet(vol_parquet)
    else:
        vh = pd.read_csv(vol_csv, parse_dates=["date"])
        vh.to_parquet(vol_parquet)

    return oc, vh

def fetch_yfinance_prices(ticker, start, end, out_path, force_refresh):
    if os.path.exists(out_path) and not force_refresh:
        return pd.read_parquet(out_path)

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns.name = None

    data.to_parquet(out_path)
    return data

if __name__ == "__main__":
    fetch_dolthub_options()
    fetch_yfinance_prices(ticker = 'SPY', start = '2020-01-01', end = '2025-12-31', out_path = 'data/spy_prices.parquet', force_refresh=True)