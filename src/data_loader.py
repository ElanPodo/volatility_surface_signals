import pandas as pd
oc = pd.read_csv("C:/Users/elanp/Documents/projects/volatility-surface-signals/data/spy_option_chain.csv", parse_dates=["date", "expiration"], encoding='utf-16')
oc.to_parquet("C:/Users/elanp/Documents/projects/volatility-surface-signals/data/spy_option_chain.parquet")
vh = pd.read_csv("C:/Users/elanp/Documents/projects/volatility-surface-signals/data/spy_volatility_history.csv", parse_dates=["date"], encoding='utf-16')
vh.to_parquet("C:/Users/elanp/Documents/projects/volatility-surface-signals/data/spy_volatility_history.parquet")
    