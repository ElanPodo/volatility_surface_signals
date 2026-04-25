import pandas as pd

oc = pd.read_csv("data/spy_option_chain.csv", parse_dates=["date", "expiration"])
oc.to_parquet("data/spy_option_chain.parquet")

vh = pd.read_csv("data/spy_volatility_history.csv", parse_dates=["date"])
vh.to_parquet("data/spy_volatility_history.parquet")
    