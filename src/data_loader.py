import os
import pandas as pd
import yfinance as yf
from pathlib import Path


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


def fetch_optionsdx_chains(raw_dir: str = "data/optionsdx_raw",
                            out_parquet: str = "data/spy_options_optionsdx.parquet",
                            force_refresh: bool = False):
    """Load OptionsDX SPY EOD text files (split across yearly/quarterly folders),
    concatenate, reshape from wide to long format, and cache as parquet.

    Returns a long-format dataframe with one row per (date, expiration, strike, cp_flag).
    """
    if os.path.exists(out_parquet) and not force_refresh:
        return pd.read_parquet(out_parquet)

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"OptionsDX raw directory not found: {raw_path}")

    files = sorted(list(raw_path.rglob("*.txt")) + list(raw_path.rglob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No .txt or .csv files found under {raw_path}")

    print(f"Found {len(files)} OptionsDX files, loading...")

    dfs = []
    for i, fp in enumerate(files):
        if i % 10 == 0:
            print(f"  [{i+1}/{len(files)}] {fp.name}")
        try:
            df = pd.read_csv(fp, low_memory=False)
            df.columns = [c.strip().strip("[]").lower().replace(" ", "_") for c in df.columns]
            dfs.append(df)
        except Exception as e:
            print(f"    failed: {fp.name} -> {e}")

    wide = pd.concat(dfs, ignore_index=True)
    print(f"Combined wide shape: {wide.shape}")
    print(f"Columns: {wide.columns.tolist()}")

    # Parse date columns (OptionsDX uses quote_date and expire_date)
    if "quote_date" in wide.columns:
        wide["quote_date"] = pd.to_datetime(wide["quote_date"], errors="coerce")
    if "expire_date" in wide.columns:
        wide["expire_date"] = pd.to_datetime(wide["expire_date"], errors="coerce")

    # Reshape wide -> long: one row per contract with cp_flag
    shared_cols = [c for c in ["quote_date", "expire_date", "strike", "underlying_last", "dte"]
                   if c in wide.columns]

    call_cols = {c: c[2:] for c in wide.columns if c.startswith("c_")}
    put_cols = {c: c[2:] for c in wide.columns if c.startswith("p_")}

    calls = wide[shared_cols + list(call_cols.keys())].rename(columns=call_cols)
    calls["cp_flag"] = "C"

    puts = wide[shared_cols + list(put_cols.keys())].rename(columns=put_cols)
    puts["cp_flag"] = "P"

    long = pd.concat([calls, puts], ignore_index=True)

    # Standardize column names to match your existing pipeline conventions
    rename_map = {
        "quote_date": "date",
        "expire_date": "expiration",
        "underlying_last": "underlying",
        "iv": "vol",
    }
    long = long.rename(columns={k: v for k, v in rename_map.items() if k in long.columns})

    # Compute dte if not present
    if "dte" not in long.columns and "expiration" in long.columns and "date" in long.columns:
        long["dte"] = (long["expiration"] - long["date"]).dt.days

    # Compute moneyness if underlying is present
    if "underlying" in long.columns and "strike" in long.columns:
        long["moneyness"] = long["strike"] / long["underlying"]

    # Sort and dedupe
    if "date" in long.columns:
        long = long.sort_values(["date", "expiration", "strike", "cp_flag"]).reset_index(drop=True)

    print(f"Final long-format shape: {long.shape}")
    print(f"Date range: {long['date'].min()} to {long['date'].max()}")

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_parquet, index=False)
    print(f"Saved -> {out_parquet} ({Path(out_parquet).stat().st_size / 1e6:.1f} MB)")

    return long


if __name__ == "__main__":
    fetch_dolthub_options()
    fetch_yfinance_prices(ticker='SPY', start='2020-01-01', end='2025-12-31',
                          out_path='data/spy_prices.parquet', force_refresh=True)
    fetch_yfinance_prices(ticker='^VIX', start='2020-01-01', end='2025-12-31',
                          out_path='data/vix_prices.parquet', force_refresh=True)
    fetch_optionsdx_chains(raw_dir='data/optionsdx_raw',
                           out_parquet='data/spy_options_optionsdx.parquet',
                           force_refresh=True)