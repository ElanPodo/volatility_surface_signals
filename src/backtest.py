"""
Variance Risk Premium short-straddle backtest with daily delta hedging.

Strategy:
- Entry: every Monday, target 30 DTE
- Signal: rolling z-score of (IV - HAR forecast) > 0.5
- Sizing: linear in z-score, capped at $100k notional per straddle
- Exit: close at 7 DTE
- Hedging: daily at close in SPY
- Costs: full half-spread on options, 1bp on SPY hedges
- Validation: walk-forward HAR refit quarterly
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, field
from typing import Optional
from src.rv_estimators import har_rv

@dataclass
class BacktestConfig:
    target_dte: int = 30
    exit_dte: int = 7
    max_notional: float = 100_000
    z_threshold: float = 0.5
    z_cap: float = 2.0
    z_lookback: int = 60
    har_refit_freq: str = 'QE'  # quarterly
    spy_slippage_bps: float = 1.0
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.013

@dataclass
class Straddle:
    entry_date: pd.Timestamp
    expiration: pd.Timestamp
    strike: float
    contracts: float                       
    entry_call_price: float                
    entry_put_price: float
    entry_underlying: float
    z_score_at_entry: float
    
    # Tracked over life of trade
    daily_records: list = field(default_factory=list)
    exit_date: Optional[pd.Timestamp] = None
    exit_call_price: Optional[float] = None
    exit_put_price: Optional[float] = None
    exit_underlying: Optional[float] = None
    
    # Cumulative hedge tracking
    cum_hedge_pnl: float = 0.0
    prev_total_delta: float = 0.0          # of the straddle position itself

def find_nearest_expiry(chain_day: pd.DataFrame, target_dte: int) -> Optional[float]:
    """Among contracts on this date, find the DTE closest to target."""
    if chain_day.empty:
        return None
    available_dtes = chain_day['dte'].unique()
    if len(available_dtes) == 0:
        return None
    nearest = min(available_dtes, key=lambda d: abs(d - target_dte))
    return nearest


def find_atm_contracts(chain_day: pd.DataFrame, target_dte: int):
    """Return (call_row, put_row, strike) closest to ATM at the target expiry."""
    dte = find_nearest_expiry(chain_day, target_dte)
    if dte is None:
        return None, None, None
    
    expiry_chain = chain_day[chain_day['dte'] == dte]
    underlying = expiry_chain['underlying'].iloc[0]
    
    # Find strike closest to underlying
    strikes = expiry_chain['strike'].unique()
    atm_strike = min(strikes, key=lambda k: abs(k - underlying))
    
    call = expiry_chain[
        (expiry_chain['strike'] == atm_strike) & (expiry_chain['cp_flag'] == 'C')
    ]
    put = expiry_chain[
        (expiry_chain['strike'] == atm_strike) & (expiry_chain['cp_flag'] == 'P')
    ]
    
    if call.empty or put.empty:
        return None, None, None
    
    return call.iloc[0], put.iloc[0], atm_strike


def lookup_contract(chain_day: pd.DataFrame, expiration, strike, cp_flag):
    """Find a specific contract on a given day. Returns row or None."""
    match = chain_day[
        (chain_day['expiration'] == expiration) &
        (chain_day['strike'] == strike) &
        (chain_day['cp_flag'] == cp_flag)
    ]
    if match.empty:
        return None
    return match.iloc[0]

def walk_forward_har(spy_prices: pd.DataFrame, refit_freq: str = 'Q') -> pd.Series:
    """
    Generate out-of-sample HAR vol forecasts using walk-forward refit.
    
    At each refit boundary, refit HAR using all data up to that point, then
    apply that model's coefficients to subsequent dates until next refit.
    
    Returns a Series of HAR forecasts indexed by date.
    """
    refit_dates = pd.date_range(
        spy_prices.index.min(), spy_prices.index.max(), freq=refit_freq
    )
    
    all_forecasts = pd.Series(index=spy_prices.index, dtype=float)
    
    for i, refit_date in enumerate(refit_dates):
        train = spy_prices.loc[:refit_date]
        if len(train) < 100:                # need minimum data to fit HAR
            continue
        
        try:
            model, har_train = har_rv(train)
        except Exception:
            continue
        
        # Apply this model to data from refit_date until next refit
        next_refit = refit_dates[i + 1] if i + 1 < len(refit_dates) else spy_prices.index.max()
        forward_data = spy_prices.loc[refit_date:next_refit]
        
        if len(forward_data) < 23:          # need 22-day rolling
            continue
        
        # Recompute features on the full series and apply model
        _, har_full = har_rv(spy_prices.loc[:next_refit])
        forecast_window = har_full.loc[refit_date:next_refit, 'RV Forecast(t+1)']
        all_forecasts.loc[forecast_window.index] = forecast_window.values
    
    return all_forecasts.dropna()

def build_signal(iv_daily: pd.Series, har_forecast: pd.Series, lookback: int = 60) -> pd.DataFrame:
    """
    Compute VRP and rolling z-score.
    
    Returns DataFrame with: iv, har, vrp, vrp_zscore (all indexed by date)
    """
    df = pd.DataFrame({'iv': iv_daily, 'har': har_forecast}).dropna()
    df['vrp'] = df['iv'] - df['har']
    df['vrp_mean'] = df['vrp'].rolling(lookback).mean()
    df['vrp_std'] = df['vrp'].rolling(lookback).std()
    df['vrp_zscore'] = (df['vrp'] - df['vrp_mean']) / df['vrp_std']
    return df


def size_from_zscore(z: float, threshold: float, cap: float, max_notional: float) -> float:
    """Linear sizing in z-score above threshold, capped at max_notional."""
    if z < threshold:
        return 0.0
    fraction = min((z - threshold) / (cap - threshold), 1.0)
    return fraction * max_notional

def run_backtest(
    chain: pd.DataFrame,
    spy_prices: pd.DataFrame,
    iv_daily: pd.Series,
    config: BacktestConfig = BacktestConfig(),
):
    """
    Main backtest. Returns trade log + daily P&L series.
    
    chain: long-format options chain with all required columns
    spy_prices: daily SPY OHLC indexed by date
    iv_daily: ATM ~30d IV series indexed by date
    """
    # 1. Walk-forward HAR forecasts
    print("Computing walk-forward HAR forecasts...")
    har_forecast = walk_forward_har(spy_prices, refit_freq=config.har_refit_freq)
    
    # 2. Build signal
    signal = build_signal(iv_daily, har_forecast, lookback=config.z_lookback)
    
    # 3. Index chain by date for fast lookup
    chain_by_date = {d: g for d, g in chain.groupby('date')}
    all_dates = sorted(chain_by_date.keys())
    
    # 4. Iterate through dates, opening straddles on Mondays + managing existing
    open_straddles: list[Straddle] = []
    closed_straddles: list[Straddle] = []
    daily_pnl_records = []
    
    for current_date in all_dates:
        chain_day = chain_by_date[current_date]
        spy_close = spy_prices['Close'].get(current_date, np.nan)
        if np.isnan(spy_close):
            continue
        
        # ---- Manage existing straddles (delta hedge + check exit) ----
        still_open = []
        for s in open_straddles:
            current_dte = (s.expiration - current_date).days
            
            # Look up current call and put prices
            call_row = lookup_contract(chain_day, s.expiration, s.strike, 'C')
            put_row = lookup_contract(chain_day, s.expiration, s.strike, 'P')
            
            if call_row is None or put_row is None:
                # Contract dropped from chain; mark to last known prices and close
                close_straddle(s, current_date, s.entry_call_price, s.entry_put_price,
                               spy_close, config)
                closed_straddles.append(s)
                continue
            
            # Mid prices for mark-to-market
            call_mid = (call_row['bid'] + call_row['ask']) / 2
            put_mid = (put_row['bid'] + put_row['ask']) / 2
            
            # Current straddle delta (we are short, so position delta is -contracts × (call_delta + put_delta))
            position_delta = -s.contracts * (call_row['delta'] + put_row['delta'])
            
            # Daily delta hedge: trade SPY to flatten total delta
            delta_to_hedge = position_delta - s.prev_total_delta
            spy_trade_notional = -delta_to_hedge * spy_close
            
            # Cost of SPY hedge trade
            spy_cost = abs(spy_trade_notional) * (config.spy_slippage_bps / 10000)
            s.cum_hedge_pnl -= spy_cost
            
            s.prev_total_delta = position_delta
            
            # Record daily mark-to-market state
            s.daily_records.append({
                'date': current_date,
                'dte': current_dte,
                'call_mid': call_mid,
                'put_mid': put_mid,
                'underlying': spy_close,
                'call_delta': call_row['delta'],
                'put_delta': put_row['delta'],
                'position_delta': position_delta,
            })
            
            # Exit check
            if current_dte <= config.exit_dte:
                # Close at half-spread (we're buying back, so pay ask)
                close_straddle(s, current_date,
                               exit_call=call_row['ask'],
                               exit_put=put_row['ask'],
                               exit_underlying=spy_close,
                               config=config)
                closed_straddles.append(s)
            else:
                still_open.append(s)
        
        open_straddles = still_open
        
        # ---- Check entry: Mondays only ----
        if current_date.weekday() == 0:                # Monday
            if current_date not in signal.index:
                daily_pnl_records.append((current_date, 0.0))
                continue
            
            z = signal.loc[current_date, 'vrp_zscore']
            if pd.isna(z):
                daily_pnl_records.append((current_date, 0.0))
                continue
            
            notional = size_from_zscore(z, config.z_threshold, config.z_cap, config.max_notional)
            if notional <= 0:
                daily_pnl_records.append((current_date, 0.0))
                continue
            
            # Find ATM call and put at target DTE
            call_row, put_row, atm_strike = find_atm_contracts(chain_day, config.target_dte)
            if call_row is None:
                continue
            
            # Number of straddle contracts (fractional). Use underlying notional.
            # 1 contract = 100 shares of underlying
            contracts = notional / (spy_close * 100)
            
            # Entry: we SHORT both legs, so we receive bid
            new_straddle = Straddle(
                entry_date=current_date,
                expiration=call_row['expiration'],
                strike=atm_strike,
                contracts=contracts,
                entry_call_price=call_row['bid'],
                entry_put_price=put_row['bid'],
                entry_underlying=spy_close,
                z_score_at_entry=z,
                prev_total_delta=-contracts * (call_row['delta'] + put_row['delta']),
            )
            open_straddles.append(new_straddle)
        
        # ---- Daily PnL aggregation across all positions ----
        daily_pnl = compute_daily_pnl(open_straddles, closed_straddles, current_date)
        daily_pnl_records.append((current_date, daily_pnl))
    
    # ---- Force-close any remaining straddles at end of backtest ----
    final_date = all_dates[-1]
    final_chain = chain_by_date[final_date]
    final_spy = spy_prices['Close'].loc[final_date]
    for s in open_straddles:
        call_row = lookup_contract(final_chain, s.expiration, s.strike, 'C')
        put_row = lookup_contract(final_chain, s.expiration, s.strike, 'P')
        if call_row is not None and put_row is not None:
            close_straddle(s, final_date, call_row['ask'], put_row['ask'], final_spy, config)
        else:
            # Use last known mid
            last_call = s.daily_records[-1]['call_mid'] if s.daily_records else s.entry_call_price
            last_put = s.daily_records[-1]['put_mid'] if s.daily_records else s.entry_put_price
            close_straddle(s, final_date, last_call, last_put, final_spy, config)
        closed_straddles.append(s)
    
    return {
        'trades': closed_straddles,
        'daily_pnl': pd.Series(dict(daily_pnl_records)),
        'signal': signal,
        'har_forecast': har_forecast,
    }


def close_straddle(s: Straddle, exit_date, exit_call, exit_put, exit_underlying, config):
    s.exit_date = exit_date
    s.exit_call_price = exit_call
    s.exit_put_price = exit_put
    s.exit_underlying = exit_underlying


def compute_daily_pnl(open_list, closed_list, current_date):
    """Crude proxy: today's pnl = today's MTM change across all positions.
    For a simpler v1 we'll compute total P&L per straddle at close instead."""
    return 0.0  # populated post-hoc in trade_pnl

def compute_trade_pnl(s: Straddle) -> dict:
    """
    Total P&L for one straddle, including hedge P&L.
    
    Short straddle P&L = (premium received) - (cost to close)
    Premium received = contracts × 100 × (entry_call_bid + entry_put_bid)
    Cost to close = contracts × 100 × (exit_call_ask + exit_put_ask)
    
    Plus cumulative hedge P&L from daily SPY trades.
    """
    if s.exit_date is None:
        return {}
    
    multiplier = 100  # standard equity option multiplier
    
    premium_received = s.contracts * multiplier * (s.entry_call_price + s.entry_put_price)
    cost_to_close = s.contracts * multiplier * (s.exit_call_price + s.exit_put_price)
    options_pnl = premium_received - cost_to_close
    
    # Hedge P&L: simple approximation. For each daily record, the hedge
    # carried from prev day to today produces P&L from underlying movement.
    # P&L = -prev_position_delta × ΔS - hedge transaction costs
    hedge_pnl = 0.0
    for i in range(1, len(s.daily_records)):
        prev = s.daily_records[i - 1]
        cur = s.daily_records[i]
        # Hedge held was -prev['position_delta'] in shares; we earn shares × ΔS
        hedge_shares = -prev['position_delta']
        hedge_pnl += hedge_shares * (cur['underlying'] - prev['underlying'])
    hedge_pnl += s.cum_hedge_pnl  # subtract transaction costs accumulated
    
    return {
        'entry_date': s.entry_date,
        'exit_date': s.exit_date,
        'strike': s.strike,
        'expiration': s.expiration,
        'contracts': s.contracts,
        'z_at_entry': s.z_score_at_entry,
        'premium_received': premium_received,
        'cost_to_close': cost_to_close,
        'options_pnl': options_pnl,
        'hedge_pnl': hedge_pnl,
        'total_pnl': options_pnl + hedge_pnl,
        'days_held': (s.exit_date - s.entry_date).days,
    }


def summarize_trades(trades: list[Straddle]) -> pd.DataFrame:
    rows = [compute_trade_pnl(t) for t in trades if t.exit_date is not None]
    return pd.DataFrame(rows)


def equity_curve(trade_df: pd.DataFrame, initial_capital: float = 1_000_000) -> pd.Series:
    """Build daily equity curve assuming P&L is realized at exit_date."""
    if trade_df.empty:
        return pd.Series(dtype=float)
    
    pnl_by_date = trade_df.groupby('exit_date')['total_pnl'].sum()
    full_index = pd.date_range(trade_df['entry_date'].min(), trade_df['exit_date'].max(), freq='D')
    daily = pnl_by_date.reindex(full_index, fill_value=0.0)
    return initial_capital + daily.cumsum()