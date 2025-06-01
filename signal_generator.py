import numpy as np
import pandas as pd

def compute_spread(series1, series2, beta, intercept=0.0):
    """
    Computes the price spread time series:
    spread(t) = series1(t) - (intercept + beta * series2(t))
    """
    return series1 - (intercept + beta * series2)

def compute_zscore(spread, window=20):
    """
    Computes the rolling z-score of the spread:
    z = (spread - rolling_mean) / rolling_std
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generates trading signals based on z-score:
      - Long spread when zscore < -entry_threshold
      - Short spread when zscore > +entry_threshold
      - Exit when |zscore| < exit_threshold
    Returns a DataFrame with columns: [position], where position ∈ {-1, 0, +1}
    """
    signals = pd.DataFrame(index=zscore.index)
    signals["zscore"] = zscore
    signals["position"] = 0  # 1=long spread, -1=short spread, 0=flat

    # Long spread: z < -entry_threshold => position=+1
    signals.loc[zscore < -entry_threshold, "position"] = 1
    # Short spread: z > +entry_threshold => position=-1
    signals.loc[zscore > +entry_threshold, "position"] = -1

    # Exit rule: if abs(z) < exit_threshold => position=0
    mask_exit = zscore.abs() < exit_threshold
    signals.loc[mask_exit, "position"] = 0

    # Forward‐fill positions to hold until an exit signal appears
    signals["position"] = signals["position"].replace(to_replace=0, method="ffill").fillna(0)

    return signals

if __name__ == "__main__":
    from data_loader import fetch_price_data
    from stats_tests import find_cointegration_pair

    tickers = ["KO", "PEP"]
    prices = fetch_price_data(tickers, "2015-01-01", "2023-12-31")
    s1 = prices["KO"]
    s2 = prices["PEP"]

    coint, β, α, _ = find_cointegration_pair(s1, s2)
    if not coint:
        print("Warning: Pair is not cointegrated. Proceed anyway.")

    spread = compute_spread(s1, s2, β, intercept=α)
    zscore = compute_zscore(spread, window=20)
    signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)

    print(signals.tail(10))
