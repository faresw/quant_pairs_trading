import matplotlib.pyplot as plt
import pandas as pd

def plot_prices_and_spread(prices_df, spread, zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    Plots:
      - Top: Price series of both tickers
      - Middle: Spread & entry/exit thresholds
      - Bottom: Z‐score & thresholds
    """
    dates = prices_df.index
    s1 = prices_df.iloc[:, 0]
    s2 = prices_df.iloc[:, 1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1) Price series
    axes[0].plot(dates, s1, label=prices_df.columns[0])
    axes[0].plot(dates, s2, label=prices_df.columns[1])
    axes[0].set_title("Price Series")
    axes[0].legend()

    # 2) Spread with thresholds
    axes[1].plot(dates, spread, label="Spread", color="black")
    axes[1].axhline(spread.mean(), color="gray", linestyle="--", label="Mean")
    axes[1].set_title("Spread")
    axes[1].legend()

    # 3) Z‐score with entry/exit
    axes[2].plot(dates, zscore, label="Z‐score", color="purple")
    axes[2].axhline(entry_threshold, color="red", linestyle="--", label="Entry Thresh")
    axes[2].axhline(-entry_threshold, color="green", linestyle="--", label="Entry Thresh")
    axes[2].axhline(exit_threshold, color="orange", linestyle=":", label="Exit Thresh")
    axes[2].axhline(-exit_threshold, color="orange", linestyle=":")
    axes[2].set_title("Z‐Score & Trading Thresholds")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def plot_equity_curve(equity_curve):
    """
    Plots the cumulative P&L (equity curve) over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values, color="blue")
    plt.title("Equity Curve (Cumulative P&L)")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage: 
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2020-01-01", periods=200)
    dummy_spread = pd.Series(np.random.randn(200).cumsum(), index=dates)
    dummy_z = (dummy_spread - dummy_spread.rolling(20).mean()) / dummy_spread.rolling(20).std()
    dummy_prices = pd.DataFrame({
        "A": np.cumsum(np.random.randn(200)) + 50,
        "B": np.cumsum(np.random.randn(200)) + 48
    }, index=dates)
    plot_prices_and_spread(dummy_prices, dummy_spread, dummy_z)
    dummy_equity = pd.Series(np.cumsum(np.random.randn(200) * 5), index=dates)
    plot_equity_curve(dummy_equity)
