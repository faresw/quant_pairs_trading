import pandas as pd

from data_loader import fetch_price_data
from stats_tests import find_cointegration_pair
from signal_generator import compute_spread, compute_zscore, generate_signals
from backtester import backtest
from performance import summary_stats
from visualize import plot_prices_and_spread, plot_equity_curve

def run_pairs_trading(
    tickers,
    start_date="2015-01-01",
    end_date="2023-12-31",
    coint_significance=0.05,
    z_window=20,
    entry_thresh=2.0,
    exit_thresh=0.5,
    transaction_cost=0.001,
    verbose=True
):
    """
    Coordinates all steps:
      1) Download data
      2) Cointegration test
      3) Compute spread & zscore
      4) Generate signals
      5) Backtest
      6) Compute performance metrics
      7) Visualize
    """
    # 1) Fetch prices
    prices = fetch_price_data(tickers, start_date, end_date, data_dir="data")
    s1 = prices[tickers[0]]
    s2 = prices[tickers[1]]
    merged_prices = pd.concat([s1, s2], axis=1).dropna()

    # 2) Cointegration test
    coint, beta, intercept, pvalue = find_cointegration_pair(s1, s2, significance=coint_significance)
    if not coint:
        print(f"Warning: {tickers[0]}/{tickers[1]} not cointegrated (p={pvalue:.4f}). Strategy may fail.\n")
    else:
        print(f"Cointegrated: β={beta:.4f}, intercept={intercept:.4f}, p-value={pvalue:.4f}\n")

    # 3) Compute spread & zscore
    spread = compute_spread(merged_prices.iloc[:, 0], merged_prices.iloc[:, 1], beta, intercept=intercept)
    zscore = compute_zscore(spread, window=z_window)

    # 4) Generate signals
    signals = generate_signals(zscore, entry_threshold=entry_thresh, exit_threshold=exit_thresh)

    # 5) Backtest
    equity_curve, trades = backtest(
        merged_prices,
        signals,
        beta,
        intercept=intercept,
        transaction_cost=transaction_cost,
        verbose=verbose
    )

    # 6) Performance metrics
    stats = summary_stats(equity_curve)
    print("Performance Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    print("\nTrades summary:")
    print(trades.head(5))
    print(f"Total trades (entry/exit pairs): {len(trades)//2}\n")

    # 7) Visualize
    plot_prices_and_spread(merged_prices, spread, zscore,
                           entry_threshold=entry_thresh, exit_threshold=exit_thresh)
    plot_equity_curve(equity_curve)

    return {
        "prices": merged_prices,
        "spread": spread,
        "zscore": zscore,
        "signals": signals,
        "equity_curve": equity_curve,
        "trades": trades,
        "stats": stats
    }

if __name__ == "__main__":
    # EDIT THESE PARAMETERS AS NEEDED
    tickers = ["KO", "PEP"]
    results = run_pairs_trading(
        tickers=tickers,
        start_date="2015-01-01",
        end_date="2023-12-31",
        coint_significance=0.05,
        z_window=20,
        entry_thresh=2.0,
        exit_thresh=0.5,
        transaction_cost=0.001,
       verbose=True
    )
