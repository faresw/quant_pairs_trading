import pandas as pd
import numpy as np

def backtest(prices_df, signals_df, beta, intercept=0.0, 
             transaction_cost=0.001, verbose=False):
    """
    Simulates a pairs-trading backtest given:
      - prices_df: DataFrame with columns [price1, price2]
      - signals_df: DataFrame with column [position] ∈ {-1, 0, +1}
      - beta, intercept: from the cointegration regression
      - transaction_cost: fraction per trade (e.g. 0.001 = 0.1%)
    Returns:
      - equity_curve: Series of cumulative P&L (in USD)
      - trades: DataFrame summarizing each trade’s P&L
    """
    price1 = prices_df.iloc[:, 0]
    price2 = prices_df.iloc[:, 1]
    dates = prices_df.index

    # Initialize DataFrame to store daily P&L contributions
    results = pd.DataFrame(index=dates)
    results["position"] = signals_df["position"]
    results["spread"] = price1 - (intercept + beta * price2)

    # Compute daily returns of each leg
    results["ret1"] = price1.pct_change().fillna(0)
    results["ret2"] = price2.pct_change().fillna(0)

    # For a “long spread” (pos=+1): +1 share of security1, -β shares of security2 
    # For “short spread” (pos=-1): -1 share of security1, +β shares of security2
    results["leg1"] = 0.0
    results["leg2"] = 0.0

    prev_pos = 0
    trades = []  # list of dicts to record individual trade P&L

    # Loop over time, day by day
    for i in range(1, len(dates)):
        date = dates[i]
        pos = int(results.at[date, "position"])

        # Determine leg weights
        if pos == 1:
            w1, w2 = 1.0, -beta
        elif pos == -1:
            w1, w2 = -1.0, beta
        else:
            w1, w2 = 0.0, 0.0

        results.at[date, "leg1"] = w1
        results.at[date, "leg2"] = w2

        # Compute daily P&L if in position
        # P&L = w1 * ret1 + w2 * ret2
        results.at[date, "daily_pnl"] = w1 * results.at[date, "ret1"] + w2 * results.at[date, "ret2"]

        # If a trade just opened or closed, apply transaction cost on both legs
        if pos != prev_pos:
            # Transaction cost on opening or closing: cost applies to notional = abs(w1)+abs(w2)
            cost = transaction_cost * (abs(w1) + abs(w2))
            results.at[date, "daily_pnl"] = results.at[date, "daily_pnl"] - cost

            # Record trade info
            trades.append({
                "date": date,
                "prev_pos": prev_pos,
                "new_pos": pos,
                "cost": cost
            })

        else:
            results.at[date, "daily_pnl"] = results.at[date, "daily_pnl"]

        prev_pos = pos

    # Fill NaN daily_pnl for first row
    results["daily_pnl"] = results["daily_pnl"].fillna(0)

    # Build equity curve
    results["equity"] = results["daily_pnl"].cumsum()

    trades_df = pd.DataFrame(trades)
    trades_df.set_index("date", inplace=True)

    if verbose:
        print("Backtest finished. Total P&L: ", results["equity"].iloc[-1])
        print("Number of trades:", len(trades_df) // 2)  # entry + exit pairs

    return results["equity"], trades_df

if __name__ == "__main__":
    from data_loader import fetch_price_data
    from stats_tests import find_cointegration_pair
    from signal_generator import compute_spread, compute_zscore, generate_signals

    # Example run for KO/PEP
    tickers = ["KO", "PEP"]
    prices = fetch_price_data(tickers, "2015-01-01", "2023-12-31")
    s1 = prices[tickers[0]]
    s2 = prices[tickers[1]]

    coint, β, α, _ = find_cointegration_pair(s1, s2)
    spread = compute_spread(s1, s2, β, intercept=α)
    zscore = compute_zscore(spread, window=20)
    signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)

    merged = pd.concat([s1, s2], axis=1).dropna()
    equity, trades = backtest(merged, signals, β, intercept=α, transaction_cost=0.001, verbose=True)
    print(trades.head())
    print("Final equity:", equity.iloc[-1])
