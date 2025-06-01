import numpy as np
import pandas as pd

def sharpe_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=252):
    """
    Computes annualized Sharpe ratio from equity curve (P&L series).
    Assumes daily P&L. 
    """
    # Compute daily returns from equity curve
    daily_returns = equity_curve.diff().fillna(0)
    mean_ret = daily_returns.mean() * periods_per_year
    std_ret = daily_returns.std() * np.sqrt(periods_per_year)

    if std_ret == 0:
        return np.nan
    return (mean_ret - risk_free_rate) / std_ret

def max_drawdown(equity_curve):
    """
    Computes maximum drawdown from the equity curve.
    """
    cum = equity_curve.cummax()
    drawdown = (cum - equity_curve) / cum
    return drawdown.max()

def cagr(equity_curve):
    """
    Computes Compound Annual Growth Rate (CAGR).
    """
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    total_return = equity_curve.iloc[-1] / (equity_curve.iloc[0] + 1e-12)  # avoid zero-division
    return total_return ** (365.0 / days) - 1

def summary_stats(equity_curve):
    """
    Returns a dictionary of key performance metrics.
    """
    sr = sharpe_ratio(equity_curve)
    md = max_drawdown(equity_curve)
    cg = cagr(equity_curve)
    total_pnl = equity_curve.iloc[-1]
    return {
        "Total P&L": total_pnl,
        "CAGR": cg,
        "Sharpe Ratio": sr,
        "Max Drawdown": md
    }

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np

    # Dummy equity curve
    dates = pd.date_range("2020-01-01", periods=100)
    equity = pd.Series(np.cumsum(np.random.randn(100) * 10), index=dates)
    stats = summary_stats(equity)
    print(stats)
