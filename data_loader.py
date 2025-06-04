import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from utils import make_folder_if_not_exists

def fetch_price_data(tickers, start_date, end_date, data_dir="data"):
    """
    Fetches daily adjusted Close prices (using auto_adjust=True) for given tickers
    from Yahoo Finance between start_date and end_date.
    Saves a CSV in data_dir for caching. Returns a DataFrame of Close prices.
    """
    make_folder_if_not_exists(data_dir)
    # Build the cache‐filename based on tickers and dates
    file_path = f"{data_dir}/prices_{'_'.join(tickers)}_{start_date}_{end_date}.csv"

    try:
        # Attempt to load from cache
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Loaded cached data from {file_path}")
    except FileNotFoundError:
        # If not cached, download from Yahoo Finance using auto_adjust=True
        print("Downloading data from Yahoo Finance…")
        df_raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,     # ensures the Close column is already adjusted
            threads=True
        )
        # If multiple tickers, df_raw is a DataFrame with MultiIndex columns,
        # otherwise it’s a single‐index DataFrame. We just want the 'Close' prices.
        if isinstance(df_raw.columns, pd.MultiIndex):
            # E.g. columns like ('Close','AAPL'), ('Close','MSFT'), etc.
            df = df_raw["Close"].copy()
        else:
            # Single‐ticker case: just take the 'Close' column
            df = df_raw[["Close"]].copy()
            # Rename column from "Close" → ticker name for consistency
            df.columns = tickers

        # Save to CSV for caching
        df.to_csv(file_path)
        print(f"Saved data to {file_path}")

    # Forward‐fill / backward‐fill any missing data
    df = df.ffill().bfill()
    return df

if __name__ == "__main__":
    # Example usage
    tickers = ["KO", "PEP"]
    start = "2015-01-01"
    end = "2023-12-31"
    prices = fetch_price_data(tickers, start, end)
    print(prices.tail())
