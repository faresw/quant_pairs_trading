# stock_ui.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(
    page_title="Stock Search UI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Search & Visualization")
st.markdown(
    """
    Enter one or more stock tickers (comma‐separated).  
    Select a date range, then click **Fetch Data** to view Adjusted Close prices, summary statistics, and a price chart.
    """
)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🔍 Search Parameters")

    # 1) Tickers input
    tickers_input = st.text_input(
        "Ticker(s)",
        value="AAPL, MSFT",
        help="Enter one or more tickers separated by commas (e.g. 'AAPL, MSFT, GOOG')."
    ).upper()

    # 2) Date range picker
    today = date.today()
    default_start = date(today.year - 1, today.month, today.day)  # One year ago
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=date(2000, 1, 1),
        max_value=today
    )
    end_date = st.date_input(
        "End Date",
        value=today,
        min_value=start_date,
        max_value=today
    )

    # 3) Fetch button
    fetch_button = st.button("Fetch Data")

# --- Main Content ---
if fetch_button:
    # 1) Parse tickers
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) == 0:
        st.error("⚠️ Please enter at least one valid ticker symbol.")
        st.stop()

    # 2) Download data using yfinance (auto‐adjusted Close)
    with st.spinner("Downloading data from Yahoo Finance…"):
        try:
            df_raw = yf.download(
                tickers,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                progress=False,
                auto_adjust=True,
                threads=True
            )
        except Exception as e:
            st.error(f"Failed to download data: {e}")
            st.stop()

    # 3) Extract the 'Close' column (auto_adjust=True ensures this is adjusted close)
    if isinstance(df_raw.columns, pd.MultiIndex):
        # Multi‐ticker case: df_raw["Close"] is a DataFrame with each column named by ticker
        try:
            df_close = df_raw["Close"].copy()
        except KeyError:
            st.error("No 'Close' column found in downloaded data.")
            st.stop()
    else:
        # Single‐ticker case: df_raw has columns like ['Open','High','Low','Close',...]
        if "Close" not in df_raw.columns:
            st.error("No 'Close' column found in downloaded data.")
            st.stop()
        df_close = df_raw[["Close"]].copy()
        df_close.columns = tickers  # Rename column to ticker for consistency

    # 4) Clean & forward/backward fill missing values
    df_close = df_close.ffill().bfill()

    # 5) Display summary
    st.subheader("📊 Price Data & Summary Statistics")

    # a) Show first + last few rows of the DataFrame
    st.write("### Adjusted Close Prices")
    st.dataframe(df_close.tail(10), use_container_width=True)

    # b) Compute summary stats
    summary_df = pd.DataFrame({
        "Mean": df_close.mean(),
        "Std Dev": df_close.std(),
        "Min": df_close.min(),
        "Max": df_close.max(),
    })
    st.write("### Summary Statistics (Over Selected Period)")
    st.dataframe(summary_df.style.format("{:.2f}"), use_container_width=True)

    # 6) Price chart
    st.subheader("📈 Price Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df_close.columns:
        ax.plot(df_close.index, df_close[col], label=col, linewidth=1.5)
    ax.set_title("Adjusted Close Price Over Time", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # 7) Download CSV button
    st.subheader("⏬ Download Data")
    csv_data = df_close.to_csv().encode("utf-8")
    st.download_button(
        label="Download Adjusted Close CSV",
        data=csv_data,
        file_name=f"prices_{'_'.join(tickers)}_{start_date}_{end_date}.csv",
        mime="text/csv"
    )

else:
    st.info("Enter tickers and date range in the sidebar, then click **Fetch Data**.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ♥ using Streamlit · Data from Yahoo Finance · (c) Fares Alhezaimi")
