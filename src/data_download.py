from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf


# -----------------------------------
# Paths
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# Equity index tickers (Europe)
# -----------------------------------
EQUITY_TICKERS: Dict[str, str] = {
    "DE": "^GDAXI",      # DAX
    "FR": "^FCHI",       # CAC40
    "IT": "FTSEMIB.MI",  # FTSE MIB
    "ES": "^IBEX",       # IBEX 35
    "NL": "^AEX",        # AEX
    "CH": "^SSMI",       # Swiss Market Index
    "UK": "^FTSE",       # FTSE 100

}


# -----------------------------------
# Helpers
# -----------------------------------
def download_prices_panel(
    tickers: Dict[str, str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download daily price levels from yfinance.
    Output: DataFrame with columns = country codes (DE, FR, ...), index = dates.
    Uses 'Adj Close'.
    """
    if not tickers:
        raise ValueError("Ticker dictionary is empty.")

    print(f"Downloading EQUITY data for {len(tickers)} tickers from yfinance...")
    yf_data = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=True,
    )

    # yfinance often returns MultiIndex columns: (field, ticker)
    if isinstance(yf_data.columns, pd.MultiIndex):
        adj_close = yf_data["Adj Close"].copy()
    else:
        # single ticker edge case
        adj_close = yf_data[["Adj Close"]].copy()

    # Map ticker -> country code
    ticker_to_country = {v: k for k, v in tickers.items()}
    adj_close = adj_close.rename(columns=ticker_to_country).sort_index(axis=1)

    # Ensure datetime index sorted
    adj_close = adj_close.sort_index()

    return adj_close


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly returns using end-of-month prices:
      R_t = P_t / P_{t-1} - 1
    """
    monthly_last = prices.resample("M").last()
    monthly_ret = monthly_last.pct_change().dropna(how="all")
    return monthly_ret


def main(start: str = "2005-01-01", end: str = "2025-01-01") -> None:
    # 1) Download daily prices
    equity_daily = download_prices_panel(EQUITY_TICKERS, start=start, end=end, interval="1d")

    raw_path = DATA_RAW_DIR / "equity_daily_prices.csv"
    equity_daily.to_csv(raw_path)
    print(f"Saved raw equity prices -> {raw_path}")

    # 2) Compute monthly returns
    equity_monthly = compute_monthly_returns(equity_daily)

    processed_path = DATA_PROCESSED_DIR / "equity_monthly_returns.csv"
    equity_monthly.to_csv(processed_path)
    print(f"Saved equity monthly returns -> {processed_path}")

    print("\nSummary:")
    print("Monthly returns shape:", equity_monthly.shape)
    print("Columns (markets):", list(equity_monthly.columns))


if __name__ == "__main__":
    main()
