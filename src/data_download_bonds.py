from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


BOND_TICKERS: Dict[str, str] = {
    "DE": "DE10Y.BOND",
    "FR": "FR10Y.BOND",
    "IT": "IT10Y.BOND",
    "ES": "ES10Y.BOND",
    "NL": "NL10Y.BOND",
    "CH": "CH10Y.BOND",
    "PL": "PL10Y.BOND",
    "UK": "GB10Y.BOND",
}


def download_bond_yields(
    tickers: Dict[str, str],
    start: str = "2005-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    print("Pobieram bond yields z Yahoo Finance...")

    data = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Adj Close"].copy()
    else:
        adj_close = data[["Adj Close"]].copy()

    inverse_map = {v: k for k, v in tickers.items()}
    adj_close = adj_close.rename(columns=inverse_map)
    adj_close = adj_close.sort_index(axis=1)

    return adj_close


def compute_bond_monthly_returns(yields: pd.DataFrame) -> pd.DataFrame:
    """
    Obligacje: niemal liniowo:
        bond_return ≈ -Δyield
    Czyli zmiany rentowności *odwrócone znakiem*.
    """
    monthly_last = yields.resample("M").last()
    dy = monthly_last.diff()

    bond_ret = -dy.dropna(how="all")
    return bond_ret


def main():
    yields_raw = download_bond_yields(BOND_TICKERS)

    raw_path = DATA_RAW_DIR / "bond_daily_yields.csv"
    yields_raw.to_csv(raw_path)
    print(f"Zapisano dzienne bond yields do: {raw_path}")

    bond_monthly = compute_bond_monthly_returns(yields_raw)
    monthly_path = DATA_PROCESSED_DIR / "bond_monthly_returns.csv"
    bond_monthly.to_csv(monthly_path)
    print(f"Zapisano miesięczne bond returns do: {monthly_path}")

    print("Kształt monthly returns:", bond_monthly.shape)


if __name__ == "__main__":
    main()
