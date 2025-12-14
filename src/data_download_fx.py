from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------------
# Ścieżki
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# Universe FX
# -----------------------------------
# Uwaga: część tickerów warto potem przetestować jak wcześniej, ale ten zestaw
# powinien działać w większości przypadków.
FX_TICKERS: Dict[str, str] = {
    "EUR": "EURUSD=X",  # euro / dolar
    "GBP": "GBPUSD=X",  # funt / dolar
    "CHF": "CHFUSD=X",  # frank / dolar
    "NOK": "NOKUSD=X",  # korona norweska / dolar
    "SEK": "SEKUSD=X",  # korona szwedzka / dolar
    # CEE jako USD/PLN itd. (notacja 'PLN=X', 'CZK=X', 'HUF=X')
    "PLN": "PLN=X",
    "CZK": "CZK=X",
    "HUF": "HUF=X",
}


# -----------------------------------
# Funkcje pomocnicze
# -----------------------------------

def download_fx_panel(
    tickers: Dict[str, str],
    start: str = "2000-01-01",
    end: str = "2025-01-01",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Pobiera dzienne kursy FX z yfinance i zwraca DataFrame:
      index = daty,
      kolumny = kody walut (EUR, GBP, ...),
      wartości = 'Adj Close'.
    """
    print(f"Pobieram FX dla {len(tickers)} par walutowych z yfinance...")

    yf_data = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=True,
    )

    if isinstance(yf_data.columns, pd.MultiIndex):
        adj_close = yf_data["Adj Close"].copy()
    else:
        adj_close = yf_data[["Adj Close"]].copy()

    ticker_to_ccy = {v: k for k, v in tickers.items()}
    adj_close = adj_close.rename(columns=ticker_to_ccy)
    adj_close = adj_close.sort_index(axis=1)

    return adj_close


def compute_monthly_fx_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Miesięczne logarytmiczne stopy zwrotu FX:
      r_t = ln(S_t) - ln(S_{t-1}),
    gdzie S_t to kurs na koniec miesiąca.
    """
    monthly_last = prices.resample("M").last()
    log_prices = np.log(monthly_last)
    monthly_ret = log_prices.diff().dropna(how="all")
    return monthly_ret


def main():
    fx_raw = download_fx_panel(FX_TICKERS, start="2005-01-01", end="2025-01-01")

    raw_path = DATA_RAW_DIR / "fx_daily_prices.csv"
    fx_raw.to_csv(raw_path)
    print(f"Zapisano dzienne kursy FX do: {raw_path}")

    fx_monthly = compute_monthly_fx_returns(fx_raw)
    monthly_path = DATA_PROCESSED_DIR / "fx_monthly_returns.csv"
    fx_monthly.to_csv(monthly_path)
    print(f"Zapisano miesięczne zwroty FX do: {monthly_path}")

    print("Kształt danych miesięcznych:", fx_monthly.shape)


if __name__ == "__main__":
    main()
