from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yfinance as yf


# -----------------------------------
# Ścieżki do katalogów z danymi
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# Tickery indeksów i 10Y yields
# (rates_tickers na razie w wersji roboczej)
# -----------------------------------

EQUITY_TICKERS: Dict[str, str] = {
    "DE": "^GDAXI",      # DAX
    "FR": "^FCHI",       # CAC40
    "IT": "FTSEMIB.MI",  # FTSE MIB
    "ES": "^IBEX",       # IBEX 35
    "NL": "^AEX",        # AEX
    "DK": "OMXC25.CO",   # OMX Copenhagen 25 (czasem C20/C25)
    "CH": "^SSMI",       # Swiss Market Index
    "UK": "^FTSE",       # FTSE 100
    "AT": "ATX.VI",      # ATX
    "FI": "^OMXH25",     # OMX Helsinki 25

}

# UWAGA: te tickery są „szkicem” – część może nie istnieć w yfinance.
# Po pierwszym uruchomieniu zobaczymy, co działa i ewentualnie poprawimy.
RATES_TICKERS: Dict[str, str] = {
    "DE": "DE10Y.BOND",
    "FR": "FR10Y.BOND",
    "IT": "IT10Y.BOND",
    "ES": "ES10Y.BOND",
    "NL": "NL10Y.BOND",
    "SE": "SE10Y.BOND",
    "DK": "DK10Y.BOND",
    "CH": "CH10Y.BOND",
    "UK": "GB10Y.BOND",
    "AT": "AT10Y.BOND",
    "FI": "FI10Y.BOND",
    "PL": "PL10Y.BOND",
}


# -----------------------------------
# Funkcje pomocnicze
# -----------------------------------

def download_panel(
    tickers: Dict[str, str],
    start: str = "2000-01-01",
    end: str = "2025-01-01",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Pobiera dane z yfinance dla podanych tickerów.
    Zwraca DataFrame z kolumnami = kody krajów (DE, FR, ...) i indexem = daty.
    Używamy 'Adj Close' jako serii cen/poziomów.
    """
    if not tickers:
        raise ValueError("Słownik tickerów jest pusty.")

    print(f"Pobieranie danych dla {len(tickers)} tickerów z yfinance...")
    yf_data = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=True,
    )

    # yfinance zwykle zwraca MultiIndex w kolumnach: (pole, ticker)
    if isinstance(yf_data.columns, pd.MultiIndex):
        adj_close = yf_data["Adj Close"].copy()
    else:
        # jeśli tylko jeden ticker
        adj_close = yf_data[["Adj Close"]].copy()

    # mapa ticker → kod kraju
    ticker_to_country = {v: k for k, v in tickers.items()}
    adj_close = adj_close.rename(columns=ticker_to_country)

    # uporządkuj kolumny alfabetycznie
    adj_close = adj_close.sort_index(axis=1)

    return adj_close


def compute_monthly_equity_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Miesięczne stopy zwrotu indeksów:
    R_m = (P_m / P_{m-1}) - 1, gdzie P_m to ostatnia cena w miesiącu.
    """
    monthly_last = prices.resample("ME").last()
    monthly_ret = monthly_last.pct_change().dropna(how="all")
    return monthly_ret


def compute_monthly_yield_changes(yields: pd.DataFrame) -> pd.DataFrame:
    """
    Miesięczne zmiany rentowności 10Y:
    Δy_m = y_m - y_{m-1}, gdzie y_m to ostatnia rentowność w miesiącu.
    Traktujemy to jako prostą miarę 'zwrotu' z obligacji.
    """
    monthly_last = yields.resample("ME").last()
    monthly_delta = monthly_last.diff().dropna(how="all")
    return monthly_delta


def download_and_save_all(
    start: str = "2000-01-01",
    end: str = "2025-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Główna funkcja:
      1) pobiera dzienne dane indeksów (equity) i rentowności 10Y (rates),
      2) zapisuje surowe dane do data/raw/,
      3) liczy miesięczne stopy zwrotu / zmiany rentowności,
      4) zapisuje je do data/processed/.

    Zwraca:
      (monthly_equity_returns, monthly_yield_changes)
    """
    # ----- EQUITY -----
    equity_raw = download_panel(EQUITY_TICKERS, start=start, end=end, interval="1d")
    equity_raw_path = DATA_RAW_DIR / "equity_daily_prices.csv"
    equity_raw.to_csv(equity_raw_path)
    print(f"Zapisano surowe dane equity do: {equity_raw_path}")

    equity_monthly = compute_monthly_equity_returns(equity_raw)
    equity_monthly_path = DATA_PROCESSED_DIR / "equity_monthly_returns.csv"
    equity_monthly.to_csv(equity_monthly_path)
    print(f"Zapisano miesięczne stopy zwrotu equity do: {equity_monthly_path}")

    # ----- RATES -----
    try:
        rates_raw = download_panel(RATES_TICKERS, start=start, end=end, interval="1d")
    except Exception as e:
        print(
            "\nUwaga: problem z pobraniem części danych dla 10Y yields.\n"
            "Sprawdź tickery w RATES_TICKERS lub usuń nieistniejące.\n"
            f"Szczegóły błędu: {e}\n"
        )
        rates_monthly = pd.DataFrame()
    else:
        rates_raw_path = DATA_RAW_DIR / "rates_daily_yields.csv"
        rates_raw.to_csv(rates_raw_path)
        print(f"Zapisano surowe dane 10Y yields do: {rates_raw_path}")

        rates_monthly = compute_monthly_yield_changes(rates_raw)
        rates_monthly_path = DATA_PROCESSED_DIR / "rates_monthly_changes.csv"
        rates_monthly.to_csv(rates_monthly_path)
        print(f"Zapisano miesięczne zmiany rentowności do: {rates_monthly_path}")

    return equity_monthly, rates_monthly


if __name__ == "__main__":
    eq, rt = download_and_save_all(start="2005-01-01", end="2025-01-01")
    print("\nPodsumowanie:")
    print("Equity monthly returns shape:", eq.shape)
    print("Rates monthly changes shape:", rt.shape)




