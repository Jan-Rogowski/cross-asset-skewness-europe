from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------
# Ścieżki
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RET_PATH = DATA_PROCESSED_DIR / "equity_monthly_returns.csv"
SKEW_PATH = DATA_PROCESSED_DIR / "equity_skewness_12m.csv"
PORTFOLIO_PATH = DATA_PROCESSED_DIR / "portfolio_skewness_ls.csv"


# -----------------------------------
# Funkcje pomocnicze
# -----------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wczytuje:
      - miesięczne stopy zwrotu indeksów,
      - 12M rolling skewness,
    i zostawia tylko wspólne daty i wspólne indeksy.
    """
    rets = pd.read_csv(RET_PATH, index_col=0, parse_dates=True)
    skew = pd.read_csv(SKEW_PATH, index_col=0, parse_dates=True)

    # wspólne kolumny (kraje)
    common_cols = sorted(set(rets.columns) & set(skew.columns))
    rets = rets[common_cols]
    skew = skew[common_cols]

    # wspólne daty
    common_idx = rets.index.intersection(skew.index)
    rets = rets.loc[common_idx].sort_index()
    skew = skew.loc[common_idx].sort_index()

    print("Używane indeksy:", common_cols)
    print("Liczba miesięcy (po dopasowaniu):", len(common_idx))

    return rets, skew


def build_skewness_portfolios(
    rets: pd.DataFrame,
    skew: pd.DataFrame,
    min_assets: int = 4,
) -> pd.DataFrame:
    """
    Buduje portfel:
      - LONG: połowa indeksów o najniższej (najbardziej negatywnej) skośności,
      - SHORT: połowa indeksów o najwyższej (najbardziej pozytywnej) skośności,
    przy czym skośność jest opóźniona o 1 miesiąc (sygnał z t-1, stopa zwrotu z t).

    Zwraca DataFrame z kolumnami: ['Long', 'Short', 'LongShort'].
    """

    # używamy skośności z poprzedniego miesiąca jako sygnału
    skew_signal = skew.shift(1)

    long_ret_list: List[float] = []
    short_ret_list: List[float] = []
    ls_ret_list: List[float] = []
    dates: List[pd.Timestamp] = []

    for date in rets.index:
        signal_row = skew_signal.loc[date]
        ret_row = rets.loc[date]

        # bierzemy tylko indeksy, dla których mamy sygnał i stopę zwrotu
        valid = signal_row.dropna().index.intersection(ret_row.dropna().index)
        if len(valid) < min_assets:
            # za mało aktywów, pomijamy miesiąc
            continue

        sig = signal_row[valid]
        r = ret_row[valid]

        # sortowanie po skośności (ascending: najbardziej negatywna → najbardziej pozytywna)
        sig_sorted = sig.sort_values()
        n = len(sig_sorted)
        k = n // 2  # połowa do long, połowa do short
        if k == 0:
            continue

        long_idx = sig_sorted.index[:k]
        short_idx = sig_sorted.index[-k:]

        long_ret = r[long_idx].mean()
        short_ret = r[short_idx].mean()
        ls_ret = long_ret - short_ret

        dates.append(date)
        long_ret_list.append(long_ret)
        short_ret_list.append(short_ret)
        ls_ret_list.append(ls_ret)

    portfolio = pd.DataFrame(
        {
            "Long": long_ret_list,
            "Short": short_ret_list,
            "LongShort": ls_ret_list,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    ).sort_index()

    return portfolio


def main():
    print("=== BUDOWANIE PORTFELA SKEWNESS ===")

    rets, skew = load_data()
    pf = build_skewness_portfolios(rets, skew, min_assets=4)

    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(PORTFOLIO_PATH)

    print(f"Zapisano portfel do: {PORTFOLIO_PATH}")
    print("Rozmiar (miesiące x kolumny):", pf.shape)
    print("Pierwsze wiersze:")
    print(pf.head())
    print("Ostatnie wiersze:")
    print(pf.tail())


if __name__ == "__main__":
    main()
