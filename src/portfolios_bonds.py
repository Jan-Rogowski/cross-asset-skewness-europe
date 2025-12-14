from pathlib import Path
from typing import Tuple, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

RET_PATH = DATA_DIR / "bond_monthly_returns.csv"
SKEW_PATH = DATA_DIR / "bond_skewness_12m.csv"
OUT_PATH = DATA_DIR / "portfolio_bonds_skewness_ls.csv"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wczytuje miesięczne zwroty obligacji oraz 12M skośność
    i wyrównuje je po wspólnych datach i krajach.
    Przycinamy okres do 2005–2025.
    """
    rets = pd.read_csv(RET_PATH, index_col=0, parse_dates=True)
    skew = pd.read_csv(SKEW_PATH, index_col=0, parse_dates=True)

    # okres 2005–2025
    rets = rets.loc["2005-01-01":"2025-01-01"]
    skew = skew.loc["2005-01-01":"2025-01-01"]

    common_cols = sorted(set(rets.columns) & set(skew.columns))
    rets = rets[common_cols]
    skew = skew[common_cols]

    common_idx = rets.index.intersection(skew.index)
    rets = rets.loc[common_idx].sort_index()
    skew = skew.loc[common_idx].sort_index()

    print("Kraje w portfelu bonds:", common_cols)
    print("Liczba miesięcy:", len(common_idx))

    return rets, skew


def build_skewness_portfolio(
    rets: pd.DataFrame,
    skew: pd.DataFrame,
    min_assets: int = 4,
) -> pd.DataFrame:
    """
    Tworzy portfel Long–Short oparty na skośności 12M.

    BONDS (zgodnie z literaturą, np. Baltas 2019):
      - LONG: obligacje o NAJWYŻSZEJ skośności (kraje bezpieczne)
      - SHORT: obligacje o NAJNIŻSZEJ skośności (kraje ryzykowne)

    W każdym miesiącu:
      1. bierzemy skośność z t-1 (signal),
      2. sortujemy kraje po skośności,
      3. dzielimy na dwie połowy: top vs bottom,
      4. obliczamy zwrot Long, Short i LongShort.
    """
    skew_signal = skew.shift(1)

    dates: List[pd.Timestamp] = []
    long_list: List[float] = []
    short_list: List[float] = []
    ls_list: List[float] = []

    for date in rets.index:
        sig_row = skew_signal.loc[date]
        ret_row = rets.loc[date]

        valid = sig_row.dropna().index.intersection(ret_row.dropna().index)
        if len(valid) < min_assets:
            continue

        sig = sig_row[valid].sort_values()
        r = ret_row[valid]

        n = len(sig)
        k = n // 2
        if k == 0:
            continue

        # BONDS: long highest skewness, short lowest skewness
        short_idx = sig.index[:k]     # najniższa skośność (kraje ryzykowne)
        long_idx = sig.index[-k:]     # najwyższa skośność (bezpieczne)

        long_ret = r[long_idx].mean()
        short_ret = r[short_idx].mean()
        ls_ret = long_ret - short_ret

        dates.append(date)
        long_list.append(long_ret)
        short_list.append(short_ret)
        ls_list.append(ls_ret)

    pf = pd.DataFrame(
        {"Long": long_list, "Short": short_list, "LongShort": ls_list},
        index=pd.DatetimeIndex(dates, name="Date"),
    ).sort_index()

    return pf


def main():
    print("=== BONDS SKEWNESS PORTFOLIO ===")
    rets, skew = load_data()
    pf = build_skewness_portfolio(rets, skew, min_assets=4)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(OUT_PATH)

    print(f"Zapisano portfel bonds LS do: {OUT_PATH}")
    print("Shape:", pf.shape)
    print(pf.head())
    print(pf.tail())


if __name__ == "__main__":
    main()
