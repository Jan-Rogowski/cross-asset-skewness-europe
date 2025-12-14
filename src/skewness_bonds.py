from pathlib import Path

import pandas as pd
from scipy.stats import skew

# -----------------------------------------
# Ścieżki
# -----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BOND_RET_PATH = DATA_PROCESSED_DIR / "bond_monthly_returns.csv"
BOND_SKEW_PATH = DATA_PROCESSED_DIR / "bond_skewness_12m.csv"


def load_monthly_bonds(min_obs: int = 80) -> pd.DataFrame:
    """
    Wczytuje miesięczne zwroty obligacji 10Y i filtruje serie,
    które mają co najmniej `min_obs` niepustych obserwacji.
    Przycinamy okres do 2005–2025 (spójnie z FX i equity).
    """
    bonds = pd.read_csv(BOND_RET_PATH, index_col=0, parse_dates=True)

    # okres 2005–2025
    bonds = bonds.loc["2005-01-01":"2025-01-01"]

    # filtracja po liczbie obserwacji
    good_cols = [c for c in bonds.columns if bonds[c].notna().sum() >= min_obs]
    bonds = bonds[good_cols].sort_index()

    print("Używane serie obligacji (kraje):", good_cols)
    print("Liczba miesięcy po filtracji:", len(bonds))

    return bonds


def compute_rolling_skewness(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Oblicza 12-miesięczną skośność (rolling skewness) dla każdej serii.
    """
    def skew_func(x):
        return skew(x, bias=False, nan_policy="omit")

    skew_df = returns.rolling(window=window, min_periods=window).apply(
        skew_func, raw=True
    )
    return skew_df


def main():
    bonds_ret = load_monthly_bonds(min_obs=80)
    bonds_skew = compute_rolling_skewness(bonds_ret, window=12)

    BOND_SKEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    bonds_skew.to_csv(BOND_SKEW_PATH)

    print(f"Skośność 12M obligacji zapisana do: {BOND_SKEW_PATH}")
    print("Shape:", bonds_skew.shape)


if __name__ == "__main__":
    main()
