from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew


# -----------------------------------
# Ścieżki
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EQUITY_RET_PATH = DATA_PROCESSED_DIR / "equity_monthly_returns.csv"
EQUITY_SKEW_PATH = DATA_PROCESSED_DIR / "equity_skewness_12m.csv"


def load_monthly_returns(path: Path) -> pd.DataFrame:
    """Wczytuje miesięczne stopy zwrotu i wyrzuca rynki z małą liczbą obserwacji."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # minimalna liczba miesięcy — wyrzuca FI (ma tylko ~130 obserwacji)
    min_obs = 180

    good_cols = [c for c in df.columns if df[c].notna().sum() >= min_obs]
    df = df[good_cols]

    print("=== LOAD MONTHLY RETURNS ===")
    print("Używane indeksy:", good_cols)
    print("Liczba miesięcy:", len(df))

    return df.sort_index()


def compute_rolling_skewness(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Liczy rolling skośność 12M dla każdego indeksu."""
    def skew_func(x):
        return skew(x, bias=False, nan_policy="omit")

    skew_df = returns.rolling(window=window, min_periods=window).apply(
        skew_func,
        raw=True,
    )
    return skew_df


def main():
    print("=== START SKEWNESS SCRIPT ===")

    monthly_ret = load_monthly_returns(EQUITY_RET_PATH)

    skew_12m = compute_rolling_skewness(monthly_ret, window=12)

    # zapis do CSV
    EQUITY_SKEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    skew_12m.to_csv(EQUITY_SKEW_PATH)

    print(f"Skośność 12M zapisana do: {EQUITY_SKEW_PATH}")
    print("Rozmiar:", skew_12m.shape)
    print("=== END SKEWNESS SCRIPT ===")


if __name__ == "__main__":
    main()
