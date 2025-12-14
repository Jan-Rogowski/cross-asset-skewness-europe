from pathlib import Path

import pandas as pd
from scipy.stats import skew


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FX_RET_PATH = DATA_PROCESSED_DIR / "fx_monthly_returns.csv"
FX_SKEW_PATH = DATA_PROCESSED_DIR / "fx_skewness_12m.csv"


def load_monthly_fx(min_obs: int = 120) -> pd.DataFrame:
    fx = pd.read_csv(FX_RET_PATH, index_col=0, parse_dates=True)

    good_cols = [c for c in fx.columns if fx[c].notna().sum() >= min_obs]
    fx = fx[good_cols].sort_index()

    print("Używane waluty:", good_cols)
    print("Liczba miesięcy:", len(fx))

    return fx


def compute_rolling_skewness(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    def skew_func(x):
        return skew(x, bias=False, nan_policy="omit")

    return returns.rolling(window=window, min_periods=window).apply(skew_func, raw=True)


def main():
    fx_ret = load_monthly_fx(min_obs=120)
    fx_skew = compute_rolling_skewness(fx_ret, window=12)

    FX_SKEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    fx_skew.to_csv(FX_SKEW_PATH)

    print(f"Skośność 12M FX zapisana do: {FX_SKEW_PATH}")
    print("Shape:", fx_skew.shape)


if __name__ == "__main__":
    main()
