from pathlib import Path
import pandas as pd
from scipy.stats import skew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

RET_PATH = DATA_PROCESSED / "commodities_monthly_returns.csv"
SKEW_PATH = DATA_PROCESSED / "commodities_skewness_12m.csv"


def load_monthly_cmdty(min_obs: int = 80) -> pd.DataFrame:
    rets = pd.read_csv(RET_PATH, index_col=0, parse_dates=True)
    rets = rets.loc["2005-01-01":"2025-01-01"]

    good_cols = [c for c in rets.columns if rets[c].notna().sum() >= min_obs]
    rets = rets[good_cols].sort_index()

    print("Używane surowce:", good_cols)
    print("Liczba miesięcy:", len(rets))
    return rets


def compute_rolling_skewness(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    def skew_func(x):
        return skew(x, bias=False, nan_policy="omit")

    skew_df = returns.rolling(window=window, min_periods=window).apply(
        skew_func, raw=True
    )
    return skew_df


def main():
    cmdty_ret = load_monthly_cmdty(min_obs=80)
    cmdty_skew = compute_rolling_skewness(cmdty_ret, window=12)

    SKEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmdty_skew.to_csv(SKEW_PATH)

    print(f"Skośność 12M surowców zapisana do: {SKEW_PATH}")
    print("Shape:", cmdty_skew.shape)


if __name__ == "__main__":
    main()
