from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PF_PATH = DATA_DIR / "portfolio_commodities_skewness_ls.csv"
OUT_PATH = DATA_DIR / "analysis_commodities_summary.csv"


def max_dd(series: pd.Series) -> float:
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def main():
    pf = pd.read_csv(PF_PATH, index_col=0, parse_dates=True)
    ls = pf["LongShort"]

    ann_mean = ls.mean() * 12
    ann_std = ls.std() * np.sqrt(12)
    sharpe = ann_mean / ann_std if ann_std != 0 else np.nan
    t_stat = ann_mean / (ls.std() / np.sqrt(len(ls))) if ls.std() != 0 else np.nan
    mdd = max_dd(ls)

    summary = pd.DataFrame(
        {
            "Annualized Mean": [ann_mean],
            "Annualized Std": [ann_std],
            "Sharpe": [sharpe],
            "t-stat Mean": [t_stat],
            "Max Drawdown": [mdd],
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    print("Commodities summary saved to:", OUT_PATH)
    print(summary)


if __name__ == "__main__":
    main()
