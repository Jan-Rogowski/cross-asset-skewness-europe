from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PF_PATH = DATA_DIR / "portfolio_fx_skewness_ls.csv"
OUT_PATH = DATA_DIR / "analysis_fx_summary.csv"


def max_drawdown(series: pd.Series) -> float:
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def main():
    pf = pd.read_csv(PF_PATH, index_col=0, parse_dates=True)
    ret = pf["LongShort"]

    ann_mean = ret.mean() * 12
    ann_std = ret.std() * np.sqrt(12)
    sharpe = ann_mean / ann_std if ann_std != 0 else np.nan
    t_stat = ann_mean / (ret.std() / np.sqrt(len(ret))) if ret.std() != 0 else np.nan
    mdd = max_drawdown(ret)

    df = pd.DataFrame(
        {
            "Annualized Mean": [ann_mean],
            "Annualized Std": [ann_std],
            "Sharpe": [sharpe],
            "t-stat Mean": [t_stat],
            "Max Drawdown": [mdd],
        }
    )

    df.to_csv(OUT_PATH, index=False)
    print("FX summary saved to:", OUT_PATH)
    print(df)


if __name__ == "__main__":
    main()
