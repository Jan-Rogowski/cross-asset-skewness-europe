from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

RET_PATH = DATA_DIR / "equity_monthly_returns.csv"
PF_PATH = DATA_DIR / "portfolio_skewness_ls.csv"
OUT_PATH = DATA_DIR / "analysis_summary.csv"


def max_drawdown(series: pd.Series) -> float:
    """Liczy maksymalne obsunięcie kapitału."""
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def compute_statistics():
    # wczytanie danych
    pf = pd.read_csv(PF_PATH, index_col=0, parse_dates=True)
    ret = pf["LongShort"]

    # annualized metrics
    ann_mean = ret.mean() * 12
    ann_std = ret.std() * np.sqrt(12)
    sharpe = ann_mean / ann_std

    # t-stat
    t_stat = ann_mean / (ret.std() / np.sqrt(len(ret)))

    # max drawdown
    mdd = max_drawdown(ret)

    # wynik do tabeli
    df = pd.DataFrame({
        "Annualized Mean": [ann_mean],
        "Annualized Std": [ann_std],
        "Sharpe": [sharpe],
        "t-stat Mean": [t_stat],
        "Max Drawdown": [mdd],
    })

    df.to_csv(OUT_PATH, index=False)

    print("Zapisano analizę do:", OUT_PATH)
    print(df)


def main():
    print("=== ANALIZA PORTFELA SKEWNESS LS ===")
    compute_statistics()


if __name__ == "__main__":
    main()

