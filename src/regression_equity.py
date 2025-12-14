# src/regression_equity.py

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ─── Ścieżki ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EQ_RET_PATH = DATA_DIR / "equity_monthly_returns.csv"
EQ_PF_PATH = DATA_DIR / "portfolio_skewness_ls.csv"
OUT_TABLE_PATH = DATA_DIR / "regression_equity_clean_table.csv"


def ols_to_table(model: sm.regression.linear_model.RegressionResultsWrapper,
                 digits: int = 6) -> pd.DataFrame:
    """Ładna tabelka OLS do pracy."""
    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues

    table = pd.DataFrame(
        {
            "Coefficient": params,
            "t-statistic": tvals,
            "p-value": pvals,
        }
    ).round(digits)

    extra = pd.DataFrame(
        {
            "Coefficient": [model.rsquared],
            "t-statistic": [np.nan],
            "p-value": [np.nan],
        },
        index=["R-squared"],
    )

    table = pd.concat([table, extra])
    return table


def main():
    # 1. Wczytanie danych
    eq_ret = pd.read_csv(EQ_RET_PATH, index_col=0, parse_dates=True)
    pf = pd.read_csv(EQ_PF_PATH, index_col=0, parse_dates=True)

    # Zakres jak w reszcie projektu
    eq_ret = eq_ret.loc["2005-01-01":"2025-01-01"]
    pf = pf.loc["2005-01-01":"2025-01-01"]

    # 2. Wyrównanie dat
    idx = eq_ret.index.intersection(pf.index)
    eq_ret = eq_ret.loc[idx].sort_index()
    pf = pf.loc[idx].sort_index()

    # 3. Rynek = średnia z wszystkich indeksów
    mkt = eq_ret.mean(axis=1)
    mkt.name = "MKT_EW"

    ls = pf["LongShort"]
    data = pd.concat([ls, mkt], axis=1).dropna()

    print("Liczba obserwacji (equity regression):", len(data))

    # 4. Regresja OLS: LS = alpha + beta * MKT
    Y = data["LongShort"]
    X = sm.add_constant(data["MKT_EW"])

    model = sm.OLS(Y, X).fit()

    print("\n=== EQUITY LS ~ MARKET EW (OLS) ===")
    print(model.summary())

    # 5. Ładna tabelka do pracy
    clean_table = ols_to_table(model)
    clean_table.to_csv(OUT_TABLE_PATH)
    print(f"\nZapisano ładną tabelę do: {OUT_TABLE_PATH}")
    print("\n=== CLEAN TABLE (EQUITY) ===")
    print(clean_table)


if __name__ == "__main__":
    main()
