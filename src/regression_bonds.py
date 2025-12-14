# src/regression_bonds.py

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ─── Ścieżki ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

BOND_RET_PATH = DATA_DIR / "bond_monthly_returns.csv"
BOND_PF_PATH = DATA_DIR / "portfolio_bonds_skewness_ls.csv"
OUT_TABLE_PATH = DATA_DIR / "regression_bonds_clean_table.csv"


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
    bond_ret = pd.read_csv(BOND_RET_PATH, index_col=0, parse_dates=True)
    pf = pd.read_csv(BOND_PF_PATH, index_col=0, parse_dates=True)

    # Spójny zakres dat
    bond_ret = bond_ret.loc["2005-01-01":"2025-01-01"]
    pf = pf.loc["2005-01-01":"2025-01-01"]

    # 2. Wyrównanie dat
    idx = bond_ret.index.intersection(pf.index)
    bond_ret = bond_ret.loc[idx].sort_index()
    pf = pf.loc[idx].sort_index()

    # 3. Bond market = średnia z wszystkich serii 10Y
    bmk = bond_ret.mean(axis=1)
    bmk.name = "BOND_MKT_EW"

    ls = pf["LongShort"]
    data = pd.concat([ls, bmk], axis=1).dropna()

    print("Liczba obserwacji (bonds regression):", len(data))

    # 4. Regresja OLS: LS = alpha + beta * BOND_MKT
    Y = data["LongShort"]
    X = sm.add_constant(data["BOND_MKT_EW"])

    model = sm.OLS(Y, X).fit()

    print("\n=== BONDS LS ~ BOND MARKET EW (OLS) ===")
    print(model.summary())

    # 5. Ładna tabelka do pracy
    clean_table = ols_to_table(model)
    clean_table.to_csv(OUT_TABLE_PATH)
    print(f"\nZapisano ładną tabelę do: {OUT_TABLE_PATH}")
    print("\n=== CLEAN TABLE (BONDS) ===")
    print(clean_table)


if __name__ == "__main__":
    main()
