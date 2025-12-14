from pathlib import Path
from fredapi import Fred
import pandas as pd
import os

# -----------------------------------------
# Ścieżki projektu
# -----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# FRED: 10Y gov bond yields (Europa + UK, PL, Nordics)
# -----------------------------------------
BOND_TICKERS = {
    "DE": "IRLTLT01DEM156N",
    "FR": "IRLTLT01FRM156N",
    "IT": "IRLTLT01ITM156N",
    "ES": "IRLTLT01ESM156N",
    "NL": "IRLTLT01NLM156N",
    "CH": "IRLTLT01CHM156N",
    "UK": "IRLTLT01GBM156N",
    "PL": "IRLTLT01PLM156N",
    "SE": "IRLTLT01SEM156N",
    "NO": "IRLTLT01NOM156N",
    "BE": "IRLTLT01BEM156N",
    "AT": "IRLTLT01ATM156N",
    "PT": "IRLTLT01PTM156N",
    "DK": "IRLTLT01DKM156N",
    "FI": "IRLTLT01FIM156N",
    "IE": "IRLTLT01IEM156N",
}


def main():
    api_key = os.getenv("FRED_API_KEY")
    if api_key is None:
        raise ValueError("Brak zmiennej środowiskowej FRED_API_KEY")

    fred = Fred(api_key=api_key)

    df_list = []
    print("\nPobieram dane 10Y obligacji z FRED...\n")

    for country, code in BOND_TICKERS.items():
        try:
            s = fred.get_series(code)
            s = s.rename(country)
            df_list.append(s)
            print(f"✔ Pobrano: {country}")
        except Exception as e:
            print(f"✖ Błąd dla {country}: {e}")

    yields = pd.concat(df_list, axis=1)
    yields.index = pd.to_datetime(yields.index)
    yields = yields.sort_index()

    # FRED podaje rentowności w %, zamieniamy na ułamki (np. 2.5% -> 0.025)
    yields = yields / 100.0

    # Zapis surowych rentowności (pełna historia)
    raw_path = DATA_RAW / "bond_yields_fred.csv"
    yields.to_csv(raw_path)
    print(f"\nZapisano dane surowe → {raw_path}")

    # Miesięczne zmiany rentowności
    monthly = yields.resample("M").last()
    dy = monthly.diff()
    ret = -dy.dropna(how="all")  # przybliżenie zwrotu obligacji: r ≈ -Δy

    # Przycinamy okres do 2005–2025, żeby było spójne z FX i equity
    ret = ret.loc["2005-01-01":"2025-01-01"]

    ret_path = DATA_PROCESSED / "bond_monthly_returns.csv"
    ret.to_csv(ret_path)
    print(f"Zapisano miesięczne zwroty → {ret_path}")
    print("Shape:", ret.shape)


if __name__ == "__main__":
    main()
