from pathlib import Path
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

CMDTY_TICKERS = {
    "CL": "CL=F",   # WTI Crude
    "BZ": "BZ=F",   # Brent
    "NG": "NG=F",   # Nat Gas
    "GC": "GC=F",   # Gold
    "SI": "SI=F",   # Silver
    "HG": "HG=F",   # Copper
    "ZC": "ZC=F",   # Corn
    "ZW": "ZW=F",   # Wheat
    "ZS": "ZS=F",   # Soybeans
    "SB": "SB=F",   # Sugar
}


def main(start: str = "2005-01-01", end: str = "2025-01-01") -> None:
    print("Pobieram dane futures na surowce z Yahoo Finance...")

    tickers = list(CMDTY_TICKERS.values())
    data = yf.download(
        tickers, start=start, end=end, interval="1d", auto_adjust=False, progress=True
    )

    # Bierzemy ceny zamknięcia
    if "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data["Close"].copy()

    # Mapujemy kolumny na nasze skróty
    prices = prices.rename(columns={v: k for k, v in CMDTY_TICKERS.items()})
    prices = prices.sort_index()

    raw_path = DATA_RAW / "commodities_daily_prices.csv"
    prices.to_csv(raw_path)
    print(f"Zapisano surowe ceny: {raw_path}")

    # Miesięczne zwroty logarytmiczne
    monthly = prices.resample("M").last()
    rets = monthly.pct_change().apply(lambda x: np.log1p(x)).dropna(how="all")

    # Przycinamy okres (dla spójności z resztą)
    rets = rets.loc["2005-01-01":"2025-01-01"]

    out_path = DATA_PROCESSED / "commodities_monthly_returns.csv"
    rets.to_csv(out_path)
    print(f"Zapisano miesięczne zwroty: {out_path}")
    print("Shape:", rets.shape)


if __name__ == "__main__":
    import numpy as np
    main()
