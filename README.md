## Cross-Asset Skewness in European Markets

This repository contains the complete codebase used in the empirical analysis conducted in the thesis
“Cross-Asset Skewness in European Markets.”

The project investigates whether skewness-based risk premia:

are common across asset classes, or

represent independent sources of risk and return.

The analysis focuses on European markets and covers the following asset classes:

- **Equities**
- **Foreign Exchange (FX)**
- **Government Bonds**
- **Commodities**


All results are generated using monthly data and skewness-based long–short strategies.

## Project Structure

```text
src/
├── data_download.py              # Data acquisition
├── data_download_fx.py
├── data_download_bonds_fred.py
├── data_download_commodities.py
├── skewness.py                   # Rolling skewness computation
├── skewness_fx.py
├── skewness_bonds.py
├── skewness_commodities.py
├── portfolios.py                 # Long–Short portfolio construction
├── portfolios_fx.py
├── portfolios_bonds.py
├── portfolios_commodities.py
├── regression_equity.py          # OLS regressions
├── regression_bonds.py
├── analysis_equity.py            # Performance statistics
├── analysis_fx.py
├── analysis_bonds.py
├── analysis_commodities.py

data/
├── raw/                          # Raw downloaded data
├── processed/                    # Processed returns, signals, portfolios

requirements.txt
README.md
```

## Environment Setup

Create and activate a Python environment (recommended: conda), then install dependencies:
```
pip install -r requirements.txt
```
## Replicating the Analysis
The empirical results can be reproduced by running the scripts in the following order.
```
1. Data Download
python src/data_download.py
python src/data_download_fx.py
python src/data_download_bonds_fred.py
python src/data_download_commodities.py
2. Rolling Skewness Estimation (12M)
python src/skewness.py
python src/skewness_fx.py
python src/skewness_bonds.py
python src/skewness_commodities.py

3. Long–Short Portfolio Construction
python src/portfolios.py
python src/portfolios_fx.py
python src/portfolios_bonds.py
python src/portfolios_commodities.py

4. Performance Analysis
python src/analysis_equity.py
python src/analysis_fx.py
python src/analysis_bonds.py
python src/analysis_commodities.py

5. Regression Analysis
python src/regression_equity.py
python src/regression_bonds.py
```
## Methodological Overview

Skewness is computed as 12-month rolling sample skewness of monthly returns.

In each month, assets are sorted by lagged skewness.

Portfolios are constructed by going long assets with highest skewness and short assets with lowest skewness.

Performance is evaluated using annualized mean returns, volatility, Sharpe ratios, t-statistics, and maximum drawdowns.

Linear regressions relate skewness portfolio returns to market benchmarks.

## Notes

The repository is designed for full reproducibility of the empirical results reported in the thesis.

Intermediate files are saved to data/processed/.

Jupyter notebooks used for exploratory analysis are intentionally excluded from the repository.


