# Hydro-financial screening of climate variability and hydropower-utility valuation

Reproducible analysis code for the study:

> Al Zaabi, G., Wehbe, Y., Zhu, Y. & Fu, H. *Slow basin storage anomalies precede valuation
> stress: Case study of Longtan Dam.*

The code screens whether basin-scale hydroclimate variability upstream of the Longtan
Hydropower Station carries a statistically defensible, physically interpretable signal in the
market-adjusted valuation of its listed operator, Guangxi Guiguan Electric Power Co., Ltd.
(GGEPC, ticker 600236). It is a **screening framework**: it reports lagged associations, not
causal attribution and not price prediction.

## Pipeline

1. **Physical.** Delineate the catchment upstream of the dam outlet from HydroBASINS, extract
   daily basin-mean hydroclimate variables (precipitation, temperature, surface runoff,
   baseflow, root-zone soil moisture, evapotranspiration) from public Earth-observation
   datasets in Google Earth Engine, derive water-balance, water-availability and
   dry-heat-stress indicators, and convert every variable into a de-seasonalized, robustly
   standardized anomaly.
2. **Financial.** Convert the daily adjusted close of GGEPC and of the market index into log
   returns, remove broad market movement with a one-factor market model, and build forward
   abnormal-return and forward 20-day valuation-stress responses.
3. **Statistical.** Screen lagged Spearman associations, assess significance with a
   circular-shift surrogate test (5,000 surrogates) that preserves autocorrelation, and control
   multiple testing with the Benjamini-Hochberg false-discovery-rate procedure.

## Contents

| File | Purpose |
|---|---|
| `longtan_hydrofinancial_screening.py` | Full pipeline: extraction, anomaly construction, market model, lagged screening, figures. |
| `robustness_checks.py` | Robustness suite: 5,000 surrogates, non-overlapping windows, moving-block bootstrap CIs, anomaly-definition sensitivity, exclusion of influential market days. |
| `requirements.txt` | Python dependencies. |

## Data sources

**Hydroclimate** (openly available in Google Earth Engine):

| Variable | Collection |
|---|---|
| Precipitation | `UCSB-CHG/CHIRPS/DAILY` |
| 2 m air temperature | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| Surface runoff, baseflow, root-zone soil moisture, evapotranspiration | `NASA/GLDAS/V021/NOAH/G025/T3H` |
| Basin and river network | `WWF/HydroSHEDS/...` |

**Equity and market index** (licensed): daily adjusted close for GGEPC (600236) and the
Shanghai Composite Index (000001) from the China Stock Market and Accounting Research (CSMAR)
database. CSMAR data are licensed and are **not** included in this repository. Supply them
locally as a CSV with the columns `Date`, `Adjusted_Close` (GGEPC) and `Market_Adjusted_Close`
(index); they are used only to compute daily returns.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`numpy`, `pandas` and `matplotlib` are sufficient to reproduce the screening from a prepared
analysis matrix. `earthengine-api`, `geemap`, `geopandas` and `pyproj` are needed only for a
fresh Earth Engine extraction and are imported lazily.

## Usage

Reuse a previously prepared analysis matrix (fastest; no Earth Engine or price data needed):

```bash
python longtan_hydrofinancial_screening.py --matrix analysis_matrix.csv --out ./outputs
```

Build the analysis matrix from scratch (requires an Earth Engine project and a CSMAR price CSV):

```bash
export GEE_PROJECT_ID=your-earth-engine-project
python longtan_hydrofinancial_screening.py --finance csmar_prices.csv --out ./outputs
```

Run the robustness suite on the matrix written by the main script:

```bash
python robustness_checks.py --matrix ./outputs/analysis_matrix.csv --out ./outputs/robustness_summary.csv
```

## Outputs

Written under `--out`:

- `analysis_matrix.csv` — the synchronized daily matrix with anomalies and valuation responses.
- `tables/screening_results.csv` — peak-lag Spearman correlation, surrogate p-value and FDR
  q-value for every hydroclimate x valuation pair.
- `tables/lag_response_*.csv` — lag-response curves for the retained storage signals.
- `figures/` — study-area map and the descriptive, anomaly, correlation, screening-matrix and
  lag-response figures.
- `run_summary.json` — sample span, settings and the number of retained pairs.

## Reproducibility

All random operations use a fixed seed (`SEED = 42`). The published headline associations are
root-zone soil moisture (peak Spearman rho 0.41 at a 90-trading-day lead) and baseflow (0.35 at a
60-trading-day lead) against forward 20-day valuation stress, both at q = 0.003 with 5,000
surrogates, with precipitation showing no defensible signal.

## License

Released under the MIT License (see `LICENSE`). The licensed CSMAR equity data are not covered by
this license and are not redistributed here.
