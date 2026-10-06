# HomeSignal

**Forward-looking HDB resale price forecasting for Singapore**

HomeSignal is a machine-learning project developed for the **Databricks AI Social Impact (DAISI) Challenge Singapore**, Track C1: HDB resale market intelligence and affordability.

Most housing tools show buyers what happened in the past. HomeSignal asks a more useful question:

> **Where are HDB resale prices likely to go next?**

Our working forecasting model predicts **median HDB resale prices 1–6 months ahead by town and flat type**.

---

## Why HomeSignal

Buying an HDB resale flat is one of the largest financial decisions many Singaporeans will make, yet most public housing tools are primarily backward-looking.

HomeSignal aims to provide buyers with forward-looking signals that can help them compare towns, flat types, and possible price movement before making a decision.

---

## Working forecasting model

The forecasting core is already built.

In our primary backtest, HomeSignal beat a simple **no-change persistence baseline at every 1–6 month forecast horizon**.

| Horizon | Persistence MAE | HomeSignal MAE | Improvement |
| --- | ---: | ---: | ---: |
| +1 month | S$45,235 | S$37,690 | 16.68% |
| +2 months | S$47,193 | S$38,146 | **19.17%** |
| +3 months | S$46,337 | S$39,482 | 14.79% |
| +4 months | S$46,753 | S$39,434 | 15.66% |
| +5 months | S$47,569 | S$40,916 | 13.98% |
| +6 months | S$49,824 | S$42,622 | 14.46% |

Across all six horizons, HomeSignal reduces MAE by approximately **14–19%** compared with persistence.

---

## What the model predicts

HomeSignal forecasts monthly median resale prices for:

**month × town × flat type**

The current model focuses on:

- 3 ROOM
- 4 ROOM
- 5 ROOM

Each forecast horizon from 1 to 6 months is handled by a separate model rather than recursively feeding predictions back into the next forecast.

---

## Data

### Currently used

1. **HDB Resale Flat Prices**
   - Source: HDB / data.gov.sg
   - Coverage used: Jan 2017 to Sep 2026
   - 241,597 transactions in the project snapshot

2. **HDB Resale Price Index**
   - Source: HDB / data.gov.sg
   - Used as a broader market-level signal

### Planned

3. **Household Employment Income**
   - Source: SingStat / data.gov.sg
   - Planned for the affordability layer

---

## Features

The model uses:

- current median resale price
- price lags from 1 to 12 months
- HDB Resale Price Index
- quarterly RPI growth
- national HDB transaction volume
- town
- flat type

To reduce time-series leakage, the RPI feature uses only the **previous completed quarter** rather than information that would not yet have been available at the forecast date.

---

## Model

HomeSignal uses direct XGBoost forecasting models.

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:absoluteerror",
    random_state=42,
    n_jobs=-1,
)
