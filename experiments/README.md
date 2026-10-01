# FlatFair ML Experiments

This folder contains the main experiments used to select the final FlatFair forecasting architecture.

The production forecasting pipeline is kept in `src/`. These experiments document why particular modelling choices were made.

## 1. Baseline

The primary benchmark is a persistence forecast:

\[
\text{future price} = \text{current price}
\]

A machine-learning model is considered useful only if it improves on this baseline.

---

## 2. Training Strategy

We tested several approaches:

- fixed historical training
- rolling-window training
- recency-weighted training
- persistence blending

Recency-weighted training performed more robustly during the 2025–2026 market regime shift.

### Final choice

- Monthly walk-forward retraining
- 24-month recency half-life
- 90% XGBoost forecast + 10% persistence forecast

---

## 3. Target Formulation

We tested:

- direct future-price prediction
- prediction of the price change from persistence

The direct price-level formulation performed better.

### 2026 Jan–Sep blend MAE

| Target | MAE |
|---|---:|
| Direct future price | $45,389.99 |
| Change from persistence | $47,829.14 |

### Final choice

Predict the future price directly.

---

## 4. Market-State Feature Ablation

Initial candidate market features:

- HDB Resale Price Index
- quarterly RPI growth
- year-on-year RPI growth
- transaction volume
- 3-month transaction-volume growth

The ablation experiments showed that:

- `hdb_rpi` was useful
- `rpi_quarterly_growth` was useful
- `transaction_volume` added useful information
- `rpi_yoy_growth` reduced performance
- `volume_growth_3m` did not improve the final model

### Final market features

```text
hdb_rpi
rpi_quarterly_growth
transaction_volume
