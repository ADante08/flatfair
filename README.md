# FlatFair

**Forward-looking HDB resale price forecasting for Singapore**

FlatFair is a machine-learning project developed for the **Databricks AI Social Impact (DAISI) Challenge Singapore**, Track C1: HDB resale market intelligence and affordability.

Instead of only showing buyers what HDB prices have done historically, FlatFair asks a more useful question:

> **Where are HDB resale prices likely to go next?**

We built a working forecasting pipeline that predicts **median HDB resale prices 1–6 months ahead by town and flat type**.

## Key Result

FlatFair beats a strong **persistence baseline** — the assumption that future prices simply remain equal to today's price — at **every forecast horizon from 1 to 6 months**.

| Horizon | Persistence MAE | FlatFair MAE | Improvement |
|---|---:|---:|---:|
| +1 month | S$45,235 | S$37,690 | 16.68% |
| +2 months | S$47,193 | S$38,146 | **19.17%** |
| +3 months | S$46,337 | S$39,482 | 14.79% |
| +4 months | S$46,753 | S$39,434 | 15.66% |
| +5 months | S$47,569 | S$40,916 | 13.98% |
| +6 months | S$49,824 | S$42,622 | 14.46% |

**FlatFair reduces MAE by approximately 14–19% across all six horizons, reaching nearly 20% improvement at the +2 month horizon.**

## What FlatFair Predicts

The model predicts monthly median resale prices at the level of:

```text
month × town × flat type
