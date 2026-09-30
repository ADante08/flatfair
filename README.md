# FlatFair

FlatFair is a project for the Databricks AI Social Impact Challenge Singapore.

## Problem

HDB resale prices vary across towns, flat types, lease lengths, floor areas, and market conditions. FlatFair aims to help users understand historical resale trends and forecast short-term resale price movements.

## Initial MVP

The first prototype will:

1. Retrieve official HDB resale transaction data from data.gov.sg.
2. Clean and transform the data.
3. Analyse resale price trends by town and flat type.
4. Train a machine learning model to forecast resale prices up to six months ahead.
5. Compare the ML model against a simple baseline.
6. Display the results in an interactive application.

## Project Structure

- `src/` — data and machine learning code
- `app/` — application code
- `data/` — local datasets, not committed to Git
- `notebooks/` — Databricks notebooks
- `docs/` — project documentation
