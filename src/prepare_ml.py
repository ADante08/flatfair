import pandas as pd

#  Load the monthly market dataset
df = pd.read_csv(
    "data/processed/monthly_market.csv",
    parse_dates=["month"]
)

# Keep the main flat types with enough transaction volume
df = df[
    df["flat_type"].isin(["3 ROOM", "4 ROOM", "5 ROOM"])
].copy()

# Measure the quality of each town × flat type time series
series_quality = (
    df.groupby(["town", "flat_type"])
    .agg(
        observed_months=("month", "nunique"),
        first_month=("month", "min"),
        last_month=("month", "max"),
        median_transactions=("transaction_count", "median")
    )
    .reset_index()
)

# Calculate how many months should exist between first and last observation
series_quality["expected_months"] = (
    (series_quality["last_month"].dt.year - series_quality["first_month"].dt.year) * 12
    + (series_quality["last_month"].dt.month - series_quality["first_month"].dt.month)
    + 1
)

# Calculate how many months are missing
series_quality["missing_months"] = (
    series_quality["expected_months"]
    - series_quality["observed_months"]
)

# Keep only sufficiently reliable time series
eligible_series = series_quality[
    (series_quality["observed_months"] >= 90)
    & (series_quality["missing_months"] <= 12)
    & (series_quality["median_transactions"] >= 5)
]

# Keep only town × flat type pairs that passed the filter
eligible_keys = eligible_series[
    ["town", "flat_type"]
]

# Merge back with the original monthly dataset
df_ml = df.merge(
    eligible_keys,
    on=["town", "flat_type"],
    how="inner"
)

# Print what survived
print("Eligible series:", len(eligible_series))
print("Rows kept for ML:", len(df_ml))

print(
    eligible_series[
        [
            "town",
            "flat_type",
            "observed_months",
            "missing_months",
            "median_transactions"
        ]
    ].to_string(index=False)
)

# Save the filtered dataset
df_ml.to_csv(
    "data/processed/ml_base.csv",
    index=False
)

print("\nSaved to data/processed/ml_base.csv")