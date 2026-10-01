import pandas as pd
from pyspark.sql import SparkSession

from config import SILVER_MONTHLY_TABLE

spark = SparkSession.builder.getOrCreate()
df = spark.table(SILVER_MONTHLY_TABLE).toPandas()
df["month"] = pd.to_datetime(df["month"])

print("\n=== BASIC DATASET INFO ===")
print("Shape:")
print(df.shape)
print("\nDate range:")
print(df["month"].min(), "to", df["month"].max())
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())

print("\n=== TRANSACTION COUNT SUMMARY ===")
print(df["transaction_count"].describe())
print("\nRows with fewer than 3 transactions:")
print((df["transaction_count"] < 3).sum())
print("\nRows with fewer than 5 transactions:")
print((df["transaction_count"] < 5).sum())
print("\nRows with fewer than 10 transactions:")
print((df["transaction_count"] < 10).sum())

print("\n=== TRANSACTION COUNTS BY FLAT TYPE ===")
print(
    df.groupby("flat_type")["transaction_count"]
    .agg(["count", "mean", "median", "min", "max"])
)

coverage = (
    df.groupby(["town", "flat_type"])
    .agg(
        first_month=("month", "min"),
        last_month=("month", "max"),
        observed_months=("month", "nunique"),
        median_transactions=("transaction_count", "median"),
        minimum_transactions=("transaction_count", "min"),
        total_transactions=("transaction_count", "sum"),
    )
    .reset_index()
)

coverage["expected_months"] = (
    (coverage["last_month"].dt.year - coverage["first_month"].dt.year) * 12
    + (coverage["last_month"].dt.month - coverage["first_month"].dt.month)
    + 1
)
coverage["missing_months"] = coverage["expected_months"] - coverage["observed_months"]

temp = df.copy()
temp["under_3"] = temp["transaction_count"] < 3
temp["under_5"] = temp["transaction_count"] < 5
low_volume = (
    temp.groupby(["town", "flat_type"])
    .agg(
        pct_months_under_3=("under_3", "mean"),
        pct_months_under_5=("under_5", "mean"),
    )
    .reset_index()
)
coverage = coverage.merge(low_volume, on=["town", "flat_type"])
coverage["pct_months_under_3"] *= 100
coverage["pct_months_under_5"] *= 100

print("\n=== SERIES COVERAGE SUMMARY ===")
print(coverage["observed_months"].describe())
print("\nGroups with at least 100 months:", (coverage["observed_months"] >= 100).sum())
print("Groups with at least 90 months:", (coverage["observed_months"] >= 90).sum())
print("Groups with at least 60 months:", (coverage["observed_months"] >= 60).sum())

print("\n=== SHORTEST SERIES ===")
print(coverage.sort_values("observed_months").head(30).to_string(index=False))
print("\n=== SERIES WITH MOST GAPS ===")
print(coverage.sort_values("missing_months", ascending=False).head(30).to_string(index=False))
print("\n=== LOWEST-VOLUME SERIES ===")
print(coverage.sort_values("median_transactions").head(30).to_string(index=False))
print("\n=== STRONGEST SERIES ===")
print(
    coverage.sort_values(
        ["observed_months", "median_transactions"],
        ascending=[False, False],
    ).head(30).to_string(index=False)
)
