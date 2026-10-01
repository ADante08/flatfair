import pandas as pd

# Load monthly market data
df = pd.read_csv(
    "data/processed/monthly_market.csv",
    parse_dates=["month"]
)

# Keep the main flat types for the first model
df = df[
    df["flat_type"].isin(["3 ROOM", "4 ROOM", "5 ROOM"])
].copy()

# We only need these columns for Model 1
df = df[
    [
        "month",
        "town",
        "flat_type",
        "median_price",
        "transaction_count"
    ]
]

# --------------------------------------------------
# Create a complete calendar for every town × flat type
# --------------------------------------------------

pairs = df[
    ["town", "flat_type"]
].drop_duplicates()

months = pd.DataFrame({
    "month": pd.date_range(
        start=df["month"].min(),
        end=df["month"].max(),
        freq="MS"
    )
})

calendar = pairs.merge(
    months,
    how="cross"
)

df = calendar.merge(
    df,
    on=["month", "town", "flat_type"],
    how="left"
)

# A missing month means zero recorded transactions
df["transaction_count"] = (
    df["transaction_count"]
    .fillna(0)
)

# Sort each market chronologically
df = df.sort_values(
    ["town", "flat_type", "month"]
)

# --------------------------------------------------
# Time features
# --------------------------------------------------

start_year = df["month"].dt.year.min()
start_month = df["month"].dt.month.min()

df["month_index"] = (
    (df["month"].dt.year - start_year) * 12
    + df["month"].dt.month
    - start_month
)

df["month_of_year"] = df["month"].dt.month

# --------------------------------------------------
# Historical price features
# --------------------------------------------------

groups = df.groupby(
    ["town", "flat_type"]
)["median_price"]

df["price_now"] = df["median_price"]

df["price_lag_1"] = groups.shift(1)
df["price_lag_2"] = groups.shift(2)
df["price_lag_3"] = groups.shift(3)
df["price_lag_4"] = groups.shift(4)
df["price_lag_5"] = groups.shift(5)
df["price_lag_6"] = groups.shift(6)
df["price_lag_7"] = groups.shift(7)
df["price_lag_8"] = groups.shift(8)
df["price_lag_9"] = groups.shift(9)
df["price_lag_10"] = groups.shift(10)
df["price_lag_11"] = groups.shift(11)
df["price_lag_12"] = groups.shift(12)

df["target_price_6m"] = groups.shift(-6)

# We cannot train on rows where the future price is unknown
df_ml = df[
    df["target_price_6m"].notna()
].copy()

# Remove original median_price because price_now already contains it
df_ml = df_ml.drop(
    columns=["median_price"]
)

# Save
df_ml.to_csv(
    "data/processed/ml_features.csv",
    index=False
)

print("ML dataset shape:", df_ml.shape)

print("\nColumns:")
print(df_ml.columns.tolist())

print("\nFirst 10 rows:")
print(df_ml.head(10))

print("\nSaved to data/processed/ml_features.csv")