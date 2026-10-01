import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load monthly market data
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/monthly_market.csv",
    parse_dates=["month"]
)

# Keep only 3, 4 and 5 room flats
df = df[
    df["flat_type"].isin(["3 ROOM", "4 ROOM", "5 ROOM"])
].copy()


# --------------------------------------------------
# 2. Create complete monthly calendar
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

# No transactions in a missing month
df["transaction_count"] = (
    df["transaction_count"]
    .fillna(0)
)

df = df.sort_values(
    ["town", "flat_type", "month"]
)


# --------------------------------------------------
# 3. Time features
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
# 4. Historical price features
# --------------------------------------------------

groups = df.groupby(
    ["town", "flat_type"]
)["median_price"]

df["price_now"] = df["median_price"]

df["price_lag_1"] = groups.shift(1)
df["price_lag_3"] = groups.shift(3)
df["price_lag_6"] = groups.shift(6)
df["price_lag_12"] = groups.shift(12)

# Price exactly 6 months later
df["target_price_6m"] = groups.shift(-6)


# --------------------------------------------------
# 5. Remove rows that cannot be used for training
# --------------------------------------------------

# Need current price and future target.
# Historical lags and enriched features may still contain NaN.
df = df[
    df["price_now"].notna()
    & df["target_price_6m"].notna()
].copy()


# --------------------------------------------------
# 6. Work out target month
# --------------------------------------------------

df["target_month"] = (
    df["month"] + pd.DateOffset(months=6)
)


# --------------------------------------------------
# 7. Chronological split
# --------------------------------------------------

train = df[
    df["target_month"] <= "2024-12-01"
].copy()

validation = df[
    (df["target_month"] >= "2025-01-01")
    & (df["target_month"] <= "2025-12-01")
].copy()

test = df[
    df["target_month"] >= "2026-01-01"
].copy()

print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Test rows:", len(test))


# --------------------------------------------------
# 8. Preserve validation labels for diagnostics
# --------------------------------------------------

validation_labels = validation[
    ["town", "flat_type"]
].copy()


# --------------------------------------------------
# 9. One-hot encode categorical variables
# --------------------------------------------------

combined = pd.concat(
    [train, validation, test],
    ignore_index=True
)

combined = pd.get_dummies(
    combined,
    columns=["town", "flat_type"],
    dtype=int
)

train_length = len(train)
validation_length = len(validation)

train = combined.iloc[
    :train_length
].copy()

validation = combined.iloc[
    train_length:
    train_length + validation_length
].copy()

test = combined.iloc[
    train_length + validation_length:
].copy()


# --------------------------------------------------
# 10. Features
# --------------------------------------------------

base_features = [
    "month_index",
    "month_of_year",
    "price_now",
    "price_lag_1",
    "price_lag_3",
    "price_lag_6",
    "price_lag_12",
    "transaction_count"
]

# NEW FEATURES FOR MODEL 3
enriched_features = [
    "median_price_per_sqm",
    "median_floor_area",
    "median_remaining_lease",
    "median_storey"
]

categorical_features = [
    column
    for column in combined.columns
    if column.startswith("town_")
    or column.startswith("flat_type_")
]

features = (
    base_features
    + enriched_features
    + categorical_features
)

target = "target_price_6m"


# --------------------------------------------------
# 11. Create X and y
# --------------------------------------------------

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]


# --------------------------------------------------
# 12. Persistence baseline
# --------------------------------------------------

baseline_prediction = validation["price_now"]

baseline_mae = mean_absolute_error(
    y_val,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        baseline_prediction
    )
)


# --------------------------------------------------
# 13. Train XGBoost
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 14. Predict validation set
# --------------------------------------------------

predictions = model.predict(X_val)

model_mae = mean_absolute_error(
    y_val,
    predictions
)

model_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

mape = np.mean(
    np.abs(
        (y_val.to_numpy() - predictions)
        / y_val.to_numpy()
    )
) * 100


# --------------------------------------------------
# 15. Compare with baseline
# --------------------------------------------------

improvement = (
    (baseline_mae - model_mae)
    / baseline_mae
    * 100
)

print("\nPersistence baseline:")
print(f"MAE:  ${baseline_mae:,.2f}")
print(f"RMSE: ${baseline_rmse:,.2f}")

print("\nEnriched XGBoost:")
print(f"MAE:  ${model_mae:,.2f}")
print(f"RMSE: ${model_rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

print(
    f"\nMAE improvement over baseline: "
    f"{improvement:.2f}%"
)

print("\nComparison with Model 1:")
print("Model 1 MAE:  $46,404.30")
print("Model 1 RMSE: $78,387.05")
print("Model 1 MAPE: 6.34%")


# --------------------------------------------------
# 16. Diagnostics
# --------------------------------------------------

results = validation_labels.copy()

results["actual"] = y_val.to_numpy()
results["prediction"] = predictions

results["absolute_error"] = abs(
    results["actual"]
    - results["prediction"]
)

results["percentage_error"] = (
    results["absolute_error"]
    / results["actual"]
    * 100
)

print("\nMAE by flat type:")
print(
    results.groupby("flat_type")["absolute_error"]
    .mean()
)

print("\nMAE by town:")
print(
    results.groupby("town")["absolute_error"]
    .mean()
    .sort_values()
)