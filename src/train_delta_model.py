import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load ML feature table
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/ml_features.csv",
    parse_dates=["month"]
)

# We need a current price in order to calculate price change
df = df[df["price_now"].notna()].copy()


# --------------------------------------------------
# 2. Target month
# --------------------------------------------------

df["target_month"] = (
    df["month"] + pd.DateOffset(months=6)
)


# --------------------------------------------------
# 3. NEW TARGET:
#    Dollar price change over the next 6 months
# --------------------------------------------------

df["target_delta_6m"] = (
    df["target_price_6m"]
    - df["price_now"]
)


# --------------------------------------------------
# 4. Chronological split
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
# 5. Preserve labels for diagnostics
# --------------------------------------------------

validation_labels = validation[
    ["town", "flat_type"]
].copy()


# --------------------------------------------------
# 6. One-hot encode town and flat type
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
# 7. Features
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

categorical_features = [
    column
    for column in combined.columns
    if column.startswith("town_")
    or column.startswith("flat_type_")
]

features = (
    base_features
    + categorical_features
)


# --------------------------------------------------
# 8. X and y
# --------------------------------------------------

X_train = train[features]
y_train = train["target_delta_6m"]

X_val = validation[features]
y_val = validation["target_delta_6m"]


# --------------------------------------------------
# 9. Persistence baseline
#
# Delta = 0 means:
# future price = current price
# --------------------------------------------------

baseline_price = validation["price_now"]

baseline_mae = mean_absolute_error(
    validation["target_price_6m"],
    baseline_price
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        validation["target_price_6m"],
        baseline_price
    )
)


# --------------------------------------------------
# 10. Train XGBoost
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
# 11. Predict price change
# --------------------------------------------------

predicted_delta = model.predict(X_val)


# Convert predicted change back into actual price
predicted_price = (
    validation["price_now"].to_numpy()
    + predicted_delta
)

actual_price = (
    validation["target_price_6m"].to_numpy()
)


# --------------------------------------------------
# 12. Evaluate
# --------------------------------------------------

model_mae = mean_absolute_error(
    actual_price,
    predicted_price
)

model_rmse = np.sqrt(
    mean_squared_error(
        actual_price,
        predicted_price
    )
)

mape = np.mean(
    np.abs(
        (actual_price - predicted_price)
        / actual_price
    )
) * 100


# --------------------------------------------------
# 13. Compare with baseline
# --------------------------------------------------

improvement = (
    (baseline_mae - model_mae)
    / baseline_mae
    * 100
)

print("\nPersistence baseline:")
print(f"MAE:  ${baseline_mae:,.2f}")
print(f"RMSE: ${baseline_rmse:,.2f}")

print("\nXGBoost delta model:")
print(f"MAE:  ${model_mae:,.2f}")
print(f"RMSE: ${model_rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

print(
    f"\nMAE improvement over baseline: "
    f"{improvement:.2f}%"
)

print("\nComparison with current best Model 1:")
print("Model 1 MAE:  $46,404.30")
print("Model 1 RMSE: $78,387.05")
print("Model 1 MAPE: 6.34%")


# --------------------------------------------------
# 14. Diagnostics
# --------------------------------------------------

results = validation_labels.copy()

results["actual"] = actual_price
results["prediction"] = predicted_price

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