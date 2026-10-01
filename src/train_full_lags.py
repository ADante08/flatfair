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

# Require a current observed price.
# Lag values may still be NaN.
df = df[df["price_now"].notna()].copy()


# --------------------------------------------------
# 2. Work out which month the target belongs to
# --------------------------------------------------

df["target_month"] = (
    df["month"] + pd.DateOffset(months=6)
)


# --------------------------------------------------
# 3. Chronological train / validation / test split
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
# 4. Save validation labels before one-hot encoding
# --------------------------------------------------

validation_labels = validation[
    ["town", "flat_type"]
].copy()


# --------------------------------------------------
# 5. Convert categorical variables into numbers
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
# 6. Choose model features
# --------------------------------------------------

base_features = [
    "month_index",
    "month_of_year",
    "price_now",
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_4",
    "price_lag_5",
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

features = base_features + categorical_features

target = "target_price_6m"


# --------------------------------------------------
# 7. Create X and y
# --------------------------------------------------

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]


# --------------------------------------------------
# 8. Persistence baseline
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

print("\nPersistence baseline:")
print(f"MAE:  ${baseline_mae:,.2f}")
print(f"RMSE: ${baseline_rmse:,.2f}")


# --------------------------------------------------
# 9. Train XGBoost
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:absoluteerror",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 10. Predict validation set
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

print("\nXGBoost:")
print(f"MAE:  ${model_mae:,.2f}")
print(f"RMSE: ${model_rmse:,.2f}")


# --------------------------------------------------
# 11. Compare with baseline
# --------------------------------------------------

improvement = (
    (baseline_mae - model_mae)
    / baseline_mae
    * 100
)

print(
    f"\nMAE improvement over baseline: "
    f"{improvement:.2f}%"
)


# --------------------------------------------------
# 12. Detailed validation diagnostics
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

print("\nOverall MAPE:")
print(
    f"{results['percentage_error'].mean():.2f}%"
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