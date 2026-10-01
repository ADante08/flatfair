import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load the already-built ML feature table
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/ml_features.csv",
    parse_dates=["month"]
)

df = df[df["price_now"].notna()].copy()

# target_month is the month whose price we are predicting
df["target_month"] = (
    df["month"] + pd.DateOffset(months=6)
)


# --------------------------------------------------
# 2. LOCKED TRAINING SET: 2019-2024 ONLY
# --------------------------------------------------

train = df[
    (df["month"] >= "2019-01-01")
    & (df["target_month"] <= "2024-12-01")
].copy()


# --------------------------------------------------
# 3. UNTOUCHED FINAL TEST SET: 2026
# --------------------------------------------------

test = df[
    (df["target_month"] >= "2026-01-01")
    & (df["target_month"] <= "2026-12-01")
].copy()


print("Training rows:", len(train))
print("2026 test rows:", len(test))

print(
    "\nTraining target range:",
    train["target_month"].min(),
    "to",
    train["target_month"].max()
)

print(
    "Test target range:",
    test["target_month"].min(),
    "to",
    test["target_month"].max()
)


# --------------------------------------------------
# 4. Keep original labels for diagnostics
# --------------------------------------------------

test_labels = test[
    [
        "month",
        "target_month",
        "town",
        "flat_type"
    ]
].copy()


# --------------------------------------------------
# 5. One-hot encode town and flat type
# --------------------------------------------------

combined = pd.concat(
    [train, test],
    ignore_index=True
)

combined = pd.get_dummies(
    combined,
    columns=["town", "flat_type"],
    dtype=int
)

train_length = len(train)

train = combined.iloc[
    :train_length
].copy()

test = combined.iloc[
    train_length:
].copy()


# --------------------------------------------------
# 6. LOCKED FEATURE SET
# --------------------------------------------------

price_features = [
    "price_now",
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_4",
    "price_lag_5",
    "price_lag_6",
    "price_lag_7",
    "price_lag_8",
    "price_lag_9",
    "price_lag_10",
    "price_lag_11",
    "price_lag_12"
]

categorical_features = [
    column
    for column in combined.columns
    if column.startswith("town_")
    or column.startswith("flat_type_")
]

features = (
    price_features
    + categorical_features
)

target = "target_price_6m"


# --------------------------------------------------
# 7. Build X and y
# --------------------------------------------------

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# --------------------------------------------------
# 8. Persistence baseline
#
# Predict:
# future price = current price
# --------------------------------------------------

baseline_predictions = test["price_now"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_mape = np.mean(
    np.abs(
        (
            y_test.to_numpy()
            - baseline_predictions.to_numpy()
        )
        / y_test.to_numpy()
    )
) * 100


# --------------------------------------------------
# 9. LOCKED XGBOOST MODEL
#
# DO NOT tune these using 2026.
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:absoluteerror",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 10. Train ONLY on 2019-2024
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 11. Predict 2026
# --------------------------------------------------

predictions = model.predict(
    X_test
)


# --------------------------------------------------
# 12. Final test metrics
# --------------------------------------------------

test_mae = mean_absolute_error(
    y_test,
    predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

test_mape = np.mean(
    np.abs(
        (
            y_test.to_numpy()
            - predictions
        )
        / y_test.to_numpy()
    )
) * 100


# --------------------------------------------------
# 13. Compare with persistence
# --------------------------------------------------

improvement = (
    (baseline_mae - test_mae)
    / baseline_mae
    * 100
)


print("\n==============================================")
print("FINAL 2026 TEST RESULTS")
print("==============================================")

print("\nPersistence baseline:")

print(
    f"MAE:  ${baseline_mae:,.2f}"
)

print(
    f"RMSE: ${baseline_rmse:,.2f}"
)

print(
    f"MAPE: {baseline_mape:.2f}%"
)


print("\nLocked XGBoost model:")

print(
    f"MAE:  ${test_mae:,.2f}"
)

print(
    f"RMSE: ${test_rmse:,.2f}"
)

print(
    f"MAPE: {test_mape:.2f}%"
)

print(
    f"\nMAE improvement over persistence: "
    f"{improvement:.2f}%"
)


# --------------------------------------------------
# 14. Save individual 2026 predictions
# --------------------------------------------------

results = test_labels.copy()

results["actual_price"] = (
    y_test.to_numpy()
)

results["predicted_price"] = (
    predictions
)

results["baseline_price"] = (
    test["price_now"].to_numpy()
)

results["absolute_error"] = np.abs(
    results["actual_price"]
    - results["predicted_price"]
)

results["percentage_error"] = (
    results["absolute_error"]
    / results["actual_price"]
    * 100
)


results.to_csv(
    "data/processed/final_2026_predictions.csv",
    index=False
)

print(
    "\nSaved predictions to "
    "data/processed/final_2026_predictions.csv"
)