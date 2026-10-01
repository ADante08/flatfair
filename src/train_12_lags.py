import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/ml_features.csv",
    parse_dates=["month"]
)

df = df[df["price_now"].notna()].copy()

df["target_month"] = (
    df["month"] + pd.DateOffset(months=6)
)


# --------------------------------------------------
# 2. Train on 2020-2024
# --------------------------------------------------

train = df[
    (df["month"] >= "2020-01-01")
    & (df["target_month"] <= "2024-12-01")
].copy()


# --------------------------------------------------
# 3. Validate on 2025
# --------------------------------------------------

validation = df[
    (df["target_month"] >= "2025-01-01")
    & (df["target_month"] <= "2025-12-01")
].copy()

print("Training rows:", len(train))
print("Validation rows:", len(validation))


# --------------------------------------------------
# 4. Encode town and flat type
# --------------------------------------------------

combined = pd.concat(
    [train, validation],
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

validation = combined.iloc[
    train_length:
].copy()


# --------------------------------------------------
# 5. Features: ALL price lags 1-12
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
    "price_lag_7",
    "price_lag_8",
    "price_lag_9",
    "price_lag_10",
    "price_lag_11",
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
# 6. X and y
# --------------------------------------------------

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]


# --------------------------------------------------
# 7. Persistence baseline
# --------------------------------------------------

baseline_predictions = validation["price_now"]

baseline_mae = mean_absolute_error(
    y_val,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        baseline_predictions
    )
)


# --------------------------------------------------
# 8. Train our current best XGBoost configuration
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

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 9. Predict
# --------------------------------------------------

predictions = model.predict(
    X_val
)


# --------------------------------------------------
# 10. Metrics
# --------------------------------------------------

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

mape = np.mean(
    np.abs(
        (
            y_val.to_numpy()
            - predictions
        )
        / y_val.to_numpy()
    )
) * 100

improvement = (
    (baseline_mae - mae)
    / baseline_mae
    * 100
)


# --------------------------------------------------
# 11. Results
# --------------------------------------------------

print("\nPersistence baseline:")
print(f"MAE:  ${baseline_mae:,.2f}")
print(f"RMSE: ${baseline_rmse:,.2f}")

print("\n12-lag XGBoost:")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

print(
    f"\nImprovement over persistence: "
    f"{improvement:.2f}%"
)

print("\nCurrent champion:")
print("Training: 2020-2024")
print("Lags: 1-6 and 12")
print("MAE:  $42,144.55")
print("RMSE: $73,613.17")
print("MAPE: 5.72%")


difference = 42144.55 - mae

print("\nComparison:")

if difference > 0:
    print(
        f"12-lag model improves MAE by "
        f"${difference:,.2f}"
    )
else:
    print(
        f"12-lag model is worse by "
        f"${-difference:,.2f}"
    )