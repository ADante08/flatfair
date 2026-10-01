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
# 2. Train on 2019-2024
# --------------------------------------------------

train = df[
    (df["month"] >= "2019-01-01")
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
# 4. One-hot encode town and flat type
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
# 5. Features: price lags 1-6 ONLY
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
# 8. Train XGBoost
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
        (y_val.to_numpy() - predictions)
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

print("\n2019-2024 / 6-lag XGBoost:")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

print(
    f"\nImprovement over persistence: "
    f"{improvement:.2f}%"
)

print("\nCurrent 12-lag champion:")
print("Training: 2019-2024")
print("MAE:  $41,817.38")
print("RMSE: $72,586.17")
print("MAPE: 5.68%")


difference = 41817.38 - mae

print("\nComparison:")

if difference > 0:
    print(
        f"6-lag model is better by "
        f"${difference:,.2f} MAE."
    )
else:
    print(
        f"6-lag model is worse by "
        f"${-difference:,.2f} MAE."
    )