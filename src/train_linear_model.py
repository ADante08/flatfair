import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load ML feature table
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
# 2. Chronological train / validation split
# --------------------------------------------------

train = df[
    df["target_month"] <= "2024-12-01"
].copy()

validation = df[
    (df["target_month"] >= "2025-01-01")
    & (df["target_month"] <= "2025-12-01")
].copy()

print("Training rows:", len(train))
print("Validation rows:", len(validation))


# --------------------------------------------------
# 3. One-hot encode town and flat type
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
# 4. Same features as our best XGBoost model
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
# 5. X and y
# --------------------------------------------------

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]


# --------------------------------------------------
# 6. LinearRegression cannot handle NaN,
#    so fill missing lag values using training medians
# --------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_val_imputed = imputer.transform(
    X_val
)


# --------------------------------------------------
# 7. Train least-squares linear regression
# --------------------------------------------------

model = LinearRegression()

model.fit(
    X_train_imputed,
    y_train
)


# --------------------------------------------------
# 8. Predict
# --------------------------------------------------

predictions = model.predict(
    X_val_imputed
)


# --------------------------------------------------
# 9. Evaluate
# --------------------------------------------------

linear_mae = mean_absolute_error(
    y_val,
    predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

linear_mape = np.mean(
    np.abs(
        (y_val.to_numpy() - predictions)
        / y_val.to_numpy()
    )
) * 100


# --------------------------------------------------
# 10. Persistence baseline
# --------------------------------------------------

baseline_predictions = validation["price_now"]

baseline_mae = mean_absolute_error(
    y_val,
    baseline_predictions
)


# --------------------------------------------------
# 11. Results
# --------------------------------------------------

print("\nPersistence baseline:")
print(f"MAE: ${baseline_mae:,.2f}")

print("\nLinear regression (line of best fit):")
print(f"MAE:  ${linear_mae:,.2f}")
print(f"RMSE: ${linear_rmse:,.2f}")
print(f"MAPE: {linear_mape:.2f}%")

print("\nBest XGBoost:")
print("MAE:  $42,702.72")
print("RMSE: $73,854.58")
print("MAPE: 5.84%")


# --------------------------------------------------
# 12. Compare directly with XGBoost
# --------------------------------------------------

xgb_mae = 42702.72

difference = (
    linear_mae - xgb_mae
)

percentage_better = (
    difference / linear_mae * 100
)

print("\nComparison:")

if linear_mae > xgb_mae:
    print(
        f"XGBoost beats linear regression by "
        f"${difference:,.2f} MAE."
    )

    print(
        f"That is {percentage_better:.2f}% lower MAE "
        f"than linear regression."
    )

else:
    print(
        f"Linear regression beats XGBoost by "
        f"${-difference:,.2f} MAE."
    )