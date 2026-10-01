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
# 2. One-hot encode town and flat type
# --------------------------------------------------

df = pd.get_dummies(
    df,
    columns=["town", "flat_type"],
    dtype=int
)


# --------------------------------------------------
# 3. Simplified winning feature set
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
    for column in df.columns
    if column.startswith("town_")
    or column.startswith("flat_type_")
]

features = (
    price_features
    + categorical_features
)

target = "target_price_6m"


# --------------------------------------------------
# 4. Fixed 2025 validation set
# --------------------------------------------------

validation = df[
    (df["target_month"] >= "2025-01-01")
    & (df["target_month"] <= "2025-12-01")
].copy()

X_val = validation[features]
y_val = validation[target]

print(
    "2025 validation rows:",
    len(validation)
)


# --------------------------------------------------
# 5. Persistence baseline
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

print("\nPersistence baseline:")
print(f"MAE:  ${baseline_mae:,.2f}")
print(f"RMSE: ${baseline_rmse:,.2f}")


# --------------------------------------------------
# 6. Training windows
# --------------------------------------------------

training_windows = [
    ("2017-01-01", "2017-2024"),
    ("2018-01-01", "2018-2024"),
    ("2019-01-01", "2019-2024"),
    ("2020-01-01", "2020-2024")
]


# --------------------------------------------------
# 7. Train identical model for each window
# --------------------------------------------------

results = []

for start_date, label in training_windows:

    train = df[
        (df["month"] >= start_date)
        & (df["target_month"] <= "2024-12-01")
    ].copy()

    X_train = train[features]
    y_train = train[target]

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

    predictions = model.predict(
        X_val
    )

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

    results.append({
        "training_window": label,
        "training_rows": len(train),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "improvement_vs_persistence": improvement
    })

    print("\n==============================")
    print(f"TRAINING WINDOW: {label}")
    print("==============================")

    print(f"Training rows: {len(train)}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")

    print(
        f"Improvement over persistence: "
        f"{improvement:.2f}%"
    )


# --------------------------------------------------
# 8. Rank results
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "MAE"
)

print("\n\n==============================================")
print("TRAINING WINDOW COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "${:,.2f}".format,
            "RMSE": "${:,.2f}".format,
            "MAPE": "{:.2f}%".format,
            "improvement_vs_persistence":
                "{:.2f}%".format
        }
    )
)


# --------------------------------------------------
# 9. Winner
# --------------------------------------------------

best = results_df.iloc[0]

print("\n==============================================")
print("BEST TRAINING WINDOW")
print("==============================================")

print(
    best["training_window"]
)

print(
    f"MAE:  ${best['MAE']:,.2f}"
)

print(
    f"RMSE: ${best['RMSE']:,.2f}"
)

print(
    f"MAPE: {best['MAPE']:.2f}%"
)


# --------------------------------------------------
# 10. Save results
# --------------------------------------------------

results_df.to_csv(
    "data/processed/start_year_simple_results.csv",
    index=False
)

print(
    "\nSaved results to "
    "data/processed/start_year_simple_results.csv"
)