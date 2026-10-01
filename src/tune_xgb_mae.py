import pandas as pd
import numpy as np

from itertools import product
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
# 3. Features
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
    for column in df.columns
    if column.startswith("town_")
    or column.startswith("flat_type_")
]

features = base_features + categorical_features

target = "target_price_6m"


# --------------------------------------------------
# 4. Rolling validation folds
# --------------------------------------------------

folds = [
    {
        "train_end": "2021-12-01",
        "val_start": "2022-01-01",
        "val_end": "2022-12-01"
    },
    {
        "train_end": "2022-12-01",
        "val_start": "2023-01-01",
        "val_end": "2023-12-01"
    },
    {
        "train_end": "2023-12-01",
        "val_start": "2024-01-01",
        "val_end": "2024-12-01"
    }
]


# --------------------------------------------------
# 5. Hyperparameter grid
# --------------------------------------------------

n_estimators_values = [200, 300, 400, 700]
learning_rate_values = [0.02, 0.03, 0.05]
max_depth_values = [3, 4, 5]

parameter_combinations = list(
    product(
        n_estimators_values,
        learning_rate_values,
        max_depth_values
    )
)

print(
    "Parameter combinations to test:",
    len(parameter_combinations)
)


# --------------------------------------------------
# 6. Test every combination
# --------------------------------------------------

results = []

for (
    n_estimators,
    learning_rate,
    max_depth
) in parameter_combinations:

    fold_maes = []

    for fold in folds:

        train = df[
            df["target_month"]
            <= fold["train_end"]
        ]

        validation = df[
            (df["target_month"] >= fold["val_start"])
            & (df["target_month"] <= fold["val_end"])
        ]

        X_train = train[features]
        y_train = train[target]

        X_val = validation[features]
        y_val = validation[target]

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
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

        predictions = model.predict(X_val)

        mae = mean_absolute_error(
            y_val,
            predictions
        )

        fold_maes.append(mae)

    average_mae = np.mean(fold_maes)

    results.append({
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "mae_2022": fold_maes[0],
        "mae_2023": fold_maes[1],
        "mae_2024": fold_maes[2],
        "average_mae": average_mae
    })

    print(
        f"Trees={n_estimators}, "
        f"LR={learning_rate}, "
        f"Depth={max_depth} "
        f"→ Average MAE: ${average_mae:,.2f}"
    )


# --------------------------------------------------
# 7. Rank models
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "average_mae"
)

print("\n==============================")
print("TOP 10 PARAMETER COMBINATIONS")
print("==============================")

print(
    results_df.head(10).to_string(index=False)
)


# --------------------------------------------------
# 8. Best parameters
# --------------------------------------------------

best = results_df.iloc[0]

best_n_estimators = int(
    best["n_estimators"]
)

best_learning_rate = float(
    best["learning_rate"]
)

best_max_depth = int(
    best["max_depth"]
)

print("\nBEST PARAMETERS:")

print(
    "n_estimators:",
    best_n_estimators
)

print(
    "learning_rate:",
    best_learning_rate
)

print(
    "max_depth:",
    best_max_depth
)


# --------------------------------------------------
# 9. Train final model through 2024
# --------------------------------------------------

train_final = df[
    df["target_month"] <= "2024-12-01"
].copy()

validation_2025 = df[
    (df["target_month"] >= "2025-01-01")
    & (df["target_month"] <= "2025-12-01")
].copy()

X_train_final = train_final[features]
y_train_final = train_final[target]

X_val_2025 = validation_2025[features]
y_val_2025 = validation_2025[target]


tuned_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    max_depth=best_max_depth,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:absoluteerror",
    random_state=42,
    n_jobs=-1
)

tuned_model.fit(
    X_train_final,
    y_train_final
)


# --------------------------------------------------
# 10. Evaluate on 2025
# --------------------------------------------------

predictions = tuned_model.predict(
    X_val_2025
)

tuned_mae = mean_absolute_error(
    y_val_2025,
    predictions
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_val_2025,
        predictions
    )
)

tuned_mape = np.mean(
    np.abs(
        (
            y_val_2025.to_numpy()
            - predictions
        )
        / y_val_2025.to_numpy()
    )
) * 100


# --------------------------------------------------
# 11. Persistence baseline
# --------------------------------------------------

baseline_predictions = (
    validation_2025["price_now"]
)

baseline_mae = mean_absolute_error(
    y_val_2025,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val_2025,
        baseline_predictions
    )
)

improvement = (
    (baseline_mae - tuned_mae)
    / baseline_mae
    * 100
)


# --------------------------------------------------
# 12. Results
# --------------------------------------------------

print("\n==============================")
print("2025 VALIDATION RESULTS")
print("==============================")

print("\nPersistence baseline:")
print(
    f"MAE:  ${baseline_mae:,.2f}"
)
print(
    f"RMSE: ${baseline_rmse:,.2f}"
)

print("\nCurrent best model:")
print("MAE:  $42,702.72")
print("RMSE: $73,854.58")
print("MAPE: 5.84%")

print("\nTuned MAE XGBoost:")

print(
    f"MAE:  ${tuned_mae:,.2f}"
)

print(
    f"RMSE: ${tuned_rmse:,.2f}"
)

print(
    f"MAPE: {tuned_mape:.2f}%"
)

print(
    f"\nImprovement over persistence: "
    f"{improvement:.2f}%"
)


# --------------------------------------------------
# 13. Save tuning results
# --------------------------------------------------

results_df.to_csv(
    "data/processed/xgb_mae_tuning_results.csv",
    index=False
)

print(
    "\nSaved tuning results to "
    "data/processed/xgb_mae_tuning_results.csv"
)