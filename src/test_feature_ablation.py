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
# 2. Training set: 2019-2024
# --------------------------------------------------

train = df[
    (df["month"] >= "2019-01-01")
    & (df["target_month"] <= "2024-12-01")
].copy()


# --------------------------------------------------
# 3. Validation set: 2025
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
# 5. Features that ALWAYS stay in the model
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

target = "target_price_6m"


# --------------------------------------------------
# 6. Optional features we want to test
# --------------------------------------------------

optional_features = [
    "month_index",
    "month_of_year",
    "transaction_count"
]


# --------------------------------------------------
# 7. Persistence baseline
# --------------------------------------------------

y_val = validation[target]

baseline_predictions = validation["price_now"]

baseline_mae = mean_absolute_error(
    y_val,
    baseline_predictions
)

print("\nPersistence baseline:")
print(f"MAE: ${baseline_mae:,.2f}")


# --------------------------------------------------
# 8. Test every combination of the three features
# --------------------------------------------------

results = []

combinations = list(
    product(
        [False, True],
        repeat=3
    )
)

for (
    use_month_index,
    use_month_of_year,
    use_transaction_count
) in combinations:

    selected_optional = []

    if use_month_index:
        selected_optional.append(
            "month_index"
        )

    if use_month_of_year:
        selected_optional.append(
            "month_of_year"
        )

    if use_transaction_count:
        selected_optional.append(
            "transaction_count"
        )

    features = (
        price_features
        + selected_optional
        + categorical_features
    )

    X_train = train[features]
    y_train = train[target]

    X_val = validation[features]


    # --------------------------------------------------
    # Same XGBoost settings for every experiment
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

    predictions = model.predict(
        X_val
    )


    # --------------------------------------------------
    # Metrics
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


    # Friendly label
    if selected_optional:
        label = " + ".join(
            selected_optional
        )
    else:
        label = "No optional features"


    results.append({
        "features": label,
        "month_index": use_month_index,
        "month_of_year": use_month_of_year,
        "transaction_count": use_transaction_count,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "improvement_vs_persistence": improvement
    })

    print("\n==============================================")
    print(label)
    print("==============================================")

    print(
        f"MAE:  ${mae:,.2f}"
    )

    print(
        f"RMSE: ${rmse:,.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )


# --------------------------------------------------
# 9. Rank all configurations
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "MAE"
)

print("\n\n==============================================")
print("FEATURE ABLATION RESULTS")
print("==============================================")

print(
    results_df[
        [
            "features",
            "MAE",
            "RMSE",
            "MAPE",
            "improvement_vs_persistence"
        ]
    ].to_string(
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
# 10. Report winner
# --------------------------------------------------

best = results_df.iloc[0]

print("\n==============================================")
print("BEST FEATURE SET")
print("==============================================")

print(
    best["features"]
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