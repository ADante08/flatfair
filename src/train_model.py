import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# SETTINGS
# ==================================================

HALF_LIFE_MONTHS = 24
BLEND_ALPHA = 0.90

TEST_START = "2025-01-01"
TEST_END = "2026-09-01"

DATA_PATH = "data/processed/ml_features.csv"
OUTPUT_PATH = "data/processed/walk_forward_predictions.csv"


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading ML features...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["month"]
)

df = df[
    df["price_now"].notna()
    & df["target_price_6m"].notna()
].copy()

df["target_month"] = (
    df["month"] + pd.DateOffset(months=6)
)

print(f"Rows available: {len(df):,}")


# ==================================================
# ONE-HOT ENCODE TOWN AND FLAT TYPE
# ==================================================

encoded = pd.get_dummies(
    df[["town", "flat_type"]],
    dtype=int
)

df = pd.concat(
    [df, encoded],
    axis=1
)


# ==================================================
# MODEL FEATURES
# ==================================================
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


market_features = [
    "hdb_rpi",
    "rpi_quarterly_growth",
    "transaction_volume"
]

categorical_features = list(
    encoded.columns
)


features = (
    price_features
    + market_features
    + categorical_features
)
print(
    f"\nModel features: {len(features)}"
)

print(
    "Market-state features:",
    ", ".join(market_features)
)
target = "target_price_6m"


# ==================================================
# MODEL
# ==================================================

def make_model():

    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        random_state=42,
        n_jobs=-1
    )


# ==================================================
# RECENCY WEIGHTS
# ==================================================

def get_weights(origin, training_data):

    age_months = (
        (origin.year - training_data["target_month"].dt.year) * 12
        + origin.month
        - training_data["target_month"].dt.month
    )

    return (
        0.5
        ** (
            age_months
            / HALF_LIFE_MONTHS
        )
    )


# ==================================================
# WALK-FORWARD BACKTEST
# ==================================================

def walk_forward(start_target, end_target):

    target_months = pd.date_range(
        start=start_target,
        end=end_target,
        freq="MS"
    )

    all_results = []

    print("\nStarting walk-forward backtest...\n")

    for target_month in target_months:

        # Six-month forecast:
        #
        # Example:
        # origin Jan 2026 -> target Jul 2026

        origin = (
            target_month
            - pd.DateOffset(months=6)
        )

        forecast_rows = df[
            df["target_month"] == target_month
        ].copy()

        if len(forecast_rows) == 0:
            continue


        # ------------------------------------------
        # STRICT INFORMATION CUTOFF
        #
        # At the forecast origin, only outcomes
        # up to that month are known.
        # ------------------------------------------

        train = df[
            df["target_month"] <= origin
        ].copy()

        weights = get_weights(
            origin,
            train
        )


        # ------------------------------------------
        # TRAIN MODEL
        # ------------------------------------------

        model = make_model()

        model.fit(
            train[features],
            train[target],
            sample_weight=weights
        )


        # ------------------------------------------
        # FORECAST
        # ------------------------------------------

        ml_prediction = model.predict(
            forecast_rows[features]
        )

        persistence_prediction = (
            forecast_rows["price_now"]
            .to_numpy()
        )


        # ------------------------------------------
        # BLEND ML WITH PERSISTENCE
        # ------------------------------------------

        blended_prediction = (
            BLEND_ALPHA
            * ml_prediction
            + (1 - BLEND_ALPHA)
            * persistence_prediction
        )


        # ------------------------------------------
        # STORE RESULTS
        # ------------------------------------------

        month_results = pd.DataFrame({
            "origin_month": origin,
            "target_month": target_month,

            "town":
                forecast_rows["town"].to_numpy(),

            "flat_type":
                forecast_rows["flat_type"].to_numpy(),

            "actual":
                forecast_rows[target].to_numpy(),

            "persistence":
                persistence_prediction,

            "xgb":
                ml_prediction,

            "blend":
                blended_prediction
        })

        all_results.append(
            month_results
        )

        print(
            f"{target_month.strftime('%Y-%m')} "
            f"| origin {origin.strftime('%Y-%m')} "
            f"| train rows {len(train):,} "
            f"| forecasts {len(forecast_rows)}"
        )

    return pd.concat(
        all_results,
        ignore_index=True
    )


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(
    data,
    prediction_column
):

    actual = data["actual"]
    prediction = data[prediction_column]

    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    mape = (
        np.mean(
            np.abs(
                (
                    actual.to_numpy()
                    - prediction.to_numpy()
                )
                / actual.to_numpy()
            )
        )
        * 100
    )

    return mae, rmse, mape


# ==================================================
# PRINT COMPARISON
# ==================================================

def print_comparison(data, title):

    baseline_mae, baseline_rmse, baseline_mape = (
        calculate_metrics(
            data,
            "persistence"
        )
    )

    xgb_mae, xgb_rmse, xgb_mape = (
        calculate_metrics(
            data,
            "xgb"
        )
    )

    blend_mae, blend_rmse, blend_mape = (
        calculate_metrics(
            data,
            "blend"
        )
    )


    xgb_improvement = (
        (
            baseline_mae
            - xgb_mae
        )
        / baseline_mae
        * 100
    )

    blend_improvement = (
        (
            baseline_mae
            - blend_mae
        )
        / baseline_mae
        * 100
    )


    print("\n==============================================")
    print(title)
    print("==============================================")

    print(
        f"{'Model':<15}"
        f"{'MAE':>14}"
        f"{'RMSE':>14}"
        f"{'MAPE':>10}"
        f"{'vs baseline':>15}"
    )

    print("-" * 68)

    print(
        f"{'Persistence':<15}"
        f"${baseline_mae:>12,.2f}"
        f"${baseline_rmse:>12,.2f}"
        f"{baseline_mape:>9.2f}%"
        f"{'—':>15}"
    )

    print(
        f"{'XGBoost':<15}"
        f"${xgb_mae:>12,.2f}"
        f"${xgb_rmse:>12,.2f}"
        f"{xgb_mape:>9.2f}%"
        f"{xgb_improvement:>14.2f}%"
    )

    print(
        f"{'Blend':<15}"
        f"${blend_mae:>12,.2f}"
        f"${blend_rmse:>12,.2f}"
        f"{blend_mape:>9.2f}%"
        f"{blend_improvement:>14.2f}%"
    )


    print("\nMAE difference from persistence:")

    print(
        f"XGBoost: "
        f"${baseline_mae - xgb_mae:+,.2f}"
    )

    print(
        f"Blend:   "
        f"${baseline_mae - blend_mae:+,.2f}"
    )


# ==================================================
# RUN BACKTEST
# ==================================================

results = walk_forward(
    TEST_START,
    TEST_END
)


# ==================================================
# OVERALL RESULTS
# ==================================================

print_comparison(
    results,
    "OVERALL 2025-2026 RESULTS"
)


# ==================================================
# RESULTS BY YEAR
# ==================================================

results["year"] = (
    results["target_month"].dt.year
)

results_2025 = results[
    results["year"] == 2025
]

results_2026 = results[
    results["year"] == 2026
]

print_comparison(
    results_2025,
    "2025 RESULTS"
)

print_comparison(
    results_2026,
    "2026 RESULTS (JAN-SEP)"
)


# ==================================================
# SAVE PREDICTIONS
# ==================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nPredictions saved to:\n"
    f"{OUTPUT_PATH}"
)