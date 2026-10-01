import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# SETTINGS
# ==================================================

ROLLING_MONTHS = 60
HALF_LIFE_MONTHS = 24

CALIBRATION_START = "2023-01-01"
CALIBRATION_END = "2024-12-01"

TEST_START = "2025-01-01"
TEST_END = "2026-09-01"


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/processed/ml_features.csv",
    parse_dates=["month"]
)

df = df[
    df["price_now"].notna()
    & df["target_price_6m"].notna()
].copy()

df["target_month"] = (
    df["month"] + pd.DateOffset(months=6)
)


# ==================================================
# 2. ONE-HOT ENCODE TOWN / FLAT TYPE
#
# We keep the original columns for diagnostics.
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
# 3. LOCKED FEATURE SET
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

categorical_features = [
    column
    for column in encoded.columns
]

features = (
    price_features
    + categorical_features
)

target = "target_price_6m"


# ==================================================
# 4. MODEL FACTORY
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
# 5. MONTH DIFFERENCE
# ==================================================

def months_old(origin, dates):

    return (
        (origin.year - dates.dt.year) * 12
        + origin.month
        - dates.dt.month
    )


# ==================================================
# 6. WALK-FORWARD PREDICTIONS
# ==================================================

def walk_forward(
    start_target,
    end_target
):

    target_months = pd.date_range(
        start=start_target,
        end=end_target,
        freq="MS"
    )

    all_results = []

    for target_month in target_months:

        # ------------------------------------------
        # We are making a 6-month-ahead forecast.
        #
        # Example:
        # target = Jul 2026
        # origin = Jan 2026
        # ------------------------------------------

        origin = (
            target_month
            - pd.DateOffset(months=6)
        )


        # ------------------------------------------
        # Rows we want to predict
        # ------------------------------------------

        forecast_rows = df[
            df["target_month"]
            == target_month
        ].copy()

        if len(forecast_rows) == 0:
            continue


        # ------------------------------------------
        # STRICT INFORMATION CUTOFF
        #
        # At origin, only targets <= origin
        # are known.
        # ------------------------------------------

        available_train = df[
            df["target_month"]
            <= origin
        ].copy()


        # ==================================================
        # A. ROLLING-WINDOW MODEL
        # ==================================================

        rolling_start = (
            origin
            - pd.DateOffset(
                months=ROLLING_MONTHS - 1
            )
        )

        rolling_train = available_train[
            available_train["target_month"]
            >= rolling_start
        ].copy()

        rolling_model = make_model()

        rolling_model.fit(
            rolling_train[features],
            rolling_train[target]
        )

        rolling_prediction = (
            rolling_model.predict(
                forecast_rows[features]
            )
        )


        # ==================================================
        # B. RECENCY-WEIGHTED MODEL
        # ==================================================

        ages = months_old(
            origin,
            available_train["target_month"]
        )

        # Every HALF_LIFE_MONTHS,
        # observation weight halves.
        weights = (
            0.5
            ** (
                ages
                / HALF_LIFE_MONTHS
            )
        )

        weighted_model = make_model()

        weighted_model.fit(
            available_train[features],
            available_train[target],
            sample_weight=weights
        )

        weighted_prediction = (
            weighted_model.predict(
                forecast_rows[features]
            )
        )


        # ==================================================
        # STORE RESULTS
        # ==================================================

        month_results = pd.DataFrame({
            "origin_month":
                origin,

            "target_month":
                target_month,

            "town":
                forecast_rows["town"]
                .to_numpy(),

            "flat_type":
                forecast_rows["flat_type"]
                .to_numpy(),

            "actual":
                forecast_rows[target]
                .to_numpy(),

            "persistence":
                forecast_rows["price_now"]
                .to_numpy(),

            "rolling_xgb":
                rolling_prediction,

            "weighted_xgb":
                weighted_prediction
        })

        all_results.append(
            month_results
        )

        print(
            f"Finished target "
            f"{target_month.strftime('%Y-%m')} "
            f"| origin "
            f"{origin.strftime('%Y-%m')} "
            f"| rolling train rows "
            f"{len(rolling_train)} "
            f"| weighted train rows "
            f"{len(available_train)}"
        )


    return pd.concat(
        all_results,
        ignore_index=True
    )


# ==================================================
# 7. FIRST:
#    WALK-FORWARD CALIBRATION 2023-2024
# ==================================================

print("\n==============================================")
print("CALIBRATION BACKTEST: 2023-2024")
print("==============================================\n")

calibration = walk_forward(
    CALIBRATION_START,
    CALIBRATION_END
)


# ==================================================
# 8. FIND BEST BLEND WEIGHT
#
# alpha = weight placed on XGBoost
#
# prediction =
# alpha * XGB
# + (1-alpha) * persistence
#
# IMPORTANT:
# alpha selected ONLY using 2023-2024.
# ==================================================

def find_best_alpha(
    data,
    model_column
):

    best_alpha = None
    best_mae = float("inf")

    for alpha in np.arange(
        0.0,
        1.01,
        0.05
    ):

        prediction = (
            alpha
            * data[model_column]
            + (1 - alpha)
            * data["persistence"]
        )

        mae = mean_absolute_error(
            data["actual"],
            prediction
        )

        if mae < best_mae:

            best_mae = mae
            best_alpha = alpha

    return best_alpha, best_mae


rolling_alpha, rolling_blend_mae = (
    find_best_alpha(
        calibration,
        "rolling_xgb"
    )
)

weighted_alpha, weighted_blend_mae = (
    find_best_alpha(
        calibration,
        "weighted_xgb"
    )
)


print("\n==============================================")
print("BLEND CALIBRATION")
print("==============================================")

print(
    f"Rolling blend alpha: "
    f"{rolling_alpha:.2f}"
)

print(
    f"Rolling calibration MAE: "
    f"${rolling_blend_mae:,.2f}"
)

print(
    f"\nWeighted blend alpha: "
    f"{weighted_alpha:.2f}"
)

print(
    f"Weighted calibration MAE: "
    f"${weighted_blend_mae:,.2f}"
)


# ==================================================
# 9. NOW RUN 2025-2026
# ==================================================

print("\n==============================================")
print("WALK-FORWARD TEST: 2025-2026")
print("==============================================\n")

results = walk_forward(
    TEST_START,
    TEST_END
)


# ==================================================
# 10. APPLY FIXED BLEND WEIGHTS
#
# These were chosen WITHOUT using 2025-2026.
# ==================================================

results["rolling_blend"] = (
    rolling_alpha
    * results["rolling_xgb"]
    + (1 - rolling_alpha)
    * results["persistence"]
)

results["weighted_blend"] = (
    weighted_alpha
    * results["weighted_xgb"]
    + (1 - weighted_alpha)
    * results["persistence"]
)


# ==================================================
# 11. METRIC FUNCTION
# ==================================================

def calculate_metrics(
    data,
    prediction_column
):

    actual = data["actual"]

    prediction = (
        data[prediction_column]
    )

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

    mape = np.mean(
        np.abs(
            (
                actual.to_numpy()
                - prediction.to_numpy()
            )
            / actual.to_numpy()
        )
    ) * 100

    return mae, rmse, mape


# ==================================================
# 12. MODEL LIST
# ==================================================

models = {
    "Persistence":
        "persistence",

    "Rolling XGB":
        "rolling_xgb",

    "Weighted XGB":
        "weighted_xgb",

    "Rolling blend":
        "rolling_blend",

    "Weighted blend":
        "weighted_blend"
}


# ==================================================
# 13. OVERALL 2025-2026 RESULTS
# ==================================================

summary_rows = []

for name, column in models.items():

    mae, rmse, mape = (
        calculate_metrics(
            results,
            column
        )
    )

    summary_rows.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })


summary = pd.DataFrame(
    summary_rows
).sort_values(
    "MAE"
)


print("\n==============================================")
print("OVERALL 2025-2026 RESULTS")
print("==============================================")

print(
    summary.to_string(
        index=False,
        formatters={
            "MAE":
                "${:,.2f}".format,

            "RMSE":
                "${:,.2f}".format,

            "MAPE":
                "{:.2f}%".format
        }
    )
)


# ==================================================
# 14. RESULTS BY YEAR
# ==================================================

results["year"] = (
    results["target_month"]
    .dt.year
)


for year in [2025, 2026]:

    yearly = results[
        results["year"] == year
    ]

    print(
        "\n=============================================="
    )

    if year == 2026:
        print(
            "2026 RESULTS (JAN-SEP)"
        )
    else:
        print(
            "2025 RESULTS"
        )

    print(
        "=============================================="
    )

    year_rows = []

    for name, column in models.items():

        mae, rmse, mape = (
            calculate_metrics(
                yearly,
                column
            )
        )

        year_rows.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    year_df = pd.DataFrame(
        year_rows
    ).sort_values(
        "MAE"
    )

    print(
        year_df.to_string(
            index=False,
            formatters={
                "MAE":
                    "${:,.2f}".format,

                "RMSE":
                    "${:,.2f}".format,

                "MAPE":
                    "{:.2f}%".format
            }
        )
    )


# ==================================================
# 15. MONTH-BY-MONTH MAE
# ==================================================

monthly_rows = []

for target_month, group in results.groupby(
    "target_month"
):

    row = {
        "target_month":
            target_month
    }

    for name, column in models.items():

        row[name] = (
            mean_absolute_error(
                group["actual"],
                group[column]
            )
        )

    monthly_rows.append(
        row
    )


monthly = pd.DataFrame(
    monthly_rows
)


print("\n==============================================")
print("MONTH-BY-MONTH MAE")
print("==============================================")

print(
    monthly.to_string(
        index=False,
        formatters={
            name:
                "${:,.2f}".format
            for name
            in models.keys()
        }
    )
)


# ==================================================
# 16. COUNT MONTHLY WINS
# ==================================================

print("\n==============================================")
print("MONTHLY WINS")
print("==============================================")

for name in models.keys():

    wins = 0

    for _, row in monthly.iterrows():

        values = [
            row[model_name]
            for model_name
            in models.keys()
        ]

        if row[name] == min(values):
            wins += 1

    print(
        f"{name}: {wins}"
    )


# ==================================================
# 17. SAVE EVERYTHING
# ==================================================

results.to_csv(
    "data/processed/"
    "walk_forward_predictions.csv",
    index=False
)

summary.to_csv(
    "data/processed/"
    "walk_forward_summary.csv",
    index=False
)

monthly.to_csv(
    "data/processed/"
    "walk_forward_monthly.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "data/processed/"
    "walk_forward_predictions.csv"
)

print(
    "data/processed/"
    "walk_forward_summary.csv"
)

print(
    "data/processed/"
    "walk_forward_monthly.csv"
)