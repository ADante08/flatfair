import os

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost

from pyspark.sql import SparkSession

from config import (
    CATALOG,
    GOLD_FEATURE_TABLE,
    GOLD_FORECAST_TABLE,
    GOLD_METRICS_TABLE,
    MLFLOW_EXPERIMENT,
)

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ==================================================
# SETTINGS
# ==================================================

HALF_LIFE_MONTHS = 24
BLEND_ALPHA = 0.90

TEST_START = "2025-01-01"
TEST_END = "2026-09-01"

REGISTER_MODELS = os.getenv("FLATFAIR_REGISTER_MODELS", "false").lower() == "true"



# ==================================================
# LOAD DATA
# ==================================================

spark = SparkSession.builder.getOrCreate()

print("\nLoading ML features from Unity Catalog...")
print(f"Table: {GOLD_FEATURE_TABLE}")

df = spark.table(GOLD_FEATURE_TABLE).toPandas()
df["month"] = pd.to_datetime(df["month"])

mlflow.set_experiment(MLFLOW_EXPERIMENT)

df = df[
    df["price_now"].notna()
].copy()

print(
    f"Rows with current price: "
    f"{len(df):,}"
)


# ==================================================
# ONE-HOT ENCODE TOWN AND FLAT TYPE
# ==================================================

encoded = pd.get_dummies(
    df[
        [
            "town",
            "flat_type"
        ]
    ],
    dtype=int
)

df = pd.concat(
    [
        df,
        encoded
    ],
    axis=1
)


# ==================================================
# WINNING FEATURE SET
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
    f"Model features: "
    f"{len(features)}"
)

print(
    "Market-state features:",
    ", ".join(
        market_features
    )
)


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

def get_weights(
    origin,
    target_months
):

    age_months = (
        (
            origin.year
            - target_months.dt.year
        ) * 12
        + origin.month
        - target_months.dt.month
    )

    weights = (
        0.5
        ** (
            age_months
            / HALF_LIFE_MONTHS
        )
    )

    return weights


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(
    actual,
    prediction
):

    actual_array = (
        actual.to_numpy()
    )

    prediction_array = (
        np.asarray(
            prediction
        )
    )

    mae = mean_absolute_error(
        actual_array,
        prediction_array
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_array,
            prediction_array
        )
    )

    mape = (
        np.mean(
            np.abs(
                (
                    actual_array
                    - prediction_array
                )
                / actual_array
            )
        )
        * 100
    )

    return mae, rmse, mape


# ==================================================
# BACKTEST ONE FORECAST HORIZON
# ==================================================

def backtest_horizon(
    horizon
):

    target = (
        f"target_price_{horizon}m"
    )

    target_month_column = (
        f"target_month_{horizon}m"
    )


    horizon_df = df[
        df[target].notna()
    ].copy()


    horizon_df[
        target_month_column
    ] = (
        horizon_df["month"]
        + pd.DateOffset(
            months=horizon
        )
    )


    test_months = pd.date_range(
        start=TEST_START,
        end=TEST_END,
        freq="MS"
    )


    all_results = []


    print(
        "\n----------------------------------------------"
    )

    print(
        f"BACKTESTING +{horizon} MONTH"
        f"{'S' if horizon > 1 else ''}"
    )

    print(
        "----------------------------------------------"
    )


    for target_month in test_months:

        origin = (
            target_month
            - pd.DateOffset(
                months=horizon
            )
        )


        forecast_rows = horizon_df[
            horizon_df[
                target_month_column
            ] == target_month
        ].copy()


        if len(
            forecast_rows
        ) == 0:

            continue


        # ==========================================
        # STRICT INFORMATION CUTOFF
        #
        # At the forecast origin, only targets
        # already observed may be used in training.
        # ==========================================

        train = horizon_df[
            horizon_df[
                target_month_column
            ] <= origin
        ].copy()


        weights = get_weights(
            origin,
            train[
                target_month_column
            ]
        )


        # ==========================================
        # TRAIN
        # ==========================================

        model = make_model()

        model.fit(
            train[features],
            train[target],
            sample_weight=weights
        )


        # ==========================================
        # FORECAST
        # ==========================================

        xgb_prediction = (
            model.predict(
                forecast_rows[
                    features
                ]
            )
        )


        persistence_prediction = (
            forecast_rows[
                "price_now"
            ].to_numpy()
        )


        blend_prediction = (
            BLEND_ALPHA
            * xgb_prediction
            + (
                1
                - BLEND_ALPHA
            )
            * persistence_prediction
        )


        month_results = (
            pd.DataFrame(
                {
                    "target_month":
                        target_month,

                    "actual":
                        forecast_rows[
                            target
                        ].to_numpy(),

                    "persistence":
                        persistence_prediction,

                    "xgb":
                        xgb_prediction,

                    "blend":
                        blend_prediction
                }
            )
        )


        all_results.append(
            month_results
        )


    results = pd.concat(
        all_results,
        ignore_index=True
    )


    # ==============================================
    # METRICS
    # ==============================================

    (
        baseline_mae,
        baseline_rmse,
        baseline_mape
    ) = calculate_metrics(
        results["actual"],
        results["persistence"]
    )


    (
        xgb_mae,
        xgb_rmse,
        xgb_mape
    ) = calculate_metrics(
        results["actual"],
        results["xgb"]
    )


    (
        blend_mae,
        blend_rmse,
        blend_mape
    ) = calculate_metrics(
        results["actual"],
        results["blend"]
    )


    improvement = (
        (
            baseline_mae
            - blend_mae
        )
        / baseline_mae
        * 100
    )


    print(
        f"Persistence MAE: "
        f"${baseline_mae:,.2f}"
    )

    print(
        f"XGBoost MAE:     "
        f"${xgb_mae:,.2f}"
    )

    print(
        f"Blend MAE:       "
        f"${blend_mae:,.2f}"
    )

    print(
        f"Blend RMSE:      "
        f"${blend_rmse:,.2f}"
    )

    print(
        f"Blend MAPE:      "
        f"{blend_mape:.2f}%"
    )

    print(
        f"Improvement:     "
        f"{improvement:.2f}%"
    )


    return {
        "horizon_months":
            horizon,

        "persistence_mae":
            baseline_mae,

        "xgb_mae":
            xgb_mae,

        "blend_mae":
            blend_mae,

        "blend_rmse":
            blend_rmse,

        "blend_mape":
            blend_mape,

        "improvement_vs_persistence":
            improvement
    }


# ==================================================
# BACKTEST ALL SIX HORIZONS
# ==================================================

print(
    "\n=============================================="
)

print(
    "DIRECT 1-6 MONTH FORECAST BACKTEST"
)

print(
    "=============================================="
)


summary_rows = []


for horizon in range(
    1,
    7
):

    result = backtest_horizon(
        horizon
    )

    summary_rows.append(
        result
    )


summary = pd.DataFrame(
    summary_rows
)


# ==================================================
# PRINT SUMMARY
# ==================================================

print(
    "\n\n=============================================="
)

print(
    "FORECAST HORIZON SUMMARY"
)

print(
    "=============================================="
)


print(
    f"{'Horizon':<10}"
    f"{'Baseline MAE':>16}"
    f"{'XGB MAE':>16}"
    f"{'Blend MAE':>16}"
    f"{'RMSE':>16}"
    f"{'MAPE':>10}"
    f"{'Improvement':>14}"
)


print(
    "-" * 98
)


for _, row in summary.iterrows():

    print(
        f"{int(row['horizon_months']):<10}"
        f"${row['persistence_mae']:>14,.2f}"
        f"${row['xgb_mae']:>14,.2f}"
        f"${row['blend_mae']:>14,.2f}"
        f"${row['blend_rmse']:>14,.2f}"
        f"{row['blend_mape']:>9.2f}%"
        f"{row['improvement_vs_persistence']:>13.2f}%"
    )


(
    spark.createDataFrame(summary)
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_METRICS_TABLE)
)

print(f"Backtest metrics written: {GOLD_METRICS_TABLE}")


# ==================================================
# LIVE FORECASTING
# ==================================================

latest_month = (
    df["month"].max()
)


latest_rows = df[
    df["month"]
    == latest_month
].copy()


print(
    "\n=============================================="
)

print(
    "LIVE FLATFAIR FORECAST"
)

print(
    "=============================================="
)


print(
    f"Latest observed month: "
    f"{latest_month.strftime('%Y-%m')}"
)

print(
    f"Segments available: "
    f"{len(latest_rows)}"
)


forecast_outputs = []


# ==================================================
# TRAIN ONE FINAL MODEL PER HORIZON
# ==================================================

for horizon in range(
    1,
    7
):

    target = (
        f"target_price_{horizon}m"
    )

    target_month_column = (
        f"target_month_{horizon}m"
    )


    training_data = df[
        df[target].notna()
    ].copy()


    training_data[
        target_month_column
    ] = (
        training_data["month"]
        + pd.DateOffset(
            months=horizon
        )
    )


    # ==============================================
    # Only outcomes already observed by
    # September 2026 may be used.
    # ==============================================

    training_data = training_data[
        training_data[
            target_month_column
        ] <= latest_month
    ].copy()


    weights = get_weights(
        latest_month,
        training_data[
            target_month_column
        ]
    )


    model = make_model()


    model.fit(
        training_data[
            features
        ],
        training_data[
            target
        ],
        sample_weight=weights
    )


    # ==============================================
    # MLFLOW TRACKING
    # One run per final horizon model.
    # ==============================================

    metrics_row = summary[
        summary["horizon_months"] == horizon
    ].iloc[0]

    with mlflow.start_run(
        run_name=f"flatfair_{horizon}m"
    ):
        mlflow.log_params({
            "horizon_months": horizon,
            "half_life_months": HALF_LIFE_MONTHS,
            "blend_alpha": BLEND_ALPHA,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:absoluteerror",
            "feature_count": len(features),
        })

        mlflow.log_metrics({
            "persistence_mae": float(metrics_row["persistence_mae"]),
            "xgb_mae": float(metrics_row["xgb_mae"]),
            "blend_mae": float(metrics_row["blend_mae"]),
            "blend_rmse": float(metrics_row["blend_rmse"]),
            "blend_mape": float(metrics_row["blend_mape"]),
            "improvement_vs_persistence": float(
                metrics_row["improvement_vs_persistence"]
            ),
        })

        mlflow.set_tag(
            "latest_observed_month",
            latest_month.strftime("%Y-%m"),
        )

        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
        )

        if REGISTER_MODELS:
            mlflow.set_registry_uri("databricks-uc")
            registered_name = (
                f"{CATALOG}.ml.hdb_forecast_{horizon}m"
            )
            mlflow.register_model(
                model_info.model_uri,
                registered_name,
            )


    # ==============================================
    # FORECAST FROM LATEST MONTH
    # ==============================================

    xgb_forecast = (
        model.predict(
            latest_rows[
                features
            ]
        )
    )


    current_price = (
        latest_rows[
            "price_now"
        ].to_numpy()
    )


    blended_forecast = (
        BLEND_ALPHA
        * xgb_forecast
        + (
            1
            - BLEND_ALPHA
        )
        * current_price
    )


    forecast_month = (
        latest_month
        + pd.DateOffset(
            months=horizon
        )
    )


    output = pd.DataFrame({
        "origin_month":
            latest_month,

        "forecast_month":
            forecast_month,

        "horizon_months":
            horizon,

        "town":
            latest_rows[
                "town"
            ].to_numpy(),

        "flat_type":
            latest_rows[
                "flat_type"
            ].to_numpy(),

        "current_price":
            current_price,

        "xgb_forecast":
            xgb_forecast,

        "forecast_price":
            blended_forecast
    })


    output[
        "forecast_change"
    ] = (
        output[
            "forecast_price"
        ]
        - output[
            "current_price"
        ]
    )


    output[
        "forecast_change_pct"
    ] = (
        output[
            "forecast_change"
        ]
        / output[
            "current_price"
        ]
        * 100
    )


    forecast_outputs.append(
        output
    )


    print(
        f"+{horizon} month"
        f"{'s' if horizon > 1 else ''}: "
        f"{forecast_month.strftime('%Y-%m')} "
        f"| training rows "
        f"{len(training_data):,}"
    )


# ==================================================
# COMBINE AND SAVE LIVE FORECASTS
# ==================================================

forecasts = pd.concat(
    forecast_outputs,
    ignore_index=True
)


(
    spark.createDataFrame(forecasts)
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_FORECAST_TABLE)
)


print(
    "\n=============================================="
)

print(
    "FORECAST FILES SAVED"
)

print(
    "=============================================="
)


print(
    f"\nBacktest summary:"
)

print(
    GOLD_METRICS_TABLE
)


print(
    f"\n1-6 month forecasts:"
)

print(
    GOLD_FORECAST_TABLE
)


# ==================================================
# SAMPLE SIX-MONTH FORECASTS
# ==================================================

print(
    "\n=============================================="
)

print(
    "SAMPLE +6 MONTH FORECASTS"
)

print(
    "=============================================="
)


sample = (
    forecasts[
        forecasts[
            "horizon_months"
        ] == 6
    ][
        [
            "town",
            "flat_type",
            "current_price",
            "forecast_price",
            "forecast_change_pct"
        ]
    ]
    .head(10)
)


print(
    sample.to_string(
        index=False,
        formatters={
            "current_price":
                "${:,.0f}".format,

            "forecast_price":
                "${:,.0f}".format,

            "forecast_change_pct":
                "{:+.2f}%".format
        }
    )
)
