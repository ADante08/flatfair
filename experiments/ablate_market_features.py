import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# ==================================================
# SETTINGS
# ==================================================

HALF_LIFE_MONTHS = 24
BLEND_ALPHA = 0.90

TEST_START = "2025-01-01"
TEST_END = "2026-09-01"

DATA_PATH = "data/processed/ml_features.csv"
OUTPUT_PATH = "data/processed/market_feature_ablation.csv"


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
    df["month"]
    + pd.DateOffset(months=6)
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
# CORE FEATURES
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
    "rpi_yoy_growth",
    "transaction_volume",
    "volume_growth_3m"
]

categorical_features = list(
    encoded.columns
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
        (
            origin.year
            - training_data["target_month"].dt.year
        ) * 12
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

def walk_forward(active_market_features):

    features = (
        price_features
        + active_market_features
        + categorical_features
    )

    target_months = pd.date_range(
        start=TEST_START,
        end=TEST_END,
        freq="MS"
    )

    results = []

    for target_month in target_months:

        # ------------------------------------------
        # Six-month forecast origin
        # ------------------------------------------

        origin = (
            target_month
            - pd.DateOffset(months=6)
        )


        # ------------------------------------------
        # Rows we are forecasting
        # ------------------------------------------

        forecast_rows = df[
            df["target_month"]
            == target_month
        ].copy()

        if len(forecast_rows) == 0:
            continue


        # ------------------------------------------
        # STRICT WALK-FORWARD TRAINING CUTOFF
        #
        # Only outcomes already known at the
        # forecast origin may be used.
        # ------------------------------------------

        train = df[
            df["target_month"]
            <= origin
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
        # XGBOOST FORECAST
        # ------------------------------------------

        xgb_prediction = model.predict(
            forecast_rows[features]
        )


        # ------------------------------------------
        # PERSISTENCE FORECAST
        # ------------------------------------------

        persistence_prediction = (
            forecast_rows["price_now"]
            .to_numpy()
        )


        # ------------------------------------------
        # 90% XGBOOST + 10% PERSISTENCE
        # ------------------------------------------

        blend_prediction = (
            BLEND_ALPHA
            * xgb_prediction
            + (1 - BLEND_ALPHA)
            * persistence_prediction
        )


        month_results = pd.DataFrame({
            "target_month":
                target_month,

            "actual":
                forecast_rows[target]
                .to_numpy(),

            "persistence":
                persistence_prediction,

            "xgb":
                xgb_prediction,

            "blend":
                blend_prediction
        })


        results.append(
            month_results
        )


    return pd.concat(
        results,
        ignore_index=True
    )


# ==================================================
# MAE
# ==================================================

def mae(data, column):

    return mean_absolute_error(
        data["actual"],
        data[column]
    )


# ==================================================
# RUN ONE VARIANT
# ==================================================

def evaluate_variant(
    name,
    active_market_features
):

    print(
        "\n=============================================="
    )

    print(name)

    print(
        "=============================================="
    )

    if active_market_features:

        print(
            "Market features:",
            ", ".join(active_market_features)
        )

    else:

        print(
            "Market features: NONE"
        )


    results = walk_forward(
        active_market_features
    )

    results["year"] = (
        results["target_month"].dt.year
    )


    results_2025 = results[
        results["year"] == 2025
    ]

    results_2026 = results[
        results["year"] == 2026
    ]


    overall_mae = mae(
        results,
        "blend"
    )

    mae_2025 = mae(
        results_2025,
        "blend"
    )

    mae_2026 = mae(
        results_2026,
        "blend"
    )


    print(
        f"Overall blend MAE: "
        f"${overall_mae:,.2f}"
    )

    print(
        f"2025 blend MAE:    "
        f"${mae_2025:,.2f}"
    )

    print(
        f"2026 blend MAE:    "
        f"${mae_2026:,.2f}"
    )


    return {
        "variant": name,
        "overall_mae": overall_mae,
        "mae_2025": mae_2025,
        "mae_2026": mae_2026
    }


# ==================================================
# ABLATION VARIANTS
# ==================================================

variants = {
    "FULL CHAMPION":
        market_features,

    "Remove hdb_rpi":
        [
            feature
            for feature in market_features
            if feature != "hdb_rpi"
        ],

    "Remove rpi_quarterly_growth":
        [
            feature
            for feature in market_features
            if feature != "rpi_quarterly_growth"
        ],

    "Remove rpi_yoy_growth":
        [
            feature
            for feature in market_features
            if feature != "rpi_yoy_growth"
        ],

    "Remove transaction_volume":
        [
            feature
            for feature in market_features
            if feature != "transaction_volume"
        ],

    "Remove volume_growth_3m":
        [
            feature
            for feature in market_features
            if feature != "volume_growth_3m"
        ]
}


# ==================================================
# RUN ABLATION
# ==================================================

print(
    "\n=============================================="
)

print(
    "MARKET FEATURE ABLATION"
)

print(
    "=============================================="
)

print(
    "\nEach experiment removes exactly one "
    "market-state feature."
)

print(
    "Everything else remains unchanged:"
)

print(
    "  - same price lags"
)

print(
    "  - same XGBoost parameters"
)

print(
    "  - same 24-month recency weighting"
)

print(
    "  - same walk-forward splits"
)

print(
    "  - same 90/10 blend"
)


rows = []

for name, active_features in variants.items():

    row = evaluate_variant(
        name,
        active_features
    )

    rows.append(
        row
    )


summary = pd.DataFrame(
    rows
)


# ==================================================
# COMPARE AGAINST FULL CHAMPION
# ==================================================

champion = summary[
    summary["variant"]
    == "FULL CHAMPION"
].iloc[0]


summary["overall_vs_champion"] = (
    summary["overall_mae"]
    - champion["overall_mae"]
)

summary["2025_vs_champion"] = (
    summary["mae_2025"]
    - champion["mae_2025"]
)

summary["2026_vs_champion"] = (
    summary["mae_2026"]
    - champion["mae_2026"]
)


# ==================================================
# PRINT FINAL TABLE
# ==================================================

print(
    "\n\n=============================================="
)

print(
    "FINAL ABLATION RESULTS"
)

print(
    "=============================================="
)

print(
    "\nPositive Δ = removing the feature made "
    "the model WORSE."
)

print(
    "Negative Δ = removing the feature made "
    "the model BETTER.\n"
)


print(
    f"{'Variant':<32}"
    f"{'Overall MAE':>14}"
    f"{'Δ Overall':>13}"
    f"{'2025 MAE':>14}"
    f"{'Δ 2025':>12}"
    f"{'2026 MAE':>14}"
    f"{'Δ 2026':>12}"
)

print(
    "-" * 111
)


for _, row in summary.iterrows():

    print(
        f"{row['variant']:<32}"
        f"${row['overall_mae']:>12,.2f}"
        f"${row['overall_vs_champion']:>+11,.2f}"
        f"${row['mae_2025']:>12,.2f}"
        f"${row['2025_vs_champion']:>+10,.2f}"
        f"${row['mae_2026']:>12,.2f}"
        f"${row['2026_vs_champion']:>+10,.2f}"
    )


# ==================================================
# FEATURE IMPORTANCE BY ABLATION
#
# Larger positive increase in MAE means removing
# the feature hurt more, so the feature contributed
# more to forecasting performance.
# ==================================================

ablation_only = summary[
    summary["variant"]
    != "FULL CHAMPION"
].copy()


ablation_only = ablation_only.sort_values(
    "2026_vs_champion",
    ascending=False
)


print(
    "\n=============================================="
)

print(
    "FEATURE CONTRIBUTION TO 2026 PERFORMANCE"
)

print(
    "=============================================="
)


for _, row in ablation_only.iterrows():

    feature_name = (
        row["variant"]
        .replace(
            "Remove ",
            ""
        )
    )

    difference = (
        row["2026_vs_champion"]
    )


    if difference > 0:

        print(
            f"{feature_name:<25} "
            f"removal worsened MAE by "
            f"${difference:,.2f}"
        )

    elif difference < 0:

        print(
            f"{feature_name:<25} "
            f"removal improved MAE by "
            f"${abs(difference):,.2f}"
        )

    else:

        print(
            f"{feature_name:<25} "
            f"no measurable change"
        )


# ==================================================
# PERSISTENCE REFERENCE
# ==================================================

full_results = walk_forward(
    market_features
)

full_results["year"] = (
    full_results["target_month"].dt.year
)

persistence_overall = mae(
    full_results,
    "persistence"
)

persistence_2025 = mae(
    full_results[
        full_results["year"] == 2025
    ],
    "persistence"
)

persistence_2026 = mae(
    full_results[
        full_results["year"] == 2026
    ],
    "persistence"
)


print(
    "\n=============================================="
)

print(
    "PERSISTENCE REFERENCE"
)

print(
    "=============================================="
)

print(
    f"Overall:      "
    f"${persistence_overall:,.2f}"
)

print(
    f"2025:         "
    f"${persistence_2025:,.2f}"
)

print(
    f"2026 Jan-Sep: "
    f"${persistence_2026:,.2f}"
)


# ==================================================
# SAVE SUMMARY
# ==================================================

summary.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nSaved ablation results to:\n"
    f"{OUTPUT_PATH}"
)
