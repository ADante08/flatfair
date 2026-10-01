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
OUTPUT_PATH = "data/processed/market_combination_results.csv"


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

categorical_features = list(
    encoded.columns
)

target = "target_price_6m"


# ==================================================
# MARKET FEATURE COMBINATIONS
# ==================================================

combinations = {

    "A": [
        "hdb_rpi",
        "rpi_quarterly_growth",
        "transaction_volume",
        "volume_growth_3m"
    ],

    "B": [
        "hdb_rpi",
        "rpi_quarterly_growth",
        "transaction_volume"
    ],

    "C": [
        "hdb_rpi",
        "rpi_quarterly_growth",
        "volume_growth_3m"
    ],

    "D": [
        "hdb_rpi",
        "rpi_quarterly_growth"
    ]
}


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

def walk_forward(market_features):

    features = (
        price_features
        + market_features
        + categorical_features
    )

    target_months = pd.date_range(
        start=TEST_START,
        end=TEST_END,
        freq="MS"
    )

    all_results = []

    for target_month in target_months:

        # Six-month forecast origin
        origin = (
            target_month
            - pd.DateOffset(months=6)
        )


        # Rows being predicted
        forecast_rows = df[
            df["target_month"] == target_month
        ].copy()

        if len(forecast_rows) == 0:
            continue


        # Strict information cutoff:
        # only outcomes already known at origin
        train = df[
            df["target_month"] <= origin
        ].copy()


        weights = get_weights(
            origin,
            train
        )


        # Train model
        model = make_model()

        model.fit(
            train[features],
            train[target],
            sample_weight=weights
        )


        # XGBoost forecast
        xgb_prediction = model.predict(
            forecast_rows[features]
        )


        # Persistence baseline
        persistence_prediction = (
            forecast_rows["price_now"]
            .to_numpy()
        )


        # 90% ML + 10% persistence
        blend_prediction = (
            BLEND_ALPHA
            * xgb_prediction
            + (1 - BLEND_ALPHA)
            * persistence_prediction
        )


        month_results = pd.DataFrame({
            "target_month": target_month,

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


        all_results.append(
            month_results
        )


    return pd.concat(
        all_results,
        ignore_index=True
    )


# ==================================================
# METRIC
# ==================================================

def mae(data, prediction_column):

    return mean_absolute_error(
        data["actual"],
        data[prediction_column]
    )


# ==================================================
# RUN COMBINATIONS A-D
# ==================================================

print("\n==============================================")
print("REDUCED MARKET-FEATURE EXPERIMENT")
print("==============================================")

print("\nEverything remains fixed except")
print("the market-state feature combination.\n")

print("A = RPI + quarterly growth + volume + volume growth")
print("B = RPI + quarterly growth + volume")
print("C = RPI + quarterly growth + volume growth")
print("D = RPI + quarterly growth")


summary_rows = []

saved_results = {}


for name, market_features in combinations.items():

    print("\n==============================================")
    print(f"MODEL {name}")
    print("==============================================")

    print(
        "Market features:",
        ", ".join(market_features)
    )

    results = walk_forward(
        market_features
    )

    saved_results[name] = results

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
        f"Overall MAE:      "
        f"${overall_mae:,.2f}"
    )

    print(
        f"2025 MAE:         "
        f"${mae_2025:,.2f}"
    )

    print(
        f"2026 Jan-Sep MAE: "
        f"${mae_2026:,.2f}"
    )


    summary_rows.append({
        "model": name,
        "features": ", ".join(market_features),
        "overall_mae": overall_mae,
        "mae_2025": mae_2025,
        "mae_2026": mae_2026
    })


summary = pd.DataFrame(
    summary_rows
)


# ==================================================
# COMPARE AGAINST MODEL A
#
# A is our previous best from the first ablation:
#
# all market features except rpi_yoy_growth
# ==================================================

model_a = summary[
    summary["model"] == "A"
].iloc[0]


summary["delta_overall_vs_A"] = (
    summary["overall_mae"]
    - model_a["overall_mae"]
)

summary["delta_2025_vs_A"] = (
    summary["mae_2025"]
    - model_a["mae_2025"]
)

summary["delta_2026_vs_A"] = (
    summary["mae_2026"]
    - model_a["mae_2026"]
)


# ==================================================
# PERSISTENCE BASELINE
# ==================================================

reference_results = saved_results["A"]

persistence_overall = mae(
    reference_results,
    "persistence"
)

persistence_2025 = mae(
    reference_results[
        reference_results["year"] == 2025
    ],
    "persistence"
)

persistence_2026 = mae(
    reference_results[
        reference_results["year"] == 2026
    ],
    "persistence"
)


# ==================================================
# IMPROVEMENT OVER PERSISTENCE
# ==================================================

summary["overall_vs_persistence_pct"] = (
    (
        persistence_overall
        - summary["overall_mae"]
    )
    / persistence_overall
    * 100
)

summary["2025_vs_persistence_pct"] = (
    (
        persistence_2025
        - summary["mae_2025"]
    )
    / persistence_2025
    * 100
)

summary["2026_vs_persistence_pct"] = (
    (
        persistence_2026
        - summary["mae_2026"]
    )
    / persistence_2026
    * 100
)


# ==================================================
# FINAL TABLE
# ==================================================

print("\n\n==============================================")
print("FINAL COMBINATION RESULTS")
print("==============================================")

print("\nΔ vs A:")
print("negative = better than A")
print("positive = worse than A\n")


print(
    f"{'Model':<8}"
    f"{'Overall':>14}"
    f"{'Δ vs A':>13}"
    f"{'2025':>14}"
    f"{'Δ vs A':>13}"
    f"{'2026':>14}"
    f"{'Δ vs A':>13}"
)

print("-" * 89)


for _, row in summary.iterrows():

    print(
        f"{row['model']:<8}"
        f"${row['overall_mae']:>12,.2f}"
        f"${row['delta_overall_vs_A']:>+11,.2f}"
        f"${row['mae_2025']:>12,.2f}"
        f"${row['delta_2025_vs_A']:>+11,.2f}"
        f"${row['mae_2026']:>12,.2f}"
        f"${row['delta_2026_vs_A']:>+11,.2f}"
    )


# ==================================================
# RANK BY OVERALL MAE
# ==================================================

ranked = summary.sort_values(
    "overall_mae"
)


print("\n==============================================")
print("RANKING BY OVERALL MAE")
print("==============================================")

for rank, (_, row) in enumerate(
    ranked.iterrows(),
    start=1
):

    print(
        f"{rank}. Model {row['model']} "
        f"| Overall ${row['overall_mae']:,.2f} "
        f"| 2025 ${row['mae_2025']:,.2f} "
        f"| 2026 ${row['mae_2026']:,.2f}"
    )


# ==================================================
# IMPROVEMENT OVER PERSISTENCE
# ==================================================

print("\n==============================================")
print("IMPROVEMENT OVER PERSISTENCE")
print("==============================================")

print(
    f"\nPersistence overall: "
    f"${persistence_overall:,.2f}"
)

print(
    f"Persistence 2025: "
    f"${persistence_2025:,.2f}"
)

print(
    f"Persistence 2026: "
    f"${persistence_2026:,.2f}\n"
)


for _, row in ranked.iterrows():

    print(
        f"Model {row['model']}: "
        f"Overall {row['overall_vs_persistence_pct']:.2f}% "
        f"| 2025 {row['2025_vs_persistence_pct']:.2f}% "
        f"| 2026 {row['2026_vs_persistence_pct']:.2f}%"
    )


# ==================================================
# SAVE
# ==================================================

summary.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nSaved results to:\n"
    f"{OUTPUT_PATH}"
)
