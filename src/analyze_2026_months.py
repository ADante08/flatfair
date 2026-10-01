import pandas as pd
import numpy as np


df = pd.read_csv(
    "data/processed/final_2026_predictions.csv",
    parse_dates=["target_month"]
)


# --------------------------------------------------
# Errors for XGBoost
# --------------------------------------------------

df["model_absolute_error"] = np.abs(
    df["actual_price"]
    - df["predicted_price"]
)

df["model_percentage_error"] = (
    df["model_absolute_error"]
    / df["actual_price"]
    * 100
)


# --------------------------------------------------
# Errors for persistence baseline
# --------------------------------------------------

df["baseline_absolute_error"] = np.abs(
    df["actual_price"]
    - df["baseline_price"]
)

df["baseline_percentage_error"] = (
    df["baseline_absolute_error"]
    / df["actual_price"]
    * 100
)


# --------------------------------------------------
# Group by target month
# --------------------------------------------------

monthly = (
    df.groupby("target_month")
    .agg(
        observations=("actual_price", "size"),

        model_mae=(
            "model_absolute_error",
            "mean"
        ),

        baseline_mae=(
            "baseline_absolute_error",
            "mean"
        ),

        model_mape=(
            "model_percentage_error",
            "mean"
        ),

        baseline_mape=(
            "baseline_percentage_error",
            "mean"
        )
    )
    .reset_index()
)


# --------------------------------------------------
# Calculate whether XGBoost won each month
# --------------------------------------------------

monthly["mae_difference"] = (
    monthly["baseline_mae"]
    - monthly["model_mae"]
)

monthly["model_better"] = (
    monthly["model_mae"]
    < monthly["baseline_mae"]
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n==============================================")
print("2026 MONTH-BY-MONTH PERFORMANCE")
print("==============================================")

print(
    monthly.to_string(
        index=False,
        formatters={
            "model_mae": "${:,.2f}".format,
            "baseline_mae": "${:,.2f}".format,
            "model_mape": "{:.2f}%".format,
            "baseline_mape": "{:.2f}%".format,
            "mae_difference": "${:,.2f}".format
        }
    )
)


# --------------------------------------------------
# How many months did XGBoost win?
# --------------------------------------------------

wins = monthly["model_better"].sum()

print(
    f"\nXGBoost beat persistence in "
    f"{wins} out of {len(monthly)} months."
)