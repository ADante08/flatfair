import pandas as pd

df = pd.read_csv(
    "data/processed/final_2026_predictions.csv",
    parse_dates=["target_month"]
)

# Actual change from forecast origin to six months later
df["actual_change"] = (
    df["actual_price"]
    - df["baseline_price"]
)

# Change that XGBoost predicted
df["predicted_change"] = (
    df["predicted_price"]
    - df["baseline_price"]
)

# Positive = model predicted too high
# Negative = model predicted too low
df["signed_error"] = (
    df["predicted_price"]
    - df["actual_price"]
)

monthly = (
    df.groupby("target_month")
    .agg(
        actual_change=("actual_change", "mean"),
        predicted_change=("predicted_change", "mean"),
        signed_error=("signed_error", "mean")
    )
    .reset_index()
)

print(
    monthly.to_string(
        index=False,
        formatters={
            "actual_change": "${:,.2f}".format,
            "predicted_change": "${:,.2f}".format,
            "signed_error": "${:,.2f}".format
        }
    )
)