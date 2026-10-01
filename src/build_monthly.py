import pandas as pd
df = pd.read_csv("data/processed/hdb_resale_clean.csv", parse_dates= ["month"])
monthly = (
    df.groupby(["month", "town", "flat_type"]).agg(
        median_price = ("resale_price","median"),
        median_price_per_sqm  = ("price_per_sqm", "median"),
        transaction_count=("resale_price", "size"),
        median_floor_area=("floor_area_sqm", "median"),
        median_remaining_lease=("remaining_lease_months", "median"),
        median_storey=("storey_mid", "median")
    )
    .reset_index()
)

print("\nShape:", monthly.shape)

monthly.to_csv(
    "data/processed/monthly_market.csv",
    index=False
)
print("Saved to data/processed/monthly_market.csv")