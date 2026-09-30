import pandas as pd

df = pd.read_csv("data/raw/hdb_resale.csv")

df = df.drop(columns=["_id"])
# This whole part changes the month format to pandas date time
df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
# Strange regex formula, after that we make the remaining lease thing into usable data, we just convert it into an integer giving us the number of months
lease_parts = df["remaining_lease"].str.extract(
    r"(?P<years>\d+)\s+years?\s*(?P<months>\d+)?"
)

lease_parts["years"] = lease_parts["years"].astype(int)
lease_parts["months"] = lease_parts["months"].fillna(0).astype(int)

df["remaining_lease_months"] = (
    lease_parts["years"] * 12 + lease_parts["months"]
)

storey_parts = df["storey_range"].str.extract(r"(?P<low>\d+)\s+TO\s+(?P<high>\d+)")
storey_parts["low"] = storey_parts["low"].astype(int)
storey_parts["high"] = storey_parts["high"].astype(int)
df["storey_mid"] = (
    storey_parts["low"] + storey_parts["high"]
) / 2
df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]
df.to_csv("data/processed/hdb_resale_clean.csv", index = False)
print("Saved cleaned data to data/processed/hdb_resale_clean.csv")