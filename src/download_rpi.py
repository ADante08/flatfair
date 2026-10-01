import requests
import pandas as pd


DATASET_ID = "d_14f63e595975691e7c24a27ae4c07c79"

URL = "https://data.gov.sg/api/action/datastore_search"

OUTPUT_PATH = "data/raw/hdb_rpi.csv"


print("\nDownloading HDB Resale Price Index...")

response = requests.get(
    URL,
    params={
        "resource_id": DATASET_ID,
        "limit": 500
    },
    timeout=30
)

response.raise_for_status()

records = response.json()["result"]["records"]

df = pd.DataFrame(records)

df = df[
    ["quarter", "index"]
].copy()

df["index"] = pd.to_numeric(
    df["index"]
)

df = df.sort_values(
    "quarter"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"Downloaded {len(df)} quarterly RPI records."
)

print(
    f"Saved to: {OUTPUT_PATH}"
)

print("\nLatest rows:")
print(df.tail())