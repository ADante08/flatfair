import os
import time

import pandas as pd
import requests
from pyspark.sql import SparkSession

from config import RAW_HDB_PATH, BRONZE_HDB_TABLE

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
URL = "https://data.gov.sg/api/action/datastore_search"
LIMIT = 5000

spark = SparkSession.builder.getOrCreate()

print("Downloading HDB resale transactions from data.gov.sg...")

offset = 0
all_records = []

while True:
    response = requests.get(
        URL,
        params={
            "resource_id": DATASET_ID,
            "limit": LIMIT,
            "offset": offset,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()["result"]
    records = payload["records"]
    total = payload["total"]

    all_records.extend(records)
    offset += len(records)
    print(f"Downloaded {len(all_records):,} / {total:,} records")

    if offset >= total:
        break

    time.sleep(2.6)

pdf = pd.DataFrame(all_records)

os.makedirs(os.path.dirname(RAW_HDB_PATH), exist_ok=True)
pdf.to_csv(RAW_HDB_PATH, index=False)

# Bronze preserves the source data as a governed Delta table.
bronze = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(RAW_HDB_PATH)
)

(
    bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_HDB_TABLE)
)

print(f"Raw file saved to: {RAW_HDB_PATH}")
print(f"Bronze table written: {BRONZE_HDB_TABLE}")
print(f"Rows: {bronze.count():,}")
