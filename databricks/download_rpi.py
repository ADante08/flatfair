import os

import pandas as pd
import requests
from pyspark.sql import SparkSession

from config import RAW_RPI_PATH, BRONZE_RPI_TABLE

DATASET_ID = "d_14f63e595975691e7c24a27ae4c07c79"
URL = "https://data.gov.sg/api/action/datastore_search"

spark = SparkSession.builder.getOrCreate()

print("Downloading HDB Resale Price Index...")

response = requests.get(
    URL,
    params={"resource_id": DATASET_ID, "limit": 500},
    timeout=30,
)
response.raise_for_status()

records = response.json()["result"]["records"]
pdf = pd.DataFrame(records)[["quarter", "index"]].copy()
pdf["index"] = pd.to_numeric(pdf["index"])
pdf = pdf.sort_values("quarter")

os.makedirs(os.path.dirname(RAW_RPI_PATH), exist_ok=True)
pdf.to_csv(RAW_RPI_PATH, index=False)

bronze = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(RAW_RPI_PATH)
)

(
    bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_RPI_TABLE)
)

print(f"Downloaded {len(pdf):,} quarterly RPI records.")
print(f"Raw file saved to: {RAW_RPI_PATH}")
print(f"Bronze table written: {BRONZE_RPI_TABLE}")
