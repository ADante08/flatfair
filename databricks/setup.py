from pyspark.sql import SparkSession
from config import (
    CATALOG,
    BRONZE_SCHEMA,
    SILVER_SCHEMA,
    GOLD_SCHEMA,
    ML_SCHEMA,
    RAW_VOLUME,
)

spark = SparkSession.builder.getOrCreate()

print(f"Preparing Unity Catalog objects under: {CATALOG}")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")

for schema in [BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA, ML_SCHEMA]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")

spark.sql(
    f"CREATE VOLUME IF NOT EXISTS "
    f"{CATALOG}.{BRONZE_SCHEMA}.{RAW_VOLUME}"
)

print("Unity Catalog setup complete.")
print(f"Catalog: {CATALOG}")
print(f"Schemas: {BRONZE_SCHEMA}, {SILVER_SCHEMA}, {GOLD_SCHEMA}, {ML_SCHEMA}")
print(f"Raw volume: {CATALOG}.{BRONZE_SCHEMA}.{RAW_VOLUME}")
