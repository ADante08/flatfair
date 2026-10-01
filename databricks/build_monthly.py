from pyspark.sql import SparkSession, functions as F

from config import SILVER_CLEAN_TABLE, SILVER_MONTHLY_TABLE

spark = SparkSession.builder.getOrCreate()

print(f"Reading Silver cleaned table: {SILVER_CLEAN_TABLE}")
df = spark.table(SILVER_CLEAN_TABLE)

monthly = (
    df.groupBy("month", "town", "flat_type")
    .agg(
        F.median("resale_price").alias("median_price"),
        F.median("price_per_sqm").alias("median_price_per_sqm"),
        F.count("resale_price").alias("transaction_count"),
        F.median("floor_area_sqm").alias("median_floor_area"),
        F.median("remaining_lease_months").alias("median_remaining_lease"),
        F.median("storey_mid").alias("median_storey"),
    )
    .orderBy("month", "town", "flat_type")
)

(
    monthly.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_MONTHLY_TABLE)
)

print(f"Silver monthly table written: {SILVER_MONTHLY_TABLE}")
print(f"Rows: {monthly.count():,}")
