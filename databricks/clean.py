from pyspark.sql import SparkSession, functions as F

from config import BRONZE_HDB_TABLE, SILVER_CLEAN_TABLE

spark = SparkSession.builder.getOrCreate()

print(f"Reading Bronze table: {BRONZE_HDB_TABLE}")
df = spark.table(BRONZE_HDB_TABLE)

# Preserve the logic from the local clean.py, but execute it with PySpark
# and persist the result as a Silver Delta table.
df = df.drop("_id")

df = (
    df
    .withColumn(
        "month",
        F.to_date(
            F.concat(F.col("month").cast("string"), F.lit("-01")),
            "yyyy-MM-dd",
        ),
    )
    .withColumn("resale_price", F.col("resale_price").cast("double"))
    .withColumn("floor_area_sqm", F.col("floor_area_sqm").cast("double"))
    .withColumn("lease_commence_date", F.col("lease_commence_date").cast("int"))
)

lease_years = F.regexp_extract(
    F.col("remaining_lease"), r"(\d+)\s+years?", 1
).cast("int")

lease_months_text = F.regexp_extract(
    F.col("remaining_lease"), r"years?\s*(\d+)?", 1
)

lease_months = F.when(
    F.length(lease_months_text) == 0,
    F.lit(0),
).otherwise(lease_months_text.cast("int"))

df = df.withColumn(
    "remaining_lease_months",
    lease_years * F.lit(12) + lease_months,
)

storey_low = F.regexp_extract(
    F.col("storey_range"), r"(\d+)\s+TO\s+(\d+)", 1
).cast("double")
storey_high = F.regexp_extract(
    F.col("storey_range"), r"(\d+)\s+TO\s+(\d+)", 2
).cast("double")

df = (
    df
    .withColumn("storey_mid", (storey_low + storey_high) / F.lit(2.0))
    .withColumn("price_per_sqm", F.col("resale_price") / F.col("floor_area_sqm"))
)

(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_CLEAN_TABLE)
)

print(f"Silver table written: {SILVER_CLEAN_TABLE}")
print(f"Rows: {df.count():,}")
