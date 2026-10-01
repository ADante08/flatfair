from pyspark.sql import SparkSession, Window, functions as F

from config import (
    BRONZE_RPI_TABLE,
    SILVER_MONTHLY_TABLE,
    GOLD_FEATURE_TABLE,
)

FLAT_TYPES = ["3 ROOM", "4 ROOM", "5 ROOM"]

spark = SparkSession.builder.getOrCreate()

print(f"Reading monthly market: {SILVER_MONTHLY_TABLE}")
market_all = spark.table(SILVER_MONTHLY_TABLE)

# National volume is calculated BEFORE filtering to the three MVP flat types,
# matching the local feature pipeline.
volume = (
    market_all
    .groupBy("month")
    .agg(F.sum("transaction_count").alias("transaction_volume"))
)

print(f"Reading Bronze RPI: {BRONZE_RPI_TABLE}")
rpi = spark.table(BRONZE_RPI_TABLE)

# Build a sortable quarter key such as 2026*4 + 2.
rpi = (
    rpi
    .withColumn("hdb_rpi", F.col("index").cast("double"))
    .withColumn(
        "rpi_year",
        F.regexp_extract(F.col("quarter"), r"(\d{4})", 1).cast("int"),
    )
    .withColumn(
        "rpi_quarter_number",
        F.regexp_extract(F.col("quarter"), r"[Qq](\d)", 1).cast("int"),
    )
    .withColumn(
        "rpi_key",
        F.col("rpi_year") * F.lit(4) + F.col("rpi_quarter_number"),
    )
)

rpi_window = Window.orderBy("rpi_key")
rpi = (
    rpi
    .withColumn("previous_rpi", F.lag("hdb_rpi", 1).over(rpi_window))
    .withColumn(
        "rpi_quarterly_growth",
        F.col("hdb_rpi") / F.col("previous_rpi") - F.lit(1.0),
    )
    .select("rpi_key", "hdb_rpi", "rpi_quarterly_growth")
)

market = market_all.filter(F.col("flat_type").isin(FLAT_TYPES))

bounds = market.agg(
    F.min("month").alias("start_month"),
    F.max("month").alias("end_month"),
).first()

start_month = bounds["start_month"]
end_month = bounds["end_month"]
num_months = (
    (end_month.year - start_month.year) * 12
    + end_month.month
    - start_month.month
    + 1
)

months = (
    spark.range(num_months)
    .select(
        F.add_months(
            F.lit(start_month),
            F.col("id").cast("int"),
        ).alias("month")
    )
)

groups = market.select("town", "flat_type").distinct()
panel = groups.crossJoin(months)

panel = (
    panel
    .join(market, ["month", "town", "flat_type"], "left")
    .fillna({"transaction_count": 0})
    .join(volume, "month", "left")
)

# Use the previous completed quarter, matching the leakage-safe local code.
# current quarter key = year*4 + quarter(month)
# previous completed quarter = current key - 1
panel = panel.withColumn(
    "rpi_key",
    F.year("month") * F.lit(4) + F.quarter("month") - F.lit(1),
)

panel = (
    panel
    .join(rpi, "rpi_key", "left")
    .drop("rpi_key")
    .withColumn("price_now", F.col("median_price"))
)

series_window = Window.partitionBy("town", "flat_type").orderBy("month")

for lag in range(1, 13):
    panel = panel.withColumn(
        f"price_lag_{lag}",
        F.lag("median_price", lag).over(series_window),
    )

for horizon in range(1, 7):
    panel = panel.withColumn(
        f"target_price_{horizon}m",
        F.lead("median_price", horizon).over(series_window),
    )

panel = panel.orderBy("town", "flat_type", "month")

(
    panel.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_FEATURE_TABLE)
)

print(f"Gold feature table written: {GOLD_FEATURE_TABLE}")
print(f"Rows: {panel.count():,}")
print(f"Months: {start_month:%Y-%m} to {end_month:%Y-%m}")
