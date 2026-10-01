import os

CATALOG = os.getenv("FLATFAIR_CATALOG", "flatfair")
BRONZE_SCHEMA = os.getenv("FLATFAIR_BRONZE_SCHEMA", "bronze")
SILVER_SCHEMA = os.getenv("FLATFAIR_SILVER_SCHEMA", "silver")
GOLD_SCHEMA = os.getenv("FLATFAIR_GOLD_SCHEMA", "gold")
ML_SCHEMA = os.getenv("FLATFAIR_ML_SCHEMA", "ml")
RAW_VOLUME = os.getenv("FLATFAIR_RAW_VOLUME", "raw_files")

RAW_VOLUME_PATH = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/{RAW_VOLUME}"
RAW_HDB_PATH = f"{RAW_VOLUME_PATH}/hdb_resale.csv"
RAW_RPI_PATH = f"{RAW_VOLUME_PATH}/hdb_rpi.csv"

BRONZE_HDB_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.hdb_resale"
BRONZE_RPI_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.hdb_rpi"

SILVER_CLEAN_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.hdb_resale_clean"
SILVER_MONTHLY_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.monthly_market"

GOLD_FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.ml_features"
GOLD_FORECAST_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.hdb_forecasts"
GOLD_METRICS_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.backtest_metrics"

MLFLOW_EXPERIMENT = os.getenv(
    "FLATFAIR_MLFLOW_EXPERIMENT",
    "/Shared/flatfair-forecasting",
)
