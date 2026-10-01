import pandas as pd
from pyspark.sql import SparkSession

from config import BRONZE_HDB_TABLE

spark = SparkSession.builder.getOrCreate()
df = spark.table(BRONZE_HDB_TABLE).toPandas()

print("Shape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())
print("\nDate range:")
print(df["month"].min(), "to", df["month"].max())
print("\nNumber of unique towns:")
print(df["town"].nunique())
print("\nTowns:")
print(sorted(df["town"].unique()))
print("\nFlat types:")
print(df["flat_type"].value_counts())
print("\nDuplicate rows:")
print(df.duplicated().sum())
if "_id" in df.columns:
    print("\nDuplicate rows excluding _id:")
    print(df.drop(columns=["_id"]).duplicated().sum())
