import pandas as pd
df = pd.read_csv("data/raw/hdb_resale.csv")

print("Shape:")
print(df.shape)

print("\n Columns:")
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

print("\nDuplicate rows excluding _id:")
print(df.drop(columns=["_id"]).duplicated().sum())