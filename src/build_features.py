import pandas as pd


# ==================================================
# SETTINGS
# ==================================================

MARKET_PATH = "data/processed/monthly_market.csv"
RPI_PATH = "data/raw/hdb_rpi.csv"
OUTPUT_PATH = "data/processed/ml_features.csv"

FLAT_TYPES = [
    "3 ROOM",
    "4 ROOM",
    "5 ROOM"
]


# ==================================================
# LOAD MONTHLY HDB MARKET DATA
# ==================================================

print("\nLoading monthly HDB market data...")

market = pd.read_csv(
    MARKET_PATH,
    parse_dates=["month"]
)

market = market.sort_values(
    ["town", "flat_type", "month"]
)

print(
    f"Monthly market rows: {len(market):,}"
)


# ==================================================
# NATIONAL TRANSACTION VOLUME
# ==================================================

print("\nBuilding national transaction volume...")

volume = (
    market
    .groupby(
        "month",
        as_index=False
    )
    .agg(
        transaction_volume=(
            "transaction_count",
            "sum"
        )
    )
    .sort_values("month")
)


# ==================================================
# LOAD HDB RESALE PRICE INDEX
# ==================================================

print("\nLoading HDB Resale Price Index...")

rpi = pd.read_csv(
    RPI_PATH
)

rpi["index"] = pd.to_numeric(
    rpi["index"]
)

rpi["quarter"] = pd.PeriodIndex(
    rpi["quarter"],
    freq="Q"
)

rpi = rpi.sort_values(
    "quarter"
)


# Quarterly RPI growth

rpi["rpi_quarterly_growth"] = (
    rpi["index"].pct_change(1)
)

rpi = rpi.rename(
    columns={
        "index": "hdb_rpi"
    }
)


# ==================================================
# KEEP MAIN FLAT TYPES
# ==================================================

market = market[
    market["flat_type"].isin(
        FLAT_TYPES
    )
].copy()


# ==================================================
# COMPLETE MONTHLY PANEL
# ==================================================

print("\nBuilding complete monthly panel...")

start_month = market["month"].min()
end_month = market["month"].max()

all_months = pd.date_range(
    start=start_month,
    end=end_month,
    freq="MS"
)

groups = (
    market[
        ["town", "flat_type"]
    ]
    .drop_duplicates()
)

panel = groups.merge(
    pd.DataFrame(
        {"month": all_months}
    ),
    how="cross"
)


# ==================================================
# MERGE LOCAL MARKET DATA
# ==================================================

panel = panel.merge(
    market,
    on=[
        "month",
        "town",
        "flat_type"
    ],
    how="left"
)

panel["transaction_count"] = (
    panel["transaction_count"]
    .fillna(0)
)


# ==================================================
# MERGE NATIONAL TRANSACTION VOLUME
# ==================================================

panel = panel.merge(
    volume,
    on="month",
    how="left"
)


# ==================================================
# MERGE HDB RPI
#
# Use previous completed quarter to avoid leakage.
#
# Example:
# October 2025 -> use 2025 Q3 RPI
# ==================================================

panel["rpi_quarter"] = (
    panel["month"].dt.to_period("Q")
    - 1
)

panel = panel.merge(
    rpi[
        [
            "quarter",
            "hdb_rpi",
            "rpi_quarterly_growth"
        ]
    ],
    left_on="rpi_quarter",
    right_on="quarter",
    how="left"
)

panel = panel.drop(
    columns=[
        "quarter",
        "rpi_quarter"
    ]
)


# ==================================================
# SORT PANEL
# ==================================================

panel = panel.sort_values(
    [
        "town",
        "flat_type",
        "month"
    ]
).reset_index(
    drop=True
)


# ==================================================
# CURRENT PRICE
# ==================================================

panel["price_now"] = (
    panel["median_price"]
)


# ==================================================
# PRICE LAGS 1-12 MONTHS
# ==================================================

print("\nCreating price lags...")

for lag in range(1, 13):

    panel[f"price_lag_{lag}"] = (
        panel
        .groupby(
            ["town", "flat_type"]
        )["median_price"]
        .shift(lag)
    )


# ==================================================
# DIRECT FORECAST TARGETS 1-6 MONTHS
# ==================================================

print(
    "Creating direct targets "
    "for horizons 1-6 months..."
)

for horizon in range(1, 7):

    panel[f"target_price_{horizon}m"] = (
        panel
        .groupby(
            ["town", "flat_type"]
        )["median_price"]
        .shift(-horizon)
    )


# ==================================================
# SAVE
# ==================================================

panel.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n==============================================")
print("FEATURE TABLE BUILT")
print("==============================================")

print(
    f"Rows: {len(panel):,}"
)

print(
    f"Months: "
    f"{panel['month'].min().strftime('%Y-%m')} "
    f"to "
    f"{panel['month'].max().strftime('%Y-%m')}"
)

print("\nForecast targets created:")

for horizon in range(1, 7):

    print(
        f"  target_price_{horizon}m"
    )

print("\nMarket-state features:")

print("  hdb_rpi")
print("  rpi_quarterly_growth")
print("  transaction_volume")

print(
    f"\nSaved to: {OUTPUT_PATH}"
)
