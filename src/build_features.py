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
#
# IMPORTANT:
# Calculate this BEFORE filtering to 3/4/5 room,
# so it represents the whole HDB resale market.
# ==================================================

print("\nBuilding national transaction-volume features...")

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


# Current 3-month average:
#
# t, t-1, t-2

volume["volume_3m_avg"] = (
    volume["transaction_volume"]
    .rolling(
        window=3,
        min_periods=3
    )
    .mean()
)


# Compare the latest 3 months against
# the preceding 3 months:
#
# recent:   t, t-1, t-2
# previous: t-3, t-4, t-5

volume["volume_growth_3m"] = (
    volume["volume_3m_avg"]
    / volume["volume_3m_avg"].shift(3)
    - 1
)


print(
    volume[
        [
            "month",
            "transaction_volume",
            "volume_growth_3m"
        ]
    ].tail()
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


# Convert values such as:
#
# 2025-Q3
#
# into Pandas quarterly periods.

rpi["quarter"] = pd.PeriodIndex(
    rpi["quarter"],
    freq="Q"
)

rpi = rpi.sort_values(
    "quarter"
)


# ==================================================
# RPI GROWTH FEATURES
# ==================================================

rpi["rpi_quarterly_growth"] = (
    rpi["index"].pct_change(1)
)

rpi["rpi_yoy_growth"] = (
    rpi["index"].pct_change(4)
)

rpi = rpi.rename(
    columns={
        "index": "hdb_rpi"
    }
)


print(
    rpi[
        [
            "quarter",
            "hdb_rpi",
            "rpi_quarterly_growth",
            "rpi_yoy_growth"
        ]
    ].tail()
)


# ==================================================
# KEEP CORE FLAT TYPES
# ==================================================

market = market[
    market["flat_type"].isin(
        FLAT_TYPES
    )
].copy()


# ==================================================
# BUILD COMPLETE MONTHLY PANEL
#
# Every town × flat type gets one row for
# every calendar month.
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

panel = (
    groups
    .merge(
        pd.DataFrame(
            {"month": all_months}
        ),
        how="cross"
    )
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


# No transaction row means zero transactions.

panel["transaction_count"] = (
    panel["transaction_count"]
    .fillna(0)
)


# ==================================================
# MERGE NATIONAL TRANSACTION VOLUME
# ==================================================

panel = panel.merge(
    volume[
        [
            "month",
            "transaction_volume",
            "volume_growth_3m"
        ]
    ],
    on="month",
    how="left"
)


# ==================================================
# ATTACH PREVIOUS COMPLETED QUARTER RPI
#
# THIS IS IMPORTANT FOR LEAKAGE.
#
# Example:
#
# Oct 2025 belongs to Q4.
#
# We attach Q3 2025,
# because Q4 is not yet completed.
# ==================================================

panel["rpi_quarter"] = (
    panel["month"]
    .dt.to_period("Q")
    - 1
)

panel = panel.merge(
    rpi[
        [
            "quarter",
            "hdb_rpi",
            "rpi_quarterly_growth",
            "rpi_yoy_growth"
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
# SORT BEFORE CREATING LAGS
# ==================================================

panel = panel.sort_values(
    [
        "town",
        "flat_type",
        "month"
    ]
).reset_index(drop=True)


# ==================================================
# PRICE FEATURES
# ==================================================

panel["price_now"] = (
    panel["median_price"]
)


for lag in range(1, 13):

    panel[f"price_lag_{lag}"] = (
        panel
        .groupby(
            ["town", "flat_type"]
        )["median_price"]
        .shift(lag)
    )


# ==================================================
# SIX-MONTH TARGET
# ==================================================

panel["target_price_6m"] = (
    panel
    .groupby(
        ["town", "flat_type"]
    )["median_price"]
    .shift(-6)
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

print("\nNew market-state features:")

print(
    "  hdb_rpi"
)

print(
    "  rpi_quarterly_growth"
)

print(
    "  rpi_yoy_growth"
)

print(
    "  transaction_volume"
)

print(
    "  volume_growth_3m"
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)