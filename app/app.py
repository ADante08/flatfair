import os

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

cfg = Config()

WAREHOUSE_ID = os.environ["WAREHOUSE_ID"]
HISTORY_TABLE = os.environ["HISTORY_TABLE"]
FORECAST_TABLE = os.environ["FORECAST_TABLE"]
HTTP_PATH = f"/sql/1.0/warehouses/{WAREHOUSE_ID}"


def get_connection():
    server_hostname = cfg.host
    if server_hostname.startswith("https://"):
        server_hostname = server_hostname.replace("https://", "")
    elif server_hostname.startswith("http://"):
        server_hostname = server_hostname.replace("http://", "")

    return sql.connect(
        server_hostname=server_hostname,
        http_path=HTTP_PATH,
        credentials_provider=lambda: cfg.authenticate,
        _use_arrow_native_complex_types=False,
    )


def query(statement: str) -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(statement)
            return cursor.fetchall_arrow().to_pandas()


def sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


st.set_page_config(
    page_title="FlatFair",
    page_icon="🏠",
    layout="wide",
)

st.title("FlatFair")
st.caption("Singapore HDB resale price forecasting powered by Databricks")

@st.cache_data(ttl=300)
def load_towns():
    return query(
        f"""
        SELECT DISTINCT town
        FROM {HISTORY_TABLE}
        WHERE flat_type IN ('3 ROOM', '4 ROOM', '5 ROOM')
        ORDER BY town
        """
    )


towns = load_towns()["town"].tolist()

if not towns:
    st.error("No towns were found in the Silver monthly-market table.")
    st.stop()

left, right = st.columns(2)
with left:
    town = st.selectbox("Town", towns)

@st.cache_data(ttl=300)
def load_flat_types(selected_town: str):
    town_sql = sql_literal(selected_town)
    return query(
        f"""
        SELECT DISTINCT flat_type
        FROM {HISTORY_TABLE}
        WHERE town = {town_sql}
          AND flat_type IN ('3 ROOM', '4 ROOM', '5 ROOM')
        ORDER BY flat_type
        """
    )


flat_types = load_flat_types(town)["flat_type"].tolist()
with right:
    flat_type = st.selectbox("Flat type", flat_types)


town_sql = sql_literal(town)
flat_sql = sql_literal(flat_type)

history = query(
    f"""
    SELECT month, median_price, transaction_count
    FROM {HISTORY_TABLE}
    WHERE town = {town_sql}
      AND flat_type = {flat_sql}
    ORDER BY month
    """
)

forecast = query(
    f"""
    SELECT
        origin_month,
        forecast_month,
        horizon_months,
        current_price,
        forecast_price,
        forecast_change_pct
    FROM {FORECAST_TABLE}
    WHERE town = {town_sql}
      AND flat_type = {flat_sql}
    ORDER BY horizon_months
    """
)

if history.empty or forecast.empty:
    st.warning("No forecast is available for this town and flat type.")
    st.stop()

for column in ["month"]:
    history[column] = pd.to_datetime(history[column])
for column in ["origin_month", "forecast_month"]:
    forecast[column] = pd.to_datetime(forecast[column])

current_price = float(forecast.iloc[0]["current_price"])
six_month = forecast.iloc[-1]

m1, m2, m3 = st.columns(3)
m1.metric("Current median", f"S${current_price:,.0f}")
m2.metric("6-month forecast", f"S${float(six_month['forecast_price']):,.0f}")
m3.metric("Expected 6-month change", f"{float(six_month['forecast_change_pct']):+.1f}%")

st.subheader("Price history and 1–6 month forecast")

history_chart = history[["month", "median_price"]].rename(
    columns={"month": "date", "median_price": "Historical"}
)

forecast_chart = forecast[["forecast_month", "forecast_price"]].rename(
    columns={"forecast_month": "date", "forecast_price": "Forecast"}
)

# Connect the forecast line to the current observed point.
origin_point = pd.DataFrame(
    {
        "date": [forecast.iloc[0]["origin_month"]],
        "Forecast": [current_price],
    }
)
forecast_chart = pd.concat([origin_point, forecast_chart], ignore_index=True)

chart = pd.merge(history_chart, forecast_chart, on="date", how="outer").sort_values("date")
st.line_chart(chart.set_index("date")[["Historical", "Forecast"]])

st.subheader("Forecast path")
forecast_table = forecast[
    ["forecast_month", "horizon_months", "forecast_price", "forecast_change_pct"]
].copy()
forecast_table.columns = ["Month", "Horizon", "Forecast price", "Change vs current (%)"]
forecast_table["Month"] = forecast_table["Month"].dt.strftime("%b %Y")
forecast_table["Horizon"] = forecast_table["Horizon"].map(lambda x: f"+{int(x)}m")
forecast_table["Forecast price"] = forecast_table["Forecast price"].map(lambda x: f"S${x:,.0f}")
forecast_table["Change vs current (%)"] = forecast_table["Change vs current (%)"].map(lambda x: f"{x:+.1f}%")
st.dataframe(forecast_table, hide_index=True, use_container_width=True)

with st.expander("How FlatFair is built"):
    st.markdown(
        """
        **Databricks architecture**

        Raw public HDB data → Bronze Delta tables → Silver cleaned/monthly market tables →
        Gold ML feature and forecast tables → MLflow-tracked XGBoost models → this Databricks App.
        """
    )
