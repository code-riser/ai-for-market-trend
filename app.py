
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Superstore_sales.csv")


# PAGE CONFIG

st.set_page_config(
    page_title="AI Market Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# PREMIUM UI CSS

st.markdown("""
<style>
body {background-color:#020617;}
.main {background-color:#020617;}
h1,h2,h3,h4 {color:#f8fafc;}
.metric {
    background: linear-gradient(135deg,#1e293b,#020617);
    padding:20px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 15px 40px rgba(0,0,0,0.5);
}
.metric h2 {color:#38bdf8;font-size:32px;}
.metric p {color:#e5e7eb;font-size:16px;}
</style>
""", unsafe_allow_html=True)


# DATA LOADER (SAFE)

@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/Superstore_sales.csv",
        encoding="latin1",
        parse_dates=["Order Date"],
        dayfirst=True
    )
    df = df.dropna(subset=["Sales"])
    return df

df = load_data()


# SIDEBAR FILTERS

st.sidebar.title("⚙️ Filters")

region = st.sidebar.multiselect(
    "Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

segment = st.sidebar.multiselect(
    "Customer Segment",
    options=df["Segment"].unique(),
    default=df["Segment"].unique()
)

category = st.sidebar.multiselect(
    "Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Segment"].isin(segment)) &
    (df["Category"].isin(category))
]


# HEADER

st.title("📊 AI-Driven Market Trend Analysis")
st.markdown(
    "Dynamic analytics & forecasting system using Machine Learning on retail sales data."
)


# KPI METRICS (CIRCLE CONCEPT)

total_sales = filtered_df["Sales"].sum()
total_orders = len(filtered_df)
avg_sales = filtered_df["Sales"].mean()

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric">
        <h2>₹ {total_sales:,.0f}</h2>
        <p>Total Sales</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric">
        <h2>{total_orders}</h2>
        <p>Total Orders</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric">
        <h2>₹ {avg_sales:,.2f}</h2>
        <p>Average Sale</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# SALES BY SEGMENT (DONUT)

seg_sales = filtered_df.groupby("Segment")["Sales"].sum().reset_index()

fig_seg = px.pie(
    seg_sales,
    values="Sales",
    names="Segment",
    hole=0.5,
    title="Sales by Customer Segment",
    template="plotly_dark"
)


# SALES BY REGION

reg_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()

fig_region = px.bar(
    reg_sales,
    x="Region",
    y="Sales",
    title="Sales by Region",
    template="plotly_dark"
)

colA, colB = st.columns(2)
colA.plotly_chart(fig_seg, width="stretch")
colB.plotly_chart(fig_region, width="stretch")

st.divider()


# MONTHLY SALES TREND

monthly = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
    .sum()
    .reset_index()
)

fig_month = px.line(
    monthly,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend",
    template="plotly_dark"
)

st.plotly_chart(fig_month, width="stretch")


# TOP 10 PRODUCTS

top_products = (
    filtered_df
    .groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_top = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top 10 Products by Sales",
    template="plotly_dark"
)

st.plotly_chart(fig_top, width="stretch")

st.divider()


# LINEAR REGRESSION FORECAST

monthly["Index"] = np.arange(len(monthly))

X = monthly[["Index"]]
y = monthly["Sales"]

model = LinearRegression()
model.fit(X, y)

monthly["Prediction"] = model.predict(X)

r2 = r2_score(y, monthly["Prediction"])
mae = mean_absolute_error(y, monthly["Prediction"])

future_steps = 12
future_index = pd.DataFrame({
    "Index": range(len(X), len(X) + future_steps)
})

future_dates = pd.date_range(
    start=monthly["Order Date"].max(),
    periods=future_steps + 1,
    freq="ME"
)[1:]

future_sales = model.predict(future_index)

forecast_df = pd.DataFrame({
    "Order Date": future_dates,
    "Sales": future_sales
})

fig_forecast = px.line(
    pd.concat([monthly[["Order Date","Sales"]], forecast_df]),
    x="Order Date",
    y="Sales",
    title="Sales Forecast (Linear Regression)",
    template="plotly_dark"
)

st.plotly_chart(fig_forecast, width="stretch")


# MODEL PERFORMANCE

m1, m2 = st.columns(2)
m1.metric("R² Score", f"{r2:.3f}")
m2.metric("Mean Absolute Error", f"₹ {mae:,.2f}")

st.divider()


# RAW & FILTERED DATA

st.subheader("📂 Filtered Dataset")
st.dataframe(filtered_df, width="stretch")

st.subheader("📂 Raw Dataset")
st.dataframe(df.head(100), width="stretch")


# FOOTER

st.divider()
st.markdown(
    "**AI Applications – Module E | Market Trend Analysis Project**"
)
