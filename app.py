import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Market Trend Analysis",
    page_icon="📊",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
body {background-color:#020617;}
h1,h2,h3 {color:#f8fafc;}
.section {margin-top:40px;}
.card {
    background:#0f172a;
    padding:18px;
    border-radius:16px;
    text-align:center;
}
.card h2 {color:#38bdf8;font-size:28px;}
.card p {color:#e5e7eb;font-size:14px;}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOAD ----------------
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

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Filters")

region = st.sidebar.multiselect(
    "Region", df["Region"].unique(), df["Region"].unique()
)
segment = st.sidebar.multiselect(
    "Segment", df["Segment"].unique(), df["Segment"].unique()
)
category = st.sidebar.multiselect(
    "Category", df["Category"].unique(), df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Segment"].isin(segment)) &
    (df["Category"].isin(category))
]

# ---------------- HEADER ----------------
st.title("📊 AI-Driven Market Trend Analysis")
st.markdown(
    "Interactive **sales analytics & forecasting dashboard** built using real retail data."
)

# ---------------- KPI CARDS ----------------
k1, k2, k3, k4 = st.columns(4)

k1.markdown(f"""
<div class="card">
<h2>₹ {filtered_df['Sales'].sum():,.0f}</h2>
<p>Total Sales</p>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="card">
<h2>{filtered_df.shape[0]}</h2>
<p>Total Orders</p>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="card">
<h2>₹ {filtered_df['Sales'].mean():,.0f}</h2>
<p>Avg Order Value</p>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="card">
<h2>{filtered_df['Customer ID'].nunique()}</h2>
<p>Unique Customers</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SALES DISTRIBUTION ----------------
c1, c2 = st.columns(2)

seg_sales = filtered_df.groupby("Segment")["Sales"].sum().reset_index()
fig_seg = px.pie(
    seg_sales,
    values="Sales",
    names="Segment",
    hole=0.45,
    title="Sales by Segment"
)
c1.plotly_chart(fig_seg, use_container_width=True)

reg_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
fig_reg = px.bar(
    reg_sales,
    x="Region",
    y="Sales",
    title="Sales by Region"
)
c2.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")

# ---------------- MONTHLY TREND ----------------
monthly = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
    .sum()
    .reset_index()
)

fig_month = px.line(
    monthly,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)
st.plotly_chart(fig_month, use_container_width=True)

st.markdown("---")

# ---------------- TOP PRODUCTS ----------------
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
    title="Top 10 Products"
)
st.plotly_chart(fig_top, use_container_width=True)

st.markdown("---")

# ---------------- ML FORECAST ----------------
monthly["Index"] = np.arange(len(monthly))
X = monthly[["Index"]]
y = monthly["Sales"]

model = LinearRegression()
model.fit(X, y)
monthly["Prediction"] = model.predict(X)

future_steps = 6
future_index = pd.DataFrame({
    "Index": range(len(X), len(X) + future_steps)
})

future_dates = pd.date_range(
    start=monthly["Order Date"].max(),
    periods=future_steps + 1,
    freq="M"
)[1:]

future_sales = model.predict(future_index)

forecast_df = pd.DataFrame({
    "Order Date": future_dates,
    "Sales": future_sales
})

forecast_plot = px.line(
    pd.concat([monthly[["Order Date","Sales"]], forecast_df]),
    x="Order Date",
    y="Sales",
    title="Sales Forecast (Linear Regression)"
)
st.plotly_chart(forecast_plot, use_container_width=True)

# ---------------- MODEL METRICS ----------------
m1, m2 = st.columns(2)
m1.metric("R² Score", round(r2_score(y, monthly["Prediction"]), 3))
m2.metric("MAE", f"₹ {mean_absolute_error(y, monthly['Prediction']):,.0f}")

st.markdown("---")

# ---------------- DATA VIEW ----------------
with st.expander("📂 View Filtered Data"):
    st.dataframe(filtered_df, use_container_width=True)

st.markdown(
    "**AI Market Trend Analysis | Streamlit + Machine Learning**"
)
