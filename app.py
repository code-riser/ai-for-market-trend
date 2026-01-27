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

# ---------------- CSS ----------------
st.markdown("""
<style>
body {background-color:#020617;}
h1,h2,h3 {color:#f8fafc;}
.metric {
    background:#0f172a;
    padding:20px;
    border-radius:16px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOAD ----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "data/Superstore_sales.csv",
            encoding="latin1",
            parse_dates=["Order Date"],
            dayfirst=True
        )
        df = df.dropna(subset=["Sales"])
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        st.stop()

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Filters")

region = st.sidebar.multiselect(
    "Region",
    df["Region"].unique(),
    df["Region"].unique()
)

segment = st.sidebar.multiselect(
    "Segment",
    df["Segment"].unique(),
    df["Segment"].unique()
)

category = st.sidebar.multiselect(
    "Category",
    df["Category"].unique(),
    df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Segment"].isin(segment)) &
    (df["Category"].isin(category))
]

# ---------------- HEADER ----------------
st.title("📊 AI-Driven Market Trend Analysis")
st.markdown("Retail sales analytics & forecasting using Machine Learning")

# ---------------- KPIs ----------------
c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div class="metric">
<h2>₹ {filtered_df['Sales'].sum():,.0f}</h2>
<p>Total Sales</p>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric">
<h2>{len(filtered_df)}</h2>
<p>Total Orders</p>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric">
<h2>₹ {filtered_df['Sales'].mean():,.2f}</h2>
<p>Average Sale</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SEGMENT PIE ----------------
seg = filtered_df.groupby("Segment")["Sales"].sum().reset_index()
fig_seg = px.pie(seg, values="Sales", names="Segment", hole=0.4)
st.plotly_chart(fig_seg, use_container_width=True)

# ---------------- MONTHLY TREND ----------------
monthly = filtered_df.groupby(
    pd.Grouper(key="Order Date", freq="M")
)["Sales"].sum().reset_index()

fig_month = px.line(monthly, x="Order Date", y="Sales", markers=True)
st.plotly_chart(fig_month, use_container_width=True)

# ---------------- ML FORECAST ----------------
monthly["Index"] = np.arange(len(monthly))
X = monthly[["Index"]]
y = monthly["Sales"]

model = LinearRegression()
model.fit(X, y)
monthly["Prediction"] = model.predict(X)

r2 = r2_score(y, monthly["Prediction"])
mae = mean_absolute_error(y, monthly["Prediction"])

st.markdown("### 📈 Model Performance")
st.metric("R² Score", round(r2, 3))
st.metric("Mean Absolute Error", f"₹ {mae:,.2f}")

# ---------------- DATA VIEW ----------------
st.markdown("### 📂 Filtered Data")
st.dataframe(filtered_df)

st.markdown("---")
st.markdown("**AI Applications – Market Trend Analysis Project**")
