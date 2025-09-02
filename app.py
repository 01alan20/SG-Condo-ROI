import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path
import re

st.set_page_config(page_title="SG Condo ROI Explorer", layout="wide")

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    use_cols = [
        "project","area_bucket","district","tenure","propertyType",
        "most_recent_buy","most_recent_rent","ROI"
    ]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Clean strings
    for col in ["project","area_bucket","district","tenure","propertyType"]:
        df[col] = df[col].astype(str).str.strip()

    # Districts → int
    df["district"] = (
        pd.to_numeric(df["district"], errors="coerce")
        .fillna(0).astype(int).astype(str)
    )

    # Tenure normalization
    def map_tenure(val):
        s = str(val).lower()
        if "freehold" in s:
            return "Freehold"
        match = re.search(r"\b(\d{2,4})\b", s)
        if match:
            yrs = int(match.group(1))
            if yrs in [99, 999, 9999]:
                return str(yrs)
        return "Other"
    df["tenure_norm"] = df["tenure"].apply(map_tenure)

    # Numeric conversions
    for col in ["most_recent_buy","most_recent_rent","ROI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat <=0 as missing
    df.loc[df["most_recent_buy"]  <= 0, "most_recent_buy"]  = np.nan
    df.loc[df["most_recent_rent"] <= 0, "most_recent_rent"] = np.nan

    # ROI ratio → %
    if df["ROI"].dropna().between(0,1).mean() > 0.8:
        df["ROI"] = df["ROI"] * 100

    return df

DATA_PATH = Path("data/pmi_roi_agg.csv")
df = load_data(DATA_PATH)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

# Property type
prop_types = st.sidebar.multiselect(
    "Property Type", sorted(df["propertyType"].dropna().unique()), placeholder="All"
)

# Area bucket → sort numeric ranges
def sort_area_buckets(values):
    specials = []
    numeric_ranges = []
    for v in values:
        if v in [">300", ">800", "<=100"]:
            specials.append(v)
        else:
            m = re.match(r"(\d+)-(\d+)", v)
            if m:
                start = int(m.group(1))
                numeric_ranges.append((start, v))
    specials = sorted(
        specials,
        key=lambda x: ["<=100", ">300", ">800"].index(x) if x in ["<=100", ">300", ">800"] else 99
    )
    numeric_ranges = [v for _, v in sorted(numeric_ranges)]
    return specials + numeric_ranges

areas = st.sidebar.multiselect(
    "Area", sort_area_buckets(df["area_bucket"].dropna().unique()), placeholder="All"
)

# District
districts = st.sidebar.multiselect(
    "District", sorted(df["district"].dropna().unique(), key=lambda x:int(x)), placeholder="All"
)

# Tenure
tenure_options = ["99","999","9999","Freehold","Other"]
chosen_tenure = st.sidebar.multiselect(
    "Tenure", tenure_options, placeholder="All"
)

# If 99 or 999 → allow year filter
year_filter = None
if any(x in chosen_tenure for x in ["99","999"]):
    year_filter = st.sidebar.selectbox("Year of Lease Start", list(range(1986,2026)))

# Price input
min_price = st.sidebar.number_input("Min Price (SGD)", value=0, step=10000)
max_price = st.sidebar.number_input("Max Price (SGD)", value=10000000, step=10000)

# ROI input
min_roi = st.sidebar.number_input("Est. ROI in Years (Min)", value=0, step=1)

# -----------------------------
# Apply filters
# -----------------------------
mask = pd.Series(True, index=df.index)

if prop_types: mask &= df["propertyType"].isin(prop_types)
if areas:      mask &= df["area_bucket"].isin(areas)
if districts:  mask &= df["district"].isin(districts)
if chosen_tenure: mask &= df["tenure_norm"].isin(chosen_tenure)

if year_filter:
    mask &= df["tenure"].str.contains(str(year_filter), errors="ignore")

# Price range
mask &= (df["most_recent_buy"].fillna(-1).between(min_price, max_price)) | df["most_recent_buy"].isna()
# ROI
mask &= (df["ROI"].fillna(-1) >= min_roi)

fdf = df.loc[mask].copy()

# -----------------------------
# KPIs
# -----------------------------
st.title("Singapore Condo ROI Explorer")
st.caption("Filters use **most recent buy/rent** only. Missing values shown as blank.")

c1, c2 = st.columns(2)
c1.metric("Rows", f"{len(fdf):,}")
avg_roi = fdf["ROI"].mean()
c2.metric("Avg ROI in Years", f"{avg_roi:.0f}" if pd.notna(avg_roi) else "—")

st.divider()

# -----------------------------
# Charts
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    if not fdf["ROI"].dropna().empty:
        fig1 = px.histogram(fdf, x="ROI", nbins=60, title="Distribution of ROI (Years)")
        fig1.update_layout(yaxis_title="Count", height=400)
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")
with col2:
    if not fdf[["most_recent_buy","most_recent_rent"]].dropna().empty:
        fig2 = px.scatter(
            fdf, x="most_recent_buy", y="most_recent_rent",
            color="area_bucket",
            hover_data=["project","district","tenure","propertyType","ROI"],
            labels={"most_recent_buy":"Most Recent Buy (SGD)",
                    "most_recent_rent":"Most Recent Rent (SGD)",
                    "area_bucket":"Area"},
            title="Most Recent Buy vs Rent"
        )
        fig2.update_traces(marker=dict(size=8, opacity=0.7))
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

st.divider()

# -----------------------------
# Helpers for formatting
# -----------------------------
def fmt_money(x):
    if pd.notna(x):
        return f"S${int(round(x)):,}"
    return "Unknown"

def fmt_roi(x):
    if pd.notna(x):
        return f"{int(round(x))}"   # flat number (rounded)
    return "Unknown"

# -----------------------------
# Table (numeric sort + pretty display)
# -----------------------------
st.subheader("Projects")

# build from filtered data
table = fdf.copy()

# 1) AREA: use the SAME ordered buckets as the sidebar
area_order = sort_area_buckets(table["area_bucket"].dropna().unique())
table["Area"] = pd.Categorical(table["area_bucket"], categories=area_order, ordered=True)

# 2) DISTRICT: numeric (already working)
table["District"] = pd.to_numeric(table["district"], errors="coerce").fillna(0).astype(int)

# 3) NUMERIC BACKEND: keep these as numbers (float so NaNs are allowed)
table["Most Recent Buy"]  = pd.to_numeric(table["most_recent_buy"], errors="coerce")
table["Most Recent Rent"] = pd.to_numeric(table["most_recent_rent"], errors="coerce")
table["Est ROI in Years"] = pd.to_numeric(table["ROI"], errors="coerce").round(0)

# 4) FINAL VIEW (only user-facing columns)
final = table[[
    "project", "propertyType", "Area", "District", "tenure",
    "Most Recent Buy", "Most Recent Rent", "Est ROI in Years"
]].rename(columns={
    "project": "Project",
    "propertyType": "Property Type",
    "tenure": "Tenure",
})

# 5) PRETTY DISPLAY via Pandas Styler (sorting still uses numeric backend)
styled = final.style.format({
    "Most Recent Buy":  lambda v: f"S${v:,.0f}" if pd.notna(v) else "—",
    "Most Recent Rent": lambda v: f"S${v:,.0f}" if pd.notna(v) else "—",
    "Est ROI in Years": lambda v: f"{v:.0f}"   if pd.notna(v) else "—",
})

st.dataframe(styled, use_container_width=True, hide_index=True)
