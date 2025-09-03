import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path
import re

st.set_page_config(page_title="SG Condo ROI Explorer", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def bucket_to_first_number(val):
    """Return first numeric value from a range/bucket string."""
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()

    # 200-300 -> 200
    m = re.match(r'^\s*(\d+)\s*-\s*\d+\s*$', s)
    if m:
        return int(m.group(1))

    # <=100 -> 100
    m = re.match(r'^\s*<=\s*(\d+)\s*$', s)
    if m:
        return int(m.group(1))

    # >300 or >=120 -> 300 / 120
    m = re.match(r'^\s*>=?\s*(\d+)\s*$', s)
    if m:
        return int(m.group(1))

    # 1200+ or plain number -> 1200
    m = re.match(r'^\s*(\d+)\s*\+?\s*$', s)
    if m:
        return int(m.group(1))

    # Fallback: first number anywhere
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else pd.NA


def lease_start_year_from_tenure(val):
    """Year of Lease Start = last 4 chars if digits else last 4-digit found anywhere."""
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if len(s) >= 4 and s[-4:].isdigit():
        return int(s[-4:])
    yrs = re.findall(r'(\d{4})', s)
    return int(yrs[-1]) if yrs else pd.NA


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    use_cols = [
        "project","area_bucket","district","tenure","propertyType",
        "most_recent_buy","most_recent_rent","ROI", "filter"  # 'filter' optional
    ]

    # --- robust CSV read (handles irregular rows on Streamlit Cloud) ---
    def _read_csv_robust(p: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(p, low_memory=False)
        except Exception:
            try:
                # sniff delimiter; tolerate bad rows
                return pd.read_csv(p, engine="python", sep=None, on_bad_lines="skip", low_memory=False)
            except Exception:
                # last resort: permissive encoding
                return pd.read_csv(p, engine="python", sep=None, on_bad_lines="skip",
                                   low_memory=False, encoding="latin-1")

    if not path.exists():
        st.error(f"Data file not found: {path}")
        st.stop()

    raw = _read_csv_robust(path)

    # Keep only columns we use (ignore extras); if some are missing, we’ll proceed with what exists
    cols = [c for c in use_cols if c in raw.columns]
    df = raw[cols].copy()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Clean strings
    for col in ["project","area_bucket","district","tenure","propertyType","filter"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Districts → int but stay as string for equality filtering
    if "district" in df.columns:
        df["district"] = (
            pd.to_numeric(df["district"], errors="coerce")
            .fillna(0).astype(int).astype(str)
        )

    # Tenure normalization (Freehold / 9999 / 999 / 99 / Other)
    def map_tenure(val):
        s = str(val).lower()
        if "freehold" in s:
            return "Freehold"
        if re.search(r'(?<!\d)9999(?!\d)', s):
            return "9999"
        if re.search(r'(?<!\d)999(?!\d)', s):
            return "999"
        if re.search(r'(?<!\d)99(?!\d)', s):
            return "99"
        return "Other"

    df["tenure_norm"] = df["tenure"].apply(map_tenure)

    # Year of Lease Start: last 4 chars if digits, else last 4-digit found
    def lease_start_year_from_tenure(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()
        if len(s) >= 4 and s[-4:].isdigit():
            return int(s[-4:])
        yrs = re.findall(r'(\d{4})', s)
        return int(yrs[-1]) if yrs else pd.NA

    df["lease_year"] = df["tenure"].apply(lease_start_year_from_tenure).astype("Int64")

    # Numerics
    for col in ["most_recent_buy","most_recent_rent","ROI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat <=0 as missing
    if "most_recent_buy" in df.columns:
        df.loc[df["most_recent_buy"]  <= 0, "most_recent_buy"]  = np.nan
    if "most_recent_rent" in df.columns:
        df.loc[df["most_recent_rent"] <= 0, "most_recent_rent"] = np.nan

    # ROI ratio → % if mostly 0..1
    if "ROI" in df.columns and not df["ROI"].dropna().empty:
        if df["ROI"].dropna().between(0,1).mean() > 0.8:
            df["ROI"] = df["ROI"] * 100

    # Numeric versions of area/filter buckets
    def bucket_to_first_number(v):
        if pd.isna(v): return pd.NA
        s = str(v).strip()
        m = re.match(r'^\s*(\d+)\s*-\s*\d+\s*$', s) or \
            re.match(r'^\s*<=?\s*(\d+)\s*$', s) or \
            re.match(r'^\s*>=?\s*(\d+)\s*$', s) or \
            re.match(r'^\s*(\d+)\s*\+?\s*$', s)
        if m: return int(m.group(1))
        m = re.search(r'(\d+)', s)
        return int(m.group(1)) if m else pd.NA

    if "area_bucket" in df.columns:
        df["area_num"] = df["area_bucket"].apply(bucket_to_first_number).astype("Int64")
    if "filter" in df.columns:
        df["filter_num"] = df["filter"].apply(bucket_to_first_number).astype("Int64")

    return df

DATA_PATH = Path("data/pmi_roi_agg.csv")
df = load_data(DATA_PATH)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

# Property type
prop_types = st.sidebar.multiselect(
    "Property Type",
    sorted(df["propertyType"].dropna().unique()) if "propertyType" in df.columns else [],
    placeholder="All"
)

# Area (numeric first values)
areas = st.sidebar.multiselect(
    "Area (first value)",
    sorted(df["area_num"].dropna().unique().tolist()) if "area_num" in df.columns else [],
    placeholder="All"
)

# Optional: Filter column (if present), also numeric
filter_vals = []
if "filter_num" in df.columns:
    filter_vals = st.sidebar.multiselect(
        "Filter (first value)",
        sorted(df["filter_num"].dropna().unique().tolist()),
        placeholder="All"
    )

# District
districts = st.sidebar.multiselect(
    "District",
    sorted(df["district"].dropna().unique(), key=lambda x:int(x)) if "district" in df.columns else [],
    placeholder="All"
)

# Tenure (normalized) — includes 99/999/9999
tenure_options = ["99","999","9999","Freehold","Other"]
chosen_tenure = st.sidebar.multiselect("Tenure", tenure_options, placeholder="All")

# Year list comes ONLY from rows that match the chosen lease tenures; include "All years"
lease_tenures = {"99","999","9999"}
selected_lease_tenures = lease_tenures.intersection(chosen_tenure or [])
year_filter = None

if selected_lease_tenures:
    years = []
    if "lease_year" in df.columns and "tenure_norm" in df.columns:
        years = (
            df.loc[df["tenure_norm"].isin(selected_lease_tenures), "lease_year"]
              .dropna().astype(int).sort_values().unique().tolist()
        )
    if years:
        year_choice = st.sidebar.selectbox("Year of Lease Start", ["All years"] + years)
        if year_choice != "All years":
            year_filter = int(year_choice)
    else:
        st.sidebar.selectbox("Year of Lease Start", ["No years found"], index=0)

# Price input
min_price = st.sidebar.number_input("Min Price (SGD)", value=0, step=10000)
max_price = st.sidebar.number_input("Max Price (SGD)", value=10000000, step=10000)

# ROI input
min_roi = st.sidebar.number_input("Est. ROI in Years (Min)", value=0, step=1)

# -----------------------------
# Apply filters
# -----------------------------
mask = pd.Series(True, index=df.index)

if prop_types and "propertyType" in df.columns:
    mask &= df["propertyType"].isin(prop_types)

if areas and "area_num" in df.columns:
    mask &= df["area_num"].isin(areas)

if filter_vals and "filter_num" in df.columns:
    mask &= df["filter_num"].isin(filter_vals)

if districts and "district" in df.columns:
    mask &= df["district"].isin(districts)

# Tenure filter (multi-select)
if chosen_tenure and "tenure_norm" in df.columns:
    mask &= df["tenure_norm"].isin(chosen_tenure)

# Apply year filter ONLY if a concrete year was selected
if (year_filter is not None) and ("lease_year" in df.columns):
    mask &= (df["lease_year"] == year_filter)

# Price range
if "most_recent_buy" in df.columns:
    mask &= (df["most_recent_buy"].fillna(-1).between(min_price, max_price)) | df["most_recent_buy"].isna()

# ROI
if "ROI" in df.columns:
    mask &= (df["ROI"].fillna(-1) >= min_roi)

fdf = df.loc[mask].copy()

# -----------------------------
# KPIs
# -----------------------------
st.title("Singapore Condo ROI Explorer")
st.caption("Filters use **most recent buy/rent** only. Missing values shown as blank.")

c1, c2 = st.columns(2)
c1.metric("Rows", f"{len(fdf):,}")
avg_roi = fdf["ROI"].mean() if "ROI" in fdf.columns else np.nan
c2.metric("Avg ROI in Years", f"{avg_roi:.0f}" if pd.notna(avg_roi) else "—")

st.divider()

# -----------------------------
# Charts
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    if "ROI" in fdf.columns and not fdf["ROI"].dropna().empty:
        fig1 = px.histogram(fdf, x="ROI", nbins=60, title="Distribution of ROI (Years)")
        fig1.update_layout(yaxis_title="Count", height=400)
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

with col2:
    # Plotly JSON can't serialize pd.NA from Int64 -> cast to float and replace pd.NA
    needed = ["most_recent_buy","most_recent_rent"]
    if all(c in fdf.columns for c in needed):
        plot_df = fdf.copy()
        if "area_num" in plot_df.columns:
            plot_df["area_num_plot"] = plot_df["area_num"].astype(float)
        if "lease_year" in plot_df.columns:
            plot_df["lease_year_plot"] = plot_df["lease_year"].astype(float)
        plot_df = plot_df.where(pd.notna(plot_df), None)
        if not plot_df[["most_recent_buy","most_recent_rent"]].dropna().empty:
            fig2 = px.scatter(
                plot_df, x="most_recent_buy", y="most_recent_rent",
                color="area_num_plot" if "area_num_plot" in plot_df.columns else None,
                hover_data=[c for c in ["project","district","tenure","propertyType","ROI","lease_year_plot"] if c in plot_df.columns],
                labels={
                    "most_recent_buy":"Most Recent Buy (SGD)",
                    "most_recent_rent":"Most Recent Rent (SGD)",
                    "area_num_plot":"Area",
                    "lease_year_plot":"Year of Lease Start"
                },
                title="Most Recent Buy vs Rent"
            )
            fig2.update_traces(marker=dict(size=8, opacity=0.7))
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

st.divider()

# -----------------------------
# Table (numeric sort + pretty display)
# -----------------------------
st.subheader("Projects")

table = fdf.copy()

# AREA: numeric first value
if "area_num" in table.columns:
    table["Area"] = table["area_num"].astype("Int64")

# DISTRICT: numeric
if "district" in table.columns:
    table["District"] = pd.to_numeric(table["district"], errors="coerce").fillna(0).astype(int)

# NUMERIC BACKEND
if "most_recent_buy" in table.columns:
    table["Most Recent Buy"]  = pd.to_numeric(table["most_recent_buy"], errors="coerce")
if "most_recent_rent" in table.columns:
    table["Most Recent Rent"] = pd.to_numeric(table["most_recent_rent"], errors="coerce")
if "ROI" in table.columns:
    table["Est ROI in Years"] = pd.to_numeric(table["ROI"], errors="coerce").round(0)

# Tenure split for table (normalized + year)
table["Tenure"] = table["tenure_norm"]
table["Year of Lease Start"] = table["lease_year"]

# Optional numeric Filter if present
if "filter_num" in table.columns:
    table["Filter"] = table["filter_num"].astype("Int64")

# FINAL VIEW (only user-facing columns)
display_cols = [c for c in [
    "project", "propertyType", "Area", "District", "Tenure", "Year of Lease Start",
    "Most Recent Buy", "Most Recent Rent", "Est ROI in Years", "Filter"
] if c in table.columns]

final = table[display_cols].rename(columns={
    "project": "Project",
    "propertyType": "Property Type",
})

# PRETTY DISPLAY via Pandas Styler
styled = final.style.format({
    "Most Recent Buy":  lambda v: f"S${v:,.0f}" if pd.notna(v) else "—",
    "Most Recent Rent": lambda v: f"S${v:,.0f}" if pd.notna(v) else "—",
    "Est ROI in Years": lambda v: f"{v:.0f}"   if pd.notna(v) else "—",
})

st.dataframe(styled, use_container_width=True, hide_index=True)
