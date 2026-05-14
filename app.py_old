import streamlit as st
import pandas as pd

from utils.loader import load_dataset, get_dataframe_info
from utils.detector import detect_data_types, convert_to_datetime, build_type_summary
from utils.quality import (
    check_numerical_quality,
    check_categorical_quality,
    check_datetime_quality,
    style_quality_table,
)
from utils.ai_interpreter import interpret_dataset_overview

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analytics App",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analytics App")
st.caption("Upload your data file to get started. The app will inspect it, check its quality, and explain the findings in plain English.")

st.divider()

# ── Step 1: File Upload ───────────────────────────────────────────────────────
st.subheader("Step 1 — Upload Your Data File")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV (.csv) and Excel (.xlsx, .xls)",
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

# Load the dataset
try:
    if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        df = load_dataset(uploaded_file)
        st.session_state["df"] = df
        st.session_state["filename"] = uploaded_file.name
        # Clear downstream state on new upload
        for key in ["col_types", "ai_summary"]:
            st.session_state.pop(key, None)
    else:
        df = st.session_state["df"]
except ValueError as e:
    st.error(str(e))
    st.stop()

st.success(f"File loaded: **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

st.divider()

# ── Step 2: DataFrame Preview ─────────────────────────────────────────────────
st.subheader("Dataset Preview")
st.caption("Showing the first 100 rows of your data.")

st.dataframe(df.head(100), use_container_width=True, height=300)

st.divider()

# ── Step 3: DataFrame Info Panel ──────────────────────────────────────────────
st.subheader("Column Overview")
st.caption("A summary of each column — its type, how many values are present, and how many are missing.")

info_df = get_dataframe_info(df)

# Colour-code the Missing % column
def highlight_missing_pct(val):
    try:
        v = float(val)
        if v >= 20:
            return "background-color: #f8d7da; color: #842029"
        elif v >= 5:
            return "background-color: #fff3cd; color: #664d03"
        return ""
    except (ValueError, TypeError):
        return ""

styled_info = info_df.style.applymap(highlight_missing_pct, subset=["Missing %"])
st.dataframe(styled_info, use_container_width=True, hide_index=True)

st.divider()

# ── Step 4: Data Type Classification ─────────────────────────────────────────
st.subheader("Column Type Classification")
st.caption(
    "Columns are classified as Numerical, Categorical (text), or Date/Time. "
    "If any date columns were not automatically detected, you can add them below."
)

if "col_types" not in st.session_state:
    col_types = detect_data_types(df)
    st.session_state["col_types"] = col_types
else:
    col_types = st.session_state["col_types"]

# Type summary table
type_summary_df = build_type_summary(col_types)
st.dataframe(type_summary_df, use_container_width=True, hide_index=True)

# Allow user to override datetime columns
non_datetime_cols = col_types["numerical"] + col_types["categorical"]
if non_datetime_cols:
    override_cols = st.multiselect(
        "Mark additional columns as Date / Time (optional)",
        options=non_datetime_cols,
        default=[],
        help="Select any column that contains dates but was not automatically detected.",
    )
    if override_cols:
        df = convert_to_datetime(df, override_cols)
        st.session_state["df"] = df
        # Re-run detection with updated df
        col_types = detect_data_types(df)
        st.session_state["col_types"] = col_types
        st.session_state.pop("ai_summary", None)

st.divider()

# ── Step 5: Data Quality Summary ──────────────────────────────────────────────
st.subheader("Data Quality Summary")
st.caption(
    "Red = 20%+ missing values (needs attention). "
    "Yellow = 5–19% missing values (worth reviewing). "
    "No colour = less than 5% missing values."
)

num_df = check_numerical_quality(df, col_types["numerical"])
cat_df = check_categorical_quality(df, col_types["categorical"])
dt_df = check_datetime_quality(df, col_types["datetime"])

tab1, tab2, tab3 = st.tabs(["Numerical Columns", "Text Columns", "Date / Time Columns"])

with tab1:
    if num_df.empty:
        st.info("No numerical columns found.")
    else:
        st.dataframe(style_quality_table(num_df), use_container_width=True, hide_index=True)

with tab2:
    if cat_df.empty:
        st.info("No text columns found.")
    else:
        st.dataframe(style_quality_table(cat_df), use_container_width=True, hide_index=True)

with tab3:
    if dt_df.empty:
        st.info("No date/time columns found.")
    else:
        st.dataframe(style_quality_table(dt_df), use_container_width=True, hide_index=True)

st.divider()

# ── Step 6: AI Interpretation ─────────────────────────────────────────────────
st.subheader("AI Interpretation")
st.caption("A plain-English explanation of your dataset based on the quality findings above.")

def build_quality_text(quality_df: pd.DataFrame) -> str:
    """Convert quality DataFrame to a compact text summary for the AI prompt."""
    if quality_df.empty:
        return ""
    lines = []
    for _, row in quality_df.iterrows():
        lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(lines)

if "ai_summary" not in st.session_state:
    with st.spinner("Analysing your dataset…"):
        try:
            summary = interpret_dataset_overview(
                shape=df.shape,
                col_types=col_types,
                num_summary=build_quality_text(num_df),
                cat_summary=build_quality_text(cat_df),
                dt_summary=build_quality_text(dt_df),
            )
            st.session_state["ai_summary"] = summary
        except Exception as e:
            st.error(f"Could not generate AI interpretation: {e}")
            st.session_state["ai_summary"] = None

if st.session_state.get("ai_summary"):
    st.info(st.session_state["ai_summary"])
    if st.button("🔄 Refresh interpretation"):
        st.session_state.pop("ai_summary", None)
        st.rerun()
