import streamlit as st
import pandas as pd

from utils.loader import load_dataset, get_dataframe_info
from utils.detector import detect_data_types, convert_to_datetime
from utils.quality import (
    check_numerical_quality,
    check_categorical_quality,
    check_datetime_quality,
    style_quality_table,
)
from utils.ai_interpreter import interpret_dataset_overview

# ── Page config ───────────────────────────────────────────────────────────────
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

# Load dataset — reset state on new upload
try:
    if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        df = load_dataset(uploaded_file)
        st.session_state["df"] = df
        st.session_state["filename"] = uploaded_file.name
        st.session_state.pop("col_types", None)
        st.session_state.pop("ai_summary", None)
        st.session_state.pop("datetime_overrides", None)
        st.session_state.pop("override_warnings", None)
    else:
        df = st.session_state["df"]
except ValueError as e:
    st.error(str(e))
    st.stop()

# ── Row / Column metrics ──────────────────────────────────────────────────────
st.success(f"File loaded: **{uploaded_file.name}**")
col_a, col_b = st.columns(2)
col_a.metric("Rows", f"{df.shape[0]:,}")
col_b.metric("Columns", df.shape[1])

st.divider()

# ── Dataset Preview ───────────────────────────────────────────────────────────
st.subheader("Dataset Preview")
st.caption("Showing the first 100 rows of your data.")

# Convert bool columns to text so they display as True / False, not checkboxes
preview_df = df.head(100).copy()
bool_cols = preview_df.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    preview_df[bool_cols] = preview_df[bool_cols].astype(str)

st.dataframe(preview_df, use_container_width=True, height=300)

st.divider()

# ── Column Overview ───────────────────────────────────────────────────────────
st.subheader("Column Overview")
st.caption(
    "Each column's data type, category, total values, and missing values at a glance. "
    "Red = 20%+ missing values. Yellow = 5-19% missing values."
)

# Apply any validated datetime overrides from a previous interaction
stored_overrides = st.session_state.get("datetime_overrides", [])
if stored_overrides:
    df = convert_to_datetime(df, stored_overrides)
    st.session_state["df"] = df

# Detect column types on the (possibly updated) dataframe
col_types = detect_data_types(df)
st.session_state["col_types"] = col_types

# Build merged info + column type table
info_df = get_dataframe_info(df)

type_label_map = {
    "numerical": "Numerical",
    "categorical": "Categorical",
    "datetime": "Date / Time",
}
col_to_type = {}
for type_key, cols in col_types.items():
    for c in cols:
        col_to_type[c] = type_label_map[type_key]

# Insert "Column Type" right after "Data Type"
dt_index = info_df.columns.tolist().index("Data Type")
info_df.insert(dt_index + 1, "Column Type", info_df["Column Name"].map(col_to_type))


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


styled_info = info_df.style.map(highlight_missing_pct, subset=["Missing %"])
st.dataframe(styled_info, use_container_width=True, hide_index=True)

# Show any warnings from the previous override attempt
if st.session_state.get("override_warnings"):
    for warning in st.session_state["override_warnings"]:
        st.warning(warning)
    st.session_state["override_warnings"] = []

# Datetime override multiselect
non_datetime_cols = col_types["numerical"] + col_types["categorical"]
if non_datetime_cols:
    new_overrides = st.multiselect(
        "Mark additional columns as Date / Time (optional)",
        options=non_datetime_cols,
        default=[c for c in stored_overrides if c in non_datetime_cols],
        help="Select any column that contains dates but was not automatically detected.",
    )

    if set(new_overrides) != set(stored_overrides):
        valid_overrides = []
        warnings = []

        for col in new_overrides:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                warnings.append(
                    f"'{col}' has no values to check — it was not converted."
                )
                continue
            converted = pd.to_datetime(non_null, errors="coerce")
            valid_ratio = converted.notna().sum() / len(non_null)

            if valid_ratio >= 0.8:
                valid_overrides.append(col)
            else:
                warnings.append(
                    f"'{col}' was not converted — only {valid_ratio * 100:.0f}% of its "
                    f"values look like dates. Please check the column and try again."
                )

        st.session_state["datetime_overrides"] = valid_overrides
        st.session_state["override_warnings"] = warnings
        if valid_overrides:
            st.session_state.pop("ai_summary", None)
        st.rerun()

st.divider()

# ── Data Quality Summary ──────────────────────────────────────────────────────
st.subheader("Data Quality Summary")
st.caption(
    "Red = 20%+ missing values (needs attention). "
    "Yellow = 5-19% missing values (worth reviewing)."
)

num_df = check_numerical_quality(df, col_types["numerical"])
cat_df = check_categorical_quality(df, col_types["categorical"])
dt_df = check_datetime_quality(df, col_types["datetime"])

tab1, tab2, tab3 = st.tabs(["Numerical Columns", "Categorical Columns", "Date / Time Columns"])

with tab1:
    if num_df.empty:
        st.info("No numerical columns found.")
    else:
        st.dataframe(style_quality_table(num_df), use_container_width=True, hide_index=True)

with tab2:
    if cat_df.empty:
        st.info("No categorical columns found.")
    else:
        st.dataframe(style_quality_table(cat_df), use_container_width=True, hide_index=True)

with tab3:
    if dt_df.empty:
        st.info("No date/time columns found.")
    else:
        st.dataframe(style_quality_table(dt_df), use_container_width=True, hide_index=True)

st.divider()

# ── AI Interpretation ─────────────────────────────────────────────────────────
st.subheader("AI Interpretation")
st.caption("A plain-English explanation of your data based on the quality findings above.")


def build_quality_text(quality_df: pd.DataFrame) -> str:
    if quality_df.empty:
        return ""
    lines = []
    for _, row in quality_df.iterrows():
        lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(lines)


if "ai_summary" not in st.session_state:
    with st.spinner("Analysing your data..."):
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
