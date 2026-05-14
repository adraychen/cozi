import pandas as pd
import streamlit as st

from utils.loader import load_dataset, get_dataframe_info
from utils.detector import detect_data_types, convert_to_datetime
from utils.charts import make_histogram_boxplot, make_bar_chart, DISCRETE_THRESHOLD
from utils.ai_interpreter import interpret_dataset_overview
from utils.quality import (
    check_numerical_quality,
    check_categorical_quality,
    check_datetime_quality,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analytics App",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analytics App")
st.caption(
    "Upload your data file to get started. The app will inspect it, "
    "check its quality, and explain the findings in plain English."
)

st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
st.subheader("Step 1 — Upload Your Data File")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV (.csv) and Excel (.xlsx, .xls)",
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

try:
    if (
        "df" not in st.session_state
        or st.session_state.get("filename") != uploaded_file.name
    ):
        df = load_dataset(uploaded_file)
        st.session_state["df"] = df
        st.session_state["filename"] = uploaded_file.name
        for key in ("col_types", "ai_summary", "datetime_overrides", "override_warnings"):
            st.session_state.pop(key, None)
    else:
        df = st.session_state["df"]
except ValueError as e:
    st.error(str(e))
    st.stop()

# Row / Column metrics
st.success(f"File loaded: **{uploaded_file.name}**")
m1, m2 = st.columns(2)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", df.shape[1])

st.divider()

# ── Dataset Preview ───────────────────────────────────────────────────────────
st.subheader("Dataset Preview")
st.caption("Showing the first 100 rows of your data.")

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
    "Red = 20%+ missing values. Yellow = 5–19% missing values."
)

stored_overrides = st.session_state.get("datetime_overrides", [])
if stored_overrides:
    df = convert_to_datetime(df, stored_overrides)
    st.session_state["df"] = df

col_types = detect_data_types(df)
st.session_state["col_types"] = col_types

info_df = get_dataframe_info(df)

type_label_map = {"numerical": "Numerical", "categorical": "Categorical", "datetime": "Date / Time"}
col_to_type = {c: type_label_map[k] for k, cols in col_types.items() for c in cols}
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


st.dataframe(
    info_df.style.map(highlight_missing_pct, subset=["Missing %"]),
    use_container_width=True,
    hide_index=True,
)

# Override warnings from previous run
if st.session_state.get("override_warnings"):
    for w in st.session_state["override_warnings"]:
        st.warning(w)
    st.session_state["override_warnings"] = []

non_datetime_cols = col_types["numerical"] + col_types["categorical"]
if non_datetime_cols:
    new_overrides = st.multiselect(
        "Mark additional columns as Date / Time (optional)",
        options=non_datetime_cols,
        default=[c for c in stored_overrides if c in non_datetime_cols],
        help="Select any column that contains dates but was not detected automatically.",
    )

    if set(new_overrides) != set(stored_overrides):
        valid, warnings = [], []
        for col in new_overrides:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                warnings.append(f"'{col}' has no values to check — it was not converted.")
                continue
            converted = pd.to_datetime(non_null, errors="coerce")
            ratio = converted.notna().sum() / len(non_null)
            if ratio >= 0.8:
                valid.append(col)
            else:
                warnings.append(
                    f"'{col}' was not converted — only {ratio * 100:.0f}% of its values "
                    f"look like dates. Please check the column and try again."
                )
        st.session_state["datetime_overrides"] = valid
        st.session_state["override_warnings"] = warnings
        if valid:
            st.session_state.pop("ai_summary", None)
        st.rerun()

st.divider()

# ── Data Quality Summary ──────────────────────────────────────────────────────
st.subheader("Data Quality Summary")


def _stats_row(label: str, value) -> None:
    st.markdown(f"<div style='padding:2px 0'><b>{label}:</b> {value}</div>",
                unsafe_allow_html=True)


def render_numerical_panel(col_name: str) -> None:
    series = df[col_name]
    total = len(series)
    missing = int(series.isna().sum())
    missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0
    unique = int(series.nunique())
    is_discrete = unique < DISCRETE_THRESHOLD
    has_data = not series.isna().all()

    mn  = f"{series.min():.2f}"  if has_data else "—"
    mx  = f"{series.max():.2f}"  if has_data else "—"
    avg = f"{series.mean():.2f}" if has_data else "—"

    with st.container(border=True):
        st.markdown(f"**{col_name}**")
        left, right = st.columns([3, 2])

        with left:
            if is_discrete:
                fig, truncated, total_unique = make_bar_chart(series)
                st.plotly_chart(fig, use_container_width=True)
                if truncated:
                    st.caption(f"Showing top 10 of {total_unique} unique values, sorted by frequency.")
            else:
                st.plotly_chart(make_histogram_boxplot(series), use_container_width=True)

        with right:
            for label, val in [
                ("Missing Values", missing),
                ("Missing %",      f"{missing_pct:.2f}%"),
                ("Unique Values",  unique),
                ("Minimum",        mn),
                ("Maximum",        mx),
                ("Average",        avg),
            ]:
                _stats_row(label, val)


def render_categorical_panel(col_name: str) -> None:
    series = df[col_name]
    total = len(series)
    missing = int(series.isna().sum())
    missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0
    unique = int(series.nunique())

    if series.dtype == object:
        lower_unique = series.dropna().str.strip().str.lower().nunique()
        case_issue = "Yes" if lower_unique < unique else "No"
        has_special = series.dropna().str.contains(r"[^a-zA-Z0-9\s]", regex=True).any()
        special_chars = "Yes" if has_special else "No"
    else:
        case_issue = "No"
        special_chars = "No"

    with st.container(border=True):
        st.markdown(f"**{col_name}**")
        left, right = st.columns([3, 2])

        with left:
            fig, truncated, total_unique = make_bar_chart(series)
            st.plotly_chart(fig, use_container_width=True)
            if truncated:
                st.caption(f"Showing top 10 of {total_unique} unique values, sorted by frequency.")

        with right:
            for label, val in [
                ("Missing Values",     missing),
                ("Missing %",          f"{missing_pct:.2f}%"),
                ("Unique Values",      unique),
                ("Case Inconsistency", case_issue),
                ("Special Characters", special_chars),
            ]:
                _stats_row(label, val)


def render_datetime_panel(col_name: str) -> None:
    series = pd.to_datetime(df[col_name], errors="coerce")
    total = len(series)
    missing = int(series.isna().sum())
    missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0

    earliest   = str(series.min().date()) if pd.notna(series.min()) else "—"
    latest     = str(series.max().date()) if pd.notna(series.max()) else "—"
    date_range = (
        f"{(series.max() - series.min()).days:,} days"
        if pd.notna(series.min()) and pd.notna(series.max())
        else "—"
    )

    with st.container(border=True):
        st.markdown(f"**{col_name}**")
        left, right = st.columns([3, 2])

        with left:
            st.plotly_chart(make_histogram_boxplot(series), use_container_width=True)

        with right:
            for label, val in [
                ("Missing Values", missing),
                ("Missing %",      f"{missing_pct:.2f}%"),
                ("Earliest Date",  earliest),
                ("Latest Date",    latest),
                ("Date Range",     date_range),
            ]:
                _stats_row(label, val)


def render_panels(col_list: list, render_fn) -> None:
    """Render panels two per row; last panel is half width when count is odd."""
    for i in range(0, len(col_list), 2):
        pair = col_list[i : i + 2]
        grid = st.columns(2)
        for j, col_name in enumerate(pair):
            with grid[j]:
                render_fn(col_name)


tab1, tab2, tab3 = st.tabs(["Numerical Columns", "Categorical Columns", "Date / Time Columns"])

with tab1:
    if not col_types["numerical"]:
        st.info("No numerical columns found.")
    else:
        render_panels(col_types["numerical"], render_numerical_panel)

with tab2:
    if not col_types["categorical"]:
        st.info("No categorical columns found.")
    else:
        render_panels(col_types["categorical"], render_categorical_panel)

with tab3:
    if not col_types["datetime"]:
        st.info("No date/time columns found.")
    else:
        render_panels(col_types["datetime"], render_datetime_panel)

st.divider()

# ── AI Interpretation ─────────────────────────────────────────────────────────
st.subheader("AI Interpretation")


def build_quality_text(quality_df: pd.DataFrame) -> str:
    if quality_df.empty:
        return ""
    return "\n".join(
        " | ".join(f"{k}: {v}" for k, v in row.items())
        for _, row in quality_df.iterrows()
    )


num_df = check_numerical_quality(df, col_types["numerical"])
cat_df = check_categorical_quality(df, col_types["categorical"])
dt_df  = check_datetime_quality(df, col_types["datetime"])

if "ai_summary" not in st.session_state:
    with st.spinner("Analysing your data…"):
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
