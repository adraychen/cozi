import io
import pandas as pd
import streamlit as st

from utils.database import record_visit, get_stats, submit_feedback

from utils.loader import load_dataset, get_dataframe_info
from utils.detector import detect_data_types, convert_to_datetime
from utils.charts import (
    make_histogram_boxplot, make_bar_chart, make_violin,
    make_time_series, make_line_chart, make_heatmap, make_scatter,
    make_density_contour, make_occurrence_heatmap, make_grouped_bar,
    make_boxplot_by_category, make_pivot_heatmap,
    make_confusion_matrix, make_roc_curve,
    make_feature_importance_chart, make_learning_curve_chart,
    make_coefficient_chart,
    DISCRETE_THRESHOLD,
)
from utils.exploration import (
    numerical_stats, detect_outliers,
    categorical_stats, frequency_table, rare_categories,
    datetime_stats,
)
from utils.ai_interpreter import (
    interpret_dataset_overview,
    interpret_single_column,
    interpret_heatmap,
)
from utils.quality import (
    check_numerical_quality,
    check_categorical_quality,
    check_datetime_quality,
)

# ── Sample datasets ───────────────────────────────────────────────────────────
_SAMPLE_DATASETS = [
    {
        "name": "heart_disease.csv",
        "id":   "1dcRKYhrdo0AlA0AAOIZAPnxWc3XjuTZC",
        "desc": "Binary classification — predict heart disease (303 rows)",
    },
    {
        "name": "BostonHousing.csv",
        "id":   "1rX_UtOz92mVjbJFrB3pv12kjwjROOluv",
        "desc": "Regression — predict housing prices (506 rows)",
    },
    {
        "name": "iris.csv",
        "id":   "1vviVuBIZvizhTTLyldzkD5wQiaFUbDXP",
        "desc": "Multi-class classification — predict iris species (150 rows)",
    },
]


# ── Feedback dialog (must be at module level for @st.dialog) ──────────────────
@st.dialog("Thank you for your feedback")
def _feedback_dialog(section: str, feedback_type: str) -> None:
    comment = st.text_area(
        "Add a comment (optional)",
        placeholder="Tell us what you think…",
        key="dlg_comment",
    )
    c1, c2 = st.columns(2)
    if c1.button("Submit", type="primary", key="dlg_submit"):
        try:
            submit_feedback(section, feedback_type, comment)
        except Exception:
            pass
        st.rerun()
    if c2.button("Skip", key="dlg_skip"):
        try:
            submit_feedback(section, feedback_type, "")
        except Exception:
            pass
        st.rerun()


def _show_feedback(section: str, key: str) -> None:
    """Render 👍 / 👎 buttons that open the feedback dialog."""
    st.divider()
    st.markdown("**Was this section helpful?**")
    fb1, fb2, _ = st.columns([1, 1, 8])
    if fb1.button("👍", key=f"like_{key}", help="Like this section"):
        _feedback_dialog(section, "like")
    if fb2.button("👎", key=f"dislike_{key}", help="Dislike this section"):
        _feedback_dialog(section, "dislike")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Analytics Assistant", page_icon="📊", layout="wide")

# ── App stats banner (above title) ───────────────────────────────────────────
_b1, _b2, _b3 = st.columns(3)
try:
    _app_stats = get_stats()
    _b1.metric("👁 Total Visits", f"{_app_stats['visits']:,}")
    _b2.metric("👍 Likes",        f"{_app_stats['likes']:,}")
    _b3.metric("👎 Dislikes",     f"{_app_stats['dislikes']:,}")
except Exception:
    _b1.metric("👁 Total Visits", "—")
    _b2.metric("👍 Likes",        "—")
    _b3.metric("👎 Dislikes",     "—")


st.title("📊 Data Analytics Assistant")
st.caption(
    "Upload your data file to get started. The assistant will inspect it, "
    "check its quality, explain the findings, and provide insights for you."
)

# ── Visit tracking ────────────────────────────────────────────────────────────
if not st.session_state.get("visit_recorded"):
    try:
        record_visit()
        st.session_state["visit_recorded"] = True
    except Exception:
        pass
st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
st.subheader("Upload Your Data File")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV (.csv) and Excel (.xlsx, .xls)",
)

# ── Sample datasets (below uploader) ─────────────────────────────────────────
with st.expander("📂 Try with sample data", expanded=False):
    st.caption("Click a dataset below to load it instantly — no upload needed.")
    for _ds in _SAMPLE_DATASETS:
        if st.button(
            f"**{_ds['name']}** — {_ds['desc']}", key=f"sample_{_ds['name']}"
        ):
            with st.spinner(f"Loading {_ds['name']}…"):
                try:
                    import requests as _req
                    _resp = _req.get(
                        f"https://drive.google.com/uc?export=download&id={_ds['id']}",
                        timeout=30,
                    )
                    _resp.raise_for_status()
                    _sample_df  = pd.read_csv(io.StringIO(_resp.text))
                    _sample_fid = f"sample__{_ds['name']}"
                    st.session_state["df"]      = _sample_df
                    st.session_state["file_id"] = _sample_fid
                    for _k in ("col_types", "ai_summary", "datetime_overrides",
                               "override_warnings", "pa_results", "pa_cache_key",
                               "pa_train_error"):
                        st.session_state.pop(_k, None)
                    for _k in list(st.session_state.keys()):
                        if any(_k.startswith(_p) for _p in
                               ("col_ai_", "heatmap_ai_", "pa_pred_", "mc_")):
                            st.session_state.pop(_k, None)
                    st.rerun()
                except Exception as _e:
                    st.error(f"Could not load {_ds['name']}: {_e}")

# Stop only when no file uploaded AND no sample data loaded
if uploaded_file is None and "df" not in st.session_state:
    st.stop()

try:
    if uploaded_file is not None:
        _file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("file_id") != _file_id:
            df = load_dataset(uploaded_file)
            st.session_state["df"]      = df
            st.session_state["file_id"] = _file_id
            for _k in ("col_types", "ai_summary", "datetime_overrides",
                       "override_warnings", "pa_results", "pa_cache_key",
                       "pa_train_error"):
                st.session_state.pop(_k, None)
            for _k in list(st.session_state.keys()):
                if any(_k.startswith(_p) for _p in
                       ("col_ai_", "heatmap_ai_", "pa_pred_", "mc_")):
                    st.session_state.pop(_k, None)
        df = st.session_state["df"]
    else:
        # Sample data already loaded into session_state
        df = st.session_state["df"]
except ValueError as e:
    st.error(str(e))
    st.stop()

_display_name = (
    uploaded_file.name if uploaded_file is not None
    else st.session_state.get("file_id", "sample").replace("sample__", "")
)

st.success(f"File loaded: **{_display_name}**")
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
    "Each column's data type, category, total values, and missing values. "
    "Red = 20%+ missing. Yellow = 5–19% missing."
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
    info_df.style
        .map(highlight_missing_pct, subset=["Missing %"])
        .format({"Missing %": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

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
    st.markdown(
        f"<div style='padding:3px 0'>"
        f"<span style='color:#aaaaaa'><b>{label}:</b></span> "
        f"<span style='color:#ffffff'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


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
                    st.caption(f"Showing top 10 of {total_unique} unique values.")
            else:
                st.plotly_chart(make_histogram_boxplot(series), use_container_width=True)
        with right:
            for label, val in [
                ("Missing Values", missing),
                ("Missing %",      f"{missing_pct:.2f}%"),
                ("Unique Values",  unique),
                ("Unique %",       f"{round(unique / total * 100, 2):.2f}%"),
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
        case_issue    = "Yes" if lower_unique < unique else "No"
        has_special   = series.dropna().str.contains(r"[^a-zA-Z0-9\s]", regex=True).any()
        special_chars = "Yes" if has_special else "No"
    else:
        case_issue    = "No"
        special_chars = "No"

    with st.container(border=True):
        st.markdown(f"**{col_name}**")
        left, right = st.columns([3, 2])
        with left:
            fig, truncated, total_unique = make_bar_chart(series)
            st.plotly_chart(fig, use_container_width=True)
            if truncated:
                st.caption(f"Showing top 10 of {total_unique} unique values.")
        with right:
            for label, val in [
                ("Missing Values",     missing),
                ("Missing %",          f"{missing_pct:.2f}%"),
                ("Unique Values",      unique),
                ("Unique %",           f"{round(unique / total * 100, 2):.2f}%"),
                ("Case Inconsistency", case_issue),
                ("Special Characters", special_chars),
            ]:
                _stats_row(label, val)


def render_datetime_panel(col_name: str) -> None:
    series      = pd.to_datetime(df[col_name], errors="coerce")
    clean       = series.dropna()
    total       = len(series)
    missing     = int(series.isna().sum())
    missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0
    unique      = int(clean.nunique())
    unique_pct  = round(unique / total * 100, 2) if total > 0 else 0.0
    invalid     = int((clean > pd.Timestamp.now()).sum())
    earliest    = str(clean.min().date()) if not clean.empty else "—"
    latest      = str(clean.max().date()) if not clean.empty else "—"
    date_range  = (
        f"{(clean.max() - clean.min()).days:,} days" if not clean.empty else "—"
    )

    with st.container(border=True):
        st.markdown(f"**{col_name}**")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(make_histogram_boxplot(series), use_container_width=True)
        with right:
            for label, val in [
                ("Missing Values",         missing),
                ("Missing %",              f"{missing_pct:.2f}%"),
                ("Unique Values",          unique),
                ("Unique %",               f"{unique_pct:.2f}%"),
                ("Earliest Date",          earliest),
                ("Latest Date",            latest),
                ("Date Range",             date_range),
                ("Future Dates", invalid),
            ]:
                _stats_row(label, val)


def render_panels(col_list: list, render_fn) -> None:
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
        st.caption(
            "ℹ️ Dates that cannot be parsed — such as text values, impossible dates "
            "(e.g. April 31), or unrecognised formats — are automatically converted to NaT "
            "and counted under **Missing Values**. The **Future Dates** field shows how many "
            "valid dates fall after today."
        )
        render_panels(col_types["datetime"], render_datetime_panel)

st.divider()

# ── AI Interpretation ─────────────────────────────────────────────────────────
st.subheader("AI Interpretation")

num_df = check_numerical_quality(df, col_types["numerical"])
cat_df = check_categorical_quality(df, col_types["categorical"])
dt_df  = check_datetime_quality(df, col_types["datetime"])


def build_quality_text(quality_df: pd.DataFrame) -> str:
    if quality_df.empty:
        return ""
    return "\n".join(
        " | ".join(f"{k}: {v}" for k, v in row.items())
        for _, row in quality_df.iterrows()
    )


if "ai_summary" not in st.session_state:
    with st.spinner("Analysing your data…"):
        try:
            st.session_state["ai_summary"] = interpret_dataset_overview(
                shape=df.shape,
                col_types=col_types,
                num_summary=build_quality_text(num_df),
                cat_summary=build_quality_text(cat_df),
                dt_summary=build_quality_text(dt_df),
            )
        except Exception as e:
            st.error(f"Could not generate AI interpretation: {e}")
            st.session_state["ai_summary"] = None

if st.session_state.get("ai_summary"):
    st.info(st.session_state["ai_summary"])
    if st.button("🔄 Refresh interpretation"):
        st.session_state.pop("ai_summary", None)
        st.rerun()

_show_feedback("Data Quality Summary", "dqs")

st.divider()

# ── Data Exploratory Analysis ─────────────────────────────────────────────────
st.subheader("Data Exploratory Analysis")

# ── Single Column ─────────────────────────────────────────────────────────────
st.markdown("#### Column Profile")
st.caption("Select a column for a detailed deep-dive: full statistics, charts, and anomaly detection.")

selected_col = st.selectbox("Select a column", options=df.columns.tolist(), index=0)

if selected_col:
    col_type = col_to_type.get(selected_col, "Unknown")
    st.markdown(f"**Column:** {selected_col} &nbsp;|&nbsp; **Type:** {col_type}")
    st.divider()

    # ── Numerical ─────────────────────────────────────────────────────────────
    if col_type == "Numerical":
        series = df[selected_col]
        unique = int(series.nunique())
        is_discrete = unique < DISCRETE_THRESHOLD

        chart_l, chart_r = st.columns([3, 2])
        with chart_l:
            st.markdown("**Distribution**")
            if is_discrete:
                fig, truncated, total_unique = make_bar_chart(series, height=360)
                st.plotly_chart(fig, use_container_width=True)
                if truncated:
                    st.caption(f"Showing top 10 of {total_unique} unique values.")
            else:
                st.plotly_chart(make_histogram_boxplot(series, height=360), use_container_width=True)
        with chart_r:
            st.markdown("**Violin Plot**")
            st.plotly_chart(make_violin(series, height=360), use_container_width=True)

        st.markdown("**Summary Statistics**")
        stats = numerical_stats(series)
        outlier_summary, _ = detect_outliers(series)
        if stats:
            all_stats = dict(stats)
            all_stats["Outliers Found (IQR)"] = outlier_summary["count"]
            all_stats["Outlier %"]            = outlier_summary["pct"]
            stats_df = pd.DataFrame({
                "Statistic": list(all_stats.keys()),
                "Value":     list(all_stats.values()),
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        ai_key = f"col_ai_{selected_col}"
        with st.expander("🤖 AI Interpretation"):
            if ai_key not in st.session_state:
                with st.spinner("Generating interpretation…"):
                    try:
                        extra = {
                            "Outliers found":    outlier_summary["count"],
                            "Outlier %":         outlier_summary["pct"],
                            "Lower bound (IQR)": outlier_summary["lower_bound"] or "—",
                            "Upper bound (IQR)": outlier_summary["upper_bound"] or "—",
                        }
                        st.session_state[ai_key] = interpret_single_column(
                            col_name=selected_col, col_type=col_type,
                            stats=stats, extra=extra,
                        )
                    except Exception as e:
                        st.session_state[ai_key] = None
                        st.error(f"Could not generate interpretation: {e}")
            if st.session_state.get(ai_key):
                st.info(st.session_state[ai_key])
                if st.button("🔄 Refresh", key=f"refresh_{ai_key}"):
                    st.session_state.pop(ai_key, None)
                    st.rerun()

    # ── Categorical ───────────────────────────────────────────────────────────
    elif col_type == "Categorical":
        series = df[selected_col]
        unique = int(series.nunique())

        if unique <= 5:
            max_bars = unique
        else:
            _slider_max = min(50, unique)
            max_bars = st.slider(
                "Number of bars to display",
                min_value=5,
                max_value=_slider_max,
                value=min(20, _slider_max),
                step=5,
                help="Controls how many category values are shown in the bar chart.",
            )

        st.markdown("**Frequency Chart**")
        fig, truncated, total_unique = make_bar_chart(series, max_bars=max_bars, height=400)
        st.plotly_chart(fig, use_container_width=True)
        if truncated:
            st.caption(f"Showing top {max_bars} of {total_unique} unique values, sorted by frequency.")

        st.markdown("**Frequency Table**")
        st.caption("Rows highlighted in yellow appear in less than 1% of records.")
        freq_df = frequency_table(series)

        def _style_rare_pct(val):
            try:
                if float(val) < 1.0:
                    return "background-color: #fff3cd; color: #333333"
            except (ValueError, TypeError):
                pass
            return ""

        st.dataframe(
            freq_df.style
                .map(_style_rare_pct, subset=["Percentage"])
                .format({"Percentage": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )

        ai_key = f"col_ai_{selected_col}"
        stats  = categorical_stats(series)
        rare_df = rare_categories(series)
        with st.expander("🤖 AI Interpretation"):
            if ai_key not in st.session_state:
                with st.spinner("Generating interpretation…"):
                    try:
                        extra = {"Rare categories (< 1%)": len(rare_df) if not rare_df.empty else 0}
                        st.session_state[ai_key] = interpret_single_column(
                            col_name=selected_col, col_type=col_type,
                            stats=stats, extra=extra,
                        )
                    except Exception as e:
                        st.session_state[ai_key] = None
                        st.error(f"Could not generate interpretation: {e}")
            if st.session_state.get(ai_key):
                st.info(st.session_state[ai_key])
                if st.button("🔄 Refresh", key=f"refresh_{ai_key}"):
                    st.session_state.pop(ai_key, None)
                    st.rerun()

    # ── Date / Time ───────────────────────────────────────────────────────────
    elif col_type == "Date / Time":
        series = pd.to_datetime(df[selected_col], errors="coerce")

        chart_l, chart_r = st.columns([3, 2])
        with chart_l:
            st.markdown("**Distribution**")
            st.plotly_chart(make_histogram_boxplot(series, height=360), use_container_width=True)
        with chart_r:
            st.markdown("**Violin Plot**")
            st.plotly_chart(make_violin(series, height=360), use_container_width=True)

        st.markdown("**Summary Statistics**")
        stats = datetime_stats(series)
        if stats:
            stats_df = pd.DataFrame({
                "Statistic": list(stats.keys()),
                "Value":     [str(v) for v in stats.values()],
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        ai_key = f"col_ai_{selected_col}"
        with st.expander("🤖 AI Interpretation"):
            if ai_key not in st.session_state:
                with st.spinner("Generating interpretation…"):
                    try:
                        st.session_state[ai_key] = interpret_single_column(
                            col_name=selected_col, col_type=col_type, stats=stats,
                        )
                    except Exception as e:
                        st.session_state[ai_key] = None
                        st.error(f"Could not generate interpretation: {e}")
            if st.session_state.get(ai_key):
                st.info(st.session_state[ai_key])
                if st.button("🔄 Refresh", key=f"refresh_{ai_key}"):
                    st.session_state.pop(ai_key, None)
                    st.rerun()

st.divider()

# ── Multi-Column Analysis ─────────────────────────────────────────────────────
st.markdown("#### Column Relationships")
st.caption("Select two or more columns using the checkboxes below. Available chart types will appear based on your selection.")

# ── Grouped checkboxes by column type ────────────────────────────────────────
selected_cols = []
checkbox_groups = [
    ("Numerical",   col_types["numerical"]),
    ("Categorical", col_types["categorical"]),
    ("Date / Time", col_types["datetime"]),
]
for group_label, group_cols in checkbox_groups:
    if not group_cols:
        continue
    st.markdown(f"**{group_label}**")
    n_per_row = 4
    for i in range(0, len(group_cols), n_per_row):
        row_cols = st.columns(n_per_row)
        for j, col_name in enumerate(group_cols[i : i + n_per_row]):
            if row_cols[j].checkbox(col_name, key=f"mc_{col_name}"):
                selected_cols.append(col_name)

if len(selected_cols) >= 2:
    sel_num = [c for c in selected_cols if col_to_type.get(c) == "Numerical"]
    sel_cat = [c for c in selected_cols if col_to_type.get(c) == "Categorical"]
    sel_dt  = [c for c in selected_cols if col_to_type.get(c) == "Date / Time"]
    n_num, n_cat, n_dt = len(sel_num), len(sel_cat), len(sel_dt)

    # Determine available charts
    available = []
    if n_cat == 2 and n_num == 1 and n_dt == 0:
        available.append("Heatmap — Numerical Values")
    if n_num == 1 and n_cat == 1 and n_dt == 0:
        available.append("Boxplot — Distribution by Category")
    if n_dt == 1 and n_num >= 1 and n_cat == 0:
        available.append("Line Chart — Trend")
    if n_num == 2 and n_cat == 0 and n_dt == 0:
        available += ["Scatter Plot — Correlation", "Interaction Chart — Density"]
    if n_num >= 2 and n_cat == 0 and n_dt == 0:
        available.append("Heatmap — Multiple Correlations")
    if n_cat == 2 and n_num == 0 and n_dt == 0:
        available += ["Heatmap — Occurrence", "Interaction Chart — Grouped Bar"]

    if not available:
        st.info(
            "No charts available for this combination. Try selecting: "
            "1 date + 1 or more numerical, 2 or more numerical, or exactly 2 categorical columns."
        )
    else:
        chart_type = st.radio("Chart type", available, horizontal=True)

        # ── Heatmap — Numerical Values ────────────────────────────────────────
        if chart_type == "Heatmap — Numerical Values":
            st.plotly_chart(
                make_pivot_heatmap(df, sel_cat[0], sel_cat[1], sel_num[0], height=460),
                use_container_width=True,
            )
            st.caption(
                f"Average of **{sel_num[0]}** for each combination of "
                f"**{sel_cat[0]}** (rows) and **{sel_cat[1]}** (columns). "
                f"Showing up to 10 most frequent values per column."
            )

        # ── Boxplot — Distribution by Category ───────────────────────────────
        elif chart_type == "Boxplot — Distribution by Category":
            fig, truncated, total_cats = make_boxplot_by_category(
                df, num_col=sel_num[0], cat_col=sel_cat[0], height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
            if truncated:
                st.caption(
                    f"Showing top 15 of {total_cats} categories, sorted by median descending."
                )

        # ── Line Chart — Trend ────────────────────────────────────────────────
        elif chart_type == "Line Chart — Trend":
            st.plotly_chart(
                make_line_chart(df, x_col=sel_dt[0], y_cols=sel_num, height=420),
                use_container_width=True,
            )

        # ── Scatter Plot — Correlation ────────────────────────────────────────
        elif chart_type == "Scatter Plot — Correlation":
            corr_val = df[sel_num].corr().iloc[0, 1]
            st.plotly_chart(
                make_scatter(df[sel_num[0]], df[sel_num[1]], sel_num[0], sel_num[1], height=420),
                use_container_width=True,
            )
            st.caption(f"Correlation coefficient (r): **{corr_val:.2f}**")

        # ── Interaction Chart — Density ───────────────────────────────────────
        elif chart_type == "Interaction Chart — Density":
            st.plotly_chart(
                make_density_contour(df[sel_num[0]], df[sel_num[1]], sel_num[0], sel_num[1], height=420),
                use_container_width=True,
            )

        # ── Heatmap — Multiple Correlations ──────────────────────────────────
        elif chart_type == "Heatmap — Multiple Correlations":
            corr_df = df[sel_num].corr()
            st.plotly_chart(make_heatmap(corr_df), use_container_width=True)

            pairs = [
                f"{corr_df.columns[i]} × {corr_df.columns[j]}: {corr_df.iloc[i, j]:.2f}"
                for i in range(len(corr_df.columns))
                for j in range(i + 1, len(corr_df.columns))
            ]
            hm_key = f"heatmap_ai_{'_'.join(sorted(sel_num))}"
            with st.expander("🤖 AI Interpretation"):
                if hm_key not in st.session_state:
                    with st.spinner("Generating interpretation…"):
                        try:
                            st.session_state[hm_key] = interpret_heatmap(
                                corr_pairs="\n".join(pairs),
                                col_names=sel_num,
                            )
                        except Exception as e:
                            st.session_state[hm_key] = None
                            st.error(f"Could not generate interpretation: {e}")
                if st.session_state.get(hm_key):
                    st.info(st.session_state[hm_key])
                    if st.button("🔄 Refresh", key="refresh_heatmap"):
                        st.session_state.pop(hm_key, None)
                        st.rerun()

        # ── Heatmap — Occurrence ──────────────────────────────────────────────
        elif chart_type == "Heatmap — Occurrence":
            st.plotly_chart(
                make_occurrence_heatmap(df, sel_cat[0], sel_cat[1], height=460),
                use_container_width=True,
            )
            st.caption(
                f"Showing up to 10 most frequent values per column. "
                f"Rows = **{sel_cat[0]}**, Columns = **{sel_cat[1]}**."
            )

        # ── Interaction Chart — Grouped Bar ───────────────────────────────────
        elif chart_type == "Interaction Chart — Grouped Bar":
            st.plotly_chart(
                make_grouped_bar(df, sel_cat[0], sel_cat[1], height=420),
                use_container_width=True,
            )
            st.caption(
                f"Showing up to 10 most frequent values per column. "
                f"X-axis = **{sel_cat[0]}**, grouped by **{sel_cat[1]}**."
            )

elif len(selected_cols) == 1:
    st.info("Select at least one more column to see available charts.")

_show_feedback("Data Exploratory Analysis", "eda")

# ── Predictive Analytics ──────────────────────────────────────────────────────
st.divider()
st.subheader("Predictive Analytics")

from utils.modelling import (
    detect_target_type, detect_id_columns, estimate_training_time,
    get_model_explanation, interpret_learning_curve, interpret_r2,
    run_random_forest_regressor, PARAM_GRID, CV_FOLDS, TOTAL_COMBOS, TOTAL_FITS,
    run_random_forest_classifier, interpret_classifier_f1,
    interpret_learning_curve_classifier,
    run_random_forest_multiclass_classifier, interpret_multiclass_f1,
    PARAM_GRID_CLASSIFIER, TOTAL_COMBOS_CLASSIFIER, TOTAL_FITS_CLASSIFIER,
    run_regression_comparison, estimate_training_time_comparison,
    LASSO_PARAM_GRID, LASSO_TOTAL_FITS, COMPARISON_TOTAL_FITS,
    run_binary_classification_comparison, estimate_binary_comparison_time,
    LOGISTIC_PARAM_GRID, LOGISTIC_TOTAL_FITS, BINARY_COMPARISON_TOTAL_FITS,
    run_multiclass_classification_comparison, estimate_multiclass_comparison_time,
    MULTICLASS_COMPARISON_TOTAL_FITS,
)
import matplotlib.pyplot as plt
import shap as _shap

# Select Target Column
st.markdown("#### Select Target Column")
st.caption("The target column is what you want to predict — for example, price, sales, or temperature.")
pa_target = st.selectbox("Select target column", options=df.columns.tolist(), key="pa_target_select")

# Predictor Columns
st.markdown("#### Predictor Columns")
st.caption("Predictor columns are the inputs your model uses to make predictions. Choose columns you believe influence the target.")

pa_pred_num = [c for c in col_types["numerical"]  if c != pa_target]
pa_pred_cat = [c for c in col_types["categorical"] if c != pa_target]
pa_pred_dt  = [c for c in col_types["datetime"]    if c != pa_target]
_all_available = pa_pred_num + pa_pred_cat + pa_pred_dt

_sa1, _sa2 = st.columns(2)
if _sa1.button("✅ Select All", key="pa_select_all"):
    for _c in _all_available:
        st.session_state[f"pa_pred_{_c}"] = True
    st.rerun()
if _sa2.button("☐ Deselect All", key="pa_deselect_all"):
    for _c in _all_available:
        st.session_state[f"pa_pred_{_c}"] = False
    st.rerun()

pa_predictors = []
for grp_label, grp_cols in [("Numerical", pa_pred_num), ("Categorical", pa_pred_cat), ("Date / Time", pa_pred_dt)]:
    if not grp_cols:
        continue
    st.markdown(f"**{grp_label}**")
    for i in range(0, len(grp_cols), 4):
        row = st.columns(4)
        for j, cname in enumerate(grp_cols[i : i + 4]):
            if row[j].checkbox(cname, key=f"pa_pred_{cname}"):
                pa_predictors.append(cname)

if pa_predictors:
    for cname, reason in detect_id_columns(df, pa_predictors):
        st.warning(f"⚠️ **{cname}** may be an identifier ({reason}). Including it could reduce model accuracy — consider deselecting it.")

# Model
if pa_predictors:
    st.markdown("#### Model")

    pa_target_type = detect_target_type(df[pa_target], col_to_type.get(pa_target, "Categorical"))
    st.markdown(get_model_explanation(pa_target_type))
    st.divider()

    # Hyperparameter configuration table
    is_regression_comparison = (pa_target_type == "continuous_numerical")
    is_binary_comparison     = (pa_target_type == "binary")
    is_multiclass_comparison = (pa_target_type in ("discrete_numerical", "multiclass"))
    is_regression            = (pa_target_type in ("continuous_numerical", "discrete_numerical"))

    if is_regression_comparison:
        config_df = pd.DataFrame([
            {"Model": "Lasso",         "Parameter": "Alpha (regularisation)", "Values Tested": "0.001, 0.01, 0.1, 1, 10", "Total Fits": str(LASSO_TOTAL_FITS)},
            {"Model": "Random Forest", "Parameter": "Max Tree Depth",          "Values Tested": "10, 15",     "Total Fits": ""},
            {"Model": "Random Forest", "Parameter": "Min Samples to Split",    "Values Tested": "5, 10",      "Total Fits": ""},
            {"Model": "Random Forest", "Parameter": "Min Samples per Leaf",    "Values Tested": "3, 5",       "Total Fits": str(TOTAL_FITS)},
            {"Model": "Combined",      "Parameter": "Cross-Validation Folds",  "Values Tested": str(CV_FOLDS),"Total Fits": str(COMPARISON_TOTAL_FITS)},
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        time_est = estimate_training_time_comparison(len(df))
    elif is_binary_comparison:
        config_df = pd.DataFrame([
            {"Model": "Logistic Regression", "Parameter": "Penalty",             "Values Tested": "l1, l2",              "Total Fits": ""},
            {"Model": "Logistic Regression", "Parameter": "C (regularisation)",  "Values Tested": "0.01, 0.1, 1, 10",   "Total Fits": str(LOGISTIC_TOTAL_FITS)},
            {"Model": "Random Forest",        "Parameter": "Max Tree Depth",      "Values Tested": "10, 15",              "Total Fits": ""},
            {"Model": "Random Forest",        "Parameter": "Min Samples to Split","Values Tested": "5, 10",               "Total Fits": ""},
            {"Model": "Random Forest",        "Parameter": "Min Samples per Leaf","Values Tested": "3, 5",               "Total Fits": str(TOTAL_FITS_CLASSIFIER)},
            {"Model": "Combined",             "Parameter": "CV Folds",            "Values Tested": str(CV_FOLDS),         "Total Fits": str(BINARY_COMPARISON_TOTAL_FITS)},
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        time_est = estimate_binary_comparison_time(len(df))
    elif is_multiclass_comparison:
        config_df = pd.DataFrame([
            {"Model": "Multinomial LR", "Parameter": "Penalty",             "Values Tested": "l1, l2",            "Total Fits": ""},
            {"Model": "Multinomial LR", "Parameter": "C (regularisation)",  "Values Tested": "0.01, 0.1, 1, 10", "Total Fits": str(LOGISTIC_TOTAL_FITS)},
            {"Model": "Random Forest",  "Parameter": "Max Tree Depth",      "Values Tested": "10, 15",            "Total Fits": ""},
            {"Model": "Random Forest",  "Parameter": "Min Samples to Split","Values Tested": "5, 10",             "Total Fits": ""},
            {"Model": "Random Forest",  "Parameter": "Min Samples per Leaf","Values Tested": "3, 5",             "Total Fits": str(TOTAL_FITS_CLASSIFIER)},
            {"Model": "Combined",       "Parameter": "CV Folds",            "Values Tested": str(CV_FOLDS),       "Total Fits": str(MULTICLASS_COMPARISON_TOTAL_FITS)},
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        time_est = estimate_multiclass_comparison_time(len(df))
    else:
        _combos = TOTAL_COMBOS if is_regression else TOTAL_COMBOS_CLASSIFIER
        _fits   = TOTAL_FITS   if is_regression else TOTAL_FITS_CLASSIFIER
        param_rows = [
            {"Parameter": "Number of Trees",     "Values Tested": "300",    "What it controls": "How many decision trees are built"},
            {"Parameter": "Max Tree Depth",       "Values Tested": "10, 15", "What it controls": "How deep each tree can grow — limits overfitting"},
            {"Parameter": "Min Samples to Split", "Values Tested": "5, 10",  "What it controls": "Minimum samples a node needs before splitting"},
            {"Parameter": "Min Samples per Leaf", "Values Tested": "3, 5",   "What it controls": "Minimum samples required at each leaf node"},
        ]
        if not is_regression:
            param_rows.append({"Parameter": "Max Features", "Values Tested": "sqrt, log2", "What it controls": "Features considered at each split"})
        param_rows += [
            {"Parameter": "CV Folds",           "Values Tested": str(CV_FOLDS), "What it controls": "Validation rounds per combination"},
            {"Parameter": "Total Combinations", "Values Tested": str(_combos),  "What it controls": "—"},
            {"Parameter": "Total Fits",         "Values Tested": str(_fits),    "What it controls": "—"},
        ]
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)
        time_est = estimate_training_time(len(df))

    st.info(f"⏱ **Estimated training time: {time_est}** based on {len(df):,} rows.")

    # Invalidate cache when selection changes
    pa_cache_key = f"{pa_target}__{'|'.join(sorted(pa_predictors))}"
    if st.session_state.get("pa_cache_key") != pa_cache_key:
        st.session_state.pop("pa_results", None)
        st.session_state["pa_cache_key"] = pa_cache_key

    # Run button
    run_btn = st.button("▶ Run", type="primary", key="pa_run_btn")

    def _train_model():
        if pa_target_type == "continuous_numerical":
            return run_regression_comparison(df, pa_target, pa_predictors)
        elif pa_target_type == "binary":
            return run_binary_classification_comparison(df, pa_target, pa_predictors)
        else:
            return run_multiclass_classification_comparison(df, pa_target, pa_predictors)

    if run_btn:
        st.session_state.pop("pa_train_error", None)
        with st.spinner(f"Training — estimated {time_est}. Please wait…"):
            try:
                st.session_state["pa_results"] = _train_model()
            except Exception as e:
                st.session_state.pop("pa_results", None)
                st.session_state["pa_train_error"] = str(e)
        if not st.session_state.get("pa_train_error"):
            st.rerun()

    if st.session_state.get("pa_train_error"):
        st.error(f"Training failed: {st.session_state['pa_train_error']}")


st.divider()

# ── Results (shown first when available) ──────────────────────────────────────
if st.session_state.get("pa_results"):
    results    = st.session_state["pa_results"]
    model_type = results.get("_model_type", "")
    clean_names = [n.split("__", 1)[-1] for n in results["feature_names"]]

    st.markdown("#### Results")
    st.caption(f"Trained on {results['n_train']:,} rows · Tested on {results['n_test']:,} rows")

    # ── Regression comparison ─────────────────────────────────────────────────
    if model_type == "regression_comparison":
        winner = results["winner"]
        st.success(f"🏆 **{winner}** performs better on this dataset (higher Test R²).")

        cdf = results["comparison_df"].copy()
        higher_better = {"Train R²", "Test R²", "CV Mean R²"}
        lower_better  = {"Train MAE", "Test MAE", "Train RMSE", "Test RMSE"}

        def _style_cmp(row):
            styles = [""] * len(row)
            metric = row["Metric"]
            lv, rv = row["Lasso"], row["Random Forest"]
            if metric in higher_better:
                win_col = "Lasso" if lv >= rv else "Random Forest"
            elif metric in lower_better:
                win_col = "Lasso" if lv <= rv else "Random Forest"
            else:
                return styles
            styles[cdf.columns.tolist().index(win_col)] = (
                "background-color: #d4edda; color: #155724; font-weight: bold"
            )
            return styles

        st.dataframe(
            cdf.style.apply(_style_cmp, axis=1).format({"Lasso": "{:.3f}", "Random Forest": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )
        st.divider()

        wm = results["winner_metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test R²",    f"{wm['test_r2']:.3f}")
        mc2.metric("Test RMSE",  f"{wm['test_rmse']:.3f}")
        mc3.metric("Test MAE",   f"{wm['test_mae']:.3f}")
        mc4.metric("CV Mean R²", f"{wm['cv_mean']:.3f} ± {wm['cv_std']:.3f}")
        st.markdown(interpret_r2(wm["test_r2"]))
        st.divider()

        st.markdown(f"**Learning Curve** — {winner}")
        st.plotly_chart(make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="R² Score"), use_container_width=True)
        st.markdown(interpret_learning_curve(results["train_scores"], results["val_scores"]))
        st.divider()

        if winner == "Random Forest":
            st.markdown("**Feature Importance** — top 20 predictors")
            st.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
        else:
            st.markdown("**Coefficient Importance** — top 20 predictors")
            st.caption("Blue = positive effect on target. Orange = negative effect.")
            st.plotly_chart(make_coefficient_chart(results["coefficients"]), use_container_width=True)
        st.divider()

        st.markdown(f"**SHAP Summary** — {winner}")
        st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
        plt.figure()
        _shap.summary_plot(results["shap_values"], results["X_sample"], feature_names=clean_names, show=False)
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.close("all")

    # ── Binary comparison ─────────────────────────────────────────────────────
    elif model_type == "binary_comparison":
        winner = results["winner"]
        st.success(f"🏆 **{winner}** performs better on this dataset (higher Test F1 Score).")

        cdf = results["comparison_df"].copy()

        def _style_binary_cmp(row):
            styles = [""] * len(row)
            lv, rv = row["Logistic Regression"], row["Random Forest"]
            win_col = "Logistic Regression" if lv >= rv else "Random Forest"
            styles[cdf.columns.tolist().index(win_col)] = (
                "background-color: #d4edda; color: #155724; font-weight: bold"
            )
            return styles

        st.dataframe(
            cdf.style.apply(_style_binary_cmp, axis=1)
               .format({"Logistic Regression": "{:.3f}", "Random Forest": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )
        st.divider()

        wm = results["winner_metrics"]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Test F1",       f"{wm['test_f1']:.3f}")
        mc2.metric("Test Accuracy", f"{wm['test_accuracy']:.3f}")
        mc3.metric("Precision",     f"{wm['test_precision']:.3f}")
        mc4.metric("Recall",        f"{wm['test_recall']:.3f}")
        mc5.metric("ROC-AUC",       f"{wm['roc_auc']:.3f}")
        st.markdown(interpret_classifier_f1(wm["test_f1"]))
        st.divider()

        cm_col, roc_col = st.columns(2)
        with cm_col:
            st.markdown("**Confusion Matrix**")
            st.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
        with roc_col:
            st.markdown("**ROC Curve**")
            st.plotly_chart(make_roc_curve(results["fpr"], results["tpr"], wm["roc_auc"]), use_container_width=True)
        st.divider()

        st.markdown(f"**Learning Curve** — {winner}")
        st.plotly_chart(
            make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="F1 Score"),
            use_container_width=True,
        )
        st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
        st.divider()

        if winner == "Random Forest":
            st.markdown("**Feature Importance** — top 20 predictors")
            st.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
        else:
            st.markdown("**Coefficient Importance** — top 20 predictors")
            st.caption("Blue = increases probability of positive class. Orange = decreases it.")
            st.plotly_chart(make_coefficient_chart(results["coefficients"]), use_container_width=True)
            if results["odds_ratio_df"] is not None:
                st.markdown("**Odds Ratios**")
                st.caption(
                    "Odds Ratio > 1: feature increases the likelihood of the positive class. "
                    "Odds Ratio < 1: feature decreases it."
                )
                st.dataframe(
                    results["odds_ratio_df"].style.format({"Coefficient": "{:.4f}", "Odds Ratio": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )
        st.divider()

        st.markdown(f"**SHAP Summary** — {winner} (positive class)")
        st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
        plt.figure()
        _shap.summary_plot(results["shap_display"], results["X_sample"], feature_names=clean_names, show=False)
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.close("all")

    # ── Binary classifier ─────────────────────────────────────────────────────
    elif model_type == "binary_classifier":
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy",  f"{results['accuracy']:.3f}")
        mc2.metric("F1 Score",  f"{results['f1']:.3f}")
        mc3.metric("Precision", f"{results['precision']:.3f}")
        mc4.metric("Recall",    f"{results['recall']:.3f}")
        mc5.metric("ROC-AUC",   f"{results['roc_auc']:.3f}")
        st.markdown(interpret_classifier_f1(results["f1"]))
        st.divider()

        cm_col, roc_col = st.columns(2)
        with cm_col:
            st.markdown("**Confusion Matrix**")
            st.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
        with roc_col:
            st.markdown("**ROC Curve**")
            st.plotly_chart(make_roc_curve(results["fpr"], results["tpr"], results["roc_auc"]), use_container_width=True)
        st.divider()

        st.markdown("**Learning Curve** — training vs validation accuracy")
        st.plotly_chart(make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="Accuracy"), use_container_width=True)
        st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
        st.divider()

        st.markdown("**Feature Importance** — top 20 predictors")
        st.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
        st.divider()

        st.markdown("**SHAP Summary** — positive class")
        st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
        plt.figure()
        _shap.summary_plot(results["shap_vals_class1"], results["X_sample"], feature_names=clean_names, show=False)
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.close("all")

    # ── Multiclass comparison ─────────────────────────────────────────────────
    elif model_type == "multiclass_comparison":
        winner = results["winner"]
        st.success(f"🏆 **{winner}** performs better on this dataset (higher weighted Test F1 Score).")

        cdf = results["comparison_df"].copy()
        _lr_col  = "Multinomial Logistic Regression"
        _rf_col  = "Random Forest"

        def _style_mc_cmp(row):
            styles  = [""] * len(row)
            lv, rv  = row[_lr_col], row[_rf_col]
            win_col = _lr_col if lv >= rv else _rf_col
            styles[cdf.columns.tolist().index(win_col)] = (
                "background-color: #d4edda; color: #155724; font-weight: bold"
            )
            return styles

        st.dataframe(
            cdf.style.apply(_style_mc_cmp, axis=1)
               .format({_lr_col: "{:.3f}", _rf_col: "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )
        st.divider()

        wm = results["winner_metrics"]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Test F1 (weighted)",        f"{wm['test_f1']:.3f}")
        mc2.metric("Test Accuracy",             f"{wm['test_accuracy']:.3f}")
        mc3.metric("Precision (weighted)",      f"{wm['test_precision']:.3f}")
        mc4.metric("Recall (weighted)",         f"{wm['test_recall']:.3f}")
        mc5.metric("Weighted ROC-AUC",          f"{wm['roc_auc']:.3f}")
        st.markdown(interpret_multiclass_f1(wm["test_f1"]))
        st.divider()

        st.markdown("**Confusion Matrix**")
        st.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
        st.divider()

        st.markdown(f"**Learning Curve** — {winner}")
        st.plotly_chart(
            make_learning_curve_chart(results["train_sizes"], results["train_scores"],
                                      results["val_scores"], score_label="Weighted F1"),
            use_container_width=True,
        )
        st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
        st.divider()

        if winner == "Random Forest":
            st.markdown("**Feature Importance** — top 20 predictors")
            st.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
        else:
            st.markdown("**Coefficient Importance** — select a class to view its top predictors")
            st.caption("Blue = increases probability of this class. Orange = decreases it.")
            sel_cls_coef = st.selectbox("Select class", options=results["class_labels"], key="pa_coef_class")
            coef_s = results["coefficients_per_class"][sel_cls_coef]
            st.plotly_chart(make_coefficient_chart(coef_s), use_container_width=True)
            if results["odds_ratio_per_class"] and sel_cls_coef in results["odds_ratio_per_class"]:
                st.markdown(f"**Odds Ratios — class: {sel_cls_coef}**")
                st.caption("Odds Ratio > 1: feature increases the likelihood of this class. Odds Ratio < 1: feature decreases it.")
                st.dataframe(
                    results["odds_ratio_per_class"][sel_cls_coef]
                       .style.format({"Coefficient": "{:.4f}", "Odds Ratio": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )
        st.divider()

        st.markdown(f"**SHAP Summary** — {winner} (select a class)")
        st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
        sel_cls_shap = st.selectbox("Select class for SHAP", options=results["class_labels"], key="pa_shap_mc_class")
        class_idx    = results["class_labels"].index(sel_cls_shap)
        sv           = results["shap_values"]
        if isinstance(sv, list):
            shap_cls = sv[class_idx]
        else:
            shap_cls = sv[:, :, class_idx]
        plt.figure()
        _shap.summary_plot(shap_cls, results["X_sample"], feature_names=clean_names, show=False)
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.close("all")

    # ── Multi-class classifier ────────────────────────────────────────────────
    elif model_type == "multiclass_classifier":
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy",             f"{results['accuracy']:.3f}")
        mc2.metric("F1 (weighted)",         f"{results['f1']:.3f}")
        mc3.metric("Precision (weighted)", f"{results['precision']:.3f}")
        mc4.metric("Recall (weighted)",    f"{results['recall']:.3f}")
        mc5.metric("ROC-AUC (OvR)",        f"{results['roc_auc']:.3f}")
        st.markdown(interpret_multiclass_f1(results["f1"]))
        st.divider()

        st.markdown("**Confusion Matrix**")
        st.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
        st.divider()

        st.markdown("**Learning Curve** — training vs validation accuracy")
        st.plotly_chart(make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="Accuracy"), use_container_width=True)
        st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
        st.divider()

        st.markdown("**Feature Importance** — top 20 predictors")
        st.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
        st.divider()

        st.markdown("**SHAP Summary** — feature impact by class")
        st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
        shap_class = st.selectbox("Select class to view", options=results["class_labels"], key="pa_shap_class")
        class_idx  = results["class_labels"].index(shap_class)
        sv         = results["shap_values"]
        shap_cls   = sv[:, :, class_idx] if not isinstance(sv, list) else sv[class_idx]
        plt.figure()
        _shap.summary_plot(shap_cls, results["X_sample"], feature_names=clean_names, show=False)
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.close("all")

    st.divider()

_show_feedback("Predictive Analytics", "pa")
