import pandas as pd
import streamlit as st

from utils.loader import load_dataset, get_dataframe_info
from utils.detector import detect_data_types, convert_to_datetime
from utils.charts import (
    make_histogram_boxplot, make_bar_chart, make_violin,
    make_time_series, make_line_chart, make_heatmap, make_scatter,
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Analytics App", page_icon="📊", layout="wide")

st.title("📊 Data Analytics App")
st.caption(
    "Upload your data file to get started. The app will inspect it, "
    "check its quality, and explain the findings in plain English."
)
st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
st.subheader("Upload Your Data File")

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
        # Clear all per-column AI caches
        for k in list(st.session_state.keys()):
            if k.startswith("col_ai_") or k.startswith("heatmap_ai_"):
                st.session_state.pop(k, None)
    else:
        df = st.session_state["df"]
except ValueError as e:
    st.error(str(e))
    st.stop()

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

LABEL_COLOR = "#666666"
VALUE_COLOR = "#111111"


def _stats_row(label: str, value) -> None:
    st.markdown(
        f"<div style='padding:3px 0'>"
        f"<b style='color:{LABEL_COLOR}'>{label}:</b> "
        f"<span style='color:{VALUE_COLOR}'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _stat_card(label: str, value) -> None:
    st.markdown(
        f"<div style='padding:8px;background:#f9f9f9;border-radius:6px;margin:4px 0'>"
        f"<div style='font-size:12px;color:{LABEL_COLOR}'>{label}</div>"
        f"<div style='font-size:16px;font-weight:600;color:{VALUE_COLOR}'>{value}</div>"
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
        if pd.notna(series.min()) and pd.notna(series.max()) else "—"
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

# ── AI Interpretation (dataset overview) ─────────────────────────────────────
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

st.divider()

# ── Data Exploratory Analysis ─────────────────────────────────────────────────
st.subheader("Data Exploratory Analysis")


# ── Single column deep dive ───────────────────────────────────────────────────
st.markdown("#### Single Column")
st.caption("Select a column for a detailed deep-dive: full statistics, charts, and outlier or anomaly detection.")

all_cols = df.columns.tolist()
selected_col = st.selectbox("Select a column", options=all_cols, index=0)

if selected_col:
    col_type = col_to_type.get(selected_col, "Unknown")
    st.markdown(
        f"<span style='color:{LABEL_COLOR}'>Column:</span> "
        f"<span style='color:{VALUE_COLOR}'><b>{selected_col}</b></span> &nbsp;|&nbsp; "
        f"<span style='color:{LABEL_COLOR}'>Type:</span> "
        f"<span style='color:{VALUE_COLOR}'><b>{col_type}</b></span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Numerical ─────────────────────────────────────────────────────────────
    if col_type == "Numerical":
        series = df[selected_col]
        unique = int(series.nunique())
        is_discrete = unique < DISCRETE_THRESHOLD

        chart_l, chart_r = st.columns([3, 2])
        with chart_l:
            st.markdown(f"<span style='color:{LABEL_COLOR}'>Distribution</span>", unsafe_allow_html=True)
            if is_discrete:
                fig, truncated, total_unique = make_bar_chart(series, height=360)
                st.plotly_chart(fig, use_container_width=True)
                if truncated:
                    st.caption(f"Showing top 10 of {total_unique} unique values.")
            else:
                st.plotly_chart(make_histogram_boxplot(series, height=360), use_container_width=True)
        with chart_r:
            st.markdown(f"<span style='color:{LABEL_COLOR}'>Violin Plot</span>", unsafe_allow_html=True)
            st.plotly_chart(make_violin(series, height=360), use_container_width=True)

        st.markdown(f"<b style='color:{LABEL_COLOR}'>Summary Statistics</b>", unsafe_allow_html=True)
        stats = numerical_stats(series)
        if stats:
            stat_items = list(stats.items())
            for i in range(0, len(stat_items), 4):
                cols = st.columns(4)
                for j, (label, val) in enumerate(stat_items[i : i + 4]):
                    with cols[j]:
                        _stat_card(label, val)

        st.markdown(f"<b style='color:{LABEL_COLOR}'>Outlier Detection</b> — IQR method", unsafe_allow_html=True)
        outlier_summary, outlier_vals = detect_outliers(series)
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Outliers Found",  outlier_summary["count"])
        o2.metric("Outlier %",       outlier_summary["pct"])
        o3.metric("Lower Bound",     outlier_summary["lower_bound"] or "—")
        o4.metric("Upper Bound",     outlier_summary["upper_bound"] or "—")
        if outlier_summary["count"] > 0:
            with st.expander(f"View {outlier_summary['count']} outlier values"):
                st.dataframe(
                    outlier_vals.reset_index(drop=True).rename("Value").to_frame(),
                    use_container_width=True, hide_index=True,
                )

        # AI interpretation
        ai_key = f"col_ai_{selected_col}"
        with st.expander("🤖 AI Interpretation"):
            if ai_key not in st.session_state:
                with st.spinner("Generating interpretation…"):
                    try:
                        extra = {
                            "Outliers found": outlier_summary["count"],
                            "Outlier %": outlier_summary["pct"],
                            "Lower bound (IQR)": outlier_summary["lower_bound"] or "—",
                            "Upper bound (IQR)": outlier_summary["upper_bound"] or "—",
                        }
                        st.session_state[ai_key] = interpret_single_column(
                            col_name=selected_col,
                            col_type=col_type,
                            stats=stats,
                            extra=extra,
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

        chart_l, chart_r = st.columns([3, 2])
        with chart_l:
            st.markdown(f"<span style='color:{LABEL_COLOR}'>Frequency Chart</span>", unsafe_allow_html=True)
            fig, truncated, total_unique = make_bar_chart(series, max_bars=10, height=360)
            st.plotly_chart(fig, use_container_width=True)
            if truncated:
                st.caption(f"Showing top 10 of {total_unique} unique values.")
        with chart_r:
            st.markdown(f"<span style='color:{LABEL_COLOR}'>Summary Statistics</span>", unsafe_allow_html=True)
            stats = categorical_stats(series)
            for label, val in stats.items():
                _stats_row(label, val)

        st.markdown(f"<b style='color:{LABEL_COLOR}'>Frequency Table</b>", unsafe_allow_html=True)
        freq_df = frequency_table(series)
        st.dataframe(
            freq_df.style.format({"Percentage": "{:.2f}%"}),
            use_container_width=True, hide_index=True,
        )

        rare_df = rare_categories(series)
        if not rare_df.empty:
            st.warning(
                f"⚠️ {len(rare_df)} rare {'category' if len(rare_df) == 1 else 'categories'} found "
                f"(appearing in less than 1% of rows). These may be data entry errors or outliers."
            )
            with st.expander("View rare categories"):
                st.dataframe(
                    rare_df.style.format({"Percentage": "{:.2f}%"}),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.success("No rare categories found (all values appear in 1% or more of rows).")

        # AI interpretation
        ai_key = f"col_ai_{selected_col}"
        with st.expander("🤖 AI Interpretation"):
            if ai_key not in st.session_state:
                with st.spinner("Generating interpretation…"):
                    try:
                        extra = {
                            "Rare categories (< 1%)": len(rare_df) if not rare_df.empty else 0,
                        }
                        st.session_state[ai_key] = interpret_single_column(
                            col_name=selected_col,
                            col_type=col_type,
                            stats=stats,
                            extra=extra,
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

        freq_options = {"Day": "D", "Week": "W", "Month": "M", "Year": "Y"}
        freq_label = st.radio(
            "Aggregate by", options=list(freq_options.keys()), index=2, horizontal=True,
        )
        st.markdown(f"<span style='color:{LABEL_COLOR}'>Records Over Time</span>", unsafe_allow_html=True)
        st.plotly_chart(
            make_time_series(series, freq=freq_options[freq_label], height=360),
            use_container_width=True,
        )

        st.markdown(f"<b style='color:{LABEL_COLOR}'>Summary Statistics</b>", unsafe_allow_html=True)
        stats = datetime_stats(series)
        stat_items = list(stats.items())
        for i in range(0, len(stat_items), 4):
            cols = st.columns(4)
            for j, (label, val) in enumerate(stat_items[i : i + 4]):
                with cols[j]:
                    _stat_card(label, val)

        # AI interpretation
        ai_key = f"col_ai_{selected_col}"
        with st.expander("🤖 AI Interpretation"):
            if ai_key not in st.session_state:
                with st.spinner("Generating interpretation…"):
                    try:
                        st.session_state[ai_key] = interpret_single_column(
                            col_name=selected_col,
                            col_type=col_type,
                            stats=stats,
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

# ── Multi-column analysis ─────────────────────────────────────────────────────
st.markdown("#### Multi-Column Analysis")
st.caption("Compare and explore relationships across multiple columns.")

num_cols    = col_types["numerical"]
dt_cols     = col_types["datetime"]
all_x_cols  = dt_cols + num_cols   # valid x-axis candidates for line chart

chart_type = st.radio(
    "Chart type",
    ["Line Chart — Trend", "Correlation Heatmap", "Scatter Plot"],
    horizontal=True,
)

# ── Line Chart ────────────────────────────────────────────────────────────────
if chart_type == "Line Chart — Trend":
    if not num_cols:
        st.info("Line chart requires at least one numerical column.")
    else:
        x_options = all_x_cols if all_x_cols else None
        if x_options:
            x_col = st.selectbox("X-axis (time or numerical)", options=x_options)
        else:
            x_col = None
            st.caption("No datetime or numerical column found for x-axis. Using row index.")

        y_cols = st.multiselect(
            "Y-axis columns (select one or more numerical columns)",
            options=[c for c in num_cols if c != x_col],
        )

        if y_cols:
            plot_df = df.copy()
            if x_col is None:
                plot_df["Row Index"] = range(len(plot_df))
                x_col_plot = "Row Index"
            else:
                x_col_plot = x_col

            st.plotly_chart(
                make_line_chart(plot_df, x_col=x_col_plot, y_cols=y_cols, height=420),
                use_container_width=True,
            )
        else:
            st.info("Select at least one Y-axis column to plot.")

# ── Correlation Heatmap ───────────────────────────────────────────────────────
elif chart_type == "Correlation Heatmap":
    if len(num_cols) < 2:
        st.info("Correlation heatmap requires at least two numerical columns.")
    else:
        selected_hm_cols = st.multiselect(
            "Select numerical columns (minimum 2)",
            options=num_cols,
            default=num_cols[:min(6, len(num_cols))],
        )

        if len(selected_hm_cols) < 2:
            st.info("Select at least 2 columns to generate the heatmap.")
        else:
            corr_df = df[selected_hm_cols].corr()
            st.plotly_chart(make_heatmap(corr_df), use_container_width=True)

            # Build upper-triangle pairs for AI
            pairs = []
            cols_list = corr_df.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    pairs.append(
                        f"{cols_list[i]} × {cols_list[j]}: {corr_df.iloc[i, j]:.2f}"
                    )
            corr_text = "\n".join(pairs)

            # AI interpretation
            hm_key = f"heatmap_ai_{'_'.join(sorted(selected_hm_cols))}"
            with st.expander("🤖 AI Interpretation"):
                if hm_key not in st.session_state:
                    with st.spinner("Generating interpretation…"):
                        try:
                            st.session_state[hm_key] = interpret_heatmap(
                                corr_pairs=corr_text,
                                col_names=selected_hm_cols,
                            )
                        except Exception as e:
                            st.session_state[hm_key] = None
                            st.error(f"Could not generate interpretation: {e}")
                if st.session_state.get(hm_key):
                    st.info(st.session_state[hm_key])
                    if st.button("🔄 Refresh", key="refresh_heatmap"):
                        st.session_state.pop(hm_key, None)
                        st.rerun()

# ── Scatter Plot ──────────────────────────────────────────────────────────────
elif chart_type == "Scatter Plot":
    if len(num_cols) < 2:
        st.info("Scatter plot requires at least two numerical columns.")
    else:
        sc1, sc2 = st.columns(2)
        with sc1:
            x_scatter = st.selectbox("X-axis column", options=num_cols, index=0)
        with sc2:
            y_options = [c for c in num_cols if c != x_scatter]
            y_scatter = st.selectbox("Y-axis column", options=y_options, index=0)

        if x_scatter and y_scatter:
            corr_val = df[[x_scatter, y_scatter]].corr().iloc[0, 1]
            st.plotly_chart(
                make_scatter(df[x_scatter], df[y_scatter], x_scatter, y_scatter, height=420),
                use_container_width=True,
            )
            st.caption(f"Correlation coefficient (r): **{corr_val:.2f}**")
