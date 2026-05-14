import pandas as pd
import streamlit as st

from utils.loader import load_dataset, get_dataframe_info
from utils.detector import detect_data_types, convert_to_datetime
from utils.charts import (
    make_histogram_boxplot, make_bar_chart, make_violin,
    make_time_series, make_line_chart, make_heatmap, make_scatter,
    make_density_contour, make_occurrence_heatmap, make_grouped_bar,
    make_boxplot_by_category, DISCRETE_THRESHOLD,
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


def _stats_row(label: str, value) -> None:
    """Theme-aware label: value row."""
    st.markdown(f"**{label}:** {value}")


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
        if stats:
            stat_items = list(stats.items())
            for i in range(0, len(stat_items), 4):
                cols = st.columns(4)
                for j, (label, val) in enumerate(stat_items[i : i + 4]):
                    with cols[j]:
                        st.metric(label=label, value=val)

        st.markdown("**Outlier Detection** — IQR method")
        outlier_summary, _ = detect_outliers(series)
        o1, o2 = st.columns(2)
        o1.metric("Outliers Found", outlier_summary["count"])
        o2.metric("Outlier %",      outlier_summary["pct"])

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

        max_bars = st.slider(
            "Number of bars to display",
            min_value=5,
            max_value=min(50, unique),
            value=min(20, unique),
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

        def highlight_rare_rows(row):
            color = "background-color: #fff3cd" if row["Percentage"] < 1.0 else ""
            return [color] * len(row)

        st.dataframe(
            freq_df.style
                .apply(highlight_rare_rows, axis=1)
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

        freq_options = {"Day": "D", "Week": "W", "Month": "M", "Year": "Y"}
        freq_label = st.radio(
            "Aggregate by", options=list(freq_options.keys()), index=2, horizontal=True,
        )
        st.markdown("**Records Over Time**")
        st.plotly_chart(
            make_time_series(series, freq=freq_options[freq_label], height=360),
            use_container_width=True,
        )

        st.markdown("**Summary Statistics**")
        stats = datetime_stats(series)
        stat_items = list(stats.items())
        for i in range(0, len(stat_items), 4):
            cols = st.columns(4)
            for j, (label, val) in enumerate(stat_items[i : i + 4]):
                with cols[j]:
                    st.metric(label=label, value=val)

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

        # ── Boxplot — Distribution by Category ───────────────────────────────
        if chart_type == "Boxplot — Distribution by Category":
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
