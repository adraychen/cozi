from utils.report_capture import reportable, offer_result
import pandas as pd
import re
import streamlit as st
from utils import presentation


from utils.charts import (
    make_histogram_boxplot, make_bar_chart, make_violin,
    make_time_series, make_line_chart, make_scatter,
    make_density_contour, make_occurrence_heatmap, make_grouped_bar,
    make_boxplot_by_category, make_pivot_heatmap,
    make_confusion_matrix, make_roc_curve,
    make_feature_importance_chart, make_learning_curve_chart,
    make_coefficient_chart,
)
from utils.exploration import (
    numerical_stats, detect_outliers,
    categorical_stats, frequency_table, rare_categories,
    datetime_stats,
)
from utils.ai_interpreter import (
    interpret_dataset_overview,
    interpret_single_column,
    interpret_predictive_results,
)
from utils.quality import (
    leading_trailing_whitespace_pct,
    check_numerical_quality,
    check_categorical_quality,
    check_datetime_quality,
)



def dataset_csv(frame):
    """Export all stored rows and columns, before analytical type conversions."""
    return frame.to_csv(index=False).encode('utf-8-sig')


def render_inspect(df, _show_feedback, dataset_name='Dataset'):
    raw = df
    with st.expander("Dataset Preview", expanded=True):
        # ── Dataset Preview ───────────────────────────────────────────────────────────
        st.caption("Showing the first 100 rows of your data.")

        preview_df = df.head(100).copy()
        bool_cols = preview_df.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            preview_df[bool_cols] = preview_df[bool_cols].astype(str)
        presentation.dataframe(preview_df, use_container_width=True, height=300)
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', dataset_name).strip(' .') or 'Dataset'
        filename = re.sub(r'\.(csv|xlsx|xls)$', '', filename, flags=re.IGNORECASE) + '.csv'
        st.download_button('Download dataset (CSV)', data=dataset_csv(df),
                           file_name=filename, mime='text/csv', key='view_dataset_download',
                           on_click='ignore', help='Download all rows and columns, not only the preview.')
        st.caption('Datasets are kept for this session only. Download a CSV to save a local copy.')

        st.divider()


    with st.expander("Column Overview", expanded=True):
        # ── Column Overview ───────────────────────────────────────────────────────────
        from utils.variable_metadata_ui import render_column_overview

        with st.spinner('Preparing column types and overview…'):
            df, col_types = render_column_overview(df, st.session_state.get("file_id"))
        # Older classification providers expose datetime without the aggregate temporal key.
        col_types.setdefault("temporal", list(dict.fromkeys(
            col_types.get("datetime", []) + col_types.get("time", []) + col_types.get("duration", [])
        )))
        variable_metadata = st.session_state["variable_metadata"]
        st.session_state["col_types"] = col_types  # Compatibility for existing analysis modules.
        col_to_type = {c: m["variable_type"] for c, m in variable_metadata.items()}

        from utils.duplicate_overview import render_duplicate_columns
        render_duplicate_columns(raw, dataset_name)

        st.divider()


    with st.expander("Row Overview", expanded=True):
        from utils.duplicate_overview import render_row_overview
        render_row_overview(raw, dataset_name)

    with st.expander("Data Quality Summary", expanded=True):
        if df.empty:
            st.info("This dataset has no rows to assess.")
            return
        # ── Data Quality Summary ──────────────────────────────────────────────────────


        def render_numerical_panel(col_name: str) -> None:
            series = df[col_name]
            total = len(series)
            missing = int(series.isna().sum())
            missing_pct = round(missing / total * 100, 2) if total > 0 else 0.0
            unique = int(series.nunique())
            is_discrete = variable_metadata[col_name]["variable_subtype"] == "Discrete"
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
                        presentation.plotly_chart(fig, use_container_width=True, key='view_quality_numerical_' + str(col_name))
                        if truncated:
                            st.caption(f"Showing top 10 of {total_unique} unique values.")
                    else:
                        presentation.plotly_chart(make_histogram_boxplot(series), use_container_width=True, key='view_quality_numerical_' + str(col_name))
                with right:
                    statistics = [
                        ("Missing Values", missing),
                        ("Missing %",      f"{missing_pct:.2f}%"),
                        ("Unique Values",  unique),
                        ("Unique %",       f"{round(unique / total * 100, 2):.2f}%"),
                        ("Minimum",        mn),
                        ("Maximum",        mx),
                        ("Average",        avg),
                    ]
                    presentation.dataframe(pd.DataFrame([(label, str(value)) for label, value in statistics], columns=["Statistic", "Value"]), hide_index=True)


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
                    presentation.plotly_chart(fig, use_container_width=True, key='view_quality_categorical_' + str(col_name))
                    if truncated:
                        st.caption(f"Showing top 10 of {total_unique} unique values.")
                with right:
                    statistics = [
                        ("Missing Values",     missing),
                        ("Missing %",          f"{missing_pct:.2f}%"),
                        ("Unique Values",      unique),
                        ("Unique %",           f"{round(unique / total * 100, 2):.2f}%"),
                        ("Case Inconsistency", case_issue),
                        ("Special Characters", special_chars),
                        ("Leading/Trailing Whitespace", f"{leading_trailing_whitespace_pct(raw[col_name]):.2f}%"),
                    ]
                    presentation.dataframe(pd.DataFrame([(label, str(value)) for label, value in statistics], columns=["Statistic", "Value"]), hide_index=True)


        def render_datetime_panel(col_name: str) -> None:
            subtype = variable_metadata[col_name]['variable_subtype']
            if subtype in ('Time', 'Duration', 'Year', 'Month', 'Year-Month'):
                from utils.temporal_ui import render_noncalendar_temporal
                st.markdown(f'**{col_name}**')
                render_noncalendar_temporal(df[col_name], subtype, 'quality_temporal_'+str(col_name))
                return
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
                    presentation.plotly_chart(make_histogram_boxplot(series), use_container_width=True, key='view_quality_temporal_' + str(col_name))
                with right:
                    statistics = [
                        ("Missing Values",         missing),
                        ("Missing %",              f"{missing_pct:.2f}%"),
                        ("Unique Values",          unique),
                        ("Unique %",               f"{unique_pct:.2f}%"),
                        ("Earliest Date",          earliest),
                        ("Latest Date",            latest),
                        ("Date Range",             date_range),
                        ("Future Dates", invalid),
                    ]
                    presentation.dataframe(pd.DataFrame([(label, str(value)) for label, value in statistics], columns=["Statistic", "Value"]), hide_index=True)


        def render_panels(col_list: list, render_fn) -> None:
            for i in range(0, len(col_list), 2):
                pair = col_list[i : i + 2]
                grid = st.columns(2)
                for j, col_name in enumerate(pair):
                    with grid[j]:
                        render_fn(col_name)


        tab1, tab2, tab3 = st.tabs(["Numerical Columns", "Categorical Columns", "Temporal Columns"])
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
            if not col_types["temporal"]:
                st.info("No temporal columns found.")
            else:
                st.caption(
                    "Temporal values that do not match their selected subtype are counted as missing in analyses; original values are preserved. "
                    "For full dates and timestamps, Future Dates counts valid dates after today. "
                    "Year, Month, and Year-Month are summarized at their stated precision."
                )
                render_panels(col_types["temporal"], render_datetime_panel)

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


        has_summary = bool(st.session_state.get('ai_summary'))
        if not has_summary:
            st.caption('Generate an AI interpretation when you are ready. Your dataset is available without waiting for it.')
        if st.button('🔄 Refresh interpretation' if has_summary else 'Generate interpretation', key='view_generate_ai'):
            with st.spinner("Analysing your data…"):
                try:
                    num_df = check_numerical_quality(df, col_types["numerical"])
                    cat_df = check_categorical_quality(df, col_types["categorical"])
                    dt_df = check_datetime_quality(df, col_types["datetime"])
                    st.session_state["ai_summary"] = interpret_dataset_overview(
                        shape=df.shape,
                        col_types=col_types,
                        num_summary=build_quality_text(num_df),
                        cat_summary=build_quality_text(cat_df),
                        dt_summary=build_quality_text(dt_df),
                    )
                except Exception as e:
                    st.error(f"Could not generate AI interpretation: {e}")
                else:
                    st.rerun()

        if st.session_state.get("ai_summary"):
            st.info(st.session_state["ai_summary"])

        _show_feedback("Data Quality Summary", "dqs")

        st.divider()




@reportable
def render_analyze(raw, _show_feedback):
    st.session_state.pop('report_latest', None)
    from utils.variable_metadata import ensure_metadata, effective_data
    variable_metadata = ensure_metadata(st.session_state, raw, st.session_state["file_id"])
    df, col_types = effective_data(raw, variable_metadata)
    st.session_state["col_types"] = col_types
    col_to_type = {c: m["variable_type"] for c, m in variable_metadata.items()}
    # ── Exploratory Data Analysis ─────────────────────────────────────────────────
    st.subheader("Exploratory Data Analysis")

    if st.session_state.get('view_analyze_tabs') == 'Predictive Analytics':
        st.session_state['view_analyze_tabs'] = 'Descriptive Statistics'
    profile_tab, group_tab, visual_tab, relationships_tab, comparison_tab = st.tabs(["Descriptive Statistics", "Group Statistics", "Visual Analysis", "Relationship", "Comparison"], key="view_analyze_tabs", on_change="rerun")

    with profile_tab:
        if profile_tab.open:
            # ── Single Column ─────────────────────────────────────────────────────────────
            st.markdown("#### Descriptive Statistics")
            st.caption("Select one variable to view its statistics, charts, and anomaly detection. Selecting another variable clears the previous selection.")

            from utils.descriptive_controls import select_descriptive_variable
            selected_col = select_descriptive_variable(col_types)
            if selected_col is None:
                st.info("Select one variable above to display its descriptive statistics.")

            if selected_col is not None:
                stats, ai_key = {}, None
                col_type = col_to_type.get(selected_col, "Unknown")
                st.markdown(f"**Column:** {selected_col} &nbsp;|&nbsp; **Type:** {'Temporal' if col_type == 'Date / Time' else col_type}")
                st.divider()

                # ── Numerical ─────────────────────────────────────────────────────────────
                if col_type == "Numerical":
                    series = df[selected_col]
                    unique = int(series.nunique())
                    is_discrete = variable_metadata[selected_col]["variable_subtype"] == "Discrete"

                    chart_l, chart_r = st.columns([3, 2])
                    with chart_l:
                        st.markdown("**Distribution**")
                        if is_discrete:
                            fig, truncated, total_unique = make_bar_chart(series, height=360)
                            presentation.plotly_chart(fig, use_container_width=True)
                            if truncated:
                                st.caption(f"Showing top 10 of {total_unique} unique values.")
                        else:
                            presentation.plotly_chart(make_histogram_boxplot(series, height=360), use_container_width=True)
                        from utils.distribution_fit_ui import render_distribution_fit
                        render_distribution_fit(series, selected_col, variable_metadata[selected_col]["variable_subtype"])
                    with chart_r:
                        st.markdown("**Violin Plot**")
                        presentation.plotly_chart(make_violin(series, height=360), use_container_width=True)

                    st.markdown("**Summary Statistics**")
                    stats = numerical_stats(series)
                    outlier_summary, _ = detect_outliers(series)
                    if stats:
                        all_stats = dict(stats)
                        all_stats["Outliers Found (IQR)"] = outlier_summary["count"]
                        all_stats["Outlier %"]            = outlier_summary["pct"]
                        from utils.descriptive_statistics_display import render_statistics
                        render_statistics(all_stats)
                        st.caption(
                            "Standard error = sample standard deviation / √n. The 95% confidence interval estimates "
                            "the population mean using Student's t distribution; it assumes independent observations "
                            "and approximately normal data or a sufficiently large sample. Missing and infinite values "
                            "are excluded. At least two valid values are required."
                        )
                        st.caption(
                            "Coefficient of variation = sample standard deviation / mean × 100%. "
                            "Shown only for a positive mean; meaningful for ratio-scale data with a true zero "
                            "and unstable when the mean is close to zero."
                        )

                    ai_key = f"col_ai_descriptive_v2_{selected_col}"
                    with st.expander("🤖 AI Interpretation"):
                        if st.button("Refresh interpretation" if st.session_state.get(ai_key) else "Generate interpretation", key="view_ai_generate_" + ai_key):
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
                        key="view_slider_100")

                    st.markdown("**Frequency Chart**")
                    fig, truncated, total_unique = make_bar_chart(series, max_bars=max_bars, height=400)
                    presentation.plotly_chart(fig, use_container_width=True)
                    if truncated:
                        st.caption(f"Showing {max_bars} of {total_unique} categories; ordinal variables follow their configured order, others follow frequency.")

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

                    presentation.dataframe(
                        freq_df.style
                            .map(_style_rare_pct, subset=["Percentage"])
                            .format({"Percentage": "{:.2f}%"}),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("**Mode**")
                    mode_counts = series.value_counts(dropna=True)
                    mode_counts = mode_counts[mode_counts > 0]
                    if mode_counts.empty:
                        st.caption("No mode available: this column has no non-missing values.")
                    else:
                        mode_frequency = int(mode_counts.max())
                        mode_values = mode_counts[mode_counts == mode_frequency].index.tolist()
                        st.text("; ".join(str(value) for value in mode_values))
                        st.caption(
                            f"Frequency: {mode_frequency:,} each. Missing values are excluded."
                            + (" All tied modes are shown." if len(mode_values) > 1 else "")
                        )

                    ai_key = f"col_ai_{selected_col}"
                    stats  = categorical_stats(series)
                    rare_df = rare_categories(series)
                    with st.expander("🤖 AI Interpretation"):
                        if st.button("Refresh interpretation" if st.session_state.get(ai_key) else "Generate interpretation", key="view_ai_generate_" + ai_key):
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

                # ── Date / Time ───────────────────────────────────────────────────────────
                elif col_type == "Date / Time" and variable_metadata[selected_col]['variable_subtype'] in ('Time', 'Duration', 'Year', 'Month', 'Year-Month'):
                    from utils.temporal_ui import render_noncalendar_temporal
                    render_noncalendar_temporal(df[selected_col], variable_metadata[selected_col]['variable_subtype'], 'descriptive_temporal', interactive_summary=True)
                elif col_type == "Date / Time":
                    series = pd.to_datetime(df[selected_col], errors="coerce")

                    chart_l, chart_r = st.columns([3, 2])
                    with chart_l:
                        st.markdown("**Distribution**")
                        presentation.plotly_chart(make_histogram_boxplot(series, height=360), use_container_width=True)
                    with chart_r:
                        st.markdown("**Violin Plot**")
                        presentation.plotly_chart(make_violin(series, height=360), use_container_width=True)

                    st.markdown("**Summary Statistics**")
                    stats = datetime_stats(series)
                    if stats:
                        stats_df = pd.DataFrame({
                            "Statistic": list(stats.keys()),
                            "Value":     [str(v) for v in stats.values()],
                        })
                        presentation.dataframe(stats_df, use_container_width=True, hide_index=True)

                    ai_key = f"col_ai_{selected_col}"
                    with st.expander("🤖 AI Interpretation"):
                        if st.button("Refresh interpretation" if st.session_state.get(ai_key) else "Generate interpretation", key="view_ai_generate_" + ai_key):
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

                if stats:
                    offer_result('Descriptive Statistics', 'Descriptive statistics of '+str(selected_col),
                                 {'variable': selected_col}, variable_metadata[selected_col], stats,
                                 warnings=['These summaries describe the supplied observations. Inference requires an appropriate study design.'],
                                 interpretation=st.session_state.get(ai_key, '') if ai_key else '', key='descriptive')

            st.divider()

    with group_tab:
        if group_tab.open:
            from utils.group_statistics_ui import render_group_statistics
            render_group_statistics(df, col_types)

    with visual_tab:
        if visual_tab.open:
            from utils.visual_analysis import render_visual_analysis
            render_visual_analysis(df, col_types)

    with comparison_tab:
        if comparison_tab.open:
            from utils.comparison_ui import render_comparison
            render_comparison(df, col_types)

    with relationships_tab:
        if relationships_tab.open:
            # ── Multi-Column Analysis ─────────────────────────────────────────────────────
            st.markdown("#### Relationship")
            st.caption("Explore relationships between variables in your data.")

            # ── Grouped checkboxes by variable type ────────────────────────────────────────
            selected_cols = []
            checkbox_groups = [
                ("Numerical",   col_types["numerical"]),
                ("Categorical", col_types["categorical"]),
                ("Temporal", col_types["datetime"]),
            ]
            def set_relationship_group(columns, selected):
                for column in columns:
                    st.session_state[f'mc_{column}'] = selected

            for group_label, group_cols in checkbox_groups:
                if not group_cols:
                    continue
                st.markdown(f"**{group_label}**")
                if group_label in ("Numerical", "Categorical"):
                    select_button, clear_button = st.columns(2)
                    select_button.button("Select all", key=f"mc_all_{group_label}",
                                         on_click=set_relationship_group, args=(group_cols, True))
                    clear_button.button("Clear all", key=f"mc_clear_{group_label}",
                                        on_click=set_relationship_group, args=(group_cols, False))
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

                from utils.relationship_ui import render_relationship_analysis
                render_relationship_analysis(df, sel_num, sel_cat, sel_dt)

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
                if n_cat == 2 and n_num == 0 and n_dt == 0:
                    available += ["Heatmap — Occurrence", "Interaction Chart — Grouped Bar"]

                if not available and n_num < 2:
                    st.info(
                        "No charts available for this combination. Try selecting: "
                        "1 date + 1 or more numerical, 2 or more numerical, or exactly 2 categorical columns."
                    )
                elif available:
                    chart_type = st.radio("Chart type", available, horizontal=True, key="view_radio_287")

                    # ── Heatmap — Numerical Values ────────────────────────────────────────
                    if chart_type == "Heatmap — Numerical Values":
                        presentation.plotly_chart(
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
                        presentation.plotly_chart(fig, use_container_width=True)
                        if truncated:
                            st.caption(
                                f"Showing top 15 of {total_cats} categories, sorted by median descending."
                            )

                    # ── Line Chart — Trend ────────────────────────────────────────────────
                    elif chart_type == "Line Chart — Trend":
                        presentation.plotly_chart(
                            make_line_chart(df, x_col=sel_dt[0], y_cols=sel_num, height=420),
                            use_container_width=True,
                        )

                    # ── Scatter Plot — Correlation ────────────────────────────────────────
                    elif chart_type == "Scatter Plot — Correlation":
                        corr_val = df[sel_num].corr().iloc[0, 1]
                        presentation.plotly_chart(
                            make_scatter(df[sel_num[0]], df[sel_num[1]], sel_num[0], sel_num[1], height=420),
                            use_container_width=True,
                        )
                        st.caption(f"Correlation coefficient (r): **{corr_val:.2f}**")

                    # ── Interaction Chart — Density ───────────────────────────────────────
                    elif chart_type == "Interaction Chart — Density":
                        presentation.plotly_chart(
                            make_density_contour(df[sel_num[0]], df[sel_num[1]], sel_num[0], sel_num[1], height=420),
                            use_container_width=True,
                        )

                    # ── Heatmap — Occurrence ──────────────────────────────────────────────
                    elif chart_type == "Heatmap — Occurrence":
                        presentation.plotly_chart(
                            make_occurrence_heatmap(df, sel_cat[0], sel_cat[1], height=460),
                            use_container_width=True,
                        )
                        st.caption(
                            f"Showing up to 10 most frequent values per column. "
                            f"Rows = **{sel_cat[0]}**, Columns = **{sel_cat[1]}**."
                        )

                    # ── Interaction Chart — Grouped Bar ───────────────────────────────────
                    elif chart_type == "Interaction Chart — Grouped Bar":
                        presentation.plotly_chart(
                            make_grouped_bar(df, sel_cat[0], sel_cat[1], height=420),
                            use_container_width=True,
                        )
                        st.caption(
                            f"Showing up to 10 most frequent values per column. "
                            f"X-axis = **{sel_cat[0]}**, grouped by **{sel_cat[1]}**."
                        )

            elif len(selected_cols) == 1:
                st.info("Select at least one more column to see available charts.")


    _show_feedback("Exploratory Data Analysis", "eda")

    st.divider()
    render_predictive(df, col_types, col_to_type, variable_metadata, _show_feedback)


@reportable
def render_predictive(df, col_types, col_to_type, variable_metadata, _show_feedback):
    st.subheader("Predictive Analytics")

    from utils.modelling_config import (
        detect_target_type, detect_id_columns, estimate_training_time,
        get_model_explanation, interpret_learning_curve, interpret_r2,
        PARAM_GRID, CV_FOLDS, TOTAL_COMBOS, TOTAL_FITS,
        interpret_classifier_f1,
        interpret_learning_curve_classifier,
        interpret_multiclass_f1,
        PARAM_GRID_CLASSIFIER, TOTAL_COMBOS_CLASSIFIER, TOTAL_FITS_CLASSIFIER,
        estimate_training_time_comparison,
        LASSO_PARAM_GRID, LASSO_TOTAL_FITS, COMPARISON_TOTAL_FITS,
        estimate_binary_comparison_time,
        LOGISTIC_PARAM_GRID, LOGISTIC_TOTAL_FITS, BINARY_COMPARISON_TOTAL_FITS,
        estimate_multiclass_comparison_time,
        MULTICLASS_COMPARISON_TOTAL_FITS,
    )

    # Select Target Column
    st.markdown("#### Select Target Column")
    st.caption("The target column is what you want to predict — for example, price, sales, or temperature.")
    from utils.predictive_controls import render_target_selector
    pa_target = render_target_selector(df.columns.tolist(), col_types)
    if pa_target is None:
        st.info('Choose a target column.')
        return
    from utils.predictive_routing import is_temporal_target
    if is_temporal_target(df[pa_target], variable_metadata[pa_target]['variable_type']):
        from utils.temporal_target_guidance import render_temporal_guidance
        render_temporal_guidance(df, pa_target, variable_metadata)
        return

    # Predictor Columns
    st.markdown("#### Predictor Columns")
    st.caption("Predictor columns are the inputs your model uses to make predictions. Choose columns you believe influence the target.")

    pa_pred_num = [c for c in col_types["numerical"]  if c != pa_target]
    pa_pred_cat = [c for c in col_types["categorical"] if c != pa_target]
    pa_pred_dt  = [c for c in col_types["datetime"]    if c != pa_target]
    def set_predictor_group(columns, selected):
        for column in columns:
            st.session_state[f"pa_pred_{column}"] = selected

    pa_predictors = []
    for grp_label, grp_cols in [("Numerical", pa_pred_num), ("Categorical", pa_pred_cat), ("Temporal", pa_pred_dt)]:
        if not grp_cols:
            continue
        st.markdown(f"**{grp_label}**")
        select_button, clear_button = st.columns(2)
        select_button.button('Select all', key=f'pa_select_all_{grp_label}',
                             on_click=set_predictor_group, args=(grp_cols, True))
        clear_button.button('Clear all', key=f'pa_clear_all_{grp_label}',
                            on_click=set_predictor_group, args=(grp_cols, False))
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

        pa_target_type = detect_target_type(df[pa_target], col_to_type.get(pa_target, "Categorical"), variable_metadata[pa_target]["variable_subtype"])
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
            presentation.dataframe(config_df, use_container_width=True, hide_index=True)
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
            presentation.dataframe(config_df, use_container_width=True, hide_index=True)
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
            presentation.dataframe(config_df, use_container_width=True, hide_index=True)
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
            presentation.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)
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
            from utils.modelling import (run_regression_comparison,
                                         run_binary_classification_comparison,
                                         run_multiclass_classification_comparison)
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
        import matplotlib.pyplot as plt
        import shap as _shap
        results    = st.session_state["pa_results"]
        model_type = results.get("_model_type", "")
        clean_names = [n.split("__", 1)[-1] for n in results["feature_names"]]

        # ── Helper: build AI interpretation inputs from results ───────────────────
        def _pa_metrics_text(r):
            wm  = r["winner_metrics"]
            if r["_model_type"] == "regression_comparison":
                gap = round(wm["train_r2"] - wm["test_r2"], 3)
                severity = "good generalisation" if gap <= 0.05 else ("mild overfitting" if gap <= 0.10 else "possible overfitting")
                return (
                    f"Test R²: {wm['test_r2']:.3f}  |  Train R²: {wm['train_r2']:.3f}  |  Gap: {gap} ({severity})\n"
                    f"CV Mean R²: {wm['cv_mean']:.3f} ± {wm['cv_std']:.3f}\n"
                    f"Test RMSE: {wm['test_rmse']:.3f}  |  Test MAE: {wm['test_mae']:.3f}"
                )
            else:
                gap = round(wm["train_f1"] - wm["test_f1"], 3)
                severity = "good generalisation" if gap <= 0.05 else ("mild overfitting" if gap <= 0.10 else "possible overfitting")
                return (
                    f"Test F1: {wm['test_f1']:.3f}  |  Train F1: {wm['train_f1']:.3f}  |  Gap: {gap} ({severity})\n"
                    f"Test Accuracy: {wm['test_accuracy']:.3f}  |  ROC-AUC: {wm['roc_auc']:.3f}\n"
                    f"Precision: {wm['test_precision']:.3f}  |  Recall: {wm['test_recall']:.3f}\n"
                    f"CV Mean F1: {wm['cv_mean']:.3f} ± {wm['cv_std']:.3f}"
                )

        def _pa_features_text(r):
            clean = lambda n: n.split("__", 1)[-1] if "__" in n else n
            if r.get("feat_imp") is not None:
                top = r["feat_imp"].head(10)
                return "\n".join(f"{i+1}. {clean(n)}: {v:.4f}" for i, (n, v) in enumerate(top.items()))
            elif r.get("coefficients") is not None:
                top = r["coefficients"].head(10)
                return "\n".join(f"{i+1}. {clean(n)}: {v:+.4f}" for i, (n, v) in enumerate(top.items()))
            elif r.get("coefficients_per_class") is not None:
                import pandas as _pd2
                all_c = _pd2.DataFrame(r["coefficients_per_class"]).abs().mean(axis=1)
                all_c = all_c.sort_values(ascending=False).head(10)
                return "\n".join(f"{i+1}. {clean(n)}: {v:.4f} (mean across classes)" for i, (n, v) in enumerate(all_c.items()))
            return "Not available"

        def _pa_shap_text(r):
            import numpy as _np2
            clean = lambda n: n.split("__", 1)[-1] if "__" in n else n
            names = [clean(n) for n in r["feature_names"]]
            sv = r["shap_display"] if "shap_display" in r else r.get("shap_values")
            if sv is None:
                return "Not available"
            if isinstance(sv, list):
                mean_abs = _np2.mean([_np2.abs(s).mean(axis=0) for s in sv], axis=0)
            elif hasattr(sv, "ndim") and sv.ndim == 3:
                mean_abs = _np2.abs(sv).mean(axis=(0, 2))
            else:
                mean_abs = _np2.abs(sv).mean(axis=0)
            pairs = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)[:10]
            return "\n".join(f"{i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(pairs))

        _pa_ai_key = f"pa_ai__{st.session_state.get('pa_cache_key', model_type)}"

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

            presentation.dataframe(
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
            presentation.plotly_chart(make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="R² Score"), use_container_width=True)
            st.markdown(interpret_learning_curve(results["train_scores"], results["val_scores"]))
            st.divider()

            if winner == "Random Forest":
                st.markdown("**Feature Importance** — top 20 predictors")
                presentation.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
            else:
                st.markdown("**Coefficient Importance** — top 20 predictors")
                st.caption("Blue = positive effect on target. Orange = negative effect.")
                presentation.plotly_chart(make_coefficient_chart(results["coefficients"]), use_container_width=True)
            st.divider()

            st.markdown(f"**SHAP Summary** — {winner}")
            st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
            plt.figure()
            _shap.summary_plot(results["shap_values"], results["X_sample"], feature_names=clean_names, show=False)
            presentation.pyplot(plt.gcf(), bbox_inches="tight")
            plt.close("all")

            with st.expander("🤖 AI Interpretation"):
                if st.button("Refresh interpretation" if st.session_state.get(_pa_ai_key) else "Generate interpretation", key="view_ai_generate_" + _pa_ai_key):
                    with st.spinner("Generating interpretation…"):
                        try:
                            st.session_state[_pa_ai_key] = interpret_predictive_results(
                                model_type  = results["_model_type"],
                                winner      = results["winner"],
                                target_col  = results.get("target_col", "target"),
                                n_rows      = results["n_train"],
                                metrics_text  = _pa_metrics_text(results),
                                features_text = _pa_features_text(results),
                                shap_text     = _pa_shap_text(results),
                            )
                        except Exception as e:
                            st.error(f"Could not generate interpretation: {e}")
                if _pa_ai_key in st.session_state:
                    st.info(st.session_state[_pa_ai_key])

        # ── Binary comparison ─────────────────────────────────────────────────────
        elif model_type == "binary_comparison":
            st.caption(f"Positive class for precision, recall, F1 and ROC: {results['class_labels'][1]}")
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

            presentation.dataframe(
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
                presentation.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
            with roc_col:
                st.markdown("**ROC Curve**")
                presentation.plotly_chart(make_roc_curve(results["fpr"], results["tpr"], wm["roc_auc"]), use_container_width=True)
            st.divider()

            st.markdown(f"**Learning Curve** — {winner}")
            presentation.plotly_chart(
                make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="F1 Score"),
                use_container_width=True,
            )
            st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
            st.divider()

            if winner == "Random Forest":
                st.markdown("**Feature Importance** — top 20 predictors")
                presentation.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
            else:
                st.markdown("**Coefficient Importance** — top 20 predictors")
                st.caption("Blue = increases probability of positive class. Orange = decreases it.")
                presentation.plotly_chart(make_coefficient_chart(results["coefficients"]), use_container_width=True)
                if results["odds_ratio_df"] is not None:
                    st.markdown("**Odds Ratios**")
                    st.caption(
                        "Odds Ratio > 1: feature increases the likelihood of the positive class. "
                        "Odds Ratio < 1: feature decreases it."
                    )
                    presentation.dataframe(
                        results["odds_ratio_df"].style.format({"Coefficient": "{:.4f}", "Odds Ratio": "{:.4f}"}),
                        use_container_width=True, hide_index=True,
                    )
            st.divider()

            st.markdown(f"**SHAP Summary** — {winner} (positive class)")
            st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
            plt.figure()
            _shap.summary_plot(results["shap_display"], results["X_sample"], feature_names=clean_names, show=False)
            presentation.pyplot(plt.gcf(), bbox_inches="tight")
            plt.close("all")

            with st.expander("🤖 AI Interpretation"):
                if st.button("Refresh interpretation" if st.session_state.get(_pa_ai_key) else "Generate interpretation", key="view_ai_generate_" + _pa_ai_key):
                    with st.spinner("Generating interpretation…"):
                        try:
                            st.session_state[_pa_ai_key] = interpret_predictive_results(
                                model_type    = results["_model_type"],
                                winner        = results["winner"],
                                target_col    = results.get("target_col", "target"),
                                n_rows        = results["n_train"],
                                metrics_text  = _pa_metrics_text(results),
                                features_text = _pa_features_text(results),
                                shap_text     = _pa_shap_text(results),
                            )
                        except Exception as e:
                            st.error(f"Could not generate interpretation: {e}")
                if _pa_ai_key in st.session_state:
                    st.info(st.session_state[_pa_ai_key])

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
                presentation.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
            with roc_col:
                st.markdown("**ROC Curve**")
                presentation.plotly_chart(make_roc_curve(results["fpr"], results["tpr"], results["roc_auc"]), use_container_width=True)
            st.divider()

            st.markdown("**Learning Curve** — training vs validation accuracy")
            presentation.plotly_chart(make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="Accuracy"), use_container_width=True)
            st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
            st.divider()

            st.markdown("**Feature Importance** — top 20 predictors")
            presentation.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
            st.divider()

            st.markdown("**SHAP Summary** — positive class")
            st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
            plt.figure()
            _shap.summary_plot(results["shap_vals_class1"], results["X_sample"], feature_names=clean_names, show=False)
            presentation.pyplot(plt.gcf(), bbox_inches="tight")
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

            presentation.dataframe(
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
            presentation.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
            st.divider()

            st.markdown(f"**Learning Curve** — {winner}")
            presentation.plotly_chart(
                make_learning_curve_chart(results["train_sizes"], results["train_scores"],
                                          results["val_scores"], score_label="Weighted F1"),
                use_container_width=True,
            )
            st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
            st.divider()

            if winner == "Random Forest":
                st.markdown("**Feature Importance** — top 20 predictors")
                presentation.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
            else:
                st.markdown("**Coefficient Importance** — select a class to view its top predictors")
                st.caption("Blue = increases probability of this class. Orange = decreases it.")
                sel_cls_coef = st.selectbox("Select class", options=results["class_labels"], key="pa_coef_class")
                coef_s = results["coefficients_per_class"][sel_cls_coef]
                presentation.plotly_chart(make_coefficient_chart(coef_s), use_container_width=True)
                if results["odds_ratio_per_class"] and sel_cls_coef in results["odds_ratio_per_class"]:
                    st.markdown(f"**Odds Ratios — class: {sel_cls_coef}**")
                    st.caption("Odds Ratio > 1: feature increases the likelihood of this class. Odds Ratio < 1: feature decreases it.")
                    presentation.dataframe(
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
            presentation.pyplot(plt.gcf(), bbox_inches="tight")
            plt.close("all")

            with st.expander("🤖 AI Interpretation"):
                if st.button("Refresh interpretation" if st.session_state.get(_pa_ai_key) else "Generate interpretation", key="view_ai_generate_" + _pa_ai_key):
                    with st.spinner("Generating interpretation…"):
                        try:
                            st.session_state[_pa_ai_key] = interpret_predictive_results(
                                model_type    = results["_model_type"],
                                winner        = results["winner"],
                                target_col    = results.get("target_col", "target"),
                                n_rows        = results["n_train"],
                                metrics_text  = _pa_metrics_text(results),
                                features_text = _pa_features_text(results),
                                shap_text     = _pa_shap_text(results),
                            )
                        except Exception as e:
                            st.error(f"Could not generate interpretation: {e}")
                if _pa_ai_key in st.session_state:
                    st.info(st.session_state[_pa_ai_key])

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
            presentation.plotly_chart(make_confusion_matrix(results["cm"], results["class_labels"]), use_container_width=True)
            st.divider()

            st.markdown("**Learning Curve** — training vs validation accuracy")
            presentation.plotly_chart(make_learning_curve_chart(results["train_sizes"], results["train_scores"], results["val_scores"], score_label="Accuracy"), use_container_width=True)
            st.markdown(interpret_learning_curve_classifier(results["train_scores"], results["val_scores"]))
            st.divider()

            st.markdown("**Feature Importance** — top 20 predictors")
            presentation.plotly_chart(make_feature_importance_chart(results["feat_imp"]), use_container_width=True)
            st.divider()

            st.markdown("**SHAP Summary** — feature impact by class")
            st.caption(f"Based on {min(1000, results['X_sample'].shape[0]):,} test rows.")
            shap_class = st.selectbox("Select class to view", options=results["class_labels"], key="pa_shap_class")
            class_idx  = results["class_labels"].index(shap_class)
            sv         = results["shap_values"]
            shap_cls   = sv[:, :, class_idx] if not isinstance(sv, list) else sv[class_idx]
            plt.figure()
            _shap.summary_plot(shap_cls, results["X_sample"], feature_names=clean_names, show=False)
            presentation.pyplot(plt.gcf(), bbox_inches="tight")
            plt.close("all")

        st.divider()

    if st.session_state.get('pa_results') and pa_predictors:
        offer_result('Predictive Analytics', 'Predictive model for '+str(pa_target),
                     {'target': pa_target, 'predictors': pa_predictors},
                     {'target_type': pa_target_type, 'validation': 'Existing train/test split and cross-validation'},
                     results, None, ['Predictive accuracy is estimated from the reported validation sample; future performance can differ.'],
                     key='predictive')
    _show_feedback("Predictive Analytics", "pa")
