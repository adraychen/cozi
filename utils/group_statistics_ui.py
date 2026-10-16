from utils.numerical_formatting import format_plot_hover
from utils.report_capture import reportable, offer_result
"""Streamlit controls and results for grouped exploratory analysis."""
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import presentation
from utils.variable_selection import variable_checkboxes
from utils.group_statistics import STATISTICS, FREQUENCIES, summary, compare, contribution, temporal
from utils.temporal import temporal_columns, period_frequencies, period_values

ANALYSES = ["Group Summary", "Segment Comparison", "Period-over-Period Comparison",
            "Running & Cumulative Statistics", "Rolling Averages", "Contribution to Total"]


GROUP_ANALYSES = ["Group Summary", "Running & Cumulative Statistics", "Rolling Averages", "Contribution to Total"]


@reportable
def render_group_statistics(df, col_types, analysis=None, key_prefix="gs", show_heading=True):
    temporal_candidates = temporal_columns(col_types)
    if show_heading:
        presentation.markdown("#### Group Statistics")
    if key_prefix == 'gs':
        presentation.caption('View summary statistics for your data by group.')
        measures = variable_checkboxes('Numerical', col_types['numerical'], 'gs_measure', single=True, hide_empty=True)
        measure = measures[0] if measures else None
        dimensions = variable_checkboxes('Categorical', col_types['categorical'], 'gs_dimensions', hide_empty=True)
        dates = variable_checkboxes('Temporal', temporal_candidates, 'gs_temporal', single=True, hide_empty=True)
    else:
        measure = st.selectbox("Numerical column", [None] + col_types["numerical"],
                               format_func=lambda value: "None (categorical data / row counts)" if value is None else value,
                               key=key_prefix + "_measure_optional")
        dimensions = st.multiselect("Categorical columns", col_types["categorical"], key=key_prefix + "_dimensions",
                                    help="Select one or more dimensions. Leave empty for the whole dataset.")
    if analysis is None:
        if st.session_state.get(key_prefix + "_analysis") not in GROUP_ANALYSES:
            st.session_state.pop(key_prefix + "_analysis", None)
        analysis = st.selectbox("Analysis type", GROUP_ANALYSES, key=key_prefix + "_analysis")
    time_based = analysis in ANALYSES[2:5]
    allowed = ["Row count"] if measure is None else (["Sum", "Non-missing count", "Row count"] if analysis == ANALYSES[5] else list(STATISTICS) + (["Period-end value"] if analysis == ANALYSES[2] else []))
    statistic = st.selectbox("Statistic", allowed, key=key_prefix + "_stat_" + analysis + ("_counts" if measure is None else "_numeric"))
    presentation.caption("Missing categories are included. Row count counts all rows, including missing measures. Non-missing count counts valid numerical values. Standard deviation uses the sample definition.")
    try:
        if time_based:
            if not temporal_candidates:
                presentation.info("Select a Temporal override in Column Overview to enable time-based analysis.")
                return
            if key_prefix == 'gs':
                if not dates:
                    presentation.info('Select one Date or DateTime variable, or a Year-Month or Year variable, above for time-based analysis.')
                    return
                date = dates[0]
            else:
                date = st.selectbox("Date column", temporal_candidates, key=key_prefix + "_date")
            if analysis == ANALYSES[2]:
                from utils.period_comparison_ui import render_period_comparison
                render_period_comparison(df, measure, dimensions, date, statistic, key_prefix)
                return
            from utils.temporal_validation_ui import guide_temporal_selection
            allowed, _, _ = guide_temporal_selection(df, date, st.session_state.get('variable_metadata', {}),
                {'tab': 'Group Statistics', 'numerical': [measure] if measure is not None else [],
                 'categorical': dimensions, 'analysis': analysis}, 'temporal_guidance_' + key_prefix + '_' + str(date))
            if not allowed:
                return
            subtype = st.session_state.get('variable_metadata', {}).get(date, {}).get('variable_subtype')
            frequencies = period_frequencies(subtype)
            if not frequencies:
                presentation.info('Period analysis requires Date, DateTime, Year-Month, or Year values.')
                return
            if subtype == 'Year':
                presentation.caption('Year values support yearly comparisons only.')
            if subtype == 'Year-Month':
                presentation.caption('Year-Month values represent whole months; daily and weekly comparisons are unavailable.')
            frequency_key = key_prefix + '_frequency'
            if st.session_state.get(frequency_key) not in frequencies:
                st.session_state.pop(frequency_key, None)
            frequency = st.selectbox("Period frequency", frequencies, index=frequencies.index('Monthly') if 'Monthly' in frequencies else 0, key=frequency_key)
            zero = st.checkbox("Treat missing period values as zero", key=key_prefix + "_zero",
                               help="Off by default. Enable only when missing values mean no activity.")
            lag, window = 1, 3
            if analysis == ANALYSES[2]:
                options = ["Previous period"]
                if frequency in ("Monthly", "Quarterly", "Yearly"):
                    options.append("Same period last year")
                mode = st.selectbox("Compare with", options, key=key_prefix + "_comparison_" + frequency)
                lag = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}.get(frequency, 1) if mode == "Same period last year" else 1
            if analysis == ANALYSES[4]:
                window = int(st.number_input("Window size (calendar periods)", min_value=2, max_value=365, value=3, key=key_prefix + "_window"))
                presentation.caption("A rolling average requires a complete window of non-missing period values.")
            result, keys = temporal(df, measure, dimensions, date, frequency, statistic, analysis, window, lag, zero, subtype)
            observed_periods = period_values(df[date], frequency, subtype)
            invalid = observed_periods.isna().sum()
            if invalid:
                presentation.caption(f"Excluded {invalid:,} rows with missing or invalid dates.")
            presentation.caption("Weekly periods run Monday–Sunday. Running and rolling statistics operate on the selected period statistic; expanding mean weights periods equally. Missing periods remain gaps unless zero filling is selected.")
            latest = observed_periods.max()
            current = pd.Timestamp.now().to_period(FREQUENCIES[frequency])
            if latest >= current:
                presentation.warning("The data includes the current or a future period. The current period may be incomplete.")
            presentation.caption("First and last periods may also be partial depending on dataset coverage.")
            plotted = ["Value", "Previous"] if analysis == ANALYSES[2] else (["Value", "Rolling average"] if analysis == ANALYSES[4] else [st.selectbox("Running statistic", ["Cumulative sum", "Expanding mean", "Observed periods"], key=key_prefix + "_running")])
        else:
            result, keys = summary(df, measure, dimensions, statistic)
            if result.empty:
                presentation.info("No data available.")
                return
            if analysis == ANALYSES[1]:
                baseline = st.selectbox("Reference segment", list(result.index),
                    format_func=lambda i: "; ".join(f"{dimensions[j] if dimensions else 'Dataset'}: {result.loc[i, key]}" for j, key in enumerate(keys)), key=key_prefix + "_reference")
                result = compare(result, result.loc[baseline, "Value"])
                presentation.caption("Percentage difference uses the absolute reference value; zero or missing references show N/A.")
            if analysis == ANALYSES[5]:
                parent = st.checkbox("Share within parent group", key=key_prefix + "_parent", disabled=len(keys) < 2)
                result = contribution(result, keys, parent)
                presentation.caption("Denominator: " + ("each combination of dimensions except the last." if parent and len(keys) > 1 else "the whole dataset."))
                if (result["Value"] < 0).any():
                    presentation.warning("Negative values produce signed contributions, which can fall outside 0–100%.")
            plotted = ["Contribution %"] if analysis == ANALYSES[5] else ["Value"]
        result["Segment"] = result[keys].apply(lambda row: "; ".join(f"{dimensions[i] if dimensions else 'Dataset'}: {v}" for i, v in enumerate(row)), axis=1)
        labels = result["Segment"].unique().tolist()
        chosen = st.multiselect("Segments to chart", labels, default=labels[:10], key=key_prefix + "_chart_" + analysis)
        chart = result[result["Segment"].isin(chosen)]
        chart_kind = {'Mean': 'mean', 'Median': 'median', 'Minimum': 'minimum', 'Maximum': 'maximum', 'Standard deviation': 'sd', 'Row count': 'count', 'Non-missing count': 'count'}.get(statistic)
        if analysis == ANALYSES[5]:
            chart_kind = 'percentage'
        if not chart.empty:
            if time_based:
                long = chart.melt(id_vars=["Period", "Segment"], value_vars=plotted, var_name="Statistic", value_name="Result")
                fig = px.line(long, x="Period", y="Result", color="Segment", line_dash="Statistic")
            else:
                fig = px.bar(chart, x="Segment", y=plotted[0], color="Segment")
            if chart_kind:
                format_plot_hover(fig, {'y': chart_kind})
            presentation.plotly_chart(fig, use_container_width=True, key=key_prefix + "_chart")
        display = result.drop(columns=keys)
        schema = {'Contribution %': 'percentage', 'Difference %': 'percentage', 'Change %': 'percentage', 'Observed periods': 'count'}
        value_type = {'Mean': 'mean', 'Median': 'median', 'Minimum': 'minimum', 'Maximum': 'maximum', 'Standard deviation': 'sd', 'Variance': 'variance', 'Row count': 'count', 'Non-missing count': 'count'}.get(statistic)
        if value_type:
            schema.update({'Value': value_type, 'Previous': value_type, 'Reference': value_type})
        schema.update({'Expanding mean': 'mean', 'Rolling average': 'mean'})
        presentation.dataframe(display.style.format(precision=2, na_rep="N/A"), result_types=schema, use_container_width=True)
        st.download_button("Download statistics (CSV)", display.to_csv(index=False).encode("utf-8-sig"),
                           file_name="comparison.csv" if key_prefix != "gs" else "group_statistics.csv", mime="text/csv", key=key_prefix + "_download")
        offer_result(analysis, analysis, {'measure': measure, 'dimensions': dimensions},
                     {'statistic': statistic, 'time_based': time_based}, {'Summary rows': len(result)},
                     warnings=['Descriptive group and period comparisons do not establish causation. Missing values and zero denominators follow the configured analysis rules.'], key=key_prefix)
    except (ValueError, TypeError) as exc:
        presentation.warning(f"Unable to calculate these statistics: {exc}")
