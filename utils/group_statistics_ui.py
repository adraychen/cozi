"""Streamlit controls and results for grouped exploratory analysis."""
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.group_statistics import STATISTICS, FREQUENCIES, summary, compare, contribution, temporal

ANALYSES = ["Group Summary", "Segment Comparison", "Period-over-Period Comparison",
            "Running & Cumulative Statistics", "Rolling Averages", "Contribution to Total"]


def render_group_statistics(df, col_types):
    st.markdown("#### Group Statistics")
    if not col_types["numerical"]:
        st.info("Load a dataset with a numerical column to use Group Statistics.")
        return
    measure = st.selectbox("Numerical column", col_types["numerical"], key="gs_measure")
    dimensions = st.multiselect("Categorical columns", col_types["categorical"], key="gs_dimensions",
                                help="Select one or more dimensions. Leave empty for the whole dataset.")
    analysis = st.selectbox("Analysis type", ANALYSES, key="gs_analysis")
    time_based = analysis in ANALYSES[2:5]
    allowed = ["Sum", "Non-missing count"] if analysis == ANALYSES[5] else list(STATISTICS)
    statistic = st.selectbox("Statistic", allowed, key="gs_stat_" + analysis)
    st.caption("Missing categories are included. Non-missing count counts valid numerical values. Standard deviation uses the sample definition.")
    try:
        if time_based:
            if not col_types["datetime"]:
                st.info("Select a Date/Time override in Column Overview to enable time-based analysis.")
                return
            date = st.selectbox("Date column", col_types["datetime"], key="gs_date")
            frequency = st.selectbox("Period frequency", list(FREQUENCIES), index=2, key="gs_frequency")
            zero = st.checkbox("Treat missing period values as zero", key="gs_zero",
                               help="Off by default. Enable only when missing values mean no activity.")
            lag, window = 1, 3
            if analysis == ANALYSES[2]:
                options = ["Previous period"]
                if frequency in ("Monthly", "Quarterly", "Yearly"):
                    options.append("Same period last year")
                mode = st.selectbox("Compare with", options, key="gs_comparison_" + frequency)
                lag = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}.get(frequency, 1) if mode == "Same period last year" else 1
            if analysis == ANALYSES[4]:
                window = int(st.number_input("Window size (calendar periods)", min_value=2, max_value=365, value=3, key="gs_window"))
                st.caption("A rolling average requires a complete window of non-missing period values.")
            result, keys = temporal(df, measure, dimensions, date, frequency, statistic, analysis, window, lag, zero)
            invalid = pd.to_datetime(df[date], errors="coerce").isna().sum()
            if invalid:
                st.caption(f"Excluded {invalid:,} rows with missing or invalid dates.")
            st.caption("Weekly periods run Monday–Sunday. Running and rolling statistics operate on the selected period statistic; expanding mean weights periods equally. Missing periods remain gaps unless zero filling is selected.")
            latest = pd.to_datetime(df[date], errors="coerce").max()
            current = pd.Timestamp.now().to_period(FREQUENCIES[frequency])
            if latest.to_period(FREQUENCIES[frequency]) >= current:
                st.warning("The data includes the current or a future period. The current period may be incomplete.")
            st.caption("First and last periods may also be partial depending on dataset coverage.")
            plotted = ["Value", "Previous"] if analysis == ANALYSES[2] else (["Value", "Rolling average"] if analysis == ANALYSES[4] else [st.selectbox("Running statistic", ["Cumulative sum", "Expanding mean", "Observed periods"], key="gs_running")])
        else:
            result, keys = summary(df, measure, dimensions, statistic)
            if result.empty:
                st.info("No data available.")
                return
            if analysis == ANALYSES[1]:
                baseline = st.selectbox("Reference segment", list(result.index),
                    format_func=lambda i: "; ".join(f"{dimensions[j] if dimensions else 'Dataset'}: {result.loc[i, key]}" for j, key in enumerate(keys)), key="gs_reference")
                result = compare(result, result.loc[baseline, "Value"])
                st.caption("Percentage difference uses the absolute reference value; zero or missing references show N/A.")
            if analysis == ANALYSES[5]:
                parent = st.checkbox("Share within parent group", key="gs_parent", disabled=len(keys) < 2)
                result = contribution(result, keys, parent)
                st.caption("Denominator: " + ("each combination of dimensions except the last." if parent and len(keys) > 1 else "the whole dataset."))
                if (result["Value"] < 0).any():
                    st.warning("Negative values produce signed contributions, which can fall outside 0–100%.")
            plotted = ["Contribution %"] if analysis == ANALYSES[5] else ["Value"]
        result["Segment"] = result[keys].apply(lambda row: "; ".join(f"{dimensions[i] if dimensions else 'Dataset'}: {v}" for i, v in enumerate(row)), axis=1)
        labels = result["Segment"].unique().tolist()
        chosen = st.multiselect("Segments to chart", labels, default=labels[:10], key="gs_chart_" + analysis)
        chart = result[result["Segment"].isin(chosen)]
        if not chart.empty:
            if time_based:
                long = chart.melt(id_vars=["Period", "Segment"], value_vars=plotted, var_name="Statistic", value_name="Result")
                fig = px.line(long, x="Period", y="Result", color="Segment", line_dash="Statistic")
            else:
                fig = px.bar(chart, x="Segment", y=plotted[0], color="Segment")
            st.plotly_chart(fig, use_container_width=True)
        display = result.drop(columns=keys)
        st.dataframe(display.style.format(precision=2, na_rep="N/A"), use_container_width=True)
        st.download_button("Download statistics (CSV)", display.to_csv(index=False).encode("utf-8-sig"),
                           file_name="group_statistics.csv", mime="text/csv", key="gs_download")
    except (ValueError, TypeError) as exc:
        st.warning(f"Unable to calculate these statistics: {exc}")
