"""Streamlit controls and results for grouped exploratory analysis."""
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.group_statistics import STATISTICS, FREQUENCIES, summary, compare, contribution, temporal, contingency

ANALYSES = ["Group Summary", "Segment Comparison", "Period-over-Period Comparison",
            "Running & Cumulative Statistics", "Rolling Averages", "Contribution to Total", "Cross Tabulation / Chi-square"]


def render_contingency(df, dimensions):
    st.caption("Counts rows for two categorical variables; the numerical selection is not used.")
    if len(dimensions) < 2:
        st.info("Select at least two categorical columns above to build a contingency table.")
        return
    row = st.selectbox("Row variable", dimensions, key="gs_ct_row")
    column = st.selectbox("Column variable", [c for c in dimensions if c != row], key="gs_ct_column")
    if len(dimensions) > 2:
        st.caption("Only the chosen row and column variables enter this test; other dimensions are not used as controls.")
    missing = st.checkbox("Include missing values as categories", key="gs_ct_missing")
    alpha = st.selectbox("Significance level", [0.05, 0.01, 0.10], key="gs_ct_alpha")
    try:
        result = contingency(df, row, column, missing)
    except (ValueError, TypeError) as exc:
        st.warning(str(exc))
        return
    st.caption(f"Included {result['n']:,} rows; excluded {result['excluded']:,} rows with missing categories. Expected counts = row total × column total ÷ grand total.")
    views = {"Observed counts": result["observed"], "Expected counts": result["expected"],
             "Observed − expected": result["difference"], "Pearson residuals": result["residual"],
             "Row percentages": result["observed"].div(result["observed"].sum(axis=1), axis=0) * 100,
             "Column percentages": result["observed"].div(result["observed"].sum(axis=0), axis=1) * 100}
    view = st.selectbox("Table view", list(views), key="gs_ct_view")
    table = views[view]
    st.dataframe(table.style.format(precision=2), use_container_width=True)
    if table.size <= 400:
        signed = view in ("Observed − expected", "Pearson residuals")
        fig = px.imshow(table, aspect="auto", color_continuous_scale="RdBu_r" if signed else "Blues",
                        color_continuous_midpoint=0 if signed else None,
                        labels={"x": str(column), "y": str(row), "color": view})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Heatmap omitted for tables larger than 400 cells; all cells remain in the table and downloads.")
    st.caption("Positive differences/residuals mean more observations than expected under independence. Residuals are descriptive, not separate significance tests.")
    # Positional cell records preserve the original category names, even when they match output labels.
    records = [{"Row category": r, "Column category": c, "Observed": int(result['observed'].loc[r, c]),
                "Expected": float(result['expected'].loc[r, c]), "Observed - expected": float(result['difference'].loc[r, c]),
                "Pearson residual": float(result['residual'].loc[r, c])}
               for r in table.index for c in table.columns]
    st.download_button("Download observed and expected counts (CSV)", pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig"),
                       file_name="contingency_comparison.csv", mime="text/csv", key="gs_ct_download")
    st.markdown("##### Chi-square test of association")
    st.caption("Pearson chi-square test without continuity correction. Null hypothesis: the two variables are independent. Rows must represent independent observations; association does not imply causation.")
    if result["chi2"] is None:
        st.info("The test requires at least two observed categories in each variable.")
        return
    a, b, c = st.columns(3)
    a.metric("Chi-square", f"{result['chi2']:.4f}")
    b.metric("Degrees of freedom", result["dof"])
    c.metric("p-value", f"{result['p']:.4g}")
    if result["low_expected"]:
        st.warning(f"{result['low_expected']} of {table.size} cells have expected counts below 5. The asymptotic p-value may be unreliable; interpret cautiously.")
    if result["p"] < alpha:
        st.write(f"At α = {alpha}, the test provides evidence of an association, subject to the assumptions above.")
    else:
        st.write(f"At α = {alpha}, the test does not provide sufficient evidence of an association. This does not prove independence.")
    st.caption("Repeatedly testing different pairs increases the chance of false positives; these p-values are unadjusted.")
    st.markdown("[Chi-square method documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html)")


def render_group_statistics(df, col_types):
    st.markdown("#### Group Statistics")
    measure = st.selectbox("Numerical column", [None] + col_types["numerical"],
                           format_func=lambda value: "None (categorical data / row counts)" if value is None else value,
                           key="gs_measure_optional")
    dimensions = st.multiselect("Categorical columns", col_types["categorical"], key="gs_dimensions",
                                help="Select one or more dimensions. Leave empty for the whole dataset.")
    analysis = st.selectbox("Analysis type", ANALYSES, key="gs_analysis")
    if analysis == ANALYSES[6]:
        render_contingency(df, dimensions)
        return
    time_based = analysis in ANALYSES[2:5]
    allowed = ["Row count"] if measure is None else (["Sum", "Non-missing count", "Row count"] if analysis == ANALYSES[5] else list(STATISTICS))
    statistic = st.selectbox("Statistic", allowed, key="gs_stat_" + analysis + ("_counts" if measure is None else "_numeric"))
    st.caption("Missing categories are included. Row count counts all rows, including missing measures. Non-missing count counts valid numerical values. Standard deviation uses the sample definition.")
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
