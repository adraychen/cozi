import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from utils.anova_analysis import prepare_anova, run_anova


def render_anova(df, category, outcome):
    prepared = prepare_anova(df, category, outcome)
    st.caption('Compare the mean of a numerical response across independent groups. Identifiers are not meaningful outcomes. Ordered ratings assume equally spaced scores.')
    method = st.selectbox('ANOVA method', ['Welch', 'Classical'], key='rel_anova_method')
    st.caption('Welch is the default to allow unequal variances. Method choice is explicit; diagnostic p-values do not switch the method automatically.')
    if method == 'Classical' and (not np.isfinite(prepared['levene_p']) or prepared['levene_p'] < .05):
        st.warning('Classical ANOVA selected despite uncertain or unequal variances. Its inference may be unreliable; consider Welch when group variances are positive.')
    st.write('Independence cannot be verified from the CSV. Repeated responses from a person or clustered measurements may require repeated-measures or mixed-effects analysis.')
    # Scope confirmation to the active dataset and variable pair.
    context = (st.session_state.get('file_id', id(df)), category, outcome)
    if st.session_state.get('rel_anova_context') != context:
        st.session_state['rel_anova_independent'] = False
        st.session_state['rel_anova_context'] = context
    independent = False
    if not prepared['errors']:
        independent = st.checkbox('I confirm the observations are independent, with no repeated or clustered measurements', key='rel_anova_independent')
    result = None
    result_error = None
    if independent and not prepared['errors']:
        try:
            result = run_anova(prepared, method, independent=True)
        except ValueError as exc:
            result_error = str(exc)
    tabs = st.tabs(['Model Overview', 'Group Analysis', 'Diagnostics', 'Visual Analysis', 'ANOVA Findings'])
    with tabs[0]:
        st.markdown('**Model summary**')
        st.table(pd.DataFrame([
            ('Response / Dependent Variable', str(outcome)),
            ('Grouping / Independent Variable', str(category)),
            ('Original rows', str(len(df))), ('Used rows', str(prepared['n'])),
            ('Excluded missing/nonfinite rows', str(prepared['excluded'])),
            ('Number of groups', str(len(prepared['groups']))),
            ('Analysis method', method + ' One-way ANOVA'),
        ], columns=['Model detail', 'Value']))
        for error in prepared['errors']:
            st.error('Cannot run ANOVA: ' + error)
        if result_error:
            st.error('Cannot run ANOVA: ' + result_error)
        if not independent and not prepared['errors']:
            st.info('Confirm the study design above to display the ANOVA result. Descriptive summaries and diagnostics remain available.')
        if result is not None:
            st.markdown(f'**{method} One-way ANOVA**')
            st.dataframe(pd.DataFrame([result]), hide_index=True, use_container_width=True)
    with tabs[1]:
        st.markdown('**Group summaries**')
        st.dataframe(prepared['summary'], hide_index=True, use_container_width=True)
    with tabs[2]:
        for note in prepared['warnings']:
            st.warning(note)
        st.write('Normality is checked within each group, not on the combined outcome. Shapiro–Wilk p-values below 0.05 flag departures; larger p-values do not prove normality. These exploratory diagnostics are unadjusted for multiple comparisons and should be interpreted with the plots.')
        st.dataframe(prepared['normality'], hide_index=True, use_container_width=True)
        if prepared['groups']:
            index = st.selectbox('Group for normal Q–Q plot', list(range(len(prepared['groups']))),
                                 format_func=lambda i: str(prepared['labels'][i]), key='rel_anova_qq')
            g = prepared['groups'][index]
            if len(g) >= 3 and np.ptp(g) > 0:
                (x, y), (slope, intercept, _) = stats.probplot(g, dist='norm')
                fig = px.scatter(x=x, y=y, labels={'x': 'Theoretical normal quantile', 'y': outcome})
                fig.add_scatter(x=x, y=intercept+slope*x, mode='lines', name='Normal reference')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('A useful normal Q–Q plot needs at least three observations and nonconstant values.')
        if np.isfinite(prepared['levene_p']):
            st.write(f"Median-centered Levene (Brown–Forsythe variance test): statistic = {prepared['levene_stat']:.4g}; p = {prepared['levene_p']:.4g}.")
        else:
            st.info('Variance test unavailable; equal variances have not been established.')
        st.write('Welch ANOVA permits unequal variances. Classical ANOVA assumes equal population variances. Both require independent observations and warrant caution with severe non-normality and outliers, especially in small groups. A non-significant variance test does not establish equality.')
        st.caption('References: [SciPy Levene](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html), [SciPy Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html), [statsmodels ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html).')

    with tabs[3]:
        st.markdown('**Outcome distribution by group**')
        if not prepared['summary'].empty:
            plot = prepared['clean'].copy()
            plot[category] = plot[category].astype(str)
            st.plotly_chart(px.box(plot, x=category, y=outcome, points='outliers'), use_container_width=True, key='rel_anova_box')
        else:
            st.info('No usable observations to plot.')
    with tabs[4]:
        if result is None:
            st.info('ANOVA findings are available after validation succeeds and independence is confirmed.')
            return
        if result['p-value'] < .05:
            st.write('At α = 0.05, there is evidence that at least one population mean differs, subject to the assumptions above. This does not identify which groups differ; follow-up comparisons with multiplicity correction are needed.')
        else:
            st.write('At α = 0.05, there is insufficient evidence of different population means. This does not establish that the means are equal.')
        st.caption('Association is not causation. Repeated exploratory testing increases false-positive risk. No post-hoc pairwise tests are run automatically.')
