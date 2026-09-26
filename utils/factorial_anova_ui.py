"""Factorial ANOVA configuration and readable model reporting."""
from itertools import combinations
import numpy as np
import plotly.express as px
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
import streamlit as st
from utils.factorial_anova import prepare_factorial, run_factorial, interaction_summary, posthoc
from utils.ttest_analysis import numerical_candidates


def _prune(key, options, multiple=False):
    if key in st.session_state:
        if multiple:
            valid = [v for v in st.session_state[key] if v in options]
            if valid != st.session_state[key]:
                st.session_state[key] = valid
        elif st.session_state[key] not in options:
            del st.session_state[key]


def render_factorial_anova(df, numerical, categorical):
    st.markdown('##### Factorial ANOVA')
    st.caption('One numerical response and two or more categorical factors, with optional two-way interactions. All selected main effects remain in the model.')
    metadata = st.session_state.get('variable_metadata', {})
    numerical = numerical_candidates(df, numerical, metadata)
    categorical = [c for c in categorical if metadata.get(c, {}).get('variable_type', 'Categorical') == 'Categorical'
                   and metadata.get(c, {}).get('variable_subtype', 'Nominal') in ('Binary', 'Nominal', 'Ordinal')]
    if not numerical or len(categorical) < 2:
        st.info('Select one Numerical response and at least two Categorical variables in Comparison. Review Variable Type / Subtype in Column Overview; numerical codes must be explicitly classified as categorical to be factors.')
        return
    _prune('fac_y', numerical)
    _prune('fac_x', categorical, True)
    outcome = st.selectbox('Response (Y)', numerical, key='fac_y')
    if 'fac_x' not in st.session_state:
        st.session_state['fac_x'] = categorical[:2]
    factors = st.multiselect('Factors (X)', categorical, key='fac_x')
    if len(factors) < 2:
        st.info('Choose at least two categorical factors.')
        return
    options = list(combinations(factors, 2))
    _prune('fac_interactions', options, True)
    interactions = st.multiselect('Interaction effects (optional)', options,
                                 format_func=lambda pair: ' × '.join(map(str, pair)), key='fac_interactions')
    st.write('Main effects: ' + ', '.join(map(str, factors)))
    st.write('Interaction effects: ' + (', '.join(' × '.join(map(str, p)) for p in interactions) or 'None'))
    st.caption('Type II Sum of Squares · α = 0.05. Type II tests respect marginality: a main-effect test does not adjust for interactions containing that factor. Review interactions before interpreting main effects. The method never switches automatically.')
    try:
        prepared = prepare_factorial(df, outcome, factors, interactions)
    except (ValueError, TypeError) as exc:
        st.warning(str(exc))
        return
    st.write(f"Original observations: {prepared['original']} · Used: {prepared['n']} · Excluded: {prepared['excluded']} (missing: {prepared['missing']}; additional nonfinite responses: {prepared['nonfinite']})")
    for note in prepared['warnings']:
        st.warning(note)
    if prepared['errors']:
        for error in prepared['errors']:
            st.warning(error)
        return
    try:
        result = run_factorial(prepared)
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.warning('Cannot report Factorial ANOVA: ' + str(exc))
        return
    st.write('Independence depends on study and data-collection design; it cannot be verified from these data. Repeated or clustered observations require an appropriate repeated-measures or mixed model.')
    table = result['table']
    if result['significant_interaction_factors']:
        st.warning('A significant interaction means the effect of one factor depends on the level of another. Do not interpret these main effects in isolation: ' +
                   ', '.join(str(f) for f in factors if f in result['significant_interaction_factors']))
    tabs = st.tabs(['ANOVA table', 'Diagnostics', 'Interaction plots', 'Post-hoc comparisons', 'Findings'])
    with tabs[0]:
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption('Partial η² = effect SS / (effect SS + residual SS). Effect sizes are conditional on this model and do not sum to 1. Statistical significance does not establish practical importance or causation.')
    with tabs[1]:
        residuals = np.asarray(result['model'].resid)
        (x, y), (slope, intercept, _) = stats.probplot(residuals, dist='norm')
        fig = px.scatter(x=x, y=y, labels={'x': 'Theoretical normal quantile', 'y': 'Model residual'})
        fig.add_scatter(x=x, y=intercept+slope*x, mode='lines', name='Normal reference')
        st.plotly_chart(fig, use_container_width=True, key='fac_qq')
        st.write('Assess residual normality using the Q–Q plot: systematic curvature or extreme tails suggest departures from normal errors. Small samples limit diagnostic sensitivity.')
        if len(residuals) >= 3:
            jb, normal_p = jarque_bera(residuals)[:2]
            st.write(f'Residual Jarque–Bera diagnostic: statistic = {jb:.4g}; p = {normal_p:.4g}.')
            st.caption('As in regression diagnostics, this is an exploratory normality check. Its asymptotic p-value can be unreliable in small samples; a non-significant result does not establish normality.')
            if normal_p < .05:
                st.warning('Residuals show evidence of non-normality. Review the Q–Q plot, outliers, and sample size before relying on F tests.')
        st.plotly_chart(px.scatter(x=result['model'].fittedvalues, y=residuals,
                                  labels={'x': 'Fitted response', 'y': 'Model residual'}),
                        use_container_width=True, key='fac_residuals')
        if result['levene'] is None:
            st.info('Median-centered Levene test unavailable: every observed factor combination needs replication and a finite test statistic. Equal variances have not been established.')
        else:
            statistic, p = result['levene']
            st.write(f'Median-centered Levene across observed factor combinations: statistic = {statistic:.4g}; p = {p:.4g}.')
            if p < .05:
                st.warning('There is evidence of unequal variances. Classical OLS F tests may be unreliable, particularly with unequal cell sizes.')
            else:
                st.caption('A non-significant test does not prove equal variances.')
    with tabs[2]:
        if not interactions:
            st.info('Select an interaction above to inspect its plot.')
        else:
            _prune('fac_plot', interactions)
            pair = st.selectbox('Interaction to inspect', interactions,
                                format_func=lambda p: ' × '.join(map(str, p)), key='fac_plot')
            summary = interaction_summary(prepared, pair)
            # Fixed plotting aliases also support user columns named Mean or N.
            plot = summary.copy()
            plot.columns = ['First factor', 'Second factor', 'Mean response', 'N']
            plot['First factor'] = plot['First factor'].astype(str)
            plot['Second factor'] = plot['Second factor'].astype(str)
            st.plotly_chart(px.line(plot, x='First factor', y='Mean response', color='Second factor', markers=True,
                                   hover_data=['N'], labels={'First factor': str(pair[0]), 'Second factor': str(pair[1])}),
                            use_container_width=True, key='fac_interaction_plot')
            st.caption('Observed means from model complete cases, pooled over other factors. These are descriptive, unadjusted means; imbalance can make them differ from model-adjusted effects. Missing cells are not imputed.')
            st.dataframe(summary, hide_index=True)
    with tabs[3]:
        eligible = [f for f in factors if len(prepared['levels'][f]) > 2 and not any(f in p for p in interactions)]
        if interactions:
            st.info('Factors in interactions require care. Significant interactions call for simple effects within relevant factor levels. This interface does not implement simple effects or marginal comparisons for factors in selected interactions, even when non-significant.')
        if eligible:
            _prune('fac_posthoc', eligible)
            factor = st.selectbox('Factor for model-adjusted comparisons', eligible, key='fac_posthoc')
            if st.checkbox('Show pairwise comparisons for this factor', key='fac_posthoc_show'):
                try:
                    st.dataframe(posthoc(prepared, result, factor), hide_index=True, use_container_width=True)
                except ValueError as exc:
                    st.warning(str(exc))
                st.caption('Contrasts use the fitted OLS covariance and residual degrees of freedom. Holm correction covers all pairs within the selected factor; it does not cover repeated exploration across factors or models.')
        else:
            st.info('No eligible factor with more than two levels outside selected interactions.')
    with tabs[4]:
        for _, row in table[table['Effect'] != 'Residual'].iterrows():
            st.markdown(f"**{row['Source']}: p = {row['p-value']:.4g}; partial η² = {row['Partial η²']:.3g}**")
            if row['Effect'] == 'Interaction effect':
                if row['Significant'] == 'Yes':
                    st.warning('There is evidence of an interaction: the effect of one factor depends on the level of the other factor. Do not interpret the corresponding main effects in isolation.')
                else:
                    st.write('There is insufficient evidence of this interaction; this does not establish that the effects are independent of factor levels.')
            elif row['Significant'] == 'Yes':
                st.write('There is evidence that mean response differs among at least some levels after accounting for other terms under the Type II hypothesis described above.')
            else:
                st.write('There is insufficient evidence of a mean difference under this Type II test. This does not establish equal means.')
        st.caption('Interpret effect sizes in the subject-matter context. Associations do not imply causation.')
