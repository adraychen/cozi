from utils.numerical_formatting import format_plot_hover
from utils.numerical_formatting import format_statistic as fmt, format_p_value, DESCRIPTIVE, INFERENCE
from utils.report_capture import reportable, offer_result
"""Factorial ANOVA configuration and readable model reporting."""
from itertools import combinations
import numpy as np
import plotly.express as px
from scipy import stats
import streamlit as st
from utils import presentation
from utils.factorial_anova import prepare_factorial, run_factorial, posthoc, interaction_summary
from utils.factorial_followup import (MEAN, estimated_marginal_means, simple_effects_family, robust_type2)
from utils.factorial_diagnostics import diagnostic_summary
from utils.ttest_analysis import numerical_candidates


def _prune(key, options, multiple=False):
    if key in st.session_state:
        if multiple:
            valid = [v for v in st.session_state[key] if v in options]
            if valid != st.session_state[key]:
                st.session_state[key] = valid
        elif st.session_state[key] not in options:
            del st.session_state[key]


def display_name(value):
    # Avoid literal Markdown emphasis markers in exported/Plotly labels.
    return str(value).replace('**', '').strip()


def render_effect_plots(prepared, result):
    include_all = st.checkbox('Include non-significant selected terms', key='fac_plot_all')
    presentation.caption(f"Showing {'all selected terms' if include_all else 'all significant terms'} using Type II ANOVA at α = {result['alpha']:.3g}. Each main effect and interaction has its own plot.")
    presentation.caption(EMM_NOTE)
    shown = 0
    # Interactions precede marginal main effects in the guided review.
    ordered = sorted(enumerate(prepared['terms'].items()), key=lambda entry: entry[1][1][1] != 'Interaction effect')
    for index, (term, (name, kind, members)) in ordered:
        p = result['term_pvalues'][term]
        if not include_all and p >= result['alpha']:
            continue
        shown += 1
        presentation.subheader(f'{kind}: {display_name(name)}')
        presentation.caption(f"{format_p_value(p, result['alpha'], context='narrative')} · {'Significant' if p < result['alpha'] else 'Insufficient evidence at this α'}")
        try:
            if kind == 'Interaction effect':
                if st.checkbox('Swap factors on this plot', key=f'fac_swap_{index}'):
                    members = tuple(reversed(members))
                summary = estimated_marginal_means(prepared, result, list(members))
                plot = summary.copy()
                for column in ['Factor 1 level', 'Factor 2 level']:
                    plot[column] = plot[column].map(display_name)
                plot['CI half-width'] = plot['95% CI upper'] - plot[MEAN]
                fig = px.line(plot, x='Factor 1 level', y=MEAN, color='Factor 2 level', markers=True,
                              error_y='CI half-width', labels={'Factor 1 level': display_name(members[0]),
                              'Factor 2 level': display_name(members[1]), MEAN: f'{display_name(prepared["outcome"])} — adjusted mean (EMM)'})
                if p < result['alpha']:
                    presentation.info(interaction_interpretation(members, prepared['outcome']))
            else:
                factor = members[0]
                summary = estimated_marginal_means(prepared, result, [factor])
                plot = summary.copy()
                plot['Level'] = plot['Level'].map(display_name)
                plot['CI half-width'] = plot['95% CI upper'] - plot[MEAN]
                fig = px.scatter(plot, x='Level', y=MEAN, error_y='CI half-width',
                                 labels={'Level': display_name(factor), MEAN: f'{display_name(prepared["outcome"])} — adjusted mean (EMM)'})
                if factor in result['significant_interaction_factors']:
                    presentation.info('This factor participates in a significant interaction. Interpret this marginal main-effect plot together with the interaction/simple effects, not in isolation.')
            fig.update_xaxes(type='category')
            format_plot_hover(fig, {'y': 'mean'})
            presentation.plotly_chart(fig, width='stretch', key=f'fac_effect_plot_{index}')
            with st.expander(f'Adjusted means and 95% confidence intervals: {display_name(name)}'):
                presentation.dataframe(summary, result_types=DESCRIPTIVE, hide_index=True)
                presentation.caption('For context: raw arithmetic means and observed counts below pool over other factors. These are descriptive summaries, not the adjusted means plotted above.')
                if len(members) == 2:
                    observed = interaction_summary(prepared, members)
                else:
                    raw = prepared['clean'][[members[0], prepared['outcome']]].copy()
                    raw.columns = ['Level', 'Response']
                    observed = raw.groupby('Level', observed=True, sort=False).Response.agg(Mean='mean', N='size').reset_index()
                presentation.dataframe(observed, result_types=DESCRIPTIVE, hide_index=True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            presentation.warning(str(exc))
    if not shown:
        presentation.info('No selected terms are statistically significant at the current significance level. Enable the option above to inspect all selected terms.')


EMM_NOTE = ('Adjusted mean (Estimated Marginal Mean): fitted-model predictions averaged equally over all levels of the other factors, not raw arithmetic means. '
            'Intervals for means are pointwise 95% confidence intervals using classical OLS uncertainty. Empty cells can require model-based extrapolation; these predictions are not new observations.')
COMPARISON_NOTE = ('Holm adjustment covers the displayed comparison family. The 95% simultaneous confidence intervals use Bonferroni correction for that family; '
                   'these conservative intervals can include zero even when Holm declares significance. Significance uses the selected α; intervals remain 95%. '
                   'Corrections do not cover later exploration of other factors, directions, or models.')


def interaction_interpretation(pair, outcome):
    a, b = map(display_name, pair)
    return (f'{a} × {b} is significant. The relationship between {b} and {display_name(outcome)} differs across {a} levels: '
            'the effect of one factor depends on the level of the other. Interpret associated main effects cautiously and examine the interaction/simple effects.')


def render_comparisons(prepared, result):
    presentation.caption(EMM_NOTE)
    reported = []
    eligible = [f for f in prepared['factors'] if f not in result['significant_interaction_factors']
                and 2 <= len(prepared['levels'][f]) <= 30]
    if eligible:
        _prune('fac_posthoc', eligible)
        factor = st.selectbox('Factor for model-adjusted comparisons', eligible, key='fac_posthoc')
        presentation.caption('Global comparisons are available when no selected interaction involving this factor is significant. A non-significant interaction is not proof of a constant effect; retain domain-relevant interactions in your interpretation.')
        if st.checkbox('Show pairwise comparisons for this factor', key='fac_posthoc_show'):
            try:
                presentation.dataframe(estimated_marginal_means(prepared, result, [factor]), result_types=DESCRIPTIVE, hide_index=True)
                comparisons = posthoc(prepared, result, factor)
                presentation.dataframe(comparisons, result_types={**INFERENCE, '95% CI lower': 'ci', '95% CI upper': 'ci'}, alpha=result['alpha'], hide_index=True, width='stretch')
                reported.append((display_name(factor), comparisons))
                presentation.caption('Comparison family: all pairs within this factor. ' + COMPARISON_NOTE)
            except (ValueError, np.linalg.LinAlgError) as exc:
                presentation.warning(str(exc))
    else:
        presentation.info('No factor is eligible for global pairwise comparisons (2–30 levels and no significant interaction involving it).')
    interacting = [f for f in prepared['factors'] if f in result['significant_interaction_factors']]
    if interacting:
        presentation.subheader('Simple effects for significant interactions')
        presentation.write('Compare one factor within the levels of its interaction partners. Select either factor to change the direction. All significant partners are held fixed when a factor has multiple interactions.')
        _prune('fac_simple_factor', interacting)
        factor = st.selectbox('Factor to compare within interaction partners', interacting, key='fac_simple_factor')
        if st.checkbox('Show simple-effects comparisons', key='fac_simple_show'):
            try:
                family = simple_effects_family(prepared, result, factor)
                presentation.caption('Comparison family: all displayed pairs across all supported strata in this direction. ' + COMPARISON_NOTE)
                for item in family:
                    context = ', '.join(f'{display_name(f)} = {display_name(v)}' for f, v in item['fixed'].items())
                    title = f'Simple effect: {display_name(factor)} within {context}'
                    with st.expander(title, expanded=True):
                        if 'error' in item:
                            presentation.warning(item['error'])
                        else:
                            presentation.dataframe(item['means'], result_types=DESCRIPTIVE, hide_index=True)
                            presentation.dataframe(item['comparisons'], result_types={**INFERENCE, '95% CI lower': 'ci', '95% CI upper': 'ci'}, alpha=result['alpha'], hide_index=True, width='stretch')
                            reported.append((title, item['comparisons']))
            except (ValueError, np.linalg.LinAlgError) as exc:
                presentation.warning(str(exc))
    return reported


def render_findings(prepared, result, diagnostics, comparisons):
    presentation.subheader('A. Model reliability')
    presentation.write(f"{diagnostics['overall']}: {diagnostics['reliability']}")
    presentation.subheader('B. Main effects')
    table = result['table']
    effects = table[table.Effect != 'Residual']
    maximum = effects['Partial η²'].max()
    for factor in prepared['factors']:
        row = effects[(effects.Source == str(factor)) & (effects.Effect == 'Main effect')].iloc[0]
        presentation.write(f"{display_name(factor)}: {format_p_value(row['p-value'], result['alpha'], context='narrative')}; partial η² = {row['Partial η²']:.3g}.")
        if row.Significant == 'Yes':
            presentation.write(f'There is evidence that {display_name(prepared["outcome"])} differs among levels under the Type II test. This test adjusts for other terms except interactions containing this factor.')
        else:
            presentation.write('Insufficient evidence of a difference under this Type II test; this does not establish equal means.')
        if np.isclose(row['Partial η²'], maximum):
            presentation.write('This factor has the largest partial η² among the tested effects; assess its practical importance in context.')
        if factor in result['significant_interaction_factors']:
            presentation.info('A significant interaction contains this factor. Its main effect is marginal and does not apply universally across the interacting factor. Prioritize the interaction and simple effects below.')
        else:
            try:
                means = estimated_marginal_means(prepared, result, [factor])
                high, low = means.loc[means[MEAN].idxmax()], means.loc[means[MEAN].idxmin()]
                presentation.write(f"The highest adjusted estimate is {display_name(high['Level'])} ({fmt(high[MEAN], 'mean')}); the lowest is {display_name(low['Level'])} ({fmt(low[MEAN], 'mean')}). This ranking alone does not establish a pairwise difference; review confidence intervals and adjusted comparisons.")
            except (ValueError, np.linalg.LinAlgError) as exc:
                presentation.info(str(exc))
    presentation.subheader('C. Interaction effects')
    interaction_rows = effects[effects.Effect == 'Interaction effect']
    if interaction_rows.empty:
        presentation.write('No interactions were selected. The current model assumes additive factor effects; consider domain-relevant interactions when that assumption is questionable.')
    for term, (name, kind, members) in prepared['terms'].items():
        if kind != 'Interaction effect':
            continue
        row = interaction_rows[interaction_rows.Source == name].iloc[0]
        presentation.write(f"{display_name(name)}: {format_p_value(row['p-value'], result['alpha'], context='narrative')}; partial η² = {row['Partial η²']:.3g}.")
        if row.Significant == 'Yes':
            presentation.write(interaction_interpretation(members, prepared['outcome']))
        else:
            presentation.write(f'The analysis provides insufficient evidence that the relationship between {display_name(members[1])} and the response differs across {display_name(members[0])} levels. This does not show that the factors are independent.')
    presentation.subheader('D. Important comparisons')
    if not comparisons:
        presentation.write('Open Post-hoc comparisons and request adjusted pairs or simple effects. No pairwise conclusions have been generated yet.')
    for context, comparison in comparisons:
        presentation.write(context)
        significant = comparison[comparison.Significant == 'Yes']
        if significant.empty:
            presentation.write('No requested pairwise difference has sufficient evidence after multiplicity adjustment.')
        else:
            presentation.dataframe(significant.drop(columns=['p-value', 'SE']), hide_index=True)
            presentation.caption('Difference = first level minus second level: positive means the first adjusted estimate is higher. Intervals and adjusted p-values describe magnitude and uncertainty together.')
    presentation.subheader('E. Suggested next steps')
    if not interaction_rows.empty and not (interaction_rows.Significant == 'Yes').any():
        presentation.write(f"No selected interaction shows sufficient evidence at α = {result['alpha']:.3g}. You may retain the interactions based on the analysis plan or explicitly fit a reduced main-effects model for simpler interpretation. No terms are removed automatically; main effects must remain whenever their interaction remains.")
    if prepared['empty_cells'] or prepared['singleton_cells']:
        presentation.write('Prioritize additional replication or missing factor combinations. Review unnecessary interactions/factors using domain knowledge; sparse designs can make effects unstable or comparisons unsupported.')
    for note in diagnostics['recommendations']:
        presentation.write(note)
    presentation.caption('No transformation, model reduction, or method switch is performed automatically. Findings refer to the classical Type II model; HC3, when requested, is a separate sensitivity comparison.')


@reportable
def render_factorial_anova(df, numerical, categorical):
    presentation.markdown('##### Factorial ANOVA')
    presentation.caption('One numerical response and two or more categorical factors, with optional two-way interactions. All selected main effects remain in the model.')
    metadata = st.session_state.get('variable_metadata', {})
    numerical = numerical_candidates(df, numerical, metadata)
    categorical = [c for c in categorical if metadata.get(c, {}).get('variable_type', 'Categorical') == 'Categorical'
                   and metadata.get(c, {}).get('variable_subtype', 'Nominal') in ('Binary', 'Nominal', 'Ordinal')]
    if not numerical or len(categorical) < 2:
        presentation.info('Select one Numerical response and at least two Categorical variables in Comparison. Review Variable Type / Subtype in Column Overview; numerical codes must be explicitly classified as categorical to be factors.')
        return
    _prune('fac_y', numerical)
    _prune('fac_x', categorical, True)
    st.markdown('<a id="fac-model-configuration"></a>', unsafe_allow_html=True)
    if st.session_state.get('fac_review_terms'):
        st.info('Review the response, factors, effect types and interactions below. Confirm changes to fit a new model.')
    outcome = st.selectbox('Response (Y)', numerical, key='fac_y')
    if 'fac_x' not in st.session_state:
        st.session_state['fac_x'] = categorical[:]
    factors = st.multiselect('Factors (X)', categorical, key='fac_x')
    from utils.anova_effects_ui import effect_controls, render_random
    effects = effect_controls(factors, outcome, 'fac')
    if len(factors) < 2:
        presentation.info('Choose at least two categorical factors.')
        return
    alpha = st.number_input('Significance level (α)', min_value=.001, max_value=.20,
                            value=.05, step=.01, format='%.3f', key='fac_alpha')
    presentation.caption('Select an interaction when the effect of one factor may depend on the level of another factor. Choose interactions using business/domain knowledge, not every possible pair.')
    presentation.caption('For example, Segment × Discount Band asks: Does the relationship between Discount Band and the response differ across Segments?')
    options = list(combinations(factors, 2))
    _prune('fac_interactions', options, True)
    interactions = st.multiselect('Interaction effects (optional)', options,
                                 format_func=lambda pair: ' × '.join(map(str, pair)), key='fac_interactions')
    from utils.factorial_guidance import analysis_context, dependence_indicators, guided_items
    from utils.guided_diagnostics import diagnostic_status
    from utils.factorial_guidance_ui import navigate, render_items
    configuration = (outcome, tuple(factors), tuple(interactions), tuple(effects.items()), alpha)
    previous = st.session_state.get('fac_applied_configuration')
    if previous is not None and previous != configuration:
        if st.session_state.get('report_latest', {}).get('analysis_type') == 'Factorial ANOVA':
            st.session_state.pop('report_latest', None)
        st.info('Model changes are pending. Apply them to run a new analysis; results for the previous configuration are withheld here.')
        if not st.button('Apply changes and rerun', key='fac_apply_model'):
            return
    st.session_state['fac_applied_configuration'] = configuration
    context = analysis_context(df, outcome, factors, interactions, effects, alpha, metadata)
    if st.session_state.get('fac_guidance_context') != context:
        st.session_state['fac_guidance_context'] = context
        st.session_state.pop('fac_study_answer', None)
        st.session_state.pop('fac_diagnostic_focus', None)
        for key in list(st.session_state):
            if key.startswith('fac_random_'):
                del st.session_state[key]
    if 'Random' in effects.values():
        return render_random(df, outcome, effects, interactions, alpha, 'fac')
    prepared, result, failure = None, None, None
    try:
        prepared = prepare_factorial(df, outcome, factors, interactions)
        result = run_factorial(prepared, alpha)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        failure = str(exc)
    from utils.report_capture import begin_result
    begin_result()
    diagnostics = diagnostic_summary(prepared, result) if result is not None else None
    evidence = dependence_indicators(df, prepared, metadata) if prepared is not None else None
    answer = st.session_state.get('fac_study_answer')
    items = guided_items(prepared, result, diagnostics, answer, evidence, failure) if prepared is not None else []
    status = diagnostic_status(items) if items else 'Model issue'
    tabs = st.tabs(['Model Overview', 'Diagnostics', 'Effect plots', 'Post-hoc comparisons', 'ANOVA Findings'],
                   key='fac_results_tab', on_change='rerun')
    with tabs[0]:
        presentation.subheader('Model Overview')
        if failure:
            st.error('Model issue — requested Factorial ANOVA inference is unavailable. Review Diagnostics.')
        else:
            presentation.write('Diagnostic status: **' + status + '**')
        st.button('Open Diagnostics', key='fac_open_diagnostics', on_click=navigate, args=('overview',))
        if st.session_state.get('fac_review_terms'):
            st.markdown('[Review response, factors and interaction controls](#fac-model-configuration)')
        presentation.write('Main effects: ' + ', '.join(map(str, factors)))
        presentation.write('Interaction effects: ' + (', '.join(' × '.join(map(str, p)) for p in interactions) or 'None'))
        presentation.caption(f'Type II Sum of Squares · α = {alpha:.3g}.')
        with st.expander('About Type II tests and interactions'):
            presentation.write('Type II tests respect marginality: a main-effect test adjusts for other terms except interactions containing that factor. Review significant interactions before interpreting associated main effects. The method never switches automatically.')
        if prepared is not None:
            presentation.write(f"Original observations: {prepared['original']} · Used: {prepared['n']} · Excluded: {prepared['excluded']} (missing: {prepared['missing']}; additional nonfinite responses: {prepared['nonfinite']}).")
            if prepared['excluded']:
                with st.expander('Excluded observation details'):
                    selected = df[[outcome] + factors]
                    missing = selected.isna().any(axis=1)
                    nonfinite = ~np.isfinite(selected[outcome].to_numpy(dtype=float))
                    excluded = selected.loc[missing | nonfinite].copy()
                    excluded['Exclusion reason'] = np.where(missing[missing | nonfinite], 'Missing selected value', 'Nonfinite response')
                    presentation.dataframe(excluded)
        if result is None:
            if st.session_state.get('report_latest', {}).get('analysis_type') == 'Factorial ANOVA':
                st.session_state.pop('report_latest', None)
            with tabs[1]:
                if items:
                    render_items(items, prepared, evidence)
                else:
                    presentation.write('What was found: ' + failure)
                    presentation.write('Review the response type and distinct factor levels, then confirm changes and rerun.')
                st.markdown('[Review model configuration](#fac-model-configuration)')
            return
        presentation.caption('Checks are screening results, not proof of all assumptions. Independence is a study-design assumption; review its recorded status in Diagnostics.')
        table = result['table']
        presentation.dataframe(table, result_types=INFERENCE, alpha=alpha, hide_index=True, width='stretch')
        presentation.caption('Partial η² = effect SS / (effect SS + residual SS). Effect sizes are conditional on this model and do not sum to 1. Statistical significance does not establish practical importance or causation.')
        if st.checkbox('Show HC3 robust inference for comparison', key='fac_hc3'):
            presentation.write('HC3 heteroskedasticity-robust inference — equivalent OLS model, Type II hypotheses')
            try:
                presentation.dataframe(robust_type2(prepared, result), result_types=INFERENCE, alpha=alpha, hide_index=True, width='stretch')
            except (ValueError, np.linalg.LinAlgError) as exc:
                presentation.warning(str(exc))
            presentation.caption('The classical ANOVA and partial η² above remain unchanged. HC3 changes covariance-based inference, not the fitted response or selected terms. It does not resolve dependence, missing cells, or an incorrect mean model. Robust F tests are an approximate sensitivity analysis, especially with small samples or high leverage.')
    with tabs[1]:
        presentation.subheader('Model validity')
        render_items(items, prepared, evidence)
        st.markdown('[Review model configuration and interactions](#fac-model-configuration)')
        presentation.caption('Automated pattern flags are exploratory, not definitive hypothesis tests. Study-design answers remain user-provided assumptions.')
        presentation.subheader('Residual plots and variance details')
        residuals = np.asarray(result['model'].resid)
        (x, y), (slope, intercept, _) = stats.probplot(residuals, dist='norm')
        fig = px.scatter(x=x, y=y, labels={'x': 'Theoretical normal quantile', 'y': 'Model residual'})
        fig.add_scatter(x=x, y=intercept+slope*x, mode='lines', name='Normal reference')
        presentation.plotly_chart(fig, width='stretch', key='fac_qq')
        presentation.write('In the Q–Q plot, points near the reference line are compatible with normal residuals. Substantial curvature suggests skew or a distribution mismatch; large departures at the ends suggest heavy tails or extreme observations. Inspect the size and pattern of departures rather than relying only on a p-value.')
        if diagnostics['jb'] is not None:
            jb, normal_p, skew, kurtosis = diagnostics['jb']
            presentation.write(f'Residual Jarque–Bera diagnostic: statistic = {jb:.4g}; {format_p_value(normal_p, alpha, context='narrative')}; skewness = {skew:.3g}; kurtosis = {kurtosis:.3g} (normal reference: 3).')
            presentation.caption('The asymptotic p-value can be unreliable in small samples. A non-significant result does not establish normality.')
        presentation.plotly_chart(px.scatter(x=result['model'].fittedvalues, y=residuals,
                                  labels={'x': 'Fitted response', 'y': 'Model residual'}),
                        width='stretch', key='fac_residuals')
        presentation.write('Look for residuals scattered around zero with roughly similar spread. A widening or narrowing funnel can indicate changing variance; curves can indicate an incomplete mean model. Extreme points warrant investigation in context.')
        for pattern in diagnostics['patterns']:
            presentation.write(pattern)
        with st.expander('How the exploratory pattern screens work'):
            presentation.write('With at least 24 observations, fitted values are divided into up to four quantile bins with at least six observations each. Spread is flagged when the last/first residual SD ratio exceeds 1.75 (or is below its reciprocal), with an absolute-residual Spearman trend exceeding ±0.25 in the same direction. A binned mean beyond half a residual SD with at least three bins flags a possible mean pattern. Internally studentized residuals beyond ±3 are flagged. These heuristics can miss patterns, especially with few distinct fitted values.')
        if result['levene'] is None:
            presentation.info('Median-centered Levene test unavailable: every observed factor combination needs replication and a finite test statistic. Equal variances could not be formally assessed using the selected factor-cell structure.')
        else:
            statistic, p = result['levene']
            presentation.write(f'Median-centered Levene across observed factor combinations: statistic = {statistic:.4g}; {format_p_value(p, result['alpha'], context='narrative')}.')
            presentation.caption('A non-significant test does not prove equal variances. Interpret this alongside the residual-vs-fitted plot.')
        presentation.subheader('Possible next steps')
        for note in diagnostics['recommendations']:
            presentation.write(note)
        presentation.caption('These are suggestions only. Perform any transformation explicitly and refit; compare diagnostics and the changed response-scale interpretation.')
    with tabs[2]:
        presentation.write('For significant interactions, inspect the interaction plot and adjusted cell means before associated main effects. Post-hoc comparisons supports simple effects with Holm-adjusted comparisons and Bonferroni simultaneous intervals. Consider magnitude and practical relevance.')
        render_effect_plots(prepared, result)
    with tabs[3]:
        comparisons = render_comparisons(prepared, result)
    with tabs[4]:
        presentation.write(next(item.observed_result for item in items if item.diagnostic_id == 'independence'))
        render_findings(prepared, result, diagnostics, comparisons)
    offer_result('Factorial ANOVA', str(outcome)+' by '+', '.join(map(str, factors)),
                 {'response': outcome, 'factors': factors}, {'interactions': interactions, 'alpha': alpha, 'study_design': answer},
                 result, {**diagnostics, 'guided_items': [item.__dict__ for item in items]},
                 [item.observed_result for item in items if item.severity != 'ok'], key='factorial', run_id=context)
