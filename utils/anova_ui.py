"""Guided Welch One-way ANOVA, diagnostics and Games–Howell reporting."""
import hashlib
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import streamlit as st
from utils import presentation
from utils.report_capture import reportable, offer_result, begin_result
from utils.anova_analysis import prepare_anova, run_anova, variation_decomposition, games_howell, EFFECT_SIZE, VARIATION_NOTE
from utils.anova_independence import assess_independence
from utils.anova_charts import group_mean_plot, pairwise_plot, residual_plot


@st.cache_data(show_spinner=False, max_entries=16)
def _prepare(df, category, outcome, metadata):
    prepared = prepare_anova(df, category, outcome)
    return prepared, assess_independence(df, prepared, metadata)


@st.cache_data(show_spinner=False, max_entries=16)
def _pairwise(prepared, alpha):
    return games_howell(prepared, alpha)


def _json_safe(value):
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict('records'))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def guidance_context(prepared, independence, result, pairwise, metadata):
    """Bounded summaries only: no individual rows, IDs, timestamps or measurements."""
    pairs = pairwise['table'] if pairwise else pd.DataFrame()
    selected = pairs[pairs.Significant] if not pairs.empty else pairs
    return _json_safe({
        'response': str(prepared['outcome']), 'factor': str(prepared['category']),
        'n': prepared['n'], 'excluded': prepared['excluded'], 'groups': prepared['summary'].head(30),
        'group_count': len(prepared['groups']), 'result': result,
        'pairwise': selected.head(20), 'significant_pair_count': len(selected),
        'pairwise_status': pairwise['reason'] if pairwise else 'Inference withheld pending study-design clarification.',
        'independence': {'state': independence['state'], 'evidence': independence['evidence'],
                         'qualification': independence['qualification']},
        'warnings': prepared['warnings'], 'errors': prepared['errors'],
        'metadata': {str(c): {key: metadata[c].get(key) for key in ('variable_type', 'variable_subtype', 'source')}
                     for c in list(metadata)[:40]},
    })


def _show_pair_plot(pairwise):
    presentation.markdown('**Games–Howell Pairwise Comparison Plot**')
    if pairwise is None or pairwise['table'].empty:
        presentation.info(pairwise['reason'] if pairwise else 'Pairwise inference is withheld until the study-design concern is resolved.')
        return
    table = pairwise['table']
    mode = st.selectbox('Comparisons to plot', ['All comparisons', 'Significant comparisons', 'Nonsignificant comparisons'],
                        key='rel_anova_pair_filter')
    if mode != 'All comparisons':
        table = table[table.Significant.eq(mode == 'Significant comparisons')]
    if table.empty:
        presentation.info('No comparisons match this filter.')
        return
    pages = math.ceil(len(table)/20)
    if pages > 1:
        if st.session_state.get('rel_anova_pair_page', 0) >= pages:
            st.session_state.pop('rel_anova_pair_page', None)
        page = st.selectbox('Comparison page', range(pages), format_func=lambda p: f'{p+1} of {pages}',
                            key='rel_anova_pair_page')
    else:
        page = 0
    shown = table.iloc[page*20:(page+1)*20]
    presentation.caption(f'Showing {len(shown)} of {len(table)} filtered comparisons. Filtering does not change the full comparison family or its adjustment.')
    fig = pairwise_plot(pairwise, shown)
    presentation.plotly_chart(fig, key='rel_anova_pair_plot')
    presentation.caption(f'{pairwise["confidence"]:.0%} simultaneous Games–Howell intervals. Above zero: A has the higher mean; below zero: B has the higher mean. An interval containing zero does not establish equality. Significance uses the adjusted p-value at α = {pairwise["alpha"]:g}.')


@reportable
def render_anova(df, category, outcome):
    from utils.anova_effects_ui import effect_controls, render_random
    effects = effect_controls([category], outcome, 'rel_anova')
    if effects[category] == 'Random':
        alpha = st.number_input('Significance level (α)', min_value=.001, max_value=.20, value=.05, step=.01, format='%.3f', key='rel_anova_alpha')
        return render_random(df, outcome, effects, (), alpha, 'rel_anova')
    metadata = st.session_state.get('variable_metadata', {})
    prepared, independence = _prepare(df, category, outcome, metadata)
    alpha = st.number_input('Significance level (α)', min_value=.001, max_value=.20, value=.05,
                            step=.01, format='%.3f', key='rel_anova_alpha')
    signature = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    context = (st.session_state.get('file_id'), category, outcome, signature)
    if st.session_state.get('rel_anova_context') != context:
        for key in ('rel_anova_design', 'rel_anova_qq', 'rel_anova_pair_page'):
            st.session_state.pop(key, None)
        st.session_state['rel_anova_context'] = context
    design = None
    if independence['requires_clarification']:
        presentation.warning('Potential dependence detected. Ordinary Welch ANOVA may be inappropriate; repeated-measures or mixed-effects analysis may be needed.')
        presentation.dataframe(independence['evidence'], hide_index=True)
        clarification = st.selectbox(
            'What do the repeated identifiers, cluster labels or chronological indicators represent?',
            ['Unresolved study design', 'Repeated or clustered measurements',
             'Independent observations; these labels/order do not identify dependent units'],
            key='rel_anova_design')
        if clarification.startswith('Independent'):
            design = 'independent'
            presentation.warning('Inference proceeds under your study-design clarification. The detected indicators remain limitations; diagnostics have not verified independence.')
    result, result_error, variation, pairwise = None, None, None, None
    if not prepared['errors']:
        try:
            result = run_anova(prepared, alpha=alpha, independence=independence, design=design)
            pairwise = _pairwise(prepared, alpha)
        except ValueError as exc:
            result_error = str(exc)
    try:
        variation = variation_decomposition(prepared)
    except ValueError as exc:
        prepared['warnings'].append(str(exc))
    begin_result()
    tabs = st.tabs(['Overview', 'Group Comparisons', 'Diagnostics', 'ANOVA Findings'])
    with tabs[0]:
        presentation.markdown('**Model summary**')
        presentation.table(pd.DataFrame([
            ('Response / Dependent Variable', str(outcome)), ('Grouping / Independent Variable', str(category)),
            ('Original rows', str(len(df))), ('Used rows', str(prepared['n'])),
            ('Excluded missing/nonfinite rows', str(prepared['excluded'])),
            ('Number of groups', str(len(prepared['groups']))), ('Effect type', 'Fixed'), ('Analysis method', 'Welch One-way ANOVA'),
            ('Significance level', str(alpha)), ('Independence assessment', independence['state']),
        ], columns=['Model detail', 'Value']))
        for error in prepared['errors']:
            presentation.error('Cannot run ANOVA: ' + error)
        if result_error:
            presentation.warning('Inference withheld: ' + result_error)
        presentation.markdown('**Statistical Results — Welch One-way ANOVA**')
        if result is not None:
            presentation.dataframe(pd.DataFrame([result]).rename(columns={
                'method': 'Method', 'df between': 'df numerator', 'alpha': 'α', EFFECT_SIZE: "Effect Size (Cohen's f²)",
            }).style.format({
                'F': '{:.6g}', 'p-value': lambda p: 'p < 0.001' if p < .001 else f'{p:.4g}', 'df numerator': '{:.6g}',
                'df denominator': '{:.6g}', "Effect Size (Cohen's f²)": '{:.6g}', 'α': '{:.3g}',
            }), hide_index=True)
            presentation.write('Statistically significant at the selected α.' if result['p-value'] < alpha else 'Not statistically significant at the selected α; this does not establish equal population means.')
            presentation.caption(f"Exact p-value: {result['p-value']:.17g}. Statistical significance does not establish practical importance.")
        else:
            presentation.info('Descriptive summaries remain available; validated Welch inference is not available.')
        presentation.markdown('**Variation Decomposition**')
        if variation is not None:
            display = variation.drop(columns=['Mean Square (MS)']).copy()
            display['Source'] = display['Source'].replace({'Within Groups (Residual)': 'Within Groups'})
            total_ss = display.loc[display['Source'] == 'Total', 'Sum of Squares (SS)'].iloc[0]
            display['Variation (%)'] = display['Sum of Squares (SS)'] / total_ss * 100 if total_ss > 0 else float('nan')
            presentation.dataframe(display.style.format({'Variation (%)': '{:.2f}%'}, na_rep='—'), hide_index=True)
            if total_ss == 0:
                presentation.caption('Percentages are unavailable because total observed variation is zero.')
            presentation.caption(VARIATION_NOTE)
            presentation.caption("Welch's F-statistic uses variance-adjusted methods; it is not calculated directly from these conventional sums of squares. These descriptive percentages are not causal contributions or Welch-adjusted effect sizes.")
        else:
            presentation.info('No numerically valid variation decomposition is available.')
    with tabs[1]:
        presentation.markdown('**Group Statistics**')
        presentation.dataframe(prepared['summary'], hide_index=True)
        presentation.caption('Individual group mean intervals use Student-t, mean ± t(0.975, n−1) × SD/√n. These pointwise 95% intervals are not multiplicity-adjusted pairwise intervals.')
        presentation.markdown('**Group Boxplot**')
        if not prepared['summary'].empty:
            plot = prepared['clean'].copy()
            plot[category] = plot[category].astype(str)
            presentation.plotly_chart(px.box(plot, x=category, y=outcome, points='outliers'), key='rel_anova_box')
        else:
            presentation.info('No usable observations to plot.')
        presentation.markdown('**Group Means with 95% Confidence Intervals**')
        means = group_mean_plot(prepared)
        if means is not None:
            presentation.plotly_chart(means, key='rel_anova_means')
            presentation.caption('Pointwise Student-t intervals for each group mean; these are not Games–Howell or multiplicity-adjusted pairwise intervals. Their overlap is not a substitute for pairwise comparisons.')
            if len(means.data[0].x) < len(prepared['summary']):
                presentation.caption('Groups without estimable mean intervals are omitted from this chart only; all groups remain in the summaries and boxplot.')
        else:
            presentation.info('No estimable group mean confidence intervals are available.')
        presentation.markdown('**Games–Howell Pairwise Comparisons**')
        presentation.caption('Compare mean differences using simultaneous confidence intervals and adjusted p-values across the full family of group pairs.')
        _show_pair_plot(pairwise)
        if pairwise is not None and not pairwise['table'].empty:
            table = pairwise['table']
            columns = ['Comparison', 'Mean Difference', 'Adjusted p-value', pairwise['lower'], pairwise['upper'], 'Significant']
            presentation.dataframe(table[columns].style.format({
                'Mean Difference': '{:.6g}', 'Adjusted p-value': '{:.4g}',
                pairwise['lower']: '{:.6g}', pairwise['upper']: '{:.6g}',
            }), hide_index=True, height=420)
            presentation.caption(f'Mean Difference = Mean(Group A) − Mean(Group B). Studentized-range adjusted p-values and {pairwise["confidence"]:.0%} simultaneous intervals use pair-specific variances and Welch–Satterthwaite degrees of freedom. All {len(table)} pairs form the family; an overall significant ANOVA is not required.')
        else:
            presentation.info(pairwise['reason'] if pairwise else 'Games–Howell is unavailable while validated Welch inference is withheld.')
    with tabs[2]:
        presentation.markdown('**Group sizes and variances**')
        presentation.dataframe(pd.DataFrame([
            ('Minimum group size', prepared['min_group_size']), ('Maximum group size', prepared['max_group_size']),
            ('Group-size ratio (maximum/minimum)', prepared['size_ratio']),
            ('Variance ratio (maximum/minimum; positive finite variances only)', prepared['variance_ratio']),
            ('Excluded missing/nonfinite rows (union)', prepared['excluded']),
            ('Missing/nonfinite response rows', prepared['missing_response']), ('Missing factor rows', prepared['missing_factor']),
        ], columns=['Diagnostic', 'Value']), hide_index=True)
        presentation.caption('Missing-factor and missing-response counts can overlap. Undefined ratios do not establish equal variances.')
        for note in prepared['warnings']:
            presentation.warning(note)
        if np.isfinite(prepared['levene_p']):
            presentation.write(f"Median-centered Levene (Brown–Forsythe variance screen): statistic = {prepared['levene_stat']:.4g}; p = {prepared['levene_p']:.4g}.")
        else:
            presentation.info('Variance screening is unavailable.')
        presentation.caption('Levene describes variance differences; its p-value never switches the method. A nonsignificant result does not establish equality.')
        presentation.markdown('**Distribution diagnostics**')
        presentation.dataframe(prepared['distribution'], hide_index=True)
        presentation.caption('Skewness, excess kurtosis and 1.5 × IQR outlier flags are descriptive screens. |Skewness| ≥ 2 or |excess kurtosis| ≥ 7 flags severe shape concerns, not a formal normality decision. No Shapiro–Wilk test or automatic Kruskal–Wallis switch is used.')
        if prepared['groups']:
            index = st.selectbox('Group for normal Q–Q plot', range(len(prepared['groups'])),
                                 format_func=lambda i: str(prepared['labels'][i]), key='rel_anova_qq')
            g = prepared['groups'][index]
            if len(g) >= 3 and np.ptp(g) > 0 and np.all(np.isfinite(g)) and np.isfinite(np.var(g)):
                (x, y), (slope, intercept, _) = stats.probplot(g, dist='norm')
                fig = px.scatter(x=x, y=y, labels={'x': 'Theoretical normal quantile', 'y': str(outcome)})
                fig.add_scatter(x=x, y=intercept+slope*x, mode='lines', name='Normal reference')
                presentation.plotly_chart(fig, key='rel_anova_qq_plot')
        if prepared['n'] and not prepared['errors']:
            presentation.markdown('**Within-group residual patterns**')
            presentation.plotly_chart(residual_plot(prepared), key='rel_anova_residual_plot')
            residual = np.concatenate([g-np.mean(g) for g in prepared['groups']])
            presentation.plotly_chart(px.histogram(x=residual, labels={'x': 'Within-group residual'}), key='rel_anova_residual_hist')
            presentation.caption('Review spread, tails and unusual residuals. Pooling groups with unequal variances can itself produce heavy tails; this is not a test that assumptions hold.')
        presentation.markdown('**Independence assessment**')
        presentation.write(independence['state'])
        presentation.caption(independence['qualification'])
        if not independence['evidence'].empty:
            presentation.dataframe(independence['evidence'], hide_index=True)
        if not independence['serial'].empty:
            presentation.dataframe(independence['serial'], hide_index=True)
            presentation.caption('Serial diagnostics require ≥12 observations with distinct, approximately equally spaced chronological times within each group. Ljung–Box screen p-values are Bonferroni-adjusted across tested groups; Durbin–Watson and lag-1 correlation are descriptive. No serial test is run on arbitrary row order.')
        elif independence['evidence'].empty:
            presentation.write('No usable entity or chronological structure was supplied. Inference proceeds under the independence assumption, with study-design limitations.')
        presentation.caption('Methods: [Welch ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html), [unequal-variance effect size](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.effectsize_oneway.html), [Games–Howell reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html).')
    with tabs[3]:
        presentation.markdown('**Overall findings**')
        if result is None:
            presentation.info('Inferential findings are withheld until numerical and study-design concerns are resolved.')
        elif result['p-value'] < alpha:
            presentation.write(f'At α = {alpha:g}, there is statistical evidence that at least one population group mean differs, under the independence assumption and the limitations below.')
        else:
            presentation.write(f'At α = {alpha:g}, there is insufficient evidence that population group means differ. This does not establish that the means are equal.')
        presentation.write('Between-group variation describes differences between observed group means; within-group variation describes the spread of observations around their own group mean.')
        if pairwise is not None and not pairwise['table'].empty:
            table = pairwise['table']
            significant = table[table.Significant]
            presentation.markdown('**Pairwise findings**')
            if significant.empty:
                presentation.write('No pair shows an adjusted p-value below the selected significance level. This does not establish equal means.')
            else:
                presentation.write(f'{len(significant)} of {len(table)} comparisons show evidence of a mean difference after Games–Howell adjustment.')
                findings = [{'Comparison': row['Comparison'],
                             'Finding': f"{row['Group A']} has a {'higher' if row['Mean Difference'] > 0 else 'lower'} mean than {row['Group B']} by {abs(row['Mean Difference']):.4g} response units.",
                             'Adjusted p-value': row['Adjusted p-value']} for _, row in significant.iterrows()]
                presentation.dataframe(pd.DataFrame(findings), hide_index=True, height=420)
        if result:
            presentation.write(f'Variance-weighted effect size f² = {result[EFFECT_SIZE]:.4g}. Statistical significance does not establish practical importance.')
        presentation.markdown('**Important limitations**')
        presentation.write(independence['state'] + '. ' + independence['qualification'])
        if design == 'independent':
            presentation.warning('Results rely on your clarification that the detected labels/order do not represent dependent units.')
        for note in prepared['warnings']:
            presentation.warning(note)
        presentation.caption('Welch and Games–Howell allow unequal variances; they do not repair dependence, severe skew or outliers in small samples. Association is not causation. Adjustment covers this pairwise family, not later exploration of other outcomes or factors.')
        ai_context = guidance_context(prepared, independence, result, pairwise, metadata)
        digest = hashlib.sha256(json.dumps(ai_context, sort_keys=True, default=str).encode()).hexdigest()
        with st.expander('🤖 AI Interpretation'):
            saved = st.session_state.get('rel_anova_ai', {})
            if st.button('Refresh interpretation' if saved.get('context') == digest else 'Generate interpretation',
                         key='rel_anova_ai_generate'):
                try:
                    from utils.ai_interpreter import interpret_anova
                    with st.spinner('Explaining the validated summaries…'):
                        text = interpret_anova(ai_context)
                    st.session_state['rel_anova_ai'] = {'context': digest, 'text': text}
                except Exception:
                    st.session_state.pop('rel_anova_ai', None)
                    presentation.info('AI guidance is unavailable. The statistical results and deterministic explanations remain available.')
            saved = st.session_state.get('rel_anova_ai', {})
            if saved.get('context') == digest:
                presentation.write(saved['text'])
                presentation.caption('AI explanation is advisory. Eligibility, calculations and limitations are determined by the statistical engine.')
    if result is not None:
        saved = st.session_state.get('rel_anova_ai', {})
        offer_result('One-way ANOVA', str(outcome)+' by '+str(category),
                     {'response': outcome, 'factor': category},
                     {'method': 'Welch', 'alpha': alpha, 'independence_state': independence['state'], 'design_clarification': design},
                     {**result, 'pairwise comparisons': len(pairwise['table']),
                      'pairwise status': pairwise['reason']},
                     {'groups': prepared['summary'], 'distribution': prepared['distribution'],
                      'independence': independence['evidence'], 'serial': independence['serial']},
                     prepared['warnings'] + [independence['qualification']],
                     interpretation=saved.get('text', '') if saved.get('context') == digest else '', key='anova')
