"""Effect controls and guided Gaussian variance-component results."""
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import presentation
from utils.anova_effects import model_family, sync_effects, prepare_random, fit_random
from utils.report_capture import begin_result, offer_result


def effect_controls(factors, outcome, prefix):
    context = (st.session_state.get('file_id'), outcome)
    base = prefix + '_effects'
    if st.session_state.get(base + '_context') != context:
        for key in list(st.session_state):
            if key.startswith(base):
                del st.session_state[key]
        st.session_state[base + '_context'] = context
    previous = st.session_state.get(base, {})
    for factor in set(previous)-set(factors):
        st.session_state.pop(base + '_' + hashlib.sha256(str(factor).encode()).hexdigest()[:16], None)
    effects = sync_effects(previous, factors)
    presentation.caption('Are you comparing these specific groups, or understanding variation across a broader population of groups? Fixed compares selected groups (for example, Machines A, B and C). Random estimates population variability (for example, machine-to-machine variation among sampled machines). Neither the number of groups nor an identifier name determines the choice.')
    for factor in factors:
        key = base + '_' + hashlib.sha256(str(factor).encode()).hexdigest()[:16]
        if key not in st.session_state:
            st.session_state[key] = effects[factor]
        effects[factor] = st.radio(f'{factor} — Effect Type', ['Fixed', 'Random'], horizontal=True, key=key)
    st.session_state[base] = effects
    if len(factors) > 1:
        presentation.write('Automatically selected model: **' + model_family(effects) + '**')
    elif 'Random' in effects.values():
        presentation.write('**' + model_family(effects) + '**')
    return effects


@st.cache_data(show_spinner=False, max_entries=8)
def _fit(df, outcome, effects, interactions, parents, alpha):
    prepared = prepare_random(df, outcome, effects, interactions, parents)
    return prepared, fit_random(prepared, alpha)


def render_random(df, outcome, effects, interactions, alpha, prefix):
    factors = list(effects)
    # Every structure change gets new confirmation widgets; unrelated reruns retain answers.
    signature = hashlib.sha256(repr((st.session_state.get('file_id'), outcome, effects, interactions)).encode()).hexdigest()[:16]
    base = prefix + '_random_' + signature
    for key in list(st.session_state):
        if key.startswith(prefix + '_random_') and not key.startswith(base):
            del st.session_state[key]
    parents, unresolved = {}, False
    presentation.caption('Random terms use independent Gaussian group intercepts and one common residual variance. Confirm how group labels identify real groups. Missing combinations alone do not establish nesting.')
    for i, factor in enumerate(factors):
        if effects[factor] != 'Random' or len(factors) == 1:
            continue
        others = [f for f in factors if f != factor]
        options = ['Choose how labels identify groups', 'The same label identifies the same group everywhere'] + [f'Each {factor} belongs to one {p}; labels identify groups within that parent' for p in others]
        choice = st.selectbox(f'How do {factor} labels relate to the other factors?', options, key=base + f'_parent_{i}')
        if choice == options[0]:
            unresolved = True
        elif choice in options[2:]:
            parents[factor] = others[options.index(choice)-2]
    selected = df[factors].dropna()
    counts = [{'Factor': str(f), 'Levels': selected[f].nunique(),
               'Minimum observations per label': int(selected[f].value_counts().min()) if len(selected) else 0,
               'Maximum observations per label': int(selected[f].value_counts().max()) if len(selected) else 0} for f in factors]
    presentation.dataframe(pd.DataFrame(counts), hide_index=True)
    from utils.anova_independence import assess_independence
    clean = df[[outcome] + factors].apply(lambda s: s.replace([float('inf'), -float('inf')], float('nan'))).dropna().reset_index()
    evidence = assess_independence(df.reset_index(drop=True), {'clean': clean.set_index('index'), 'category': factors[0], 'outcome': outcome}, st.session_state.get('variable_metadata', {}))
    if not evidence['evidence'].empty:
        presentation.dataframe(evidence['evidence'], hide_index=True)
    answer = st.selectbox('After accounting for the selected random groups, what dependence remains between observations?',
        ['Choose the study design', 'None expected: group intercepts and selected interactions describe the dependence',
         'Effects vary within the same entity and need correlated slopes', 'Time, repeated measurements, or another grouping needs additional modeling', 'I do not know'], key=base + '_dependence')
    presentation.caption(('For factorial designs, if treatment effects vary across sampled groups, select the relevant factor × group interaction above. Independent interaction components are supported; correlated random slopes, serial correlation and unequal residual variances are not. Do not omit necessary terms to obtain a fit.') if len(factors) > 1 else 'This one-factor model accounts for shared group intercepts only. If other factors or within-entity changes are needed, select them in Comparison and use a supported factorial structure.')
    tabs = st.tabs(['Overview', 'Group Comparisons', 'Diagnostics', 'ANOVA Findings']) if prefix == 'rel_anova' else None
    if unresolved or answer != 'None expected: group intercepts and selected interactions describe the dependence':
        if tabs:
            with tabs[0]:
                presentation.info('Model awaiting study-design confirmation. No random-effects estimates are available.')
            for tab in tabs[1:]:
                with tab:
                    presentation.info('Results will be available after a valid model is fitted.')
        presentation.info('Confirm group identity and dependence before fitting. Correlated slopes or additional repeated-measure structure are not supported here; select all necessary grouping variables or use a suitable mixed/repeated-measures analysis.')
        return
    try:
        with st.spinner('Checking estimability and fitting the requested REML model…'):
            prepared, result = _fit(df, outcome, effects, tuple(interactions), parents, alpha)
    except (ValueError, TypeError) as exc:
        if tabs:
            with tabs[0]:
                presentation.warning('Model unavailable: ' + str(exc))
            for tab in tabs[1:]:
                with tab:
                    presentation.info('No fitted model results are available. Review the model status in Overview.')
        presentation.warning('Cannot report this model: ' + str(exc))
        return
    if tabs:
        return _render_oneway_random(prepared, result, outcome, effects, interactions, parents, alpha, answer, prefix, evidence, tabs)
    begin_result()
    presentation.write('**' + model_family(effects) + '** · REML · ' + str(result['n']) + ' observations')
    presentation.caption(f"Excluded missing/nonfinite observations: {prepared['excluded']}. Balanced full crossing: {prepared['balanced']}. Unobserved combinations: {prepared['empty_cells']} (some may be structurally absent in a nested design).")
    presentation.dataframe(prepared['groups'], hide_index=True)
    if not prepared['relationships'].empty:
        presentation.dataframe(prepared['relationships'], hide_index=True)
    presentation.caption(f"Converged: {result['converged']}. Positive-definite information: {result['information_positive_definite']}. Near a variance boundary: {result['near_boundary']}.")
    for note in result['warnings']:
        presentation.warning(note)
    presentation.subheader('Random effects')
    presentation.dataframe(result['variance'], hide_index=True)
    presentation.write(f"Total modeled variance: {result['total_variance']:.5g}")
    if result['icc'] is not None:
        presentation.write(f"ICC: {result['icc']:.4f} — correlation between two observations sharing the same random group.")
    else:
        presentation.caption('There is no single ICC for this structure. The correlation for two observations is the sum of variances of the components they share, divided by total modeled variance. Percentages are conditional on the fixed predictors.')
    presentation.caption('Variance-component confidence intervals and significance tests are not supplied. Zero is a boundary; an ordinary chi-square likelihood-ratio test is not valid here.')
    if 'Fixed' in effects.values():
        presentation.subheader('Fixed effects')
        presentation.dataframe(result['fixed'], hide_index=True, column_config={'p-value (asymptotic)': st.column_config.NumberColumn(format='%.4g')})
        presentation.caption(f'{1-alpha:.1%} asymptotic normal Wald intervals and two-sided z tests, when available; these are large-sample coefficient tests, not Type II F tests or small-sample corrected inference. Reference levels use first observed order. Estimates compare the selected fixed groups conditional on random variation. Small group counts can make inference unreliable even above the five-group suppression threshold.')
        reference = pd.DataFrame([{'Factor': str(f), 'Reference group': str(prepared['raw'][f].iloc[0])} for f in factors if effects[f] == 'Fixed'])
        presentation.dataframe(reference, hide_index=True)
        if result['inference']:
            plot = result['fixed'].copy()
            plot['Interval'] = plot['Upper CI']-plot['Estimate']
            presentation.plotly_chart(px.scatter(plot, x='Estimate', y='Coefficient', error_x='Interval', title='Fixed-effect estimates'))
    presentation.plotly_chart(px.bar(result['variance'].assign(Component=lambda frame: frame.Component.replace({'Residual (within groups)': 'Residual'})), x='Component', y='Variance', title='Estimated variance components'))
    factor = st.selectbox('Distribution by factor', factors, key=base + '_plot')
    presentation.plotly_chart(px.box(prepared['raw'], x=factor, y=outcome, title='Response distributions'))
    presentation.plotly_chart(px.scatter(x=result['fitted'], y=result['residuals'], labels={'x': 'Conditional fitted response', 'y': 'Conditional residual'}, title='Residuals versus fitted response'))
    from scipy import stats
    (quantiles, ordered), (slope, intercept, _) = stats.probplot(result['residuals'], dist='norm')
    qq = px.scatter(x=quantiles, y=ordered, labels={'x': 'Theoretical normal quantile', 'y': 'Conditional residual'}, title='Residual normal Q–Q plot')
    qq.add_scatter(x=quantiles, y=intercept+slope*quantiles, mode='lines', name='Normal reference')
    presentation.plotly_chart(qq)
    presentation.caption('Review residuals for unequal spread, curves and outliers. Gaussian random effects, Gaussian residuals, common residual variance and conditional independence are assumptions, not conclusions from convergence.')
    interpretation = ('This model compares the selected fixed groups while accounting for sampled-group variation.' if 'Fixed' in effects.values() else 'This model estimates variation among groups representing a broader population.') + ' Random variances are not differences between particular group means. Population generalization depends on sampling; these results do not establish causation.'
    presentation.write(interpretation)
    offer_result('ANOVA — ' + model_family(effects), str(outcome) + ' by ' + ', '.join(map(str, factors)),
                 {'response': outcome, 'factors': factors}, {'effects': effects, 'parents': parents, 'interactions': interactions, 'alpha': alpha, 'dependence': answer},
                 result, {'groups': prepared['groups'], 'balanced': prepared['balanced'], 'empty_cells': prepared['empty_cells']}, result['warnings'], interpretation, key=prefix + '_random')


def _render_oneway_random(prepared, result, outcome, effects, interactions, parents, alpha, answer, prefix, evidence, tabs):
    """One-way navigation only; the shared factorial renderer stays unchanged."""
    from scipy import stats
    from utils.anova_analysis import prepare_anova
    from utils.anova_charts import group_mean_plot
    factor = next(iter(effects))
    descriptive = prepare_anova(prepared['raw'], factor, outcome)
    begin_result()
    with tabs[0]:
        presentation.markdown('**Analysis Summary**')
        presentation.table(pd.DataFrame([
            ('Response variable', str(outcome)), ('Grouping factor', str(factor)),
            ('Effect type', 'Random'), ('Model family', model_family(effects)),
            ('Estimation method', 'REML'), ('Observations used', result['n']),
            ('Number of groups', len(descriptive['groups'])), ('Significance level (α)', alpha),
        ], columns=['Model detail', 'Value']))
        presentation.caption(f"Excluded missing/nonfinite observations: {prepared['excluded']}.")
        presentation.markdown('**Variance Components**')
        presentation.dataframe(result['variance'], hide_index=True)
        presentation.write(f"Total modeled variance: {result['total_variance']:.5g}")
        if result['icc'] is not None:
            presentation.write(f"ICC: {result['icc']:.4f} — correlation between two observations sharing the same random group.")
        presentation.caption('Variance-component confidence intervals and significance tests are not supplied. Zero is a boundary; an ordinary chi-square likelihood-ratio test is not valid here.')
        presentation.plotly_chart(px.bar(result['variance'].assign(Component=lambda frame: frame.Component.replace({'Residual (within groups)': 'Residual'})), x='Component', y='Variance', title='Estimated variance components'))
    with tabs[1]:
        presentation.markdown('**Group Statistics**')
        presentation.dataframe(descriptive['summary'], hide_index=True)
        presentation.markdown('**Group Boxplot**')
        presentation.plotly_chart(px.box(prepared['raw'], x=factor, y=outcome, points='outliers', title='Response distributions'))
        presentation.markdown('**Group Means with 95% Confidence Intervals**')
        means = group_mean_plot(descriptive)
        if means is not None:
            presentation.plotly_chart(means)
        else:
            presentation.info('No estimable group mean confidence intervals are available.')
        presentation.caption('Descriptive, individual Student-t intervals for observed group means; these are not model-based pairwise inferences. Games–Howell is not reported as a random-effects conclusion. Model-appropriate pairwise inference is not implemented.')
    with tabs[2]:
        presentation.markdown('**Data and Group Checks**')
        presentation.dataframe(prepared['groups'], hide_index=True)
        presentation.caption(f"Excluded missing/nonfinite observations: {prepared['excluded']}. Balanced full crossing: {prepared['balanced']}. Unobserved combinations: {prepared['empty_cells']}.")
        if not prepared['relationships'].empty:
            presentation.dataframe(prepared['relationships'], hide_index=True)
        presentation.caption(f"Converged: {result['converged']}. Positive-definite information: {result['information_positive_definite']}. Near a variance boundary: {result['near_boundary']}.")
        for note in result['warnings']:
            presentation.warning(note)
        presentation.markdown('**Conditional Residual Diagnostics**')
        presentation.plotly_chart(px.scatter(x=result['fitted'], y=result['residuals'], labels={'x': 'Conditional fitted response', 'y': 'Conditional residual'}, title='Residuals versus fitted response'))
        (quantiles, ordered), (slope, intercept, _) = stats.probplot(result['residuals'], dist='norm')
        qq = px.scatter(x=quantiles, y=ordered, labels={'x': 'Theoretical normal quantile', 'y': 'Conditional residual'}, title='Residual normal Q–Q plot')
        qq.add_scatter(x=quantiles, y=intercept+slope*quantiles, mode='lines', name='Normal reference')
        presentation.plotly_chart(qq)
        presentation.markdown('**Independence Assessment**')
        presentation.write(evidence['state'])
        presentation.caption(evidence['qualification'])
        if not evidence['evidence'].empty:
            presentation.dataframe(evidence['evidence'], hide_index=True)
        presentation.caption('Review residuals for unequal spread, curves and outliers. Gaussian random effects, Gaussian residuals, common residual variance and conditional independence are assumptions, not conclusions from convergence. No dedicated random-effect normality diagnostic is implemented.')
    interpretation = 'This model estimates variation among groups representing a broader population. Random variances are not differences between particular group means. Population generalization depends on sampling; these results do not establish causation.'
    with tabs[3]:
        presentation.markdown('**Overall Findings**')
        presentation.write(interpretation)
        for _, row in result['variance'].iterrows():
            presentation.write(f"{row['Component']}: estimated variance {row['Variance']:.5g}.")
        if result['icc'] is not None:
            presentation.write(f"ICC = {result['icc']:.4f}: the estimated share of modeled variance associated with shared group membership.")
        presentation.markdown('**Reliability and Limitations**')
        for note in result['warnings']:
            presentation.warning(note)
        presentation.caption('REML estimates rely on the selected group structure, conditional independence and a common residual variance. Variance components do not establish fixed group-mean differences or causation. Review the diagnostic plots and the study design.')
    offer_result('ANOVA — ' + model_family(effects), str(outcome) + ' by ' + str(factor),
                 {'response': outcome, 'factors': [factor]}, {'effects': effects, 'parents': parents, 'interactions': interactions, 'alpha': alpha, 'dependence': answer},
                 result, {'groups': prepared['groups'], 'balanced': prepared['balanced'], 'empty_cells': prepared['empty_cells']}, result['warnings'], interpretation, key=prefix + '_random')
