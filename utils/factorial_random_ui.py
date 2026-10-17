"""Factorial REML results in the existing guided results navigation."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from utils import presentation
from utils.anova_effects import model_family
from utils.factorial_interactions import support_table, random_assessment
from utils.factorial_interactions_ui import insights, refinement, INTERACTION_TYPES
from utils.factorial_guidance_ui import render_items, navigate
from utils.guided_diagnostics import Diagnostic, diagnostic_status
from utils.numerical_formatting import REGRESSION, format_plot_hover
from utils.report_capture import begin_result, offer_result
from utils.factorial_study_design_ui import render_study_design


def render_factorial_random(df, outcome, effects, interactions, alpha, mode, context):
    from utils.anova_effects_ui import _fit
    factors = list(effects)
    panel_status = st.session_state.get('fac_design_panel_status', 'review')
    with st.expander({'review': 'Review advised', 'confirmed': 'Study Design', 'issue': 'Model issue'}[panel_status]):
        study = render_study_design(df, outcome, effects, interactions)
    ready = study['decision']['fit'] and study['decision']['inference']
    next_status = 'confirmed' if ready else 'issue' if study['decision']['severity'] == 'blocking' else 'review'
    if next_status != panel_status:
        st.session_state['fac_design_panel_status'] = next_status
        st.rerun()
    if not ready:
        st.session_state.pop('report_latest', None)
        return
    tabs = st.tabs(['Model Overview', 'Diagnostics', 'Effect plots', 'Post-hoc comparisons', 'ANOVA Findings'],
                   key='fac_results_tab', on_change='rerun')
    begin_result()
    from utils.report_capture import record_text
    for item in study['items']:
        record_text(item.observed_result + ' ' + item.explanation)
    parents, effective = study['parents'], study['effects']
    answer = study['answers']['collection']
    prepared, result, failure = None, None, None
    structure = (study['answers']['context'], tuple(effective.items()), tuple(parents.items()))
    if ready:
        st.session_state['fac_design_applied_structure'] = structure
        try:
            # The engine expands explicit nesting and checks projected covariance
            # kernels; child × parent duplication is blocked, never crossed silently.
            with st.spinner('Checking interaction identifiability and fitting the REML model…'):
                prepared, result = _fit(df, outcome, effective, tuple(interactions), parents, alpha)
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            failure = str(exc)
    items = list(study['items'])

    def diagnostic(key, severity, title, found, why, action, target='terms'):
        items.append(Diagnostic(key, 'model', severity, title, found, why, action,
                                'review' if target else None, target))

    if failure:
        diagnostic('random_estimability', 'blocking', 'Model Estimability', failure,
                   'The requested joint model is unavailable; no partial full-model results or interaction p-values are presented.',
                   'Review the listed interactions, nested ownership and replication. Collect missing information or confirm a scientifically justified revision.')
    else:
        diagnostic('random_estimability', 'ok', 'Model Estimability',
                   f"{prepared['n']} usable observations; fixed design rank {prepared['fixed_rank']}; residual-subspace dimension {prepared['residual_df']}; "
                   f"covariance-kernel rank {prepared['covariance_rank']} (including residual variance).",
                   'Fixed coefficients and variance components pass separate rank checks before fitting; rank alone does not ensure precise estimates.',
                   'Review support, boundary status and residuals before interpretation.')
        diagnostic('random_coverage', 'review' if prepared['empty_cells'] or prepared['singleton_cells'] else 'ok', 'Cell Coverage',
                   f"{prepared['observed_cells']} observed / {prepared['possible_cells']} possible combinations; {prepared['empty_cells']} empty; {prepared['singleton_cells']} single observations. Excluded rows: {prepared['excluded']}.",
                   'Sparse combinations limit precision. Missing combinations may be structural in nested designs; full crossed coverage is not assumed.',
                   'Review counts below and collect replicated combinations where scientifically possible.', None)
        for index, note in enumerate(result['warnings']):
            diagnostic('random_fit_' + str(index), 'review', 'Fit and Uncertainty', note,
                       'Convergence alone does not establish reliable inference.',
                       'Review boundary components, sampling and model complexity before using estimates.')
        diagnostic('random_uncertainty', 'review', 'Interaction Uncertainty',
                   'Random interaction estimates are descriptive. Component confidence intervals and calibrated boundary tests are unavailable.',
                   'A positive variance or percentage does not establish practical importance; no ordinary chi-square or OLS F test is substituted.',
                   'Assess magnitudes in context and seek additional replicated groups or externally validated component inference.', None)
    status = diagnostic_status(items)
    with tabs[0]:
        presentation.subheader('Model Overview')
        presentation.markdown('**Model summary**')
        presentation.table(pd.DataFrame([
            ('Response / Dependent Variable', str(outcome)), ('Factors / Independent Variables', ', '.join(map(str, factors))),
            ('Effect types', ', '.join(f'{f}: {v}' for f, v in effective.items())),
            ('Analysis method', model_family(effective)), ('Estimation method', 'REML'),
            ('Interaction selection', mode), ('Significance level', str(alpha)),
        ], columns=['Model detail', 'Value']))
        if status == 'Model issue':
            st.error('Model issue — the requested model is unavailable. Review Diagnostics.')
        else:
            presentation.write('Diagnostic status: **' + status + '**')
        st.button('Open Diagnostics', key='fac_open_diagnostics', on_click=navigate, args=('overview',))
    with tabs[1]:
        render_items([i for i in items if i.category != 'study_design'])
        if interactions:
            presentation.write('Requested interactions: ' + ', '.join(' × '.join(map(str, p)) for p in interactions))
            support = support_table(df, outcome, effects, interactions)
            if failure:
                support['Assessment'] = 'Not estimable in requested model'
                support['Reason'] = failure
            presentation.dataframe(support, hide_index=True)
            presentation.caption('Pair counts combine measurements across other factors. Cozy also checks whether the full model can distinguish its sources of variation.')
    if result is None:
        st.session_state.pop('report_latest', None)
        for tab in (tabs[2], tabs[3], tabs[4]):
            with tab:
                presentation.info('Results await study-design confirmation or resolution of the model issue. Review Diagnostics; no terms were dropped.')
        return
    result = {**result, 'alpha': alpha, 'study_design': study['answers']}
    assessment = random_assessment(prepared, result, interactions, alpha)
    diagnostics = [i.__dict__ for i in items]
    run_context = context + repr(study['answers'])
    with tabs[0]:
        if not study['decision']['inference']:
            presentation.info('Conditional exploration only: measurement collection remains unconfirmed. Fixed-effect p-values, confidence intervals and inferential interaction conclusions are withheld. Review Diagnostics.')
        presentation.markdown('**Variance Components**')
        presentation.dataframe(result['variance'], result_types={**INTERACTION_TYPES, 'Standard deviation': 'sd'}, hide_index=True)
        chart = px.bar(result['variance'], x='Component', y='Percentage of modeled variance',
                       title='Contribution to modeled variance')
        format_plot_hover(chart, {'y': 'percentage'})
        presentation.plotly_chart(chart, key='fac_variance_components')
        if 'Fixed' in effective.values():
            presentation.markdown('**Fixed Effects**')
            presentation.dataframe(result['fixed'], result_types={**REGRESSION, 'p-value (asymptotic)': 'p',
                                                                  'Wald z (asymptotic)': 'z'}, alpha=alpha, hide_index=True)
            presentation.caption('When available, coefficient inference uses asymptotic normal Wald intervals and tests. Fixed-interaction omnibus tests use asymptotic Wald chi-square, with Holm adjustment across fixed interactions. These are not OLS Type II F tests; small-sample corrected inference and mixed-model adjusted means/simple-effects comparisons are not implemented.')
        insights(prepared, result, assessment, mode, random=True)
        refinement(prepared, result, random=True, context=run_context)
    with tabs[1]:
        presentation.dataframe(prepared['groups'], hide_index=True)
        presentation.dataframe(prepared['relationships'], hide_index=True)
        presentation.caption(f"Converged: {result['converged']}. Positive-definite information: {result['information_positive_definite']}. Near a variance boundary: {result['near_boundary']}.")
        presentation.subheader('Conditional Residual Diagnostics')
        presentation.plotly_chart(px.scatter(x=result['fitted'], y=result['residuals'],
            labels={'x': 'Conditional fitted response', 'y': 'Conditional residual'}, title='Residuals versus fitted response'), key='fac_random_residuals')
        (x, y), (slope, intercept, _) = stats.probplot(result['residuals'], dist='norm')
        qq = px.scatter(x=x, y=y, labels={'x': 'Theoretical normal quantile', 'y': 'Conditional residual'}, title='Conditional residual Q–Q plot')
        qq.add_scatter(x=x, y=intercept+slope*x, mode='lines', name='Normal reference')
        presentation.plotly_chart(qq, key='fac_random_qq')
        presentation.caption('Inspect curves, outliers and unequal spread. Gaussian random effects and residuals, common residual variance and conditional independence remain assumptions. Convergence does not verify them.')
    with tabs[2]:
        presentation.caption('These observed distributions are descriptive. Variance-component estimates above provide the model-based view of random interactions; crossing lines alone do not establish a random variance component.')
        if interactions:
            pair = st.selectbox('Combination distribution', interactions, key='fac_random_plot_pair',
                                format_func=lambda p: ' × '.join(map(str, p)))
            raw = prepared['raw'][[*pair, outcome]].copy()
            raw.columns = ['First factor', 'Second factor', 'Response']
            presentation.plotly_chart(px.box(raw, x='First factor', color='Second factor', y='Response', title=f'Observed {outcome} by factor combination'), key='fac_random_combinations')
    with tabs[3]:
        presentation.info('Model-based mixed/random pairwise and simple-effects inference is not implemented. Review fixed coefficients and descriptive combination plots; Games–Howell and OLS post-hoc tests are not substituted.')
    with tabs[4]:
        interpretation = insights(prepared, result, assessment, mode, random=True, diagnostics=diagnostics, ai=True)
    offer_result('ANOVA — ' + model_family(effective), str(outcome) + ' by ' + ', '.join(map(str, factors)),
                 {'response': outcome, 'factors': factors},
                 {'effects': effective, 'selected_effects': effects, 'parents': parents, 'study_design': study['answers'],
                  'interactions': interactions, 'interaction_selection': mode,
                  'alpha': alpha, 'dependence': answer, 'method': 'REML'},
                 {**result, 'interaction_assessment': assessment}, {'guided_items': diagnostics},
                 [i.observed_result for i in items if i.severity != 'ok'], interpretation,
                 key='fac_random', run_id=run_context)
