"""Guided diagnostic rendering; interactive controls are excluded from reports."""
import streamlit as st
from utils import presentation
from utils.factorial_guidance import coverage_table, INDEPENDENT, REPEATED, UNSURE


def navigate(target):
    st.session_state['fac_results_tab'] = 'Model Overview' if target in ('terms', 'hc3') else 'Diagnostics'
    st.session_state['fac_diagnostic_focus'] = target
    if target == 'terms':
        st.session_state['fac_review_terms'] = True


def render_items(items, prepared=None, evidence=None):
    labels = {'ok': 'No concern detected', 'review': 'Review advised',
              'unresolved': 'Study design unresolved', 'blocking': 'Model issue'}
    from utils.report_capture import record_text
    for item in items:
        record_text(item.title, heading=True)
        with st.expander(item.title + ' · ' + labels[item.severity], expanded=item.severity != 'ok'):
            presentation.write('What was found: ' + item.observed_result)
            presentation.write('Why it matters: ' + item.explanation)
            presentation.write('Recommended action: ' + item.recommended_action)
            if item.action_target == 'coverage':
                st.button('Review Cell Coverage', key='fac_action_' + item.diagnostic_id,
                          on_click=navigate, args=('coverage',))
            elif item.action_target == 'terms':
                st.markdown('[Review Interactions / Model Terms](#fac-model-configuration)')
            elif item.action_target == 'residuals':
                st.markdown('[View residual plots](#residual-plots-and-variance-details)')
            elif item.action_target == 'variance':
                st.button('Review HC3 sensitivity option', key='fac_action_hc3', on_click=navigate, args=('hc3',))
        if item.diagnostic_id == 'cell_coverage' and prepared is not None:
            render_coverage(prepared)
        if item.diagnostic_id == 'independence' and evidence is not None:
            render_study_design(evidence)


def render_coverage(prepared):
    with st.expander('Cell-count details', expanded=st.session_state.get('fac_diagnostic_focus') == 'coverage'):
        pairs = prepared['interactions'] or ([tuple(prepared['factors'])] if len(prepared['factors']) == 2 else [])
        for index, pair in enumerate(pairs):
            table, limited = coverage_table(prepared, pair)
            if not limited and len(table) <= 400:
                matrix = table.pivot(index=pair[0], columns=pair[1], values='Observations').fillna(0).astype(int)
                # Use theme-compatible muted backgrounds; the text counts remain accessible.
                styled = matrix.style.map(lambda value: 'background-color: rgba(210,150,40,0.22)' if value == 0
                                          else 'background-color: rgba(210,150,40,0.10)' if value <= 2 else '')
                presentation.write(' × '.join(map(str, pair)) + ' — observed counts (other factors pooled)')
                presentation.dataframe(styled)
        table, limited = coverage_table(prepared)
        if limited:
            presentation.caption('The possible design exceeds 5,000 combinations. Only observed combinations are listed; the total empty count is reported above. No full Cartesian allocation is made.')
        selection = st.selectbox('Cell-count view', ['All listed combinations', 'Empty', 'Single observation', 'Two observations'], key='fac_cell_filter')
        shown = table if selection == 'All listed combinations' else table[table.Coverage == selection]
        presentation.dataframe(shown, hide_index=True, width='stretch')
        presentation.caption('Shading marks empty cells and cells with one or two observations. Two is a descriptive replication band, not a requirement for valid inference.')


def render_study_design(evidence):
    presentation.subheader('Check Study Design')
    if not evidence.empty:
        presentation.dataframe(evidence, hide_index=True, width='stretch')
    else:
        presentation.write('No named identifier, grouping or time indicators were detected. Their absence does not establish independence.')
    st.selectbox('Were multiple measurements collected from the same product, batch, subject, or other experimental unit?',
                 [None, INDEPENDENT, REPEATED, UNSURE], index=0,
                 format_func=lambda value: value or 'Choose the study design', key='fac_study_answer')
    presentation.caption('Your answer is a study-design assumption for this analysis configuration, not a statistical verification. Uncertainty does not prevent reviewing otherwise valid results.')
    if st.session_state.get('fac_study_answer') == REPEATED:
        presentation.write('This independent-observations ANOVA does not account for the reported dependence. Use the existing effect-type controls to assess a supported mixed model and confirm the grouping structure before fitting. A repeated treatment label alone does not establish the grouping unit.')
        st.markdown('[Review fixed/random effect controls](#fac-model-configuration)')
