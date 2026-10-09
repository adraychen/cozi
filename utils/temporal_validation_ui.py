"""Contextual guidance that never constructs columns during analysis."""
import streamlit as st
from utils import presentation
from utils.temporal_validation import assess_temporal_selection
from utils.temporal_navigation import open_component_transform, select_existing_temporal


def guide_temporal_selection(df, column, metadata, context, key_prefix, allow_seasonal=False):
    assessment = assess_temporal_selection(df, column, metadata, dimensions=context.get('categorical', []))
    objective = 'Chronological reporting periods'
    if assessment.kind != 'Month':
        return True, assessment, objective
    if assessment.status == 'Blocked' and assessment.year_column is None and len(assessment.year_columns) > 1:
        presentation.warning(assessment.message)
        year_key = key_prefix + '_year'
        options = [None] + assessment.year_columns
        if st.session_state.get(year_key) not in options:
            st.session_state.pop(year_key, None)
        chosen = st.selectbox('Corresponding year column', options,
                              format_func=lambda value: 'Choose the Year column…' if value is None else value, key=year_key)
        if chosen is not None:
            assessment = assess_temporal_selection(df, column, metadata, dimensions=context.get('categorical', []), year_column=chosen)
    if assessment.status == 'Blocked' and allow_seasonal:
        presentation.warning(assessment.message)
        objective = st.selectbox('Analysis objective', ['Chronological reporting periods', 'Monthly seasonal profile (pool years)'],
                                 key=key_prefix + '_objective')
        if objective == 'Monthly seasonal profile (pool years)':
            # The explicit pooled objective does not require choosing a calendar year.
            presentation.caption('Years are pooled intentionally. These comparisons describe a seasonal month profile, not dated period changes.')
            return True, assessment, objective
    if assessment.status == 'Blocked':
        if not allow_seasonal:
            presentation.warning(assessment.message)
        if assessment.existing_columns:
            presentation.info('Select ' + ', '.join(assessment.existing_columns) + ' to compare reporting periods in chronological order.')
            for index, existing in enumerate(assessment.existing_columns[:5]):
                st.button('Use ' + str(existing), key='temporal_select_button_' + key_prefix + '_' + str(index),
                          on_click=select_existing_temporal, args=(st.session_state, existing, context))
        else:
            presentation.info('Create a Month-Year variable in Transform to distinguish reporting periods.')
            if state_has_workspace():
                st.button('Create Month-Year Variable', key='temporal_create_button_' + key_prefix,
                          on_click=open_component_transform,
                          args=(st.session_state, column, assessment.year_column, context))
        return False, assessment, objective
    if assessment.status == 'Warning':
        presentation.caption(assessment.message)
        if not assessment.years:
            objective = 'Month-of-year comparison (year unavailable)'
    return True, assessment, objective


def state_has_workspace():
    return st.session_state.get('ws_active') in st.session_state.get('ws_datasets', {})
