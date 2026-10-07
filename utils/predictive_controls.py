"""Grouped single-target controls for predictive analytics."""
import hashlib
import streamlit as st


def target_key(column):
    return 'pa_target_check_' + hashlib.sha256(repr(column).encode()).hexdigest()[:16]


def choose_target(column, options):
    st.session_state['pa_target_select'] = column
    for option in options:
        st.session_state[target_key(option)] = option == column


def render_target_selector(options, col_types):
    target = st.session_state.get('pa_target_select')
    if target not in options:
        target = options[0] if options else None
    st.session_state['pa_target_select'] = target
    st.caption('Choose one target. Checking another column switches the target.')
    for title, kind in [('Numerical', 'numerical'), ('Categorical', 'categorical'),
                        ('Temporal', 'datetime')]:
        members = col_types.get('temporal', col_types[kind]) if kind == 'datetime' else col_types[kind]
        columns = [column for column in options if column in members]
        st.markdown(f'**{title}**')
        if not columns:
            st.caption('No available columns.')
        for start in range(0, len(columns), 4):
            row = st.columns(4)
            for container, column in zip(row, columns[start:start + 4]):
                key = target_key(column)
                st.session_state[key] = column == target
                container.checkbox(str(column), key=key, on_change=choose_target,
                                   args=(column, options))
    return target
