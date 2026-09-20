"""Checkbox grids for single- and multiple-variable analytical roles."""
import streamlit as st


def variable_checkboxes(label, options, prefix, single=False, default=None, hide_empty=False):
    options = list(options)
    state_key = prefix + '_selected'
    if state_key not in st.session_state:
        st.session_state[state_key] = list(default or [])
    selected = [c for c in st.session_state[state_key] if c in options]
    if single:
        selected = selected[:1]
    st.session_state[state_key] = selected

    def changed(column):
        checked = st.session_state[prefix + '_column_' + str(column)]
        values = st.session_state[state_key]
        st.session_state[state_key] = ([column] if checked else []) if single else (
            list(dict.fromkeys(values + [column])) if checked else [v for v in values if v != column])

    def set_all(value):
        st.session_state[state_key] = options[:] if value else []

    if hide_empty and not options:
        return []
    st.markdown(f'**{label}**')
    if not options:
        st.caption('No eligible variables.')
        return []
    if not single:
        left, right = st.columns(2)
        left.button('Select all', key=prefix+'_all', on_click=set_all, args=(True,))
        right.button('Clear all', key=prefix+'_clear', on_click=set_all, args=(False,))
    for col in options:
        st.session_state[prefix+'_column_'+str(col)] = col in selected
    for start in range(0, len(options), 4):
        cells = st.columns(4)
        for cell, col in zip(cells, options[start:start+4]):
            cell.checkbox(str(col), key=prefix+'_column_'+str(col), on_change=changed, args=(col,))
    return selected
