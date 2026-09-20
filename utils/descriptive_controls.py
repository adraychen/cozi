"""Single-variable checkbox selection for Descriptive Statistics."""
import streamlit as st


def select_descriptive_variable(col_types):
    groups = [('Numerical', col_types['numerical']),
              ('Categorical', col_types['categorical']),
              ('Temporal', col_types.get('temporal', col_types['datetime']))]
    columns = [column for _, values in groups for column in values]
    selected_key = 'dist_profile_selected'
    if selected_key not in st.session_state:
        st.session_state[selected_key] = columns[0] if columns else None
    if st.session_state[selected_key] not in columns:
        st.session_state[selected_key] = None

    def select(column):
        st.session_state[selected_key] = column if st.session_state[f'dist_profile_column_{column}'] else None

    # Synchronize all checkbox states before creating this run's widgets.
    # The callback runs first, so switching across groups stays exclusive.
    for column in columns:
        st.session_state[f'dist_profile_column_{column}'] = column == st.session_state[selected_key]
    for label, values in groups:
        if not values:
            continue
        st.markdown(f'**{label}**')
        for start in range(0, len(values), 4):
            cells = st.columns(4)
            for cell, column in zip(cells, values[start:start + 4]):
                cell.checkbox(str(column), key=f'dist_profile_column_{column}', on_change=select, args=(column,))
    return st.session_state[selected_key]
