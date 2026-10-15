"""Duplicate summaries and cleanup actions in the Inspect workflow."""
import pandas as pd
import streamlit as st
from utils import presentation
from utils.dataset_workspace import add_dataset
from utils.table_editing import duplicate_columns, remove_duplicate_columns, remove_duplicate_rows


def _create_cleaned(frame, name, operation, settings):
    dataset_id = st.session_state['ws_active']
    add_dataset(st.session_state, frame, name, operation, parents=[dataset_id],
                operation={'operation': operation, 'source': dataset_id, **settings})
    st.rerun()


def render_duplicate_columns(frame, dataset_name):
    st.markdown('**Duplicated columns**')
    duplicates = duplicate_columns(frame)
    st.caption('Columns are duplicates when their stored values, data types, and missing-value positions match in every row. The first column in each group is kept.')
    if not duplicates:
        if len(frame):
            st.write('No duplicated columns found.')
        else:
            st.info('Duplicate columns cannot be assessed without rows.')
        return
    presentation.dataframe(pd.DataFrame({'Column to remove': list(duplicates), 'Column to keep': list(duplicates.values())}),
                 hide_index=True, width='stretch')
    name = st.text_input('Output dataset name', dataset_name + ' · Unique columns', key='view_unique_columns_name')
    st.caption(f'{len(duplicates):,} columns will be removed. Creates a new dataset; the source remains available.')
    if st.button('Remove duplicated columns', key='view_remove_duplicate_columns', disabled=not name.strip()):
        result, removed = remove_duplicate_columns(frame)
        _create_cleaned(result, name, 'Remove duplicated columns', {'removed_columns': removed, 'keep': 'first'})


def render_row_overview(frame, dataset_name):
    duplicate_mask = frame.duplicated(keep='first')
    count = int(duplicate_mask.sum())
    total, duplicated = st.columns(2)
    total.metric('Number of rows', f'{len(frame):,}')
    duplicated.metric('Duplicated rows', f'{count:,}')
    st.caption('Duplicated rows are extra copies after the first occurrence, matching all stored columns. Matching missing values count as equal. The row index is ignored.')
    if not count:
        st.write('No duplicated rows found.')
        return
    # Factorize values only, so repeated source indices never affect grouping.
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)
    groups = {}
    for position, code in enumerate(codes):
        groups.setdefault(code, []).append(position)
    positions = [position for group in groups.values() if len(group) > 1 for position in group]
    display = frame.iloc[positions].reset_index(drop=True)

    def available_label(label):
        while label in display.columns:
            label += ' (overview)'
        return label

    index_label = available_label('Row index')
    display.insert(0, index_label, list(frame.index[positions]))
    status_label = available_label('Row status')
    statuses = ['Duplicate' if duplicate_mask.iloc[position] else 'Earlier matching' for position in positions]
    display.insert(1, status_label, statuses)
    styled = display.style.apply(
        lambda row: ['background-color: #E8EEF7; color: #1A1A1A' if statuses[row.name] == 'Duplicate' else ''] * len(row),
        axis=1,
    )
    st.caption('Each earlier matching row is followed by its duplicates. Original row indices are shown; shaded rows are duplicates to remove.')
    presentation.dataframe(styled, hide_index=True, width='stretch')
    if st.button('Remove duplicate rows', key='view_remove_duplicate_rows'):
        st.session_state['view_duplicate_rows_pending'] = True
    if not st.session_state.get('view_duplicate_rows_pending', False):
        return
    name = st.text_input('Output dataset name', dataset_name + ' · Unique rows', key='view_unique_rows_name')
    st.caption(f'Keep the first occurrence of each row: {len(frame) - count:,} rows will remain. Creates a new dataset; the source remains available.')
    confirm, cancel = st.columns(2)
    if confirm.button('Confirm removal', key='view_duplicate_rows_confirm', disabled=not name.strip()):
        st.session_state.pop('view_duplicate_rows_pending', None)
        _create_cleaned(remove_duplicate_rows(frame), name.strip(), 'Remove duplicated rows', {'keep': 'first', 'columns': 'all', 'removed_rows': count})
    if cancel.button('Cancel', key='view_duplicate_rows_cancel'):
        st.session_state.pop('view_duplicate_rows_pending', None)
        st.rerun()
