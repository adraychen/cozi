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
    preview = frame.loc[duplicate_mask].head(100)
    first_occurrences = frame.loc[~duplicate_mask]
    # Compare against the complete source, since matches may lie outside the preview.
    combined = pd.concat([first_occurrences, preview], ignore_index=True)
    matching_firsts = combined.duplicated(keep=False).iloc[:len(first_occurrences)].to_numpy()
    st.caption('Earlier matching rows — these first occurrences will be kept. Original row indices are shown.')
    presentation.dataframe(first_occurrences.iloc[matching_firsts], width='stretch')
    st.caption('Rows to remove — showing the first 100 extra copies of the earlier rows above. These rows need not match one another.')
    presentation.dataframe(preview, width='stretch')
    name = st.text_input('Output dataset name', dataset_name + ' · Unique rows', key='view_unique_rows_name')
    st.caption(f'Keep the first occurrence of each row: {len(frame) - count:,} rows will remain. Creates a new dataset; the source remains available.')
    if st.button('Remove duplicated rows', key='view_remove_duplicate_rows', disabled=not name.strip()):
        _create_cleaned(remove_duplicate_rows(frame), name, 'Remove duplicated rows', {'keep': 'first', 'columns': 'all', 'removed_rows': count})
