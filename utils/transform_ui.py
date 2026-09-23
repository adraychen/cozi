"""Guided transformations with validated previews and explicit result creation."""
import json
import pandas as pd
import streamlit as st
from utils.dataset_workspace import add_dataset
from utils.table_transforms import (
    FUNCTIONS, join_tables, append_tables, aggregate_table, pivot_table, unpivot_table,
)


def render_transform(dataset_id, registry):
    entry = registry[dataset_id]
    frame = entry['frame']
    columns = list(frame)
    operation = st.radio('Operation', ['Join', 'Append', 'Aggregate', 'Pivot', 'Unpivot'],
                         horizontal=True, key='tf_operation')
    st.caption('Transformations use stored data values. Preview the result, then save it as a new dataset; source tables are preserved.')
    config = {'operation': operation, 'source': dataset_id}
    parents = [dataset_id]
    others = [key for key in registry if key != dataset_id]
    if operation in ('Join', 'Append') and not others:
        st.info(f'Add another dataset in ＋ Add datasets to use {operation}.')
        return
    if operation == 'Join':
        right_id = st.selectbox('Right dataset', others, format_func=lambda k: registry[k]['name'], key='tf_right')
        parents.append(right_id)
        right = registry[right_id]['frame']
        left_box, right_box = st.columns(2)
        common = [c for c in columns if c in right]
        with left_box:
            left_keys = st.multiselect('Left keys (in matching order)', columns, default=common[:1], key='tf_left_keys')
        with right_box:
            right_keys = st.multiselect('Right keys (in matching order)', list(right), default=common[:1], key='tf_right_keys_' + right_id)
        labels = {'left': 'Left — keep all left rows', 'inner': 'Inner — matched rows only',
                  'right': 'Right — keep all right rows', 'outer': 'Full outer — keep all rows'}
        how = st.selectbox('Join type', list(labels), format_func=labels.get, key='tf_join_type')
        missing = st.checkbox('Match missing keys to other missing keys', value=False, key='tf_match_missing')
        st.caption('Columns with overlapping names use _left and _right suffixes. Duplicate keys can multiply rows.')
        config.update(right=right_id, left_keys=left_keys, right_keys=right_keys, how=how, match_missing=missing)
        def calculate():
            result, report = join_tables(frame, right, left_keys, right_keys, how, missing)
            warnings = []
            if report['Many-to-many matches']:
                warnings.append('Many-to-many matches multiply rows. Check the duplicate-key counts before creating the result.')
            if (how == 'left' and len(result) > len(frame)) or (how == 'right' and len(result) > len(right)):
                warnings.append('The output has more rows than the retained input because some keys have multiple matches.')
            return result, report, warnings
    elif operation == 'Append':
        extra = st.multiselect('Additional datasets (in output order)', others, format_func=lambda k: registry[k]['name'], key='tf_append_inputs')
        parents += extra
        policy = st.selectbox('Columns to keep', ['all', 'common'],
                              format_func=lambda x: 'All columns — fill gaps with missing values' if x == 'all' else 'Common columns only', key='tf_append_columns')
        include_source = st.checkbox('Include source dataset column', value=True, key='tf_include_source')
        source_name = st.text_input('Source column name', 'source_dataset', key='tf_source_name') if include_source else None
        config.update(inputs=extra, columns=policy, source_column=source_name)
        st.caption('Columns align by name. Rows retain their input order; duplicates are retained.')
        def calculate():
            result, warnings = append_tables([registry[k]['frame'] for k in parents], [registry[k]['name'] for k in parents], policy, source_name)
            alignment = {registry[k]['name']: ', '.join(c for c in result if c not in registry[k]['frame'] and c != source_name) or 'None' for k in parents}
            return result, {'Rows': len(result), 'Columns filled with missing values': alignment}, warnings
    elif operation == 'Aggregate':
        groups = st.multiselect('Group by (optional)', columns, key='tf_groups')
        functions = st.multiselect('Calculations', list(FUNCTIONS), default=['Sum'], key='tf_functions')
        measures = st.multiselect('Measure columns', columns, key='tf_measures')
        keep_missing = st.checkbox('Keep missing grouping values', value=True, key='tf_aggregate_missing')
        specs = []
        for function in functions:
            targets = columns[:1] if function == 'Count rows' else measures
            for col in targets:
                default = 'row_count' if function == 'Count rows' else col + '_' + FUNCTIONS[function]
                name = st.text_input(f'Output name: {function}' + ('' if function == 'Count rows' else f' of {col}'), default,
                                     key=f'tf_measure_name_{function}_{col}')
                specs.append({'column': col, 'function': function, 'name': name})
        config.update(groups=groups, measures=specs, keep_missing=keep_missing)
        def calculate():
            if any(f != 'Count rows' for f in functions) and not measures:
                raise ValueError('Choose measure columns for the selected calculations.')
            result = aggregate_table(frame, groups, specs, keep_missing)
            return result, {'Input rows': len(frame), 'Output groups': len(result)}, []
    elif operation == 'Pivot':
        rows = st.multiselect('Row identifiers', columns, key='tf_pivot_rows')
        column = st.selectbox('New columns from', columns, key='tf_pivot_column')
        value = st.selectbox('Values from', columns, key='tf_pivot_value')
        function = st.selectbox('Repeated combinations', ['Require unique combinations'] + list(FUNCTIONS), key='tf_pivot_function')
        function = None if function == 'Require unique combinations' else function
        fill_zero = st.checkbox('Fill missing result cells with zero', value=False, key='tf_pivot_fill')
        config.update(rows=rows, columns=column, values=value, function=function, fill_zero=fill_zero)
        def calculate():
            if not rows:
                raise ValueError('Choose at least one row identifier.')
            result = pivot_table(frame, rows, column, value, function, fill_zero)
            return result, {'Output rows': len(result), 'Output columns': len(result.columns)}, []
    else:
        identifiers = st.multiselect('Keep as identifiers', columns, key='tf_unpivot_ids')
        values = st.multiselect('Columns to turn into rows', columns, key='tf_unpivot_values')
        variable_name = st.text_input('Variable column name', 'variable', key='tf_variable_name')
        value_name = st.text_input('Value column name', 'value', key='tf_value_name')
        drop_missing = st.checkbox('Drop rows with missing values', value=False, key='tf_unpivot_missing')
        config.update(identifiers=identifiers, columns=values, variable_name=variable_name, value_name=value_name, drop_missing=drop_missing)
        def calculate():
            result = unpivot_table(frame, identifiers, values, variable_name, value_name, drop_missing)
            return result, {'Input rows': len(frame), 'Output rows': len(result)}, []
    signature = json.dumps(config, sort_keys=True)
    saved = st.session_state.get('tf_result')
    if saved and saved['signature'] != signature:
        st.session_state.pop('tf_result', None)
        saved = None
    if st.button('Preview result', key='tf_preview', type='primary'):
        try:
            with st.spinner('Calculating and validating the result…'):
                result, report, warnings = calculate()
            saved = {'signature': signature, 'frame': result, 'report': report, 'warnings': warnings}
            st.session_state['tf_result'] = saved
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.session_state.pop('tf_result', None)
            saved = None
            st.error(f'Cannot create this result: {exc}')
    if saved:
        st.subheader('Result preview')
        st.caption(f"{len(saved['frame']):,} rows · {len(saved['frame'].columns):,} columns · Showing the first 100 rows")
        st.dataframe(saved['frame'].head(100), use_container_width=True)
        with st.expander('Validation details', expanded=True):
            st.json(saved['report'])
        for warning in saved['warnings']:
            st.warning(warning)
        if saved['frame'].empty:
            st.info('The result has no rows. You can save it, or adjust the configuration.')
        name = st.text_input('Output dataset name', entry['name'] + ' · ' + operation, key='tf_output_name_' + operation)
        if st.button('Create dataset', key='tf_create', disabled=not name.strip()):
            add_dataset(st.session_state, saved['frame'], name, operation, parents=parents, operation=config)
            st.rerun()
