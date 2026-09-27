"""Guided transformations with validated previews and explicit result creation."""
import json
import pandas as pd
import streamlit as st
from utils.currency_conversion import convert_currency_to_numeric, CurrencyConversionError, detect_currency_columns
from utils.dataset_workspace import add_dataset
from utils.table_transforms import (
    FUNCTIONS, join_tables, append_tables, aggregate_table, pivot_table, unpivot_table,
)
from utils.table_editing import rename_column, remove_columns, add_column, replace_values, remove_characters, trim_spaces


def render_validation(report, operation):
    """Present validation using the same metrics and tables as inspection."""
    if operation == 'Join':
        output, matched = st.columns(2)
        output.metric('Output rows', f"{report['Output rows']:,}")
        matched.metric('Matched left rows', f"{report['Matched left rows']:,}")
        rows = []
        for label, left_key, right_key in [
            ('Unmatched rows', 'Unmatched left rows', 'Unmatched right rows'),
            ('Rows with duplicate keys', 'Left rows with duplicate keys', 'Right rows with duplicate keys'),
            ('Rows with missing keys', 'Left rows with missing keys', 'Right rows with missing keys'),
        ]:
            rows.append({'Check': label, 'Left dataset': f'{report[left_key]:,}',
                         'Right dataset': f'{report[right_key]:,}'})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.markdown('**Many-to-many matches:** ' + ('Yes' if report['Many-to-many matches'] else 'No'))
        st.caption('Duplicate-key counts include every row sharing a repeated key. Many-to-many matches can multiply output rows.')
    else:
        rows = []
        for label, value in report.items():
            if isinstance(value, dict):
                rows.extend({'Check': f'{label} — {name}', 'Result': str(detail)} for name, detail in value.items())
            else:
                rows.append({'Check': label, 'Result': f'{value:,}' if isinstance(value, int) else str(value)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def render_transform(dataset_id, registry):
    entry = registry[dataset_id]
    draft = st.session_state.get('tf_draft')
    frame = draft['frame'] if draft else entry['frame']
    if draft:
        st.subheader('Pending changes')
        for index, step in enumerate(draft['steps'], 1):
            st.write(f"{index}. {step['operation']}" + (f" — {step['column'] if step['column'] != 0 else 'All columns'}" if 'column' in step else ''))
        st.caption('These changes are retained while you configure the next operation. Create dataset saves the applied changes listed here.')
        with st.expander('Preview accumulated changes'):
            st.dataframe(frame.head(100), width='stretch')
        draft_name = st.text_input('Dataset name for accumulated changes', entry['name'] + ' · Transformed', key='tf_draft_name')
        if st.button('Create dataset', key='tf_draft_create', type='primary', disabled=not draft_name.strip()):
            add_dataset(st.session_state, frame, draft_name, 'Transform', parents=draft['parents'], operation={'steps': draft['steps']})
            st.rerun()
        if st.button('Discard accumulated changes', key='tf_draft_discard'):
            for key in list(st.session_state):
                if key.startswith('tf_'):
                    del st.session_state[key]
            st.rerun()
    columns = list(frame)
    operation = st.radio('Operation', ['Join', 'Append', 'Aggregate', 'Pivot', 'Unpivot',
                                      'Rename column', 'Remove column', 'Add column',
                                      'Replace Values', 'Remove Characters', 'Trim Spaces', 'Convert Currency to Numeric'],
                         horizontal=True, key='tf_operation')
    st.caption('Preview a change, then choose Apply and continue to edit another column or run another operation. Create dataset saves the combined result; source tables are preserved.')
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
            result, warnings = append_tables([frame] + [registry[k]['frame'] for k in extra], [registry[k]['name'] for k in parents], policy, source_name)
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
    elif operation == 'Unpivot':
        identifiers = st.multiselect('Keep as identifiers', columns, key='tf_unpivot_ids')
        values = st.multiselect('Columns to turn into rows', columns, key='tf_unpivot_values')
        variable_name = st.text_input('Variable column name', 'variable', key='tf_variable_name')
        value_name = st.text_input('Value column name', 'value', key='tf_value_name')
        drop_missing = st.checkbox('Drop rows with missing values', value=False, key='tf_unpivot_missing')
        config.update(identifiers=identifiers, columns=values, variable_name=variable_name, value_name=value_name, drop_missing=drop_missing)
        def calculate():
            result = unpivot_table(frame, identifiers, values, variable_name, value_name, drop_missing)
            return result, {'Input rows': len(frame), 'Output rows': len(result)}, []
    elif operation == 'Rename column':
        column = st.selectbox('Column to rename', columns, key='tf_rename_column')
        new_name = st.text_input('New column name', key='tf_rename_name').strip()
        config.update(column=column, name=new_name)
        def calculate():
            result = rename_column(frame, column, new_name)
            return result, {'Original name': column, 'New name': new_name, 'Rows': len(result)}, []
    elif operation == 'Remove column':
        removed = st.multiselect('Columns to remove', columns, key='tf_remove_columns')
        st.caption('Choose one or more columns. At least one column must remain.')
        config.update(columns=removed)
        def calculate():
            result = remove_columns(frame, removed)
            return result, {'Removed columns': ', '.join(removed), 'Remaining columns': len(result.columns), 'Rows': len(result)}, []
    elif operation == 'Replace Values':
        column = st.selectbox('Column', columns + [0], format_func=lambda c: 'All columns' if c == 0 else (f'{c} (column)' if c == 'All columns' else c), key='tf_replace_column')
        selected_columns = columns if column == 0 else [column]
        find_type = st.selectbox('Value to find type', ['Text', 'Number', 'True/False'], key='tf_replace_find_type')
        find = (st.selectbox('Value to find', ['True', 'False'], key='tf_replace_find_boolean')
                if find_type == 'True/False' else st.text_input('Value to find', key='tf_replace_find_' + find_type))
        replacement_type = st.selectbox('Replace with', ['Missing value', 'Text', 'Number', 'True/False'], key='tf_replace_type')
        replacement = None
        if replacement_type == 'True/False':
            replacement = st.selectbox('Replacement value', ['True', 'False'], key='tf_replace_boolean')
        elif replacement_type != 'Missing value':
            replacement = st.text_input('Replacement value', key='tf_replace_value_' + replacement_type)
        ignore_spaces = st.checkbox('Ignore surrounding spaces when matching', value=True, key='tf_replace_ignore_spaces') if find_type == 'Text' else False
        st.caption('Matches the whole value and is case-sensitive. Ignore surrounding spaces matches values such as " June " when you enter "June". Replacement text is used exactly as entered; unmatched cells stay unchanged. Existing missing values stay missing. Choose Missing value for a true missing cell; Text None creates the word None.')
        config.update(column=column, columns=selected_columns, find=find, find_type=find_type, replacement=replacement, replacement_type=replacement_type, ignore_surrounding_spaces=ignore_spaces)
        def calculate():
            result, counts = frame, {}
            for target in selected_columns:
                result, counts[target] = replace_values(result, target, find, replacement, find_type, replacement_type, ignore_surrounding_spaces=ignore_spaces)
            matched = sum(counts.values())
            warnings = ['No exact matches were found. The result is unchanged.'] if not matched else []
            return result, {'Column': 'All columns' if column == 0 else column, 'Matched cells': matched, 'Matches by column': counts, 'Rows': len(result), 'Replacement type': replacement_type}, warnings
    elif operation == 'Remove Characters':
        column = st.selectbox('Column', columns + [0], format_func=lambda c: 'All columns' if c == 0 else (f'{c} (column)' if c == 'All columns' else c), key='tf_remove_char_column')
        selected_columns = columns if column == 0 else [column]
        characters = st.text_input('Characters to remove', placeholder='$ or , or $,', key='tf_remove_characters')
        st.caption('Enter characters directly, without quotes or separators. For `$3,000`: enter `$` to get `3,000`, enter `,` to get `$3000`, or enter `$,` to get `3000`. Each character is removed wherever it occurs. Spaces and symbols are literal characters. Text stays text; missing and non-text values are preserved.')
        config.update(column=column, columns=selected_columns, characters=characters)
        def calculate():
            result, counts = frame, {}
            for target in selected_columns:
                result, counts[target] = remove_characters(result, target, characters)
            changed = sum(counts.values())
            warnings = ['No text cells contained the selected characters. The result is unchanged.'] if not changed else []
            return result, {'Column': 'All columns' if column == 0 else column, 'Changed cells': changed, 'Changes by column': counts, 'Rows': len(result)}, warnings
    elif operation == 'Trim Spaces':
        column = st.selectbox('Column', columns + [0], format_func=lambda c: 'All columns' if c == 0 else (f'{c} (column)' if c == 'All columns' else c), key='tf_trim_column')
        selected_columns = columns if column == 0 else [column]
        mode = st.selectbox('Spaces to remove', ['Leading and trailing', 'Leading only', 'Trailing only'], key='tf_trim_mode')
        st.caption('Removes whitespace only at the selected edges, including tabs and line breaks. Spaces between words remain exactly as they are. Missing and non-text values are preserved.')
        config.update(column=column, columns=selected_columns, mode=mode)
        def calculate():
            result, counts = frame, {}
            for target in selected_columns:
                result, counts[target] = trim_spaces(result, target, mode)
            changed = sum(counts.values())
            return result, {'Changed cells': changed, 'Changes by column': counts, 'Rows': len(result)}, ([] if changed else ['No cells needed trimming. The result is unchanged.'])
    elif operation == 'Convert Currency to Numeric':
        text_columns = [c for c in columns if any(isinstance(v, str) for v in frame[c].dropna()) or pd.api.types.is_string_dtype(frame[c].dtype)]
        if not text_columns:
            st.info('Choose a dataset with a text column to convert currency values.')
            st.session_state.pop('tf_result', None)
            return
        column = st.selectbox('Column', text_columns + [0], format_func=lambda c: 'All columns' if c == 0 else (f'{c} (column)' if c == 'All columns' else c), key='tf_currency_column')
        if column == 0:
            detected = detect_currency_columns(frame)
            selected_columns = st.multiselect('Columns to convert', text_columns, default=detected, key='tf_currency_targets')
            st.caption('Currency columns are selected automatically when at least one value is a valid amount with $, £, ¥, or €. Review the selection; add or remove columns as needed. Names, dates, existing numbers, and plain numeric text are skipped unless selected. Invalid values within selected columns still block the whole conversion.')
            if not detected:
                st.info('No symbol-marked currency amounts detected. Select the text columns you want to convert.')
        else:
            selected_columns = [column]
        st.caption('Blank and missing cells stay missing. Every value must be valid before you can apply the conversion.')
        config.update(column=column, columns=selected_columns)
        def calculate():
            if not selected_columns:
                raise ValueError('Select at least one currency column to convert.')
            result, failures, types = frame, [], {}
            for target in selected_columns:
                try:
                    result = convert_currency_to_numeric(result, target)
                    types[target] = 'Whole Number' if pd.api.types.is_integer_dtype(result[target].dtype) else 'Decimal Number'
                except CurrencyConversionError as exc:
                    details = exc.invalid.copy()
                    details.insert(0, 'Column', target)
                    failures.append(details)
            if failures:
                raise CurrencyConversionError(pd.concat(failures, ignore_index=True))
            return result, {'Columns converted': ', '.join(selected_columns), 'Columns unchanged': ', '.join(c for c in columns if c not in selected_columns) or 'None', 'Data type': types, 'Numeric values': int(result[selected_columns].notna().sum().sum()), 'Missing values': int(result[selected_columns].isna().sum().sum()), 'Rows': len(result)}, []
    else:  # Add column
        new_name = st.text_input('New column name', key='tf_add_name').strip()
        method = st.selectbox('Column values', ['Blank', 'Constant value', 'Copy existing column', 'Row number', 'Calculation'], key='tf_add_method')
        options = {'method': method}
        if method == 'Constant value':
            value_type = st.selectbox('Value type', ['Text', 'Number', 'True/False'], key='tf_add_value_type')
            if value_type == 'True/False':
                value = st.selectbox('Value', ['True', 'False'], key='tf_add_boolean')
            else:
                value = st.text_input('Value', key='tf_add_value_' + value_type)
            options.update(value=value, value_type=value_type)
        elif method == 'Copy existing column':
            options['source'] = st.selectbox('Source column', columns, key='tf_add_source')
        elif method == 'Calculation':
            numerical = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
            if not numerical:
                st.info('A calculation requires numerical columns.')
                st.session_state.pop('tf_result', None)
                return
            options['left'] = st.selectbox('Left operand', numerical, key='tf_add_left')
            options['calculation'] = st.selectbox('Calculation', ['Add', 'Subtract', 'Multiply', 'Divide'], key='tf_add_calculation')
            options['right'] = st.selectbox('Right operand', numerical, key='tf_add_right')
            st.caption('Calculations use the stored numerical values and preserve missing values. Division by zero is not allowed.')
        elif method == 'Row number':
            st.caption('Numbers rows from 1 in their current order.')
        else:
            st.caption('Adds a column of missing values.')
        config.update(name=new_name, **options)
        def calculate():
            result = add_column(frame, new_name, **options)
            return result, {'New column': new_name, 'Column values': method, 'Columns': len(result.columns), 'Rows': len(result)}, []
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
        except CurrencyConversionError as exc:
            st.session_state.pop('tf_result', None)
            saved = None
            st.error(str(exc))
            st.dataframe(exc.invalid, hide_index=True, width='stretch')
            st.caption('Use Replace Values to resolve these values, then Apply and continue and try the conversion again. No conversion has been applied.')
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.session_state.pop('tf_result', None)
            saved = None
            st.error(f'Cannot create this result: {exc}')
    if saved:
        st.subheader('Result preview')
        st.caption(f"{len(saved['frame']):,} rows · {len(saved['frame'].columns):,} columns · Showing the first 100 rows")
        st.dataframe(saved['frame'].head(100), use_container_width=True)
        with st.expander('Validation details', expanded=True):
            render_validation(saved['report'], operation)
        for warning in saved['warnings']:
            st.warning(warning)
        if saved['frame'].empty:
            st.info('The result has no rows. You can save it, or adjust the configuration.')
        combined_parents = list(dict.fromkeys((draft['parents'] if draft else []) + parents))
        steps = (draft['steps'] if draft else []) + [config]
        if st.button('Apply and continue', key='tf_apply'):
            pending = {'frame': saved['frame'], 'parents': combined_parents, 'steps': steps}
            output_name = st.session_state.get('tf_output_name_' + operation, st.session_state.get('tf_draft_name', entry['name'] + ' · Transformed'))
            # Clear controls whose column choices may have changed after this step.
            for key in list(st.session_state):
                if key.startswith('tf_'):
                    del st.session_state[key]
            st.session_state['tf_draft'] = pending
            st.session_state['tf_draft_name'] = output_name
            st.session_state['tf_operation'] = operation
            st.rerun()
        if draft:
            st.caption('To include this preview, choose Apply and continue, then Create dataset under Pending changes above.')
        else:
            name = st.text_input('Output dataset name', entry['name'] + ' · ' + operation, key='tf_output_name_' + operation)
            st.caption('Create dataset keeps the result in this session. To keep a local copy, download the full dataset from Inspect → Dataset Preview.')
            if st.button('Create dataset', key='tf_create', disabled=not name.strip()):
                add_dataset(st.session_state, saved['frame'], name, operation, parents=combined_parents,
                            operation={'steps': steps} if draft else config)
                st.rerun()
