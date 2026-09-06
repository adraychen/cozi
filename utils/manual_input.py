"""Bounded manual dataset parsing and Streamlit entry controls."""
import csv
import io
import uuid
import pandas as pd
import streamlit as st


def validate_frame(frame):
    if not 1 <= len(frame) <= 500 or not 1 <= len(frame.columns) <= 25:
        raise ValueError('Enter 1–500 data rows and 1–25 columns.')
    names = [str(c).strip() for c in frame.columns]
    if any(not c for c in names) or len(set(names)) != len(names):
        raise ValueError('Column names must be non-empty and unique.')
    frame = frame.copy()
    frame.columns = names
    frame = frame.replace(r'^\s*$', None, regex=True)
    if frame.isna().all().all():
        raise ValueError('Enter at least one data value.')
    for c in frame:
        values = frame[c].dropna()
        converted = pd.to_numeric(values, errors='coerce')
        # Preserve identifiers such as 001 rather than silently removing zeros.
        leading_zero = values.astype(str).str.match(r'^[+-]?0\d+').any()
        if len(values) and converted.notna().all() and not leading_zero:
            frame[c] = pd.to_numeric(frame[c], errors='coerce')
    return frame


def parse_csv(text):
    if len(text) > 5_000_000:
        raise ValueError('CSV text must be at most 5 million characters.')
    try:
        rows = list(csv.reader(io.StringIO(text.lstrip('\ufeff')), strict=True))
    except csv.Error as exc:
        raise ValueError(f'Invalid CSV: {exc}') from exc
    rows = [r for r in rows if r]  # Ignore physically blank lines, retain blank-field records.
    if len(rows) < 2:
        raise ValueError('Include column headers and at least one data row.')
    if any(len(r) != len(rows[0]) for r in rows[1:]):
        raise ValueError('Every CSV row must have the same number of fields as the header.')
    return validate_frame(pd.DataFrame(rows[1:], columns=rows[0]))


def activate_dataset(frame, name):
    for key in list(st.session_state):
        if key in ('col_types', 'ai_summary', 'datetime_overrides', 'override_warnings', 'pa_results', 'pa_cache_key', 'pa_train_error') or key.startswith(('col_ai_', 'heatmap_ai_', 'pa_pred_', 'mc_', 'gs_', 'rel_', 'dist_')):
            del st.session_state[key]
    st.session_state['df'] = frame
    st.session_state['file_id'] = f'manual__{uuid.uuid4().hex}'
    st.session_state['manual_name'] = name
    st.session_state['upload_epoch'] = st.session_state.get('upload_epoch', 0) + 1


def render_manual_input():
    if st.button('Input Data', key='manual_open'):
        st.session_state['manual_visible'] = not st.session_state.get('manual_visible', False)
    if not st.session_state.get('manual_visible', False):
        return
    with st.container(border=True):
        mode = st.radio('Entry method', ['Editable table', 'CSV text'], horizontal=True, key='manual_mode')
        st.caption('Maximum 500 data rows and 25 columns. The CSV header row is not counted. Blank cells become missing values.')
        if mode == 'Editable table':
            rows = int(st.number_input('Rows', 1, 500, 10, key='manual_rows'))
            columns = int(st.number_input('Columns', 1, 25, 3, key='manual_columns'))
            names_text = st.text_input('Column names (CSV format)', value=','.join(f'Column {i+1}' for i in range(columns)), key=f'manual_headers_{columns}')
            try:
                names = next(csv.reader([names_text], strict=True))
                if len(names) != columns or len(set(n.strip() for n in names)) != columns or any(not n.strip() for n in names):
                    raise ValueError('Provide one unique, non-empty name per column.')
            except (csv.Error, ValueError, StopIteration) as exc:
                st.error(str(exc))
                return
            st.caption('Set the table size and headers before entering values. Changing them starts a new grid. Empty trailing rows are removed on submission.')
            edited = st.data_editor(pd.DataFrame('', index=range(rows), columns=names), num_rows='fixed',
                                    use_container_width=True, key=f'manual_grid_{rows}_{columns}_{names_text}')
        else:
            text = st.text_area('CSV-formatted data', height=240, key='manual_csv',
                                placeholder='Region,Sales\nEast,120\nWest,95')
        if st.button('Input Data', key='manual_submit'):
            try:
                if mode == 'CSV text':
                    frame = parse_csv(text)
                else:
                    edited = edited.replace(r'^\s*$', None, regex=True)
                    while len(edited) and edited.iloc[-1].isna().all():
                        edited = edited.iloc[:-1]
                    frame = validate_frame(edited)
                activate_dataset(frame, 'Manual table' if mode == 'Editable table' else 'Manual CSV')
                st.session_state['manual_visible'] = False
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
