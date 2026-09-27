"""Bounded manual dataset parsing and Streamlit entry controls."""
import csv
import io
import uuid
import pandas as pd
import streamlit as st
from utils import presentation


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
        if key in ('variable_metadata', 'variable_dataset_id', 'col_types', 'ai_summary', 'datetime_overrides', 'override_warnings', 'pa_results', 'pa_cache_key', 'pa_train_error') or key.startswith(('vm_', 'va_', 'col_ai_', 'heatmap_ai_', 'pa_pred_', 'mc_', 'gs_', 'rel_', 'dist_')):
            del st.session_state[key]
    st.session_state['df'] = frame
    st.session_state['file_id'] = f'manual__{uuid.uuid4().hex}'
    st.session_state['manual_name'] = name
    st.session_state['upload_epoch'] = st.session_state.get('upload_epoch', 0) + 1


def parse_grid(grid, header_mode='Auto detect'):
    """Trim unused trailing space; retain interior missing cells/rows."""
    frame = grid.replace(r'^\s*$', None, regex=True).copy()
    occupied_rows = frame.notna().any(axis=1).to_numpy().nonzero()[0]
    occupied_cols = frame.notna().any(axis=0).to_numpy().nonzero()[0]
    if not len(occupied_rows) or not len(occupied_cols):
        raise ValueError('Enter or paste a table first.')
    frame = frame.iloc[:occupied_rows[-1]+1, :occupied_cols[-1]+1]
    first = frame.iloc[0]
    # Text-like first rows are presumed headers; ambiguous categorical data can be overridden.
    textual = first.notna().all() and pd.to_numeric(first, errors='coerce').isna().all()
    has_header = header_mode == 'First row is headers' or (header_mode == 'Auto detect' and len(frame)>1 and textual)
    if has_header:
        names = [str(v).strip() if pd.notna(v) else '' for v in first]
        frame = frame.iloc[1:].copy()
        frame.columns = names
    else:
        frame.columns = [f'Column {i+1}' for i in range(len(frame.columns))]
    return validate_frame(frame.reset_index(drop=True)), bool(has_header)


def render_manual_input(on_add=None):
    if st.button('Input Data', key='manual_open'):
        st.session_state['manual_visible'] = not st.session_state.get('manual_visible', False)
    if not st.session_state.get('manual_visible', False):
        return
    with st.container(border=True):
        mode = st.radio('Entry method', ['Editable table', 'CSV text'], horizontal=True, key='manual_mode')
        st.caption('Maximum 500 data rows and 25 columns. The CSV header row is not counted. Blank cells become missing values.')
        if mode == 'Editable table':
            st.caption('Paste your table, including its column headers, into the top-left cell. The grid is fixed at 500 rows × 25 columns; a pasted header uses one row, leaving 499 data rows. Unused trailing rows and columns are removed.')
            edited = st.data_editor(pd.DataFrame('', index=range(500), columns=[f'Column {i+1}' for i in range(25)]),
                                    num_rows='fixed', height=420, use_container_width=True, key='manual_fixed_grid')
            with st.expander('Header detection and preview'):
                header_mode = st.selectbox('Header handling', ['Auto detect', 'First row is headers', 'No header row'], key='manual_header_mode')
                st.caption('Auto detection treats a populated, text-only first row as headers. For text-only datasets or numeric headers, check the preview and override if needed.')
                try:
                    preview, detected = parse_grid(edited, header_mode)
                    st.caption('First row used as headers.' if detected else 'Generated column names; first row kept as data.')
                    presentation.dataframe(preview.head(5), hide_index=True, use_container_width=True)
                except ValueError as exc:
                    st.caption(str(exc))
        else:
            text = st.text_area('CSV-formatted data', height=240, key='manual_csv',
                                placeholder='Region,Sales\nEast,120\nWest,95')
        name = st.text_input('Dataset name', value='Manual data', key='manual_dataset_name')
        if st.button('Add dataset' if on_add else 'Input Data', key='manual_submit'):
            try:
                if mode == 'CSV text':
                    frame = parse_csv(text)
                else:
                    frame, _ = parse_grid(edited, header_mode)
                if on_add:
                    on_add(frame, name.strip() or 'Manual data')
                else:
                    activate_dataset(frame, name.strip() or 'Manual data')
                st.session_state['manual_visible'] = False
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
