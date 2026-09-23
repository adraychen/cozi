"""Dataset registry and lazy, full-width workspace navigation."""
import hashlib
import io
import uuid
import pandas as pd
import streamlit as st
from utils.table_transforms import validate_table

ADD_TAB = '＋ Add datasets'
_DATA_KEYS = {'variable_metadata', 'variable_dataset_id', 'col_types', 'ai_summary',
              'datetime_overrides', 'override_warnings', 'regression_cached_fit'}
_PREFIXES = ('vm_', 'va_', 'rel_', 'reg_', 'pred_', 'dist_', 'gs_', 'pa_', 'mc_',
             'col_ai_', 'heatmap_ai_', 'tf_', 'view_')
_TRIGGERS = {'vm_apply', 'pa_select_all', 'pa_deselect_all', 'pa_run_btn', 'tf_preview', 'tf_create'}


def dataset_state_key(key):
    """Persist data controls/results, never ephemeral button/download events."""
    if not isinstance(key, str):
        return False
    owned = key in _DATA_KEYS or key.startswith(_PREFIXES)
    trigger = (key in _TRIGGERS or key.endswith(('_download', '_all', '_clear'))
               or key.startswith(('va_graph_button_', 'mc_all_', 'mc_clear_',
                                  'gs_cmp_all_', 'gs_cmp_clear_', 'rel_log_predict_')))
    return owned and not trigger


def save_active(state):
    active = state.get('ws_active')
    registry = state.get('ws_datasets', {})
    if active in registry:
        registry[active]['state'] = {k: state[k] for k in state if dataset_state_key(k)}


def activate(state, dataset_id):
    if dataset_id == state.get('ws_active'):
        return
    save_active(state)
    for key in list(state):
        if dataset_state_key(key) or (isinstance(key, str) and key.startswith(_PREFIXES)):
            del state[key]
    entry = state['ws_datasets'][dataset_id]
    for key, value in entry['state'].items():
        state[key] = value
    state['ws_active'] = dataset_id
    state['df'] = entry['frame']
    state['file_id'] = dataset_id


def add_dataset(state, frame, name, source, parents=None, operation=None):
    frame = validate_table(frame.copy())
    frame.columns = [str(c) for c in frame.columns]
    validate_table(frame)
    registry = state.setdefault('ws_datasets', {})
    base = str(name).strip() or 'Untitled dataset'
    name, number = base, 2
    existing = {d['name'] for d in registry.values()} | {ADD_TAB}
    while name in existing:
        name = f'{base} ({number})'
        number += 1
    dataset_id = uuid.uuid4().hex
    registry[dataset_id] = {'name': name, 'frame': frame, 'source': source,
                            'parents': list(parents or []), 'operation': operation, 'state': {}}
    state['ws_pending'] = dataset_id
    return dataset_id


def initialize(state):
    if 'ws_datasets' not in state:
        state['ws_datasets'] = {}
        if 'df' in state:
            dataset_id = add_dataset(state, state['df'], state.get('manual_name', 'Existing dataset'), 'Existing session')
            state['ws_active'] = dataset_id
            state['file_id'] = dataset_id
            if 'variable_dataset_id' in state:
                state['variable_dataset_id'] = dataset_id
            save_active(state)


def render_add_datasets(samples):
    st.subheader('Add datasets')
    st.caption('Upload, input, or choose sample data. Each addition opens its own dataset tab.')
    epoch = st.session_state.get('ws_upload_epoch', 0)
    files = st.file_uploader('Drop or upload CSV or Excel files', type=['csv', 'xlsx', 'xls'],
                             accept_multiple_files=True, key=f'ws_upload_{epoch}')
    candidates = []
    for index, file in enumerate(files or []):
        identity = hashlib.sha256(file.getvalue()).hexdigest()
        try:
            if file.name.lower().endswith('.csv'):
                candidates.append((file.name, file, None))
            else:
                book = pd.ExcelFile(io.BytesIO(file.getvalue()))
                sheets = st.multiselect(f'Sheets in {file.name}', book.sheet_names, default=book.sheet_names[:1],
                                        key=f'ws_sheets_{index}_{identity}')
                candidates.extend((f'{file.name} · {sheet}', file, sheet) for sheet in sheets)
        except Exception as exc:
            st.error(f'Cannot read {file.name}: {exc}')
    if files and st.button('Add uploaded datasets', type='primary', disabled=not candidates, key='ws_import'):
        loaded, errors = [], []
        for name, file, sheet in candidates:
            try:
                buffer = io.BytesIO(file.getvalue())
                frame = pd.read_csv(buffer) if sheet is None else pd.read_excel(buffer, sheet_name=sheet)
                loaded.append((name, validate_table(frame)))
            except Exception as exc:
                errors.append(f'{name}: {exc}')
        if errors:
            for message in errors:
                st.error(message)
            st.info('No datasets were added. Fix or remove the files with errors and try again.')
        else:
            for name, frame in loaded:
                add_dataset(st.session_state, frame, name, 'Uploaded file')
            st.session_state['ws_upload_epoch'] = epoch + 1
            st.rerun()
    from utils.manual_input import render_manual_input
    render_manual_input(on_add=lambda frame, name: add_dataset(st.session_state, frame, name, 'Manual input'))
    with st.expander('📂 Try with sample data', expanded=False):
        st.caption('Choose a sample to add it to the workspace.')
        for sample in samples:
            if st.button(f"{sample['name']} — {sample['desc']}", key='ws_sample_' + sample['name']):
                try:
                    import requests
                    with st.spinner(f"Loading {sample['name']}…"):
                        response = requests.get(f"https://drive.google.com/uc?export=download&id={sample['id']}", timeout=30)
                        response.raise_for_status()
                        frame = pd.read_csv(io.StringIO(response.text))
                        add_dataset(st.session_state, frame, sample['name'], 'Sample dataset')
                    st.rerun()
                except Exception as exc:
                    st.error(f'Could not load the sample: {exc}')


def render_workspace(samples, show_feedback):
    initialize(st.session_state)
    # Detach persistent controls from widget cleanup while another workflow is shown.
    for key in list(st.session_state):
        if dataset_state_key(key) or key in ('ws_dataset_tabs', 'ws_workflow'):
            st.session_state[key] = st.session_state[key]
    pending = st.session_state.pop('ws_pending', None)
    registry = st.session_state['ws_datasets']
    if pending in registry:
        st.session_state['ws_dataset_tabs'] = registry[pending]['name']
        st.session_state['ws_workflow'] = 'Inspect'
    labels = [d['name'] for d in registry.values()] + [ADD_TAB]
    if st.session_state.get('ws_dataset_tabs') not in labels:
        st.session_state['ws_dataset_tabs'] = labels[0]
    tabs = st.tabs(labels, key='ws_dataset_tabs', on_change='rerun')
    if tabs[-1].open:
        save_active(st.session_state)
        with tabs[-1]:
            render_add_datasets(samples)
        return
    for (dataset_id, entry), tab in zip(registry.items(), tabs):
        if not tab.open:
            continue
        activate(st.session_state, dataset_id)
        try:
            with tab:
                st.subheader(entry['name'])
                st.caption(f"{len(entry['frame']):,} rows · {len(entry['frame'].columns):,} columns · {entry['source']}")
                if entry['parents']:
                    st.caption('Sources: ' + ' + '.join(registry[p]['name'] for p in entry['parents']))
                    with st.expander('Transformation settings'):
                        st.json(entry['operation'])
                inspect, transform, analyze = st.tabs(['Inspect', 'Transform', 'Analyze'], key='ws_workflow', on_change='rerun')
                if inspect.open:
                    from utils.dataset_views import render_inspect
                    with inspect:
                        render_inspect(entry['frame'], show_feedback)
                elif transform.open:
                    from utils.transform_ui import render_transform
                    with transform:
                        render_transform(dataset_id, registry)
                elif analyze.open:
                    from utils.dataset_views import render_analyze
                    with analyze:
                        if entry['frame'].empty:
                            st.info('This dataset has no rows. Choose another dataset or adjust the transformation.')
                        else:
                            render_analyze(entry['frame'], show_feedback)
        finally:
            save_active(st.session_state)
        break
