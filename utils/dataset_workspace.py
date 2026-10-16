"""Dataset registry and lazy, full-width workspace navigation."""
import hashlib
import io
import uuid
from pathlib import Path
import pandas as pd
import streamlit as st
from utils.table_transforms import validate_table

ADD_TAB = '＋ Add datasets'
_DATA_KEYS = {'variable_metadata', 'variable_dataset_id', 'col_types', 'ai_summary',
              'datetime_overrides', 'override_warnings', 'regression_cached_fit'}
_PREFIXES = ('vm_', 'va_', 'rel_', 'reg_', 'pred_', 'dist_', 'gs_', 'fac_', 'pa_', 'mc_',
             'col_ai_', 'heatmap_ai_', 'tf_', 'view_', 'temporal_')
_TRIGGERS = {'vm_apply', 'pa_select_all', 'pa_deselect_all', 'pa_run_btn', 'pa_temporal_choose_target', 'tf_preview', 'tf_create',
             'rel_anova_ai_generate', 'fac_open_diagnostics', 'fac_apply_model',
             'tf_apply', 'tf_draft_create', 'tf_draft_discard', 'tf_draft_finish',
             'view_remove_duplicate_columns', 'view_remove_duplicate_rows', 'view_duplicate_rows_confirm', 'view_duplicate_rows_cancel', 'view_generate_ai', 'temporal_resume_button'}


def dataset_state_key(key):
    """Persist data controls/results, never ephemeral button/download events."""
    if not isinstance(key, str):
        return False
    owned = key in _DATA_KEYS or key.startswith(_PREFIXES)
    trigger = (key in _TRIGGERS or key.endswith(('_download', '_all', '_clear'))
               or key.startswith(('va_graph_button_', 'mc_all_', 'mc_clear_',
                                  'gs_cmp_all_', 'gs_cmp_clear_', 'rel_log_predict_', 'fac_action_',
                                  'pa_select_all_', 'pa_clear_all_', 'view_ai_generate_', 'temporal_create_button_', 'temporal_select_button_')))
    return owned and not trigger


def save_active(state):
    active = state.get('ws_active')
    registry = state.get('ws_datasets', {})
    if active in registry:
        registry[active]['state'] = {k: state[k] for k in state if dataset_state_key(k)}
        registry[active]['workflow'] = state.get('ws_workflow', 'Inspect')


def activate(state, dataset_id):
    if dataset_id == state.get('ws_active'):
        return
    save_active(state)
    for key in list(state):
        if dataset_state_key(key) or (isinstance(key, str) and key.startswith(_PREFIXES)):
            del state[key]
    entry = state['ws_datasets'][dataset_id]
    for key, value in entry['state'].items():
        if dataset_state_key(key):
            state[key] = value
    state['ws_active'] = dataset_id
    state['df'] = entry['frame']
    state['file_id'] = dataset_id
    state['ws_workflow'] = entry.get('workflow', 'Inspect')


def _workflow_changed(dataset_id, widget_key):
    """Only the active dataset's tab widget may update its workflow."""
    if st.session_state.get('ws_active') == dataset_id:
        st.session_state['ws_workflow'] = st.session_state[widget_key]


def _select_dataset_tab(label):
    st.session_state['ws_dataset_tabs'] = label


def _request_remove_dataset(dataset_id):
    st.session_state['ws_remove_pending'] = dataset_id


@st.dialog('Remove dataset')
def _confirm_remove_dataset(dataset_id):
    registry = st.session_state['ws_datasets']
    if dataset_id not in registry:
        st.rerun()
    st.write(f"Remove {registry[dataset_id]['name']}?")
    remove, cancel = st.columns(2)
    if remove.button('Remove', type='primary', key='ws_remove_confirm'):
        from utils.project_file import remove_dataset
        st.session_state.pop('ws_remove_pending', None)
        remove_dataset(st.session_state, dataset_id)
        st.rerun()
    if cancel.button('Cancel', key='ws_remove_cancel'):
        st.session_state.pop('ws_remove_pending', None)
        st.rerun()


def add_dataset(state, frame, name, source, parents=None, operation=None, metadata=None):
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
    if metadata is not None:
        from copy import deepcopy
        registry[dataset_id]['state'].update(variable_metadata=deepcopy(metadata), variable_dataset_id=dataset_id)
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


@st.cache_data(show_spinner=False, max_entries=16)
def uploaded_sheet_names(content):
    with pd.ExcelFile(io.BytesIO(content)) as book:
        return book.sheet_names


def render_add_datasets(samples):
    st.subheader('Add datasets')
    st.caption('Upload, input, or choose sample data. Each addition opens its own dataset tab.')
    epoch = st.session_state.get('ws_upload_epoch', 0)
    st.html('''<style>
        [class*="st-key-ws_upload_"] [data-testid="stFileUploaderDropzoneInstructions"] > * {display:none;}
        [class*="st-key-ws_upload_"] [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content:"Drop or upload a CSV or Excel file. 200 MB max.";
            font-size:1rem; color:var(--text-color); opacity:.65;
        }
        [class*="st-key-ws_upload_"] [data-testid="stFileUploaderDropzone"],
        .st-key-manual_entry_panel {
            background:var(--secondary-background-color, #f0f2f6); border-radius:.5rem; padding:.5rem .75rem;
            min-height:56px; align-items:center;
        }
        [class*="st-key-ws_upload_"] [data-testid="stFileUploaderDropzone"] > div:has(+ [data-testid="stFileUploaderDropzoneInstructions"]) button,
        .st-key-manual_open button {
            width:120px; height:40px; min-height:40px; padding:.25rem .75rem;
            background:var(--background-color, #fff); border:1px solid #8884; border-radius:.5rem;
            font-size:1rem;
        }
        .st-key-manual_entry_panel p {color:var(--text-color); opacity:.65; margin:0;}
    </style>''')
    files = st.file_uploader('Drop or upload a CSV or Excel file. 200 MB max.', type=['csv', 'xlsx', 'xls'],
                             label_visibility='collapsed', accept_multiple_files=True, key=f'ws_upload_{epoch}')
    candidates = []
    for index, file in enumerate(files or []):
        identity = hashlib.sha256(file.getvalue()).hexdigest()
        try:
            if file.name.lower().endswith('.csv'):
                candidates.append((file.name, file, None))
            else:
                sheet_names = uploaded_sheet_names(file.getvalue())
                sheets = st.multiselect(f'Sheets in {file.name}', sheet_names, default=sheet_names[:1],
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
                        if 'path' in sample:
                            frame = pd.read_csv(Path(__file__).resolve().parents[1] / sample['path'])
                        else:
                            response = requests.get(f"https://drive.google.com/uc?export=download&id={sample['id']}", timeout=30)
                            response.raise_for_status()
                            frame = pd.read_csv(io.StringIO(response.text))
                        add_dataset(st.session_state, frame, sample['name'], 'Sample dataset')
                    st.rerun()
                except Exception as exc:
                    st.error(f'Could not load the sample: {exc}')


@st.fragment
def render_workspace(samples, show_feedback):
    # Scope the larger navigation treatment to the workflow row, leaving dataset
    # tabs and nested analytical tabs at their existing sizes.
    st.html('''<style>
        [class*="st-key-ws_workflow_"] > div > [role="tablist"] {
            gap: 0.6rem;
            padding: 0.35rem 0 0.65rem;
        }
        [class*="st-key-ws_workflow_"] > div > [role="tablist"] > [role="tab"] {
            min-height: 3rem;
            height: auto;
            min-width: 8rem;
            justify-content: center;
            padding: 0.7rem 1.35rem;
            border: 1px solid #4c72b04d;
            border-radius: 0.6rem;
            background: #4c72b014;
        }
        [class*="st-key-ws_workflow_"] > div > [role="tablist"] > [role="tab"] p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        [class*="st-key-ws_workflow_"] > div > [role="tablist"] > [role="tab"]:hover {
            background: #4c72b026;
            border-color: #4c72b0;
        }
        [class*="st-key-ws_workflow_"] > div > [role="tablist"] > [role="tab"][aria-selected="true"] {
            background: #4c72b0;
            border-color: #4c72b0;
            color: #ffffff;
        }
        [class*="st-key-ws_workflow_"] > div > [role="tablist"] > [role="tab"]:focus-visible {
            outline: 2px solid #4c72b0;
            outline-offset: 2px;
        }
        @media (max-width: 540px) {
            [class*="st-key-ws_workflow_"] > div > [role="tablist"] { gap: 0.35rem; }
            [class*="st-key-ws_workflow_"] > div > [role="tablist"] > [role="tab"] {
                min-width: 0;
                flex: 1 0 auto;
                padding: 0.65rem 0.6rem;
            }
            [class*="st-key-ws_workflow_"] > div > [role="tablist"] > [role="tab"] p {
                font-size: 1rem;
            }
        }
    </style>''')
    initialize(st.session_state)
    # Detach persistent controls from widget cleanup while another workflow is shown.
    for key in list(st.session_state):
        if dataset_state_key(key) or key in ('ws_dataset_tabs', 'ws_workflow'):
            st.session_state[key] = st.session_state[key]
    pending = st.session_state.pop('ws_pending', None)
    registry = st.session_state['ws_datasets']
    if pending in registry:
        st.session_state['ws_dataset_tabs'] = registry[pending]['name']
    labels = [d['name'] for d in registry.values()] + [ADD_TAB]
    if st.session_state.get('ws_dataset_tabs') not in labels:
        st.session_state['ws_dataset_tabs'] = labels[0]
    # Native Streamlit tabs cannot contain a second, independent close control.
    # Keep the dataset navigation in one tab-style row with adjacent buttons.
    st.html('''<style>
        .st-key-ws_dataset_navigation {border-bottom: 2px solid #8883; padding-bottom: 0;}
        .st-key-ws_dataset_navigation button {border: 0; border-radius: 0; background: transparent;
            padding: .4rem .2rem; box-shadow: none;}
        .st-key-ws_dataset_navigation [class*="st-key-ws_dataset_selected_"] button {
            color: #4c72b0; border-bottom: 3px solid #4c72b0;}
    </style>''')
    with st.container(horizontal=True, gap='small', key='ws_dataset_navigation'):
        for dataset_id, entry in registry.items():
            with st.container(horizontal=True, gap='xxsmall', width='content'):
                selected = entry['name'] == st.session_state['ws_dataset_tabs']
                st.button(entry['name'], key=('ws_dataset_selected_' if selected else 'ws_dataset_select_') + dataset_id,
                          type='tertiary', on_click=_select_dataset_tab, args=(entry['name'],))
                st.button('×', key='ws_remove_' + dataset_id, type='tertiary', help='Remove ' + entry['name'],
                          on_click=_request_remove_dataset, args=(dataset_id,))
        st.button(ADD_TAB, key='ws_add_dataset_tab', type='tertiary', on_click=_select_dataset_tab, args=(ADD_TAB,))
    removal = st.session_state.get('ws_remove_pending')
    if removal in registry:
        _confirm_remove_dataset(removal)
    if st.session_state['ws_dataset_tabs'] == ADD_TAB:
        save_active(st.session_state)
        render_add_datasets(samples)
        return
    for dataset_id, entry in registry.items():
        if entry['name'] != st.session_state['ws_dataset_tabs']:
            continue
        activate(st.session_state, dataset_id)
        try:
            with st.container():
                st.subheader(entry['name'])
                st.caption(f"{len(entry['frame']):,} rows · {len(entry['frame'].columns):,} columns · {entry['source']}")
                context = st.session_state.get('temporal_return_context', {})
                if context.get('target_id') == dataset_id:
                    from utils.temporal_navigation import return_to_analysis
                    st.button('Return to Analysis', key='temporal_resume_button', on_click=return_to_analysis, args=(st.session_state,))
                if entry['parents']:
                    st.caption('Sources: ' + ' + '.join(registry[p]['name'] if p in registry else 'Removed source dataset' for p in entry['parents']))
                    with st.expander('Transformation settings'):
                        st.json(entry['operation'])
                # A new dataset needs a new browser widget identity. Reusing the
                # source widget lets its old Transform value overwrite Inspect
                # when the next classification input triggers a rerun.
                workflow_key = f'ws_workflow_{dataset_id}'
                st.session_state[workflow_key] = st.session_state.get('ws_workflow', 'Inspect')
                inspect, transform, analyze, report_tab, help_tab = st.tabs(
                    ['Inspect', 'Transform', 'Analyze', 'Report', 'Help'], key=workflow_key,
                    on_change=_workflow_changed, args=(dataset_id, workflow_key))
                if inspect.open:
                    from utils.dataset_views import render_inspect
                    with inspect:
                        render_inspect(entry['frame'], show_feedback, dataset_name=entry['name'])
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
                elif report_tab.open:
                    from utils.report_ui import render_report
                    with report_tab:
                        render_report(entry, registry, dataset_id)
                elif help_tab.open:
                    from utils.help_ui import render_help
                    with help_tab:
                        render_help()
        finally:
            save_active(st.session_state)
        break
