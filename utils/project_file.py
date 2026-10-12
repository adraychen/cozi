"""Versioned data-only project archives. Never deserialize executable objects."""
import hashlib
import io
import json
import zipfile
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from utils.dataset_workspace import save_active, activate, dataset_state_key, ADD_TAB, _PREFIXES
from utils.table_transforms import validate_table

VERSION = 1
MAX_BYTES = 512 * 1024 * 1024


def project_payload(state):
    save_active(state)
    datasets = {}
    for identity, entry in state.get('ws_datasets', {}).items():
        copied = dict(entry)
        copied['state'] = {k: v for k, v in entry['state'].items()
                           if dataset_state_key(k) and not any(x in k for x in ('cached_fit', 'pipeline', 'model', 'explainer'))}
        datasets[identity] = copied
    return dict(version=VERSION, name=state.get('project_name', 'Untitled project'),
                datasets=datasets, active=state.get('ws_active'),
                report=state.get('cozy_report'), latest=state.get('report_latest'))


def dumps(payload):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        def encode(value):
            if isinstance(value, pd.DataFrame):
                path = f'data/{len(archive.namelist())}.parquet'
                archive.writestr(path, value.to_parquet(index=True))
                return {'tag': 'frame', 'value': path}
            if isinstance(value, pd.Series):
                return {'tag': 'series', 'value': encode(value.to_frame())}
            if isinstance(value, dict):
                return {'tag': 'dict', 'value': [[encode(k), encode(v)] for k, v in value.items()]}
            if isinstance(value, (list, tuple, set)):
                return {'tag': type(value).__name__, 'value': [encode(v) for v in value]}
            if isinstance(value, bytes):
                path = f'assets/{len(archive.namelist())}.bin'
                archive.writestr(path, value)
                return {'tag': 'bytes', 'value': path}
            if value is pd.NA or value is pd.NaT:
                return {'tag': 'missing', 'value': 'NA' if value is pd.NA else 'NaT'}
            if isinstance(value, pd.Period):
                return {'tag': 'period', 'value': str(value), 'freq': value.freqstr}
            if isinstance(value, (datetime, date, pd.Timestamp)):
                return {'tag': 'timestamp', 'value': value.isoformat()}
            if isinstance(value, (timedelta, pd.Timedelta)):
                return {'tag': 'duration', 'value': pd.Timedelta(value).isoformat()}
            if isinstance(value, np.ndarray):
                return {'tag': 'array', 'value': encode(value.tolist())}
            if isinstance(value, np.generic):
                return encode(value.item())
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            raise ValueError(f'Cannot save this project value: {type(value).__name__}.')
        archive.writestr('project.json', json.dumps(encode(payload), ensure_ascii=False))
    return output.getvalue()


def loads(content):
    if len(content) > MAX_BYTES:
        raise ValueError('Project file exceeds the 512 MB limit.')
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > MAX_BYTES:
                raise ValueError('Invalid or oversized project archive.')
            def decode(value):
                if not isinstance(value, dict):
                    return value
                tag, data = value['tag'], value['value']
                if tag == 'dict': return {decode(k): decode(v) for k, v in data}
                if tag in ('list', 'tuple', 'set'):
                    values = [decode(v) for v in data]
                    return values if tag == 'list' else tuple(values) if tag == 'tuple' else set(values)
                if tag == 'frame': return pd.read_parquet(io.BytesIO(archive.read(data)))
                if tag == 'series': return decode(data).iloc[:, 0]
                if tag == 'bytes': return archive.read(data)
                if tag == 'timestamp': return pd.Timestamp(data)
                if tag == 'duration': return pd.Timedelta(data)
                if tag == 'period': return pd.Period(data, freq=value['freq'])
                if tag == 'array': return np.array(decode(data))
                if tag == 'missing': return pd.NA if data == 'NA' else pd.NaT
                raise ValueError('Unsupported project value type.')
            payload = decode(json.loads(archive.read('project.json')))
        if payload['version'] != VERSION:
            raise ValueError('Unsupported Cozy project version.')
        if not isinstance(payload['name'], str) or any(payload[k] is not None and not isinstance(payload[k], dict) for k in ('report', 'latest')):
            raise ValueError('Invalid project metadata or report.')
        if not isinstance(payload['datasets'], dict): raise ValueError('Invalid dataset registry.')
        dataset_names = set()
        for identity, entry in payload['datasets'].items():
            if not isinstance(entry['frame'], pd.DataFrame): raise ValueError('Invalid dataset table.')
            validate_table(entry['frame'])
            if (not isinstance(identity, str) or not isinstance(entry['name'], str) or not entry['name'].strip()
                    or entry['name'] in dataset_names or entry['name'] == ADD_TAB
                    or not isinstance(entry['source'], str) or not isinstance(entry['parents'], list)
                    or any(not isinstance(parent, str) for parent in entry['parents'])
                    or 'operation' not in entry or not isinstance(entry['state'], dict)):
                raise ValueError('Invalid dataset metadata.')
            dataset_names.add(entry['name'])
        if payload['active'] is not None and payload['active'] not in payload['datasets']:
            raise ValueError('Invalid active dataset.')
        return payload
    except (KeyError, TypeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError('This is not a valid Cozy project file.') from exc


def fingerprint(state):
    # Hash uncompressed content; ZIP timestamps must not affect dirty detection.
    with zipfile.ZipFile(io.BytesIO(dumps(project_payload(state)))) as archive:
        digest = hashlib.sha256()
        for name in sorted(archive.namelist()):
            digest.update(name.encode()); digest.update(archive.read(name))
        return digest.hexdigest()


def replace_project(state, payload=None):
    for key in list(state):
        if dataset_state_key(key) or key.startswith(_PREFIXES + ('ws_', 'report_', 'manual_', 'cozy_report', 'project_')) or key in ('df', 'file_id'):
            del state[key]
    state['ws_datasets'] = payload['datasets'] if payload else {}
    state['project_name'] = payload['name'] if payload else 'Untitled Project'
    if payload:
        if payload['report'] is not None: state['cozy_report'] = payload['report']
        if payload['latest'] is not None: state['report_latest'] = payload['latest']
        if payload['active']:
            activate(state, payload['active'])
            state['ws_dataset_tabs'] = state['ws_datasets'][payload['active']]['name']
    state['project_saved_hash'] = fingerprint(state)


def remove_dataset(state, identity):
    save_active(state)
    del state['ws_datasets'][identity]
    if state.get('ws_active') == identity:
        state.pop('ws_active', None)
        for key in list(state):
            if dataset_state_key(key) or key in ('df', 'file_id'): del state[key]
        remaining = list(state['ws_datasets'])
        if remaining:
            activate(state, remaining[0])
            state['ws_dataset_tabs'] = state['ws_datasets'][remaining[0]]['name']
        else:
            state['ws_dataset_tabs'] = ADD_TAB


def cancel_project_action():
    import streamlit as st
    st.session_state.pop('project_dialog', None)
    st.session_state.pop('project_pending', None)


def request_project_action(action):
    import streamlit as st
    state = st.session_state
    if action in ('new', 'close'):
        queue_project_replacement(action, None)
    else:
        state['project_dialog'] = action


def queue_project_replacement(action, payload):
    import streamlit as st
    state = st.session_state
    if fingerprint(state) != state['project_saved_hash']:
        state['project_pending'] = (action, payload)
        state['project_dialog'] = 'confirm'
    else:
        replace_project(state, payload)


def finish_project_replacement():
    import streamlit as st
    pending = st.session_state.get('project_pending')
    replace_project(st.session_state, pending[1] if pending else None)


def saved_project_download(snapshot_hash):
    import streamlit as st
    state = st.session_state
    state['project_saved_hash'] = snapshot_hash
    if state.get('project_dialog') == 'save_before':
        finish_project_replacement()
    else:
        cancel_project_action()
    state['project_saved_message'] = True


import streamlit as st


@st.dialog('Project', on_dismiss=cancel_project_action)
def project_action_dialog():
    state = st.session_state
    action = state.get('project_dialog')
    if action == 'confirm':
        st.warning('The current project has unsaved changes. Save it before closing?')
        save, discard, cancel = st.columns(3)
        save.button('Save', key='project_save_before', on_click=lambda: state.update(project_dialog='save_before'))
        discard.button('Discard', key='project_confirm', on_click=finish_project_replacement)
        cancel.button('Cancel', key='project_cancel', on_click=cancel_project_action)
    elif action in ('save', 'save_before'):
        st.text_input('Project name', key='project_name')
        prepare_download(state)
        if state.get('project_save_error'):
            st.error(state['project_save_error'])
        else:
            prepared = state['project_download']
            st.caption('The browser saves this .cozy file. Save status confirms the download was requested.')
            st.download_button('Save project and continue' if action == 'save_before' else 'Save project',
                               prepared['content'], prepared['name'], 'application/octet-stream',
                               key='project_download_button', on_click=saved_project_download, args=(prepared['hash'],))
        st.button('Cancel', key='project_cancel', on_click=cancel_project_action)


def render_file_menu():
    from utils.dataset_workspace import initialize
    from utils.project_menu import render_menu
    state = st.session_state
    initialize(state)
    # Retain the editable name after the save dialog's widget is removed.
    state['project_name'] = state.get('project_name', 'Untitled Project')
    if 'project_saved_hash' not in state:
        state['project_saved_hash'] = fingerprint(state)
    event = render_menu(state['project_name'], MAX_BYTES)
    if event and event != state.get('project_last_menu_event'):
        try:
            handle_menu_event(event)
            state['project_last_menu_event'] = event
            st.rerun()
        except (ValueError, TypeError) as exc:
            state['project_last_menu_event'] = event
            st.error(f'Cannot open project: {exc}')
    if state.pop('project_saved_message', False):
        st.success('Project is saved.')
    if state.get('project_dialog'):
        project_action_dialog()


def handle_menu_event(event):
    import base64
    action = event.get('action')
    if action == 'rename':
        name = event.get('name', '').strip()
        if not name or len(name) > 200:
            raise ValueError('Use a project name between 1 and 200 characters.')
        st.session_state['project_name'] = name
    elif action == 'open':
        content = event.get('content', '')
        if len(content) > MAX_BYTES * 4 // 3 + 4:
            raise ValueError('Project file exceeds the 512 MB limit.')
        queue_project_replacement('open', loads(base64.b64decode(content, validate=True)))
    elif action in ('new', 'close', 'save'):
        request_project_action(action)
    else:
        raise ValueError('Unknown project action.')


def prepare_download(state):
    try:
        payload = project_payload(state)
        name = ''.join(c for c in state['project_name'] if c.isalnum() or c in ' -_').strip() or 'Untitled project'
        state['project_download'] = dict(content=dumps(payload), name=name + '.cozy', hash=fingerprint(state))
        state.pop('project_save_error', None)
    except Exception as exc:
        state['project_save_error'] = str(exc)
