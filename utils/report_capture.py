"""Copy authored analysis output and data, without scraping UI or fitted objects."""
from contextvars import ContextVar
from functools import wraps
from copy import deepcopy
import io
import hashlib
import json
import pandas as pd
import streamlit as st
from utils.report_state import structured, key_results, ensure_report, add_snapshot

_capture = ContextVar('report_capture', default=None)


def begin_result():
    """Start at completed result output, leaving configuration controls out."""
    capture = _capture.get()
    if capture is not None:
        capture.update(tables=[], charts=[], content=[])


def reportable(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _capture.set({'tables': [], 'charts': [], 'content': []})
        try:
            return function(*args, **kwargs)
        finally:
            _capture.reset(token)
    return wrapped


def record_table(data, title=None):
    capture = _capture.get()
    if capture is not None:
        frame = data.data if isinstance(getattr(data, 'data', None), pd.DataFrame) else data
        if isinstance(frame, pd.DataFrame):
            table = structured(frame.copy(deep=True))
            table['title'] = title or f'Table {len(capture["tables"])+1}'
            table['source_output'] = title is None
            if not isinstance(frame.index, pd.RangeIndex):
                table = structured(frame.reset_index()) | {'title': table['title'], 'source_output': title is None}
            schema = getattr(data, '_cozy_result_types', None)
            if schema:
                from utils.numerical_formatting import display_rows
                rendered = display_rows(frame, schema, alpha=data._cozy_alpha, settings=data._cozy_settings)
                if not isinstance(frame.index, pd.RangeIndex):
                    rendered = [[index, *row] for index, row in zip(frame.index, rendered)]
                table['display_rows'] = structured(rendered)
            if hasattr(data, '_cozy_display_rows'):
                table['display_rows'] = structured(data._cozy_display_rows)
            capture['tables'].append(table)
            capture['content'].append(('Table', deepcopy(table)))


def record_chart(figure):
    capture = _capture.get()
    if capture is not None:
        capture['charts'].append({'title': figure.layout.title.text or f'Chart {len(capture["charts"])+1}',
                                  'plotly': figure.to_json()})
        capture['content'].append(('Chart', deepcopy(capture['charts'][-1])))


def record_pyplot(figure):
    capture = _capture.get()
    if capture is not None and figure is not None:
        output = io.BytesIO()
        figure.savefig(output, format='png', dpi=180, bbox_inches='tight')
        capture['charts'].append({'title': f'Chart {len(capture["charts"])+1}', 'png': output.getvalue()})
        capture['content'].append(('Chart', deepcopy(capture['charts'][-1])))


def record_text(value, heading=False):
    capture = _capture.get()
    if capture is not None and isinstance(value, str) and value.strip():
        text = value.strip()
        if heading or (text.startswith('**') and text.endswith('**') and text.count('**') == 2):
            capture['content'].append(('Subheading', text.strip('#* ')))
        elif text.startswith('#'):
            capture['content'].append(('Subheading', text.lstrip('# ')))
        else:
            capture['content'].append(('Text', value))


def _add(payload, identity, fingerprint):
    state = st.session_state
    status = state.setdefault('report_add_status', {})
    if status.get(identity) == (fingerprint, True):
        return
    dataset = state.get('ws_datasets', {}).get(state.get('ws_active'))
    add_snapshot(ensure_report(state, dataset), payload)
    status[identity] = (fingerprint, True)


def offer_result(analysis_type, title, variables, settings, results, diagnostics=None,
                 warnings=None, interpretation='', key=None, run_id=None):
    capture = _capture.get() or {'tables': [], 'charts': []}
    dataset = st.session_state.get('ws_datasets', {}).get(st.session_state.get('ws_active'), {})
    frame = dataset.get('frame')
    payload = {'analysis_type': analysis_type, 'title': title, 'variables': structured(variables),
               'settings': structured(settings), 'results': key_results(results),
               'diagnostics': structured(diagnostics), 'warnings': structured(warnings),
               'interpretation': str(interpretation or ''), 'dataset': {
                   'name': dataset.get('name', 'Dataset'), 'id': st.session_state.get('file_id'),
                   'observations': len(frame) if frame is not None else None,
                   'variables': len(frame.columns) if frame is not None else None},
               **deepcopy(capture)}
    st.session_state['report_latest'] = deepcopy(payload)
    button_key = 'report_add_'+(key or analysis_type)
    identity = (button_key, st.session_state.get('ws_active'))
    digest = hashlib.sha256(json.dumps({'type': analysis_type, 'variables': payload['variables'],
        'settings': payload['settings'], 'results': structured(results), 'run_id': run_id},
        sort_keys=True, default=str).encode())
    if frame is not None:
        digest.update(frame.to_json(date_format='iso', default_handler=str).encode())
    fingerprint = digest.hexdigest()
    status = st.session_state.setdefault('report_add_status', {})
    if status.get(identity, (None, False))[0] != fingerprint:
        status[identity] = (fingerprint, False)
    added = status[identity][1]
    st.button('✓ Added to Report' if added else 'Add to Report', key=button_key,
              disabled=added, on_click=_add, args=(payload, identity, fingerprint))
