"""Independent, curated report state and immutable-by-convention evidence copies."""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
import numpy as np
import pandas as pd

BLOCK_TYPES = ['Heading', 'Text', 'Dataset Overview', 'Data Quality Summary',
               'Data Preparation', 'Analysis Result', 'Chart', 'Table', 'Key Findings',
               'AI Interpretation', 'Executive Summary', 'Recommendations',
               'Limitations', 'Conclusion', 'Page Break']
COMPONENTS = ['Variables', 'Method/settings', 'Key results', 'Tables', 'Charts',
              'Diagnostics', 'Warnings/assumptions', 'Interpretation']
EXCLUDED = {'model', 'pipeline', 'winner_model', 'grid', 'preprocessor', 'clean',
            'data', 'df', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test',
            'X_sample', 'shap_values', 'shap_explainer', 'fitted_model'}


def now():
    return datetime.now(ZoneInfo('America/Toronto')).isoformat(timespec='seconds')


def structured(value, depth=0):
    """Copy analytical values, omitting fitted objects and raw training records."""
    if depth > 8:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return {'columns': list(map(str, value.columns)),
                'rows': [[structured(v, depth+1) for v in row] for row in value.itertuples(index=False, name=None)]}
    if isinstance(value, pd.Series):
        return structured(value.to_dict(), depth+1)
    if isinstance(value, dict):
        return {str(k): structured(v, depth+1) for k, v in value.items()
                if str(k) not in EXCLUDED and not callable(v)}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [structured(v, depth+1) for v in value]
    return None


def key_results(result):
    if not isinstance(result, dict):
        return structured(result)
    return {str(k): structured(v) for k, v in result.items()
            if str(k) not in EXCLUDED and isinstance(v, (str, bool, int, float, np.number))}


def ensure_report(state, dataset=None):
    if 'cozy_report' not in state:
        dataset = dataset or {}
        frame = dataset.get('frame')
        state['cozy_report'] = {'id': uuid.uuid4().hex, 'metadata': {
            'title': (dataset.get('name', 'Analysis') + ' Report'), 'author': '',
            'organization': '', 'period': '', 'created': now(),
            'dataset': dataset.get('name', ''),
            'observations': len(frame) if frame is not None else None,
            'variables': len(frame.columns) if frame is not None else None}, 'blocks': []}
    return state['cozy_report']


def new_block(kind, title=None, snapshot=None):
    if kind not in BLOCK_TYPES:
        raise ValueError('Unsupported report block type.')
    return {'id': uuid.uuid4().hex, 'kind': kind, 'title': title or kind, 'text': '',
            'generated': '', 'previous_text': [], 'included': COMPONENTS.copy(),
            'snapshot': deepcopy(snapshot) if snapshot is not None else None}


def add_snapshot(report, payload):
    snapshot = deepcopy(payload)
    snapshot['id'] = uuid.uuid4().hex
    snapshot['timestamp'] = now()
    block = new_block('Analysis Result', payload['title'], snapshot)
    block['text'] = '' if snapshot.get('content') else payload.get('interpretation', '')
    report['blocks'].append(block)
    return block


def move_block(report, block_id, offset):
    index = next(i for i, b in enumerate(report['blocks']) if b['id'] == block_id)
    destination = index + offset
    if 0 <= destination < len(report['blocks']):
        report['blocks'].insert(destination, report['blocks'].pop(index))


def remove_block(report, block_id):
    report['blocks'][:] = [b for b in report['blocks'] if b['id'] != block_id]


def set_draft(block, text):
    block['previous_text'].append(block['text'])
    block['generated'] = str(text)
    block['text'] = str(text)


def restore_text(block):
    if block['previous_text']:
        block['text'] = block['previous_text'].pop()


def block_parts(block):
    """Canonical included content used by UI, AI evidence, and both exports."""
    snapshot = block.get('snapshot') or {}
    parts = []
    included = block['included']
    if block['kind'] == 'Analysis Result' and snapshot.get('content'):
        for kind, value in snapshot['content']:
            component = {'Table': 'Tables', 'Chart': 'Charts'}.get(kind)
            if component is None or component in included:
                parts.append((kind, value))
        if 'Interpretation' in included and block['text']:
            parts.append(('Text', block['text']))
        return parts
    for component, key in [('Variables', 'variables'), ('Method/settings', 'settings'),
                           ('Key results', 'results'), ('Diagnostics', 'diagnostics'),
                           ('Warnings/assumptions', 'warnings')]:
        if component in included and snapshot.get(key):
            parts.append((component, snapshot[key]))
    if 'Tables' in included:
        parts.extend(('Table', table) for table in snapshot.get('tables', []))
    if 'Charts' in included:
        parts.extend(('Chart', chart) for chart in snapshot.get('charts', []))
    if 'Interpretation' in included and block['text']:
        parts.append(('Text', block['text']))
    return parts


def report_evidence(report):
    evidence = []
    for block in report['blocks']:
        if block['kind'] == 'Page Break' or block['kind'] == 'Executive Summary':
            continue
        parts = [(kind, value) for kind, value in block_parts(block) if kind != 'Chart']
        evidence.append({'title': block['title'], 'kind': block['kind'],
                         'source': (block.get('snapshot') or {}).get('dataset'), 'content': parts})
    return evidence
