"""Report snapshots from existing Inspect and Transform calculations."""
from copy import deepcopy
import pandas as pd
from utils.report_state import structured, now


def transformation_history(registry, dataset_id, visited=None):
    visited = set() if visited is None else visited
    if dataset_id in visited or dataset_id not in registry:
        return []
    visited.add(dataset_id)
    entry = registry[dataset_id]
    history = []
    for parent in entry.get('parents', []):
        history.extend(transformation_history(registry, parent, visited))
    operation = entry.get('operation')
    if operation:
        for step in operation.get('steps', [operation]):
            history.append({'dataset': entry['name'], 'operation': entry['source'],
                            'settings': deepcopy(step)})
    return history


def dataset_snapshot(kind, entry, metadata, registry, dataset_id, pending=None):
    from utils.loader import get_dataframe_info
    from utils.variable_metadata import effective_data
    from utils.quality import check_numerical_quality, check_categorical_quality, check_datetime_quality
    frame = entry['frame']
    tables = []
    results = {}
    if kind == 'Dataset Overview':
        info = get_dataframe_info(frame).copy()
        info['Variable Type'] = [metadata[c]['variable_type'] for c in info['Column Name']]
        info['Variable Subtype'] = [metadata[c]['variable_subtype'] for c in info['Column Name']]
        tables.append(structured(info) | {'title': 'Column overview'})
        results = {'Observations': len(frame), 'Variables': len(frame.columns)}
    elif kind == 'Data Quality Summary':
        from utils.table_editing import duplicate_columns
        data, groups = effective_data(frame, metadata)
        for title, checker, columns in [
            ('Numerical quality', check_numerical_quality, groups['numerical']),
            ('Categorical quality', check_categorical_quality, groups['categorical']),
            ('Calendar date quality', check_datetime_quality, groups['datetime'])]:
            if columns:
                tables.append(structured(checker(data, columns)) | {'title': title})
        # The loader's missing summary covers every temporal subtype as well.
        info = get_dataframe_info(frame)
        tables.append(structured(info[['Column Name', 'Missing Values', 'Missing %']]) | {'title': 'Missing values'})
        results = {'Extra duplicate rows': int(frame.duplicated(keep='first').sum()),
                   'Duplicate columns': duplicate_columns(frame)}
    else:
        history = transformation_history(registry, dataset_id)
        if pending:
            history.extend({'dataset': entry['name'], 'operation': 'Applied pending changes',
                            'settings': deepcopy(step)} for step in pending['steps'])
        results = {'Performed transformations': structured(history)}
    return {'title': kind, 'analysis_type': kind, 'timestamp': now(),
            'dataset': {'name': entry['name'], 'id': dataset_id, 'observations': len(frame),
                        'variables': len(frame.columns)}, 'results': results, 'tables': tables, 'charts': []}
