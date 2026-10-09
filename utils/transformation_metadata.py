"""Carry explicit analytical classifications through table transformations."""
from copy import deepcopy
from utils.variable_metadata import infer_metadata, validate_type_subtype


def inherit_metadata(result, left, metadata, config, registry):
    """Map column lineage, validate overrides, and infer new or incompatible columns."""
    inferred = infer_metadata(result)
    candidates = {column: [metadata[column]] for column in result if column in metadata}
    operation = config['operation']

    def parent_metadata(dataset_id):
        entry = registry[dataset_id]
        return entry['state'].get('variable_metadata') or infer_metadata(entry['frame'])

    if operation == 'Rename column':
        candidates.pop(config['column'], None)
        candidates[config['name']] = [metadata[config['column']]]
    elif operation == 'Add column':
        candidates.pop(config['name'], None)
        if config['method'] == 'Copy existing column':
            candidates[config['name']] = [metadata[config['source']]]
    elif operation == 'Convert Currency to Numeric':
        # Conversion explicitly changes the meaning to a numerical measure.
        for column in config['columns']:
            candidates.pop(column, None)
    elif operation in ('Aggregate', 'Pivot', 'Unpivot'):
        preserved = config['groups'] if operation == 'Aggregate' else config['rows'] if operation == 'Pivot' else config['identifiers']
        candidates = {column: [metadata[column]] for column in preserved}
    elif operation == 'Append':
        for dataset_id in config['inputs']:
            other = parent_metadata(dataset_id)
            for column in result:
                if column in other:
                    candidates.setdefault(column, []).append(other[column])
        candidates.pop(config.get('source_column'), None)
    elif operation == 'Join':
        right = registry[config['right']]['frame']
        other = parent_metadata(config['right'])
        merged_keys = {a for a, b in zip(config['left_keys'], config['right_keys']) if a == b}
        overlaps = set(left) & set(right) - merged_keys
        candidates = {}
        for source, items, suffix in ((left, metadata, '_left'), (right, other, '_right')):
            for column in source:
                output = column + suffix if column in overlaps else column
                if output in result:
                    candidates.setdefault(output, []).append(items[column])

    warnings = []
    for column, items in candidates.items():
        overrides = [item for item in items if item.get('source') == 'User override']
        if not overrides:
            continue
        first = overrides[0]
        roles = {(item['variable_type'], item['variable_subtype'], tuple(item.get('order', []))) for item in overrides}
        if len(roles) > 1:
            warnings.append(f'{column}: conflicting user classifications; automatic classification used. Review in Inspect.')
            continue
        try:
            validate_type_subtype(result[column], first['variable_type'], first['variable_subtype'], first.get('order'))
        except (ValueError, TypeError, OverflowError):
            warnings.append(f'{column}: the previous classification no longer fits the transformed values; automatic classification used. Review in Inspect.')
        else:
            inferred[column] = deepcopy(first)
    return inferred, warnings
