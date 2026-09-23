"""Non-mutating table operations shared by previews and saved datasets."""
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_dtype_equal

MAX_ROWS = 1_000_000
MAX_CELLS = 10_000_000
FUNCTIONS = {'Sum': 'sum', 'Mean': 'mean', 'Minimum': 'min', 'Maximum': 'max',
             'Count nonmissing': 'count', 'Count distinct': 'nunique', 'Count rows': 'size'}


def validate_table(frame):
    if not len(frame.columns):
        raise ValueError('The table must have at least one column.')
    if not frame.columns.is_unique or any(not str(c).strip() for c in frame.columns):
        raise ValueError('Column names must be non-empty and unique.')
    check_size(len(frame), len(frame.columns))
    return frame


def check_size(rows, columns):
    if rows > MAX_ROWS or rows * columns > MAX_CELLS:
        raise ValueError('Result exceeds the workspace limit of 1,000,000 rows or 10,000,000 cells. Reduce the inputs or groups first.')


def _columns(frame, columns):
    if not columns or len(set(columns)) != len(columns) or any(c not in frame for c in columns):
        raise ValueError('Choose at least one valid column, without duplicates.')


def _compatible(left, right):
    if is_dtype_equal(left.dtype, right.dtype):
        return True
    return (is_numeric_dtype(left) and is_numeric_dtype(right)
            and not is_bool_dtype(left) and not is_bool_dtype(right))


def _temporary(columns, name):
    while name in columns:
        name += '_'
    return name


def join_tables(left, right, left_keys, right_keys, how='left', match_missing=False):
    _columns(left, left_keys)
    _columns(right, right_keys)
    if len(left_keys) != len(right_keys):
        raise ValueError('Choose the same number of left and right keys, in matching order.')
    if how not in ('left', 'inner', 'right', 'outer'):
        raise ValueError('Choose a supported join type.')
    for a, b in zip(left_keys, right_keys):
        if not _compatible(left[a], right[b]):
            raise ValueError(f'Join keys {a} and {b} have different data types. Convert them explicitly before joining.')
    l, r = left.copy(), right.copy()
    flag = _temporary(set(left) | set(right), '__join_missing')
    l[flag] = 0 if match_missing else l[left_keys].isna().any(axis=1).astype(int)
    r[flag] = 0 if match_missing else r[right_keys].isna().any(axis=1).astype(int) * 2
    lk, rk = left_keys + [flag], right_keys + [flag]
    # Estimate the complete result before allocating a potentially explosive join.
    # Groupby ordering need not match first appearance; use indexed counts.
    lg = l.groupby(lk, dropna=False, observed=True).size().reset_index(name=_temporary(lk, '__left_count'))
    rg = r.groupby(rk, dropna=False, observed=True).size().reset_index(name=_temporary(rk, '__right_count'))
    # Canonical key names avoid collisions when the original tables have different key labels.
    key_names = list(range(len(lk)))
    lg.columns = key_names + ['left_count']
    rg.columns = key_names + ['right_count']
    counts = lg.merge(rg, on=key_names, how='outer').fillna({'left_count': 0, 'right_count': 0})
    estimated = matched_left = matched_right = 0
    many_to_many = False
    for a, b in counts[['left_count', 'right_count']].itertuples(index=False, name=None):
        a, b = int(a), int(b)
        if a and b:
            estimated += a * b
            matched_left += a
            matched_right += b
            many_to_many |= a > 1 and b > 1
        elif a and how in ('left', 'outer'):
            estimated += a
        elif b and how in ('right', 'outer'):
            estimated += b
    check_size(estimated, len(left.columns) + len(right.columns))
    result = l.merge(r, left_on=lk, right_on=rk, how=how, suffixes=('_left', '_right'), sort=False).drop(columns=flag)
    validate_table(result)
    report = {'Output rows': len(result), 'Matched left rows': matched_left,
              'Unmatched left rows': len(left) - matched_left, 'Unmatched right rows': len(right) - matched_right,
              'Left rows with duplicate keys': int(left.duplicated(left_keys, keep=False).sum()),
              'Right rows with duplicate keys': int(right.duplicated(right_keys, keep=False).sum()),
              'Left rows with missing keys': int(left[left_keys].isna().any(axis=1).sum()),
              'Right rows with missing keys': int(right[right_keys].isna().any(axis=1).sum()),
              'Many-to-many matches': many_to_many}
    return result, report


def append_tables(frames, names, columns='all', source_column=None):
    if len(frames) < 2 or len(frames) != len(names):
        raise ValueError('Choose at least two datasets.')
    if columns not in ('all', 'common'):
        raise ValueError('Choose all or common columns.')
    selected = set(frames[0])
    for frame in frames[1:]:
        selected = selected | set(frame) if columns == 'all' else selected & set(frame)
    if not selected:
        raise ValueError('These datasets have no common columns.')
    if source_column is not None and (not source_column.strip() or any(source_column in f for f in frames)):
        raise ValueError('The source column must have a new, non-empty name.')
    check_size(sum(map(len, frames)), len(selected) + bool(source_column))
    warnings = []
    for col in selected:
        series = [f[col] for f in frames if col in f and f[col].notna().any()]
        if any(not _compatible(series[0], s) for s in series[1:]):
            warnings.append(f'{col}: different data types; values retained without forced conversion.')
    copies = [f.assign(**{source_column: name}) if source_column else f.copy() for f, name in zip(frames, names)]
    result = pd.concat(copies, ignore_index=True, join='outer' if columns == 'all' else 'inner', sort=False)
    return validate_table(result), warnings


def aggregate_table(frame, groups, measures, keep_missing=True):
    if groups:
        _columns(frame, groups)
    if not measures:
        raise ValueError('Add at least one measure.')
    names = [m['name'].strip() for m in measures]
    if any(not n for n in names) or len(set(names)) != len(names) or set(names) & set(groups):
        raise ValueError('Output measure names must be unique, non-empty, and different from grouping columns.')
    named = {}
    for spec, name in zip(measures, names):
        column, function = spec['column'], spec['function']
        if function not in FUNCTIONS:
            raise ValueError('Choose a supported calculation.')
        _columns(frame, [column])
        if function in ('Sum', 'Mean') and (not is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column])):
            raise ValueError(f'{function} requires a numerical column: {column}.')
        fn = (lambda x: x.sum(min_count=1)) if function == 'Sum' else FUNCTIONS[function]
        named[name] = pd.NamedAgg(column=column, aggfunc=fn)
    if groups:
        result = frame.groupby(groups, dropna=not keep_missing, observed=True, sort=False).agg(**named).reset_index()
    else:
        result = pd.DataFrame([{name: (getattr(frame[spec.column], spec.aggfunc)() if isinstance(spec.aggfunc, str) and spec.aggfunc != 'size'
                                      else len(frame) if spec.aggfunc == 'size' else spec.aggfunc(frame[spec.column]))
                                for name, spec in named.items()}])
    return validate_table(result)


def pivot_table(frame, rows, columns, values, function=None, fill_zero=False):
    _columns(frame, rows + [columns, values])
    keys = rows + [columns]
    if function is None and frame.duplicated(keys).any():
        raise ValueError('Repeated row/column combinations found. Choose an aggregation to pivot them.')
    check_size(len(frame[rows].drop_duplicates()), len(frame[columns].drop_duplicates()) + len(rows))
    spec = {'column': values, 'function': function or 'Count rows', 'name': '__pivot_value'}
    while spec['name'] in keys:
        spec['name'] += '_'
    if function:
        grouped = aggregate_table(frame, keys, [spec], keep_missing=True).set_index(keys)[spec['name']]
    else:
        grouped = frame.set_index(keys)[values]
    result = grouped.unstack(columns)
    if fill_zero:
        result = result.fillna(0)
    labels = ['(missing)' if pd.isna(c) else str(c) for c in result.columns]
    if len(set(labels)) != len(labels) or set(labels) & set(rows):
        raise ValueError('Pivot values produce duplicate column names or conflict with row identifiers. Rename the input values first.')
    result.columns = labels
    return validate_table(result.reset_index())


def unpivot_table(frame, identifiers, columns, variable_name='variable', value_name='value', drop_missing=False):
    _columns(frame, identifiers + columns)
    if not columns:
        raise ValueError('Select columns to unpivot.')
    if not variable_name.strip() or not value_name.strip() or variable_name == value_name or {variable_name, value_name} & set(identifiers):
        raise ValueError('Output names must be non-empty, distinct, and different from identifier columns.')
    check_size(len(frame) * len(columns), len(identifiers) + 2)
    # Temporary value name allows a source column named "value" to be unpivoted.
    temporary = _temporary(frame.columns, '__unpivot_value')
    temporary_variable = _temporary(list(frame.columns) + [temporary], '__unpivot_variable')
    result = frame.melt(id_vars=identifiers, value_vars=columns, var_name=temporary_variable, value_name=temporary)
    result = result.rename(columns={temporary: value_name, temporary_variable: variable_name})
    if drop_missing:
        result = result.dropna(subset=[value_name])
    return validate_table(result.reset_index(drop=True))
