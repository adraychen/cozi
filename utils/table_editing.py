"""Non-mutating column edits and exact duplicate cleanup on stored data."""
import math
import operator
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from utils.table_transforms import validate_table, check_size


def _new_name(frame, name, replacing=None):
    name = str(name).strip()
    if not name:
        raise ValueError('Enter a non-empty column name.')
    if name in frame and name != replacing:
        raise ValueError(f'A column named {name} already exists.')
    return name


def rename_column(frame, column, name):
    if column not in frame:
        raise ValueError('Choose an existing column.')
    name = _new_name(frame, name, replacing=column)
    if name == column:
        raise ValueError('Enter a different column name.')
    return validate_table(frame.rename(columns={column: name}).copy())


def remove_columns(frame, columns):
    if not columns or len(set(columns)) != len(columns) or any(c not in frame for c in columns):
        raise ValueError('Choose valid columns to remove, without duplicates.')
    if len(columns) >= len(frame.columns):
        raise ValueError('Keep at least one column in the dataset.')
    return validate_table(frame.drop(columns=columns).copy())


def add_column(frame, name, method='Blank', value=None, value_type='Text', source=None,
               left=None, right=None, calculation=None):
    name = _new_name(frame, name)
    check_size(len(frame), len(frame.columns) + 1)
    result = frame.copy()
    if method == 'Blank':
        result[name] = pd.Series(pd.NA, index=frame.index, dtype='string')
    elif method == 'Constant value':
        if value_type == 'Number':
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('Enter a finite number.')
        elif value_type == 'True/False':
            if value not in (True, False, 'True', 'False'):
                raise ValueError('Choose True or False.')
            value = value is True or value == 'True'
        elif value_type == 'Text':
            value = str(value) if value is not None else ''
        else:
            raise ValueError('Choose a supported value type.')
        result[name] = value
    elif method == 'Copy existing column':
        if source not in frame:
            raise ValueError('Choose an existing source column.')
        result[name] = frame[source].copy()
    elif method == 'Row number':
        result[name] = range(1, len(frame) + 1)
    elif method == 'Calculation':
        for column in (left, right):
            if column not in frame or not is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column]):
                raise ValueError('Choose two numerical columns for the calculation.')
        operations = {'Add': operator.add, 'Subtract': operator.sub, 'Multiply': operator.mul, 'Divide': operator.truediv}
        if calculation not in operations:
            raise ValueError('Choose a supported calculation.')
        if calculation == 'Divide' and frame[right].eq(0).any():
            raise ValueError('The denominator contains zero. Choose another column or resolve zero values first.')
        # Float arithmetic avoids integer overflow; existing missing values remain missing.
        result[name] = operations[calculation](frame[left].astype('Float64'), frame[right].astype('Float64'))
        if result[name].isin([float('inf'), -float('inf')]).any():
            raise ValueError('The calculation produces infinite values. Check the input values.')
    else:
        raise ValueError('Choose a supported column creation method.')
    return validate_table(result)


def duplicate_columns(frame):
    """Map extra copies to the first equivalent column; names are not compared."""
    if frame.empty:
        return {}
    buckets, duplicates = {}, {}
    for name in frame:
        series = frame[name]
        # Short scalar signatures filter candidates; full equality always verifies
        # every row, data type, and missing-value position before declaring a match.
        sample = tuple(None if pd.isna(v) else v for v in series.iloc[:32])
        signature = (str(series.dtype), sample)
        candidates = buckets.setdefault(signature, [])
        original = next((c for c in candidates if series.equals(frame[c])), None)
        if original is None:
            candidates.append(name)
        else:
            duplicates[name] = original
    return duplicates


def remove_duplicate_columns(frame):
    duplicates = duplicate_columns(frame)
    return validate_table(frame.drop(columns=list(duplicates)).copy()), duplicates


def remove_duplicate_rows(frame):
    """Keep the first occurrence in source order; ignore the dataframe index."""
    return validate_table(frame.drop_duplicates(keep='first').reset_index(drop=True).copy())
