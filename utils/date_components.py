"""Validated, non-mutating construction of temporal columns from components."""
import numpy as np
import pandas as pd
from utils.temporal import parse_temporal

COMBINATIONS = {'Year + Month': 'Year-Month', 'Year + Quarter': 'Year-Quarter', 'Year + Month + Day': 'Date'}
DISPLAY_FORMATS = {'Year + Month': ['YYYY-MM', 'MMM YYYY', 'MMMM YYYY'],
                   'Year + Quarter': ['YYYY-[Q]Q', '[Q]Q YYYY'], 'Year + Month + Day': ['YYYY-MM-DD']}
STRICT = 'Stop on invalid or missing components'
KEEP_MISSING = 'Leave affected values missing'


class DateComponentError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(f'{len(issues):,} rows contain invalid or missing date components. Resolve them, or explicitly choose to leave affected values missing.')


def component_metadata(config):
    combination = config['combination']
    roles = ['year', 'quarter'] if combination == 'Year + Quarter' else ['year', 'month'] + (['day'] if combination == 'Year + Month + Day' else [])
    return {'variable_type': 'Date / Time', 'variable_subtype': COMBINATIONS[combination],
            'source': 'User override', 'order': [], 'origin': 'Combine Date Components',
            'display_format': config.get('display_format', DISPLAY_FORMATS[combination][0]),
            'components': {role: config[role] for role in roles}}


def combine_date_components(frame, combination, year, month=None, quarter=None, day=None,
                            missing_policy=STRICT, display_format=None):
    if combination not in COMBINATIONS:
        raise ValueError('Choose a supported date component combination.')
    roles = {'year': year, 'quarter': quarter} if combination == 'Year + Quarter' else {'year': year, 'month': month}
    if combination == 'Year + Month + Day':
        roles['day'] = day
    if any(column not in frame for column in roles.values()):
        raise ValueError('Choose an existing column for every required date component.')
    if len(set(roles.values())) != len(roles):
        raise ValueError('Choose distinct source columns for the date components.')
    if missing_policy not in (STRICT, KEEP_MISSING):
        raise ValueError('Choose how invalid or missing components should be handled.')
    if display_format is not None and display_format not in DISPLAY_FORMATS[combination]:
        raise ValueError('Choose a supported display format for this combination.')
    reasons = pd.Series('', index=frame.index, dtype='string')
    parsed = {}
    for role, column in roles.items():
        raw = frame[column]
        if role == 'day':
            text = raw.astype('string').str.strip()
            number = pd.to_numeric(text, errors='coerce')
            valid = text.str.fullmatch(r'\d{1,2}(?:\.0+)?', na=False) & number.between(1, 31)
            parsed[role] = number.where(valid).astype('Int64')
        else:
            parsed[role] = parse_temporal(raw, {'year': 'Year', 'month': 'Month', 'quarter': 'Quarter'}[role])
        missing = raw.isna() | raw.astype('string').str.strip().eq('').fillna(False)
        invalid = parsed[role].isna() & ~missing
        reasons.loc[missing] += f'Missing {role}; '
        reasons.loc[invalid] += f'Invalid {role}; '
    valid_components = reasons.eq('')
    if combination == 'Year + Quarter':
        labels = parsed['year'].fillna(2000).astype(str) + 'Q' + parsed['quarter'].fillna(1).astype(str)
        values = pd.Series(pd.PeriodIndex(labels, freq='Q'), index=frame.index).where(valid_components)
    else:
        parts = pd.DataFrame({'year': parsed['year'].fillna(2000).astype(int),
                              'month': parsed['month'].fillna(1).astype(int),
                              'day': parsed['day'].fillna(1).astype(int) if 'day' in parsed else 1}, index=frame.index)
        values = pd.to_datetime(parts, errors='coerce').where(valid_components)
        invalid_dates = valid_components & values.isna()
        reasons.loc[invalid_dates] += 'Invalid calendar date or unsupported date range; '
    affected = reasons.ne('')
    issues = pd.DataFrame({'Row': np.arange(1, len(frame) + 1)}, index=frame.index)
    for role, column in roles.items():
        issues[role.title() + ' (' + str(column) + ')'] = frame[column]
    issues['Reason'] = reasons.str.rstrip('; ')
    issues = issues.loc[affected].reset_index(drop=True)
    if len(issues) and missing_policy == STRICT:
        raise DateComponentError(issues)
    return values, issues


def display_component_values(series, metadata, canonical=False):
    """Present declared precision rather than an internal month-start anchor."""
    subtype = metadata.get('variable_subtype')
    fmt = metadata.get('display_format')
    if subtype == 'Year-Month':
        pattern = '%Y-%m' if canonical or fmt == 'YYYY-MM' else '%b %Y' if fmt == 'MMM YYYY' else '%B %Y'
        return series.dt.strftime(pattern).where(series.notna(), pd.NA)
    if subtype == 'Year-Quarter':
        return series.map(lambda value: pd.NA if pd.isna(value) else
                          f'{value.year}-Q{value.quarter}' if canonical or fmt == 'YYYY-[Q]Q' else f'Q{value.quarter} {value.year}')
    return series.dt.strftime('%Y-%m-%d').where(series.notna(), pd.NA)


def component_preview(frame, metadata):
    preview = frame.copy()
    for column, item in metadata.items():
        if column in preview and item.get('origin') == 'Combine Date Components':
            preview[column] = display_component_values(preview[column], item)
    return preview
