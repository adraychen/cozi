"""Temporal subtype inference and parsing without inventing dates for clock times."""
import datetime as dt
import calendar
import re
import pandas as pd

SUBTYPES = ['Date', 'Time', 'DateTime', 'Duration', 'Year', 'Month', 'Year-Month', 'Quarter', 'Year-Quarter']


def temporal_columns(groups):
    return list(dict.fromkeys(groups.get('temporal', []) + sum(
        (groups.get(kind, []) for kind in ('datetime', 'time', 'duration', 'year', 'month', 'quarter', 'year_quarter')), [])))


def period_frequencies(subtype):
    if subtype == 'Year':
        return ['Yearly']
    if subtype == 'Year-Month':
        return ['Monthly', 'Quarterly', 'Yearly']
    if subtype == 'Year-Quarter':
        return ['Quarterly', 'Yearly']
    if subtype in ('Time', 'Duration', 'Month', 'Quarter'):
        return []
    return ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']


def period_values(series, frequency, subtype=None):
    if frequency not in period_frequencies(subtype):
        raise ValueError('The selected Temporal subtype does not support this period frequency.')
    codes = {'Daily': 'D', 'Weekly': 'W-SUN', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    if subtype == 'Year':
        years = parse_temporal(series, 'Year')
        periods = pd.Series(pd.PeriodIndex(years.fillna(2000).astype(str), freq='Y'), index=series.index)
        return periods.where(years.notna())
    if subtype == 'Year-Quarter':
        return parse_temporal(series, subtype).dt.asfreq(codes[frequency])
    return parse_temporal(series, 'DateTime').dt.to_period(codes[frequency])
MONTH_NAMES = {name.lower(): i for i in range(1, 13)
               for name in (calendar.month_name[i], calendar.month_abbr[i])}
MONTH_NAMES['sept'] = 9


def is_month_name_series(series):
    values = series.dropna()
    return bool(len(values) and values.astype('string').str.strip().str.lower().isin(MONTH_NAMES).mean() >= .8)


CLOCK = re.compile(r'^\s*(\d{1,2}):(\d{2})(?::(\d{2})(\.\d{1,9})?)?\s*([AP]M)?\s*$', re.I)
DURATION = re.compile(r'(^[+-]?P(?=\d|T)|\b(?:days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|milliseconds?|microseconds?|nanoseconds?)\b|\d\s*(?:d|h|m|s|ms|us|ns)\b)', re.I)


def _clock(value):
    if isinstance(value, dt.time):
        if value.tzinfo is not None:
            return pd.NaT
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second, microseconds=value.microsecond)
    match = CLOCK.fullmatch(str(value))
    if not match:
        return pd.NaT
    hour, minute, second, fraction, meridiem = match.groups()
    hour, minute, second = int(hour), int(minute), int(second or 0)
    if minute >= 60 or second >= 60:
        return pd.NaT
    if meridiem:
        if not 1 <= hour <= 12:
            return pd.NaT
        hour = hour % 12 + (12 if meridiem.upper() == 'PM' else 0)
    if hour >= 24:
        return pd.NaT
    return pd.Timedelta(hours=hour, minutes=minute, seconds=second) + pd.Timedelta(int((fraction or '.')[1:].ljust(9, '0') or '0'), unit='ns')


def _duration(value):
    if isinstance(value, (dt.timedelta, pd.Timedelta)):
        return pd.Timedelta(value)
    text = str(value).strip()
    match = CLOCK.fullmatch(text)
    if match and not match.group(5):
        # HH:MM is elapsed hours/minutes under an explicit Duration override.
        if int(match.group(2)) >= 60 or int(match.group(3) or 0) >= 60:
            return pd.NaT
        if match.group(3) is None:
            text += ':00'
    elif not DURATION.search(text):
        return pd.NaT  # Unitless numbers must not silently become nanoseconds.
    try:
        return pd.to_timedelta(text, errors='coerce')
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _calendar(value):
    if isinstance(value, (dt.time, dt.timedelta, pd.Timedelta)):
        return pd.NaT
    if not isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        text = str(value).strip()
        if text.lower() in MONTH_NAMES:
            return pd.NaT
        if CLOCK.fullmatch(text) or DURATION.search(text) or not re.search(r'[-/.A-Za-z]', text):
            return pd.NaT
    try:
        return pd.to_datetime(value, errors='coerce')
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _month_year(value):
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return pd.Timestamp(value).normalize() if value.day == 1 else pd.NaT
    text = str(value).strip()
    match = re.fullmatch(r'(\d{4})[-/](0[1-9]|1[0-2])', text)
    if match:
        year, month = map(int, match.groups())
    else:
        match = re.fullmatch(r'([A-Za-z]+)[\s/-]+(\d{4})', text)
        reverse = re.fullmatch(r'(\d{4})[\s/-]+([A-Za-z]+)', text)
        if not match and not reverse:
            return pd.NaT
        name, year = match.groups() if match else reverse.groups()[::-1]
        month = MONTH_NAMES.get(name.lower())
        if not month:
            return pd.NaT
        year = int(year)
    try:
        return pd.Timestamp(year=year, month=month, day=1)
    except (ValueError, OverflowError):
        return pd.NaT


def parse_temporal(series, subtype):
    if subtype not in SUBTYPES:
        raise ValueError('Choose ' + ', '.join(SUBTYPES) + ' for a Temporal variable.')
    if subtype == 'Year':
        # Explicit year components, not epoch offsets or fabricated full dates.
        text = series.astype('string').str.strip()
        numeric = pd.to_numeric(text, errors='coerce')
        valid = text.str.fullmatch(r'\d{4}(?:\.0+)?', na=False) & numeric.between(1000, 9999)
        return numeric.where(valid).astype('Int64').rename(series.name)
    if subtype == 'Month':
        text = series.astype('string').str.strip().str.lower()
        numeric = pd.to_numeric(text, errors='coerce')
        valid = text.str.fullmatch(r'\d{1,2}(?:\.0+)?', na=False) & numeric.between(1, 12)
        return numeric.where(valid).fillna(text.map(MONTH_NAMES)).astype('Int64').rename(series.name)
    if subtype == 'Quarter':
        text = series.astype('string').str.strip().str.upper()
        numeric = pd.to_numeric(text.str.replace(r'^Q', '', regex=True), errors='coerce')
        valid = text.str.fullmatch(r'Q?[1-4](?:\.0+)?', na=False)
        return numeric.where(valid).astype('Int64').rename(series.name)
    if subtype == 'Year-Quarter':
        if isinstance(series.dtype, pd.PeriodDtype):
            return series.dt.asfreq('Q')
        def quarter(value):
            text = str(value).strip().upper()
            match = re.fullmatch(r'(\d{4})[\s/-]?Q([1-4])', text)
            reverse = re.fullmatch(r'Q([1-4])[\s/-]+(\d{4})', text)
            if not match and not reverse:
                return pd.NaT
            year, q = match.groups() if match else reverse.groups()[::-1]
            return pd.Period(f'{year}Q{q}', freq='Q')
        return pd.Series(series.astype('string').map(quarter), index=series.index, dtype='period[Q-DEC]', name=series.name)
    if subtype == 'Year-Month':
        # The first day is an internal period anchor; input may not contain a day.
        parsed = {value: _month_year(value) for value in series.dropna().unique()}
        return pd.to_datetime(series.map(parsed), errors='coerce').rename(series.name)
    parser = _clock if subtype == 'Time' else _duration if subtype == 'Duration' else _calendar
    # Repeated labels/dates must not invoke the scalar date parser for every row.
    # Keep this cache local to the call and retain full-row weighting for inference.
    parsed_text = {}
    values = []
    for value in series:
        if pd.isna(value):
            parsed = pd.NaT
        elif isinstance(value, str):
            if value not in parsed_text:
                parsed_text[value] = parser(value)
            parsed = parsed_text[value]
        else:
            parsed = parser(value)
        values.append(parsed)
    if subtype in ('Time', 'Duration'):
        return pd.Series(pd.to_timedelta(values), index=series.index, name=series.name)
    # Normalize timezone-aware inputs to UTC; never attach today's date to a clock.
    has_zone = any(pd.notna(v) and getattr(v, 'tzinfo', None) is not None for v in values)
    result = pd.Series(pd.to_datetime(values, errors='coerce', utc=has_zone), index=series.index, name=series.name)
    if has_zone:
        result = result.dt.tz_convert(None)
    if subtype == 'Date':
        # A Date override must not silently discard meaningful time components.
        result = result.where(result == result.dt.normalize())
    return result


def infer_temporal_subtype(series):
    values = series.dropna()
    if not len(values):
        if pd.api.types.is_timedelta64_dtype(series):
            return 'Duration'
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'DateTime'
        return None
    if pd.api.types.is_timedelta64_dtype(series):
        return 'Duration'
    if isinstance(series.dtype, pd.PeriodDtype):
        return 'Year-Quarter' if series.dtype.freq.freqstr.startswith('Q') else 'Year-Month'
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    if is_month_name_series(values):
        return 'Month'
    text = values.astype('string').str.strip()
    if text.str.fullmatch(r'[Qq][1-4]', na=False).mean() >= .8:
        return 'Quarter'
    if parse_temporal(values, 'Year-Quarter').notna().mean() >= .8:
        return 'Year-Quarter'
    if not pd.api.types.is_datetime64_any_dtype(series) and parse_temporal(values, 'Year').notna().mean() >= .8:
        return 'Year'
    if not pd.api.types.is_datetime64_any_dtype(series) and parse_temporal(values, 'Year-Month').notna().mean() >= .8:
        return 'Year-Month'
    times = parse_temporal(values, 'Time')
    if times.notna().mean() >= .8:
        return 'Time'
    durations = parse_temporal(values, 'Duration')
    if durations.notna().mean() >= .8:
        return 'Duration'
    dates = parse_temporal(values, 'DateTime')
    if dates.notna().mean() < .8:
        return None
    # An explicit timestamp remains DateTime, including textual midnight values.
    if any(isinstance(v, str) and re.search(r'\d[T\s]+\d{1,2}:', v) for v in values):
        return 'DateTime'
    clean = dates.dropna()
    return 'Date' if (clean == clean.dt.normalize()).all() else 'DateTime'


def temporal_value(value, subtype):
    if pd.isna(value):
        return 'N/A'
    if subtype == 'Year':
        return str(int(value))
    if subtype == 'Month':
        return calendar.month_name[int(value)]
    if subtype == 'Quarter':
        return f'Q{int(value)}'
    if subtype == 'Year-Quarter':
        return f'Q{value.quarter} {value.year}'
    if subtype == 'Year-Month':
        return pd.Timestamp(value).strftime('%Y-%m')
    if subtype == 'Date':
        return str(pd.Timestamp(value).date())
    if subtype == 'Time':
        text = str(pd.Timedelta(value))
        return text.split('days ', 1)[-1]
    return str(value)
