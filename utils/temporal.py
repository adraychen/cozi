"""Temporal subtype inference and parsing without inventing dates for clock times."""
import datetime as dt
import re
import pandas as pd

SUBTYPES = ['Date', 'Time', 'DateTime', 'Duration']
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
        if CLOCK.fullmatch(text) or DURATION.search(text) or not re.search(r'[-/.A-Za-z]', text):
            return pd.NaT
    try:
        return pd.to_datetime(value, errors='coerce')
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def parse_temporal(series, subtype):
    if subtype not in SUBTYPES:
        raise ValueError('Choose Date, Time, DateTime, or Duration for a Temporal variable.')
    parser = _clock if subtype == 'Time' else _duration if subtype == 'Duration' else _calendar
    values = [pd.NaT if pd.isna(v) else parser(v) for v in series]
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
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
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
    if subtype == 'Date':
        return str(pd.Timestamp(value).date())
    if subtype == 'Time':
        text = str(pd.Timedelta(value))
        return text.split('days ', 1)[-1]
    return str(value)
