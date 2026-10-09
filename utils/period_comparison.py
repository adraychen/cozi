"""Validated period comparisons; calendar components never acquire invented years."""
from dataclasses import dataclass
import calendar
import numpy as np
import pandas as pd
from utils.temporal import infer_temporal_subtype, parse_temporal
from utils.group_statistics import prepare, aggregate, FREQUENCIES


@dataclass
class TimeAxis:
    kind: str
    values: pd.Series
    labels: dict


def interpret_time(series, subtype=None, source=None):
    if not len(series.dropna()):
        raise ValueError('No valid temporal observations are available.')
    inferred = infer_temporal_subtype(series)
    kind = subtype or inferred
    if inferred in ('Month', 'Quarter', 'Year', 'Year-Month', 'Year-Quarter') and kind in (None, 'Date', 'DateTime'):
        kind = inferred
    if kind is None and parse_temporal(series, 'Year').notna().mean() >= .8:
        kind = 'Year'
    if kind not in ('Date', 'DateTime', 'Month', 'Quarter', 'Year', 'Year-Month', 'Year-Quarter'):
        raise ValueError('Choose dates, month names, month and year, quarter labels, quarter and year, or four-digit years. Time and Duration do not identify reporting periods.')
    if kind in ('Date', 'DateTime') and not pd.api.types.is_datetime64_any_dtype(series):
        partial = parse_temporal(series, 'Year-Month').notna()
        if partial.any() and not partial.all():
            raise ValueError('Mixed month-and-year labels and full dates have different temporal precision. Use a consistent format or a separate reporting-period column.')
    values = parse_temporal(series, kind)
    if len(series.dropna()) and values.notna().sum() / len(series.dropna()) < .8:
        raise ValueError('The temporal column contains mixed or invalid formats. At least 80% of non-missing values must use the selected temporal format; review its classification in Inspect.')
    labels = {}
    if kind == 'Month':
        original = source if source is not None else series
        for number, label in zip(values, original):
            if pd.notna(number) and pd.notna(label):
                text = str(label).strip()
                labels.setdefault(int(number), text if text.isalpha() else calendar.month_name[int(number)])
    return TimeAxis(kind, values, labels)


def frequency_rules(axis, statistic=None):
    supported = {
        'Date': list(FREQUENCIES), 'DateTime': list(FREQUENCIES),
        'Month': ['Monthly', 'Quarterly'], 'Quarter': ['Quarterly'],
        'Year': ['Yearly'], 'Year-Month': ['Monthly', 'Quarterly', 'Yearly'],
        'Year-Quarter': ['Quarterly', 'Yearly'],
    }[axis.kind]
    rules = {}
    for frequency in FREQUENCIES:
        if statistic == 'Period-end value' and axis.kind not in ('Date', 'DateTime'):
            rules[frequency] = ('Unavailable', 'Period-end value needs full dates or timestamps to order observations within a reporting period.')
        elif frequency not in supported:
            reason = 'The source does not contain the year or finer temporal detail needed for this frequency.'
            rules[frequency] = ('Unavailable', reason)
        elif axis.kind == 'Month' and frequency == 'Quarterly':
            rules[frequency] = ('Warning', 'Month labels alone do not establish quarterly reporting. Clarify whether these are monthly observations or quarterly reports.')
        else:
            rules[frequency] = ('Supported', 'Compatible with the temporal precision.')
    return rules


def period_keys(axis, frequency):
    status, reason = frequency_rules(axis)[frequency]
    if status == 'Unavailable':
        raise ValueError(reason)
    if axis.kind in ('Month', 'Quarter'):
        return ((axis.values - 1) // 3 + 1).astype('Int64') if axis.kind == 'Month' and frequency == 'Quarterly' else axis.values
    if axis.kind == 'Year':
        periods = pd.Series(pd.PeriodIndex(axis.values.fillna(2000).astype(str), freq='Y'), index=axis.values.index).where(axis.values.notna())
    elif axis.kind == 'Year-Quarter':
        periods = axis.values.dt.asfreq(FREQUENCIES[frequency])
    else:
        periods = axis.values.dt.to_period(FREQUENCIES[frequency])
    # Nullable integer ordinals keep missingness without timestamp bounds.
    return pd.Series(periods.array.asi8, index=periods.index).where(periods.notna()).astype('Int64')


def period_label(axis, frequency, key):
    if axis.kind == 'Month' and frequency == 'Monthly':
        return axis.labels.get(int(key), calendar.month_name[int(key)])
    if axis.kind in ('Month', 'Quarter'):
        return f'Q{int(key)}'
    period = pd.Period(ordinal=int(key), freq=FREQUENCIES[frequency])
    if frequency == 'Monthly':
        return period.strftime('%b %Y')
    if frequency == 'Quarterly':
        return f'Q{period.quarter} {period.year}'
    if frequency == 'Yearly':
        return str(period.year)
    if frequency == 'Weekly':
        return f'{period.start_time:%d %b %Y} – {period.end_time:%d %b %Y}'
    return period.strftime('%d %b %Y')


def build_timeline(df, measure, dimensions, axis, frequency, statistic, month_reporting=None,
                   objective='Chronological reporting periods', temporal_metadata=None, year_column=None):
    """Aggregate once and detect absent periods within each segment's observed range."""
    if measure is None and statistic != 'Row count':
        raise ValueError('Choose a numerical metric or use Row count.')
    year_assessment = None
    if axis.kind == 'Month' and axis.values.name in df:
        from utils.temporal_validation import assess_temporal_selection
        declared = dict(temporal_metadata or {})
        declared[axis.values.name] = {'variable_type': 'Date / Time', 'variable_subtype': 'Month'}
        year_assessment = assess_temporal_selection(df, axis.values.name, declared, objective, dimensions, year_column)
        if year_assessment.status == 'Blocked':
            raise ValueError(year_assessment.message + ' Select or create a Month-Year variable, or explicitly choose a monthly seasonal profile.')
    if axis.kind == 'Month' and frequency == 'Quarterly' and month_reporting not in ('Monthly observations', 'Quarterly reports labelled by month'):
        raise ValueError('Clarify what the month labels represent before calculating quarterly comparisons.')
    data, keys = prepare(df, measure, dimensions)
    data['_period'] = period_keys(axis, frequency)
    data['_time'] = axis.values
    data = data.dropna(subset=['_period'])
    if data.empty:
        raise ValueError('No valid temporal observations are available.')
    warnings = []
    invalid = int(axis.values.isna().sum())
    if invalid:
        warnings.append(f'Excluded {invalid:,} rows with missing or invalid temporal values.')
    if axis.kind in ('Month', 'Quarter'):
        if year_assessment is None or not year_assessment.years:
            warnings.append('No unambiguous year is available for chronology. Repeated labels are pooled within each segment; these results cannot identify year-to-year transitions or year-over-year changes.')
        if year_assessment is not None and year_assessment.status == 'Warning':
            warnings.append(year_assessment.message)
    if axis.kind == 'Month' and frequency == 'Quarterly':
        counts = data.groupby(keys + ['_period'], observed=True)['_time'].nunique()
        if month_reporting == 'Quarterly reports labelled by month':
            if counts.gt(1).any():
                raise ValueError('Multiple month labels occur in a quarter within a segment. They cannot all be treated as a single quarterly report; choose monthly observations or use an explicit quarter column.')
            warnings.append('Month labels are treated as quarterly reports based on your clarification; the selected statistic aggregates duplicate report rows.')
        elif counts.lt(3).any():
            warnings.append('Some quarters contain fewer than three observed months. The selected statistic summarizes only the available monthly observations; these are partial-quarter results, not complete quarterly totals.')
    if axis.kind in ('Year-Month', 'Year-Quarter') and frequency in ('Quarterly', 'Yearly'):
        base = data['_time'].dt.to_period('M') if axis.kind == 'Year-Month' else data['_time']
        distinct = data.assign(_coverage=base).groupby(keys + ['_period'], observed=True)['_coverage'].nunique()
        expected = 3 if axis.kind == 'Year-Month' and frequency == 'Quarterly' else 12 if axis.kind == 'Year-Month' else 4 if frequency == 'Yearly' else 1
        if distinct.lt(expected).any():
            warnings.append('Some reporting periods have incomplete finer-period coverage. Aggregation uses only the available observations, not complete reporting totals.')
    if statistic == 'Period-end value':
        if axis.kind not in ('Date', 'DateTime'):
            raise ValueError('Period-end value requires full dates or timestamps to order observations. Choose another statistic or supply a dated column.')
        last_time = data.groupby(keys + ['_period'], observed=True)['_time'].transform('max')
        last = data[data['_time'].eq(last_time)]
        if last.groupby(keys + ['_period'], observed=True)['_value'].nunique(dropna=False).gt(1).any():
            raise ValueError('Conflicting values share the final timestamp in a period. Resolve the ties or choose another statistic.')
        grouped = last.groupby(keys + ['_period'], observed=True)['_value'].agg(lambda values: values.iloc[0]).rename('Value').reset_index()
    else:
        grouped = aggregate(data, keys + ['_period'], statistic)
    # Track missing measurements even where count statistics legitimately return zero.
    blanks = data.groupby(keys + ['_period'], observed=True)['_value'].count().eq(0).sum() if measure is not None else 0
    if blanks:
        warnings.append(f'{blanks:,} observed segment-periods contain no valid metric values. Filling absent periods does not fill these blanks; count statistics still count valid observations.')
    spans = grouped.groupby(keys, observed=True)['_period'].agg(['min', 'max'])
    size = int((spans['max'] - spans['min'] + 1).sum())
    if size > 250_000:
        raise ValueError('Too many segment/period combinations. Choose a coarser frequency or fewer segments.')
    parts = []
    for group, values in grouped.groupby(keys, observed=True, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        values = values.set_index('_period')
        timeline = pd.RangeIndex(int(values.index.min()), int(values.index.max()) + 1, name='_period')
        observed = timeline.isin(values.index)
        values = values.reindex(timeline)
        values['Missing period'] = ~observed
        values['Missing value'] = observed & values['Value'].isna()
        for key, label in zip(keys, group):
            values[key] = label
        parts.append(values.reset_index())
    result = pd.concat(parts, ignore_index=True)
    result['Period'] = result['_period'].map(lambda value: period_label(axis, frequency, value))
    result['Segment'] = result[keys].apply(lambda row: '; '.join(f'{dimensions[i] if dimensions else "Dataset"}: {v}' for i, v in enumerate(row)), axis=1)
    gaps = result.loc[result['Missing period'], ['Period', 'Segment']].copy()
    if len(gaps):
        warnings.append(f'{len(gaps):,} missing segment-periods within the observed reporting ranges. Comparisons use the preceding calendar period, not the preceding available observation.')
    if axis.kind in ('Date', 'DateTime'):
        warnings.append('Boundary periods may be incomplete. Aggregation uses observed records and does not establish complete reporting coverage.')
    return {'timeline': result, 'gaps': gaps, 'warnings': warnings, 'keys': keys,
            'frequency': frequency, 'axis': axis, 'invalid': invalid, 'blank_periods': int(blanks)}


def calculate(timeline, keys, zero_fill=False, lag=1):
    """Fill absent periods only; preserve blanks and strict calendar baselines."""
    result = timeline.copy(deep=True)
    if zero_fill:
        result.loc[result['Missing period'], 'Value'] = 0.
    result['Inserted zero'] = result['Missing period'] & zero_fill
    result['Previous'] = result.groupby(keys, observed=True)['Value'].shift(lag)
    baseline_period = result.groupby(keys, observed=True)['Period'].shift(lag)
    result['Previous period'] = baseline_period
    result['Change'] = result['Value'] - result['Previous']
    denominator = result['Previous'].abs().where(result['Previous'].ne(0))
    result['Change %'] = (result['Change'] / denominator * 100).replace([np.inf, -np.inf], np.nan)
    result['Comparison'] = np.where(result['Previous'].isna() | result['Value'].isna(), 'Unavailable', 'Consecutive periods' if lag == 1 else 'Same period last year')
    result.loc[result['Previous'].eq(0) & result['Value'].notna(), 'Comparison'] = 'Previous value is zero; percentage N/A'
    return result


def interpret_result(result, frequency, max_segments=20):
    narratives = []
    for segment, rows in result.groupby('Segment', sort=False):
        valid = rows[rows['Value'].notna() & ~rows['Inserted zero']]
        if valid.empty:
            narratives.append(f'{segment}: no valid metric observations are available.')
            continue
        if len(valid) == 1:
            narratives.append(f'{segment}: only one valid observed period; no overall change can be estimated.')
            continue
        first, last = valid.iloc[0], valid.iloc[-1]
        change = last['Value'] - first['Value']
        percent = f'{change / abs(first["Value"]) * 100:+.2f}%' if first['Value'] != 0 else 'percentage change N/A because the starting value is zero'
        direction = 'increased' if change > 0 else 'decreased' if change < 0 else 'was unchanged'
        text = f'{segment}: the value {direction} from {first["Period"]} to {last["Period"]} (absolute change {change:+.2f}; {percent}). This describes the overall endpoint movement, not a fitted trend.'
        if rows['Missing period'].any() or rows['Missing value'].any():
            text += ' Gaps or blank values interrupt the observations; the endpoint comparison may span nonconsecutive periods.'
        comparisons = rows.dropna(subset=['Change %'])
        positives, negatives = comparisons[comparisons['Change %'] > 0], comparisons[comparisons['Change %'] < 0]
        for label, subset, position in [('increase', positives, 'idxmax'), ('decrease', negatives, 'idxmin')]:
            if len(subset):
                row = subset.loc[getattr(subset['Change %'], position)()]
                text += f' Largest comparable percentage {label}: {row["Change %"]:+.2f}% from {row["Previous period"]} to {row["Period"]}.'
        if comparisons.empty:
            text += ' No defined period percentage comparisons are available.'
        narratives.append(text)
    if len(narratives) > max_segments:
        return '\n\n'.join(narratives[:max_segments]) + f'\n\nInterpretation shows {max_segments} segments; the results table contains all {len(narratives)}.'
    return '\n\n'.join(narratives)
