"""Reusable, deterministic assessment of month components and dated chronology."""
from dataclasses import dataclass, field
import re
import pandas as pd
from utils.temporal import SUBTYPES, infer_temporal_subtype, parse_temporal


@dataclass
class TemporalAssessment:
    kind: str | None
    status: str = 'Supported'
    year_columns: list = field(default_factory=list)
    year_column: str | None = None
    years: list = field(default_factory=list)
    existing_columns: list = field(default_factory=list)
    repeated_months: list = field(default_factory=list)
    missing_year_rows: int = 0
    message: str = ''


def temporal_granularity(series, metadata=None):
    item = metadata or {}
    if item.get('variable_type') and item['variable_type'] != 'Date / Time':
        return None
    subtype = item.get('variable_subtype')
    if subtype in SUBTYPES:
        return subtype
    return infer_temporal_subtype(series)


def assess_temporal_selection(df, column, metadata=None, objective='Chronological reporting periods',
                              dimensions=None, year_column=None):
    """Recommend only row-aligned variables, not columns with convenient names."""
    metadata = metadata or {}
    kind = temporal_granularity(df[column], metadata.get(column))
    assessment = TemporalAssessment(kind)
    if kind != 'Month':
        return assessment
    months = parse_temporal(df[column], 'Month')
    candidates = {}
    dated = {}
    for other in df:
        if other == column:
            continue
        item = metadata.get(other, {})
        granularity = temporal_granularity(df[other], item)
        text = df[other].dropna()
        named_year = bool(re.search(r'\byear\b', str(other).replace('_', ' '), re.I))
        if granularity == 'Year' or named_year:
            years = parse_temporal(df[other], 'Year')
            if len(text) and years.notna().sum() / len(text) >= .8:
                candidates[other] = years
        if granularity in ('Year-Month', 'Date', 'DateTime'):
            dates = parse_temporal(df[other], granularity)
            relevant = months.notna()
            if relevant.any() and dates.loc[relevant].notna().all() and dates.loc[relevant].dt.month.eq(months.loc[relevant]).all():
                dated[other] = dates
    assessment.year_columns = list(candidates)
    conflicting = False
    if year_column is not None:
        if year_column not in candidates:
            assessment.status = 'Blocked'
            assessment.message = 'The chosen year column does not contain valid four-digit year components.'
            return assessment
        year_values = candidates[year_column]
        assessment.year_column = year_column
    elif candidates:
        first = next(iter(candidates))
        year_values = candidates[first]
        for values in candidates.values():
            both = months.notna() & year_values.notna() & values.notna()
            if year_values.loc[both].ne(values.loc[both]).any():
                conflicting = True
        # Differing missing coverage also needs a conscious choice.
        conflicting |= any(not values.loc[months.notna()].equals(year_values.loc[months.notna()]) for values in candidates.values())
        assessment.year_column = first if not conflicting else None
    elif dated:
        first = next(iter(dated))
        year_values = dated[first].dt.year.astype('Int64')
        conflicting = any(dates.dt.year.loc[months.notna()].ne(year_values.loc[months.notna()]).any() for dates in dated.values())
    else:
        assessment.status = 'Warning'
        assessment.message = 'Month alone cannot identify dated periods across years. Year information is unavailable; month-of-year comparisons can still summarize these observations.'
        return assessment
    if conflicting:
        if objective == 'Monthly seasonal profile (pool years)':
            assessment.status = 'Warning'
            assessment.message = 'Year sources disagree, but the explicitly selected seasonal profile pools years and does not assign dated periods.'
            return assessment
        assessment.status = 'Blocked'
        if not candidates:
            assessment.existing_columns = list(dated)
            assessment.message = 'The dated variables disagree about the reporting years. Select the dated variable that represents the intended reporting calendar.'
            return assessment
        assessment.message = 'The possible year sources disagree or have different coverage. Choose the corresponding Year column; Cozy will not guess which components belong together.'
        return assessment
    relevant = months.notna()
    assessment.missing_year_rows = int((relevant & year_values.isna()).sum())
    assessment.years = sorted(int(year) for year in year_values.loc[relevant].dropna().unique())
    for other, dates in dated.items():
        known = relevant & year_values.notna()
        if known.any() and dates.loc[known].dt.year.eq(year_values.loc[known]).all():
            assessment.existing_columns.append(other)
    # Prefer declared monthly variables over more precise event dates.
    assessment.existing_columns.sort(key=lambda c: temporal_granularity(df[c], metadata.get(c)) != 'Year-Month')
    pairs = pd.DataFrame({'month': months, 'year': year_values}).dropna()
    assessment.repeated_months = [int(value) for value in pairs.groupby('month')['year'].nunique().loc[lambda values: values > 1].index]
    if dimensions:
        grouped = df[list(dimensions)].copy()
        grouped['_temporal_year'] = year_values
        multiple = grouped.loc[relevant].groupby(list(dimensions), observed=True, dropna=False)['_temporal_year'].nunique().gt(1).any()
    else:
        multiple = len(assessment.years) > 1
    seasonal = objective == 'Monthly seasonal profile (pool years)'
    if seasonal:
        assessment.status = 'Warning'
        assessment.message = 'Monthly seasonality pools years intentionally. Changes compare month-of-year aggregates, not consecutive dated reporting periods.'
    elif multiple or assessment.missing_year_rows:
        assessment.status = 'Blocked'
        assessment.message = ('Your dataset contains observations from multiple years. Using Month alone for chronological analysis may combine the same month from different years or place observations in the wrong order.'
                              if multiple else 'Some month observations have no corresponding year. Their dated order cannot be verified.')
    return assessment
