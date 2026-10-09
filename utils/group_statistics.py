"""Group statistics independent of Streamlit. Internal columns avoid user-name collisions."""
import numpy as np
import pandas as pd
from utils.temporal import period_values

STATISTICS = {"Sum": "sum", "Mean": "mean", "Median": "median", "Minimum": "min",
              "Maximum": "max", "Standard deviation": "std", "Non-missing count": "count", "Row count": "size"}
FREQUENCIES = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}


def prepare(df, measure, dimensions):
    data = pd.DataFrame({"_value": pd.Series(1, index=df.index) if measure is None else pd.to_numeric(df[measure], errors="coerce").replace([np.inf, -np.inf], np.nan)})
    keys = []
    for i, column in enumerate(dimensions):
        key = f"_g{i}"
        keys.append(key)
        values = df[column].astype("string")
        missing = "(Missing)"
        while missing in values.dropna().values:
            missing += "*"
        data[key] = values.fillna(missing)
    if not keys:
        data["_g0"] = "All data"
        keys = ["_g0"]
    return data, keys


def aggregate(data, keys, statistic):
    grouped = data.groupby(keys, observed=True, dropna=False)["_value"]
    result = grouped.sum(min_count=1) if statistic == "Sum" else grouped.agg(STATISTICS[statistic])
    return result.rename("Value").reset_index()


def summary(df, measure, dimensions, statistic):
    if measure is None and statistic != "Row count":
        raise ValueError("Select a numerical column or use Row count.")
    data, keys = prepare(df, measure, dimensions)
    return aggregate(data, keys, statistic), keys


def compare(result, baseline):
    result = result.copy()
    result["Difference"] = result["Value"] - baseline
    result["Difference %"] = result["Difference"] / abs(baseline) * 100 if pd.notna(baseline) and baseline != 0 else np.nan
    return result


def contribution(result, keys, parent=False):
    result = result.copy()
    denominator = result.groupby(keys[:-1], dropna=False)["Value"].transform("sum") if parent and len(keys) > 1 else pd.Series(result["Value"].sum(min_count=1), index=result.index)
    result["Contribution %"] = result["Value"].div(denominator.where(denominator != 0)) * 100
    return result


def temporal(df, measure, dimensions, date_column, frequency, statistic,
             analysis, window=3, lag=1, zero_fill=False, subtype=None):
    if analysis == 'Period-over-Period Comparison':
        from utils.period_comparison import interpret_time, build_timeline, calculate
        axis = interpret_time(df[date_column], subtype)
        planned = build_timeline(df, measure, dimensions, axis, frequency, statistic)
        result = calculate(planned['timeline'], planned['keys'], zero_fill, lag)
        return result.drop(columns=['_period']), planned['keys']
    data, keys = prepare(df, measure, dimensions)
    data["Period"] = period_values(df[date_column], frequency, subtype)
    data = data.dropna(subset=["Period"])
    if data.empty:
        raise ValueError("The selected date column has no valid dates.")
    periods = pd.period_range(data["Period"].min(), data["Period"].max(), freq=FREQUENCIES[frequency])
    grouped = aggregate(data, keys + ["Period"], statistic)
    if len(periods) * grouped[keys].drop_duplicates().shape[0] > 250_000:
        raise ValueError("Too many group/period combinations. Use a coarser frequency or fewer dimensions.")
    parts = []
    for group, values in grouped.groupby(keys, observed=True, dropna=False):
        values = values.set_index("Period").reindex(periods)
        group = group if isinstance(group, tuple) else (group,)
        for key, value in zip(keys, group):
            values[key] = value
        if zero_fill:
            values["Value"] = values["Value"].fillna(0)
        series = values["Value"]
        if analysis == "Running & Cumulative Statistics":
            values["Cumulative sum"] = series.cumsum()
            values["Expanding mean"] = series.expanding().mean()
            values["Observed periods"] = series.notna().cumsum()
        else:
            values["Rolling average"] = series.rolling(window, min_periods=window).mean()
        values.index.name = "Period"
        values = values.reset_index()
        values["Period"] = values["Period"].astype(str) if subtype == 'Year' else values["Period"].dt.to_timestamp()
        parts.append(values)
    return pd.concat(parts, ignore_index=True), keys


def contingency(df, row_column, column_column, include_missing=False):
    """Unweighted Pearson test of independence; no continuity correction."""
    from scipy.stats import chi2_contingency
    if row_column == column_column:
        raise ValueError("Choose two different categorical columns.")
    selected = df[[row_column, column_column]]
    excluded = 0
    if not include_missing:
        selected = selected.dropna()
        excluded = len(df) - len(selected)
    data, keys = prepare(selected, None, [row_column, column_column])
    if data.empty:
        raise ValueError("No rows remain for cross-tabulation.")
    if data[keys[0]].nunique() * data[keys[1]].nunique() > 10_000:
        raise ValueError("The table exceeds 10,000 cells. Choose columns with fewer categories.")
    observed = pd.crosstab(data[keys[0]], data[keys[1]])
    observed.index.name, observed.columns.name = row_column, column_column
    expected = pd.DataFrame(np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.to_numpy().sum(),
                            index=observed.index, columns=observed.columns)
    difference = observed - expected
    result = dict(observed=observed, expected=expected, difference=difference,
                  residual=difference / np.sqrt(expected), excluded=excluded,
                  n=int(observed.to_numpy().sum()), chi2=None, p=None, dof=None,
                  low_expected=int((expected < 5).to_numpy().sum()))
    if min(observed.shape) >= 2:
        chi2, p, dof, _ = chi2_contingency(observed.to_numpy(), correction=False)
        result.update(chi2=float(chi2), p=float(p), dof=int(dof))
    return result
