"""Group statistics independent of Streamlit. Internal columns avoid user-name collisions."""
import numpy as np
import pandas as pd

STATISTICS = {"Sum": "sum", "Mean": "mean", "Median": "median", "Minimum": "min",
              "Maximum": "max", "Standard deviation": "std", "Non-missing count": "count"}
FREQUENCIES = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}


def prepare(df, measure, dimensions):
    data = pd.DataFrame({"_value": pd.to_numeric(df[measure], errors="coerce").replace([np.inf, -np.inf], np.nan)})
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
             analysis, window=3, lag=1, zero_fill=False):
    data, keys = prepare(df, measure, dimensions)
    dates = pd.to_datetime(df[date_column], errors="coerce")
    data["Period"] = dates.dt.to_period(FREQUENCIES[frequency])
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
        if analysis == "Period-over-Period Comparison":
            values["Previous"] = series.shift(lag)
            values["Change"] = series - values["Previous"]
            values["Change %"] = values["Change"].div(values["Previous"].abs().where(values["Previous"] != 0)) * 100
        elif analysis == "Running & Cumulative Statistics":
            values["Cumulative sum"] = series.cumsum()
            values["Expanding mean"] = series.expanding().mean()
            values["Observed periods"] = series.notna().cumsum()
        else:
            values["Rolling average"] = series.rolling(window, min_periods=window).mean()
        values.index.name = "Period"
        values = values.reset_index()
        values["Period"] = values["Period"].dt.to_timestamp()
        parts.append(values)
    return pd.concat(parts, ignore_index=True), keys
