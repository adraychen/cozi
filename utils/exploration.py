import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── Numerical ─────────────────────────────────────────────────────────────────

def numerical_stats(series: pd.Series) -> dict:
    """Compute descriptive stats for a numerical column (no Count)."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {}

    n = len(clean)
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n >= 2 else np.nan
    standard_error = std / np.sqrt(n) if n >= 2 else np.nan
    cv = std / mean * 100 if n >= 2 and mean > 0 else np.nan
    margin = float(scipy_stats.t.ppf(0.975, df=n - 1)) * standard_error if n >= 2 else np.nan
    confidence_interval = f"[{mean - margin:.2f}, {mean + margin:.2f}]" if np.isfinite(margin) else "N/A"
    q1, q2, q3 = clean.quantile([0.25, 0.50, 0.75]).values
    return {
        "Mean":             f"{clean.mean():.2f}",
        "Median":           f"{clean.median():.2f}",
        "Std Dev":          f"{clean.std():.2f}",
        "Standard Error (Mean)": f"{standard_error:.2f}" if np.isfinite(standard_error) else "N/A",
        "Coefficient of Variation": f"{cv:.2f}%" if np.isfinite(cv) else "N/A",
        "95% Confidence Interval (Mean)": confidence_interval,
        "Skewness":         f"{clean.skew():.2f}",
        "Kurtosis":         f"{clean.kurtosis():.2f}",
        "Minimum":          f"{clean.min():.2f}",
        "Quartile 1 (25%)": f"{q1:.2f}",
        "Quartile 2 (50%)": f"{q2:.2f}",
        "Quartile 3 (75%)": f"{q3:.2f}",
        "Maximum":          f"{clean.max():.2f}",
    }


def detect_outliers(series: pd.Series) -> dict:
    """
    Detect outliers using the IQR method.
    Outliers are values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR.
    Returns a summary dict and the outlier values as a Series.
    """
    clean = series.dropna()
    if clean.empty:
        return {"count": 0, "pct": "0.00%", "lower_bound": None, "upper_bound": None}, pd.Series(dtype=float)

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = clean[(clean < lower) | (clean > upper)]
    pct = outliers.count() / len(clean) * 100

    summary = {
        "count":       int(outliers.count()),
        "pct":         f"{pct:.2f}%",
        "lower_bound": f"{lower:.2f}",
        "upper_bound": f"{upper:.2f}",
    }
    return summary, outliers.sort_values()


# ── Categorical ───────────────────────────────────────────────────────────────

def categorical_stats(series: pd.Series) -> dict:
    """Compute summary stats for a categorical column."""
    total = len(series)
    missing = int(series.isna().sum())
    clean = series.dropna()

    top_val = clean.value_counts().idxmax() if not clean.empty else "—"
    top_pct = f"{clean.value_counts().max() / total * 100:.2f}%" if not clean.empty else "—"

    return {
        "Count":          total,
        "Missing Values": missing,
        "Missing %":      f"{missing / total * 100:.2f}%",
        "Unique Values":  int(clean.nunique()),
        "Top Value":      str(top_val),
        "Top Value %":    top_pct,
    }


def frequency_table(series: pd.Series) -> pd.DataFrame:
    """Return a frequency table with Count and Percentage columns."""
    total = len(series)
    counts = series.value_counts(dropna=False, sort=not (isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered))
    pct = (counts / total * 100).round(2)

    df = pd.DataFrame({
        "Value":      counts.index.astype(str),
        "Count":      counts.values,
        "Percentage": pct.values,
    })
    df["Value"] = df["Value"].replace("nan", "Missing")
    return df


def rare_categories(series: pd.Series, threshold_pct: float = 1.0) -> pd.DataFrame:
    """
    Return categories that appear in fewer than threshold_pct % of rows.
    """
    total = len(series)
    counts = series.value_counts(dropna=True)
    pct = counts / total * 100
    rare = pct[pct < threshold_pct]
    if rare.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Value":      rare.index.astype(str),
        "Count":      rare.values,
        "Percentage": rare.round(2).values,
    })


# ── Date / Time ───────────────────────────────────────────────────────────────

def datetime_stats(series: pd.Series) -> dict:
    """Compute summary stats for a date/time column."""
    parsed = pd.to_datetime(series, errors="coerce")
    total = len(parsed)
    missing = int(parsed.isna().sum())
    clean = parsed.dropna()

    if clean.empty:
        return {
            "Count": total,
            "Missing Values": missing,
            "Missing %": f"{missing / total * 100:.2f}%",
        }

    date_range_days = (clean.max() - clean.min()).days

    # Most common year and month name
    most_common_year  = int(clean.dt.year.mode().iloc[0])
    most_common_month = clean.dt.month_name().mode().iloc[0]

    return {
        "Count":             total,
        "Missing Values":    missing,
        "Missing %":         f"{missing / total * 100:.2f}%",
        "Earliest Date":     str(clean.min().date()),
        "Latest Date":       str(clean.max().date()),
        "Date Range":        f"{date_range_days:,} days",
        "Most Common Year":  most_common_year,
        "Most Common Month": most_common_month,
    }
