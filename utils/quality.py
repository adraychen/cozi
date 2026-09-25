import pandas as pd
from pandas.io.formats.style import Styler
from typing import List


def leading_trailing_whitespace_pct(series: pd.Series) -> float:
    """Percentage of actual, nonmissing text cells with boundary whitespace."""
    text_values = [value for value in series.dropna() if isinstance(value, str)]
    if not text_values:
        return 0.0
    return sum(value != value.strip() for value in text_values) / len(text_values) * 100


def check_numerical_quality(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Quality report for numerical columns."""
    if not columns:
        return pd.DataFrame()

    rows = []
    total_rows = len(df)

    for col in columns:
        missing    = df[col].isna().sum()
        missing_pct = round((missing / total_rows) * 100, 1) if total_rows > 0 else 0.0
        unique     = df[col].nunique()
        unique_pct = round((unique / total_rows) * 100, 1) if total_rows > 0 else 0.0

        rows.append({
            "Column":       col,
            "Missing Values": missing,
            "Missing %":    missing_pct,
            "Unique Values": unique,
            "Unique %":     unique_pct,
            "Minimum":      df[col].min(),
            "Maximum":      df[col].max(),
            "Average":      round(df[col].mean(), 2) if not df[col].isna().all() else "—",
        })

    return pd.DataFrame(rows)


def check_categorical_quality(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Quality report for categorical (text) columns."""
    if not columns:
        return pd.DataFrame()

    rows = []
    total_rows = len(df)

    for col in columns:
        missing     = df[col].isna().sum()
        missing_pct = round((missing / total_rows) * 100, 1) if total_rows > 0 else 0.0
        unique      = df[col].nunique()
        unique_pct  = round((unique / total_rows) * 100, 1) if total_rows > 0 else 0.0

        if df[col].dtype == object:
            lower_unique = df[col].dropna().str.strip().str.lower().nunique()
            case_issue   = "Yes" if lower_unique < unique else "No"
            has_special  = df[col].dropna().str.contains(r"[^a-zA-Z0-9\s]", regex=True).any()
            special_chars = "Yes" if has_special else "No"
        else:
            case_issue    = "No"
            special_chars = "No"

        rows.append({
            "Column":                  col,
            "Missing Values":          missing,
            "Missing %":               missing_pct,
            "Unique Values":           unique,
            "Unique %":                unique_pct,
            "Case Inconsistency":      case_issue,
            "Special Characters Found": special_chars,
        })

    return pd.DataFrame(rows)


def check_datetime_quality(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Quality report for date/time columns."""
    if not columns:
        return pd.DataFrame()

    rows = []
    total_rows = len(df)
    now        = pd.Timestamp.now()

    for col in columns:
        series      = pd.to_datetime(df[col], errors="coerce")
        clean       = series.dropna()
        missing     = series.isna().sum()
        missing_pct = round((missing / total_rows) * 100, 1) if total_rows > 0 else 0.0
        unique      = clean.nunique()
        unique_pct  = round((unique / total_rows) * 100, 1) if total_rows > 0 else 0.0
        invalid     = int((clean > now).sum())  # dates after today

        rows.append({
            "Column":                   col,
            "Missing Values":           missing,
            "Missing %":                missing_pct,
            "Unique Values":            unique,
            "Unique %":                 unique_pct,
            "Earliest Date":            clean.min().date() if not clean.empty else "—",
            "Latest Date":              clean.max().date() if not clean.empty else "—",
            "Date Range (days)":        (clean.max() - clean.min()).days if not clean.empty else "—",
            "Future Dates":   invalid,
        })

    return pd.DataFrame(rows)


def colour_missing(val, col_name: str):
    """Return a background colour based on missing % severity."""
    if col_name != "Missing %":
        return ""
    try:
        v = float(val)
        if v >= 20:
            return "background-color: #f8d7da"
        elif v >= 5:
            return "background-color: #fff3cd"
        return ""
    except (ValueError, TypeError):
        return ""


def style_quality_table(df: pd.DataFrame) -> Styler:
    """Apply missing % colour coding to a quality DataFrame."""
    return df.style.map(
        lambda v: colour_missing(v, "Missing %"),
        subset=["Missing %"] if "Missing %" in df.columns else [],
    )
