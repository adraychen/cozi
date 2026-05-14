import pandas as pd
from typing import List


def check_numerical_quality(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Quality report for numerical columns."""
    if not columns:
        return pd.DataFrame()

    rows = []
    total_rows = len(df)

    for col in columns:
        missing = df[col].isna().sum()
        missing_pct = round((missing / total_rows) * 100, 1) if total_rows > 0 else 0.0
        unique = df[col].nunique()

        rows.append({
            "Column": col,
            "Missing Values": missing,
            "Missing %": missing_pct,
            "Unique Values": unique,
            "Minimum": df[col].min(),
            "Maximum": df[col].max(),
            "Average": round(df[col].mean(), 2) if not df[col].isna().all() else "—",
        })

    return pd.DataFrame(rows)


def check_categorical_quality(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Quality report for categorical (text) columns."""
    if not columns:
        return pd.DataFrame()

    rows = []
    total_rows = len(df)

    for col in columns:
        missing = df[col].isna().sum()
        missing_pct = round((missing / total_rows) * 100, 1) if total_rows > 0 else 0.0
        unique = df[col].nunique()

        # Detect case inconsistencies (e.g. "Apple" vs "apple")
        if df[col].dtype == object:
            lower_unique = df[col].dropna().str.strip().str.lower().nunique()
            case_issue = "Yes" if lower_unique < unique else "No"

            # Detect special characters (non-alphanumeric, non-space)
            has_special = df[col].dropna().str.contains(r"[^a-zA-Z0-9\s]", regex=True).any()
            special_chars = "Yes" if has_special else "No"
        else:
            case_issue = "No"
            special_chars = "No"

        rows.append({
            "Column": col,
            "Missing Values": missing,
            "Missing %": missing_pct,
            "Unique Values": unique,
            "Case Inconsistency": case_issue,
            "Special Characters Found": special_chars,
        })

    return pd.DataFrame(rows)


def check_datetime_quality(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Quality report for date/time columns."""
    if not columns:
        return pd.DataFrame()

    rows = []
    total_rows = len(df)

    for col in columns:
        series = pd.to_datetime(df[col], errors="coerce")
        missing = series.isna().sum()
        missing_pct = round((missing / total_rows) * 100, 1) if total_rows > 0 else 0.0

        rows.append({
            "Column": col,
            "Missing Values": missing,
            "Missing %": missing_pct,
            "Earliest Date": series.min().date() if pd.notna(series.min()) else "—",
            "Latest Date": series.max().date() if pd.notna(series.max()) else "—",
            "Date Range (days)": (series.max() - series.min()).days
            if pd.notna(series.min()) and pd.notna(series.max()) else "—",
        })

    return pd.DataFrame(rows)


def colour_missing(val, col_name: str):
    """Return a background colour based on missing % severity."""
    if col_name != "Missing %":
        return ""
    try:
        v = float(val)
        if v >= 20:
            return "background-color: #f8d7da"   # red
        elif v >= 5:
            return "background-color: #fff3cd"   # yellow
        else:
            return ""
    except (ValueError, TypeError):
        return ""


def style_quality_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply missing % colour coding to a quality DataFrame."""
    return df.style.map(lambda v: colour_missing(v, "Missing %"),
                        subset=["Missing %"] if "Missing %" in df.columns else [])
