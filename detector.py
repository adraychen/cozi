import pandas as pd
from typing import Dict, List


def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify each column as Numerical, Categorical, or Datetime.
    Goes beyond raw dtype by attempting to parse object columns as dates.
    """
    numerical: List[str] = []
    categorical: List[str] = []
    datetime_cols: List[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            # Attempt to detect date-like object columns
            try:
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    categorical.append(col)
                    continue
                converted = pd.to_datetime(non_null, errors="coerce")
                valid_ratio = converted.notna().sum() / len(non_null)
                if valid_ratio >= 0.8:
                    datetime_cols.append(col)
                else:
                    categorical.append(col)
            except Exception:
                categorical.append(col)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_cols,
    }


def convert_to_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to datetime dtype."""
    df = df.copy()
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_type_summary(col_types: Dict[str, List[str]]) -> pd.DataFrame:
    """Return a summary table of column type counts."""
    rows = [
        {"Column Type": "Numerical", "Count": len(col_types["numerical"]),
         "Columns": ", ".join(col_types["numerical"]) or "—"},
        {"Column Type": "Categorical (Text)", "Count": len(col_types["categorical"]),
         "Columns": ", ".join(col_types["categorical"]) or "—"},
        {"Column Type": "Date / Time", "Count": len(col_types["datetime"]),
         "Columns": ", ".join(col_types["datetime"]) or "—"},
    ]
    return pd.DataFrame(rows)
