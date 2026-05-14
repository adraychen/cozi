import pandas as pd


def load_dataset(file) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    filename = file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file)
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    else:
        raise ValueError(f"Unsupported file type: {file.name}. Please upload a CSV or Excel file.")


def get_dataframe_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a plain-language summary of the DataFrame,
    replicating df.info() as a structured table.
    """
    rows = []
    total_rows = len(df)

    for col in df.columns:
        non_missing = df[col].notna().sum()
        missing = total_rows - non_missing
        missing_pct = round((missing / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # Map pandas dtype to plain language
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            plain_type = "Whole Number"
        elif pd.api.types.is_float_dtype(dtype):
            plain_type = "Decimal Number"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            plain_type = "Date / Time"
        elif pd.api.types.is_bool_dtype(dtype):
            plain_type = "True / False"
        else:
            plain_type = "Text"

        rows.append({
            "Column Name": col,
            "Data Type": plain_type,
            "Total Values": total_rows,
            "Missing Values": missing,
            "Missing %": missing_pct,
        })

    return pd.DataFrame(rows)
