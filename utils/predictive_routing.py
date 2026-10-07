"""Deterministic target validation shared by controls and training."""
import pandas as pd


def is_temporal_target(series, variable_type=None):
    if variable_type in ('Date / Time', 'Temporal', 'Date', 'DateTime', 'Datetime'):
        return True
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return True
    # Explicit categorical overrides are respected; no implicit date classification.
    if variable_type == 'Categorical' or isinstance(series.dtype, pd.CategoricalDtype):
        return False
    from utils.temporal import infer_temporal_subtype
    return not pd.api.types.is_numeric_dtype(series) and infer_temporal_subtype(series) is not None


def validate_standard_target(series):
    if is_temporal_target(series):
        raise ValueError('Temporal targets cannot be used directly for Regression or Classification. Choose a numerical or categorical target.')
