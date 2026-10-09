"""Dataset-scoped analytical classifications, independent of Streamlit."""
import numpy as np
import pandas as pd
from utils.temporal import SUBTYPES, infer_temporal_subtype, parse_temporal, is_month_name_series

ALLOWED = {'Numerical': ['Discrete', 'Continuous'], 'Categorical': ['Binary', 'Nominal', 'Ordinal'], 'Date / Time': SUBTYPES}
KINDS = {'Numerical':'numerical', 'Categorical':'categorical', 'Date / Time':'datetime'}
ORDERS = [('low','medium','high'), ('small','medium','large'), ('poor','fair','good','excellent')]

def get_allowed_subtypes(variable_type):
    return ALLOWED[variable_type][:]

def category_order(series):
    values = list(series.dropna().unique())
    if isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
        return [v for v in series.cat.categories if v in values], True
    normalized = {str(v).strip().lower():v for v in values}
    for pattern in ORDERS:
        if set(normalized)==set(pattern) and len(values)==len(pattern):
            return [normalized[v] for v in pattern], True
    return values, False

def infer_variable_subtype(series, variable_type):
    if variable_type == 'Numerical':
        return 'Discrete' if pd.api.types.is_integer_dtype(series.dtype) else 'Continuous'
    if variable_type == 'Date / Time':
        return infer_temporal_subtype(series) or 'DateTime'
    if series.nunique(dropna=True)==2:
        return 'Binary'
    return 'Ordinal' if category_order(series)[1] else 'Nominal'

def infer_metadata(df):
    metadata = {}
    for column in df:
        series = df[column]
        temporal = None
        if pd.api.types.is_bool_dtype(series):
            variable_type = 'Categorical'
        elif pd.api.types.is_numeric_dtype(series):
            # A named year column has explicit semantic evidence; arbitrary IDs do not.
            if str(column).strip().lower() in ('year', 'calendar year') and len(series.dropna()) and parse_temporal(series.dropna(), 'Year').notna().mean() >= .8:
                temporal = 'Year'
                variable_type = 'Date / Time'
            else:
                variable_type = 'Numerical'
        else:
            temporal = infer_temporal_subtype(series)
            variable_type = 'Date / Time' if temporal else 'Categorical'
        subtype = temporal or infer_variable_subtype(series, variable_type)
        metadata[column] = dict(variable_type=variable_type, variable_subtype=subtype,
                                order=category_order(series)[0] if subtype == 'Ordinal' else [], source='Automatic')
    return metadata

def validate_type_subtype(series, variable_type, subtype, order=None):
    if subtype not in get_allowed_subtypes(variable_type):
        raise ValueError('Choose a subtype compatible with the selected Variable Type.')
    if variable_type=='Numerical':
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            raise ValueError('Date/time values cannot be treated as numeric measurements.')
        try:
            numeric=pd.to_numeric(series.dropna(),errors='raise')
        except (ValueError,TypeError):
            raise ValueError('Numerical variables must contain safely convertible numeric values; text is not coerced.')
        finite=numeric[np.isfinite(numeric)]
        if subtype=='Discrete' and not (finite==np.floor(finite)).all():
            raise ValueError('Discrete variables must contain integer-valued numbers.')
    if subtype=='Binary' and series.nunique(dropna=True)!=2:
        raise ValueError('Binary variables must contain exactly two unique non-missing values.')
    if subtype=='Ordinal':
        values=list(series.dropna().unique())
        if not order or len(order)!=len(values) or len(set(order))!=len(order) or set(order)!=set(values):
            raise ValueError('Ordinal order must include every observed category exactly once.')
    if variable_type=='Date / Time':
        values=series.dropna()
        if (pd.api.types.is_numeric_dtype(series) and subtype not in ('Year', 'Month', 'Quarter')) or not len(values):
            raise ValueError('Use temporal values with explicit formats or units; unitless numeric values require conversion first.')
        parsed = parse_temporal(values, subtype)
        if parsed.notna().mean() < .8:
            raise ValueError(f'At least 80% of non-missing values must be valid {subtype} values.')
        if subtype == 'Date':
            calendar = parse_temporal(values, 'DateTime')
            if (calendar.notna() & parsed.isna()).any():
                raise ValueError('Date would discard time components. Use DateTime for timestamps.')


def apply_user_override(metadata, column, series, variable_type, subtype, order=None):
    validate_type_subtype(series,variable_type,subtype,order)
    updated={c:dict(m) for c,m in metadata.items()}
    updated[column]=dict(variable_type=variable_type,variable_subtype=subtype,order=list(order or []) if subtype=='Ordinal' else [],source='User override')
    return updated

def ensure_metadata(state, df, dataset_id):
    if 'variable_metadata' not in state or state.get('variable_dataset_id')!=dataset_id:
        for key in list(state):
            if key.startswith('vm_'):
                state.pop(key,None)
        state['variable_dataset_id']=dataset_id
        state['variable_metadata']=infer_metadata(df)
    metadata=state['variable_metadata']
    missing=[c for c in df if c not in metadata]
    inferred=infer_metadata(df[missing]) if missing else {}
    state['variable_metadata']={c:metadata.get(c,inferred.get(c)) for c in df}
    for column, item in state['variable_metadata'].items():
        if (item.get('source') == 'Automatic' and item['variable_type'] == 'Date / Time'
                and item['variable_subtype'] in ('Date', 'DateTime')
                and is_month_name_series(df[column])):
            item['variable_subtype'] = 'Month'
        elif (item.get('source') == 'Automatic' and item['variable_type'] == 'Date / Time'
              and item['variable_subtype'] in ('Date', 'DateTime')
              and not pd.api.types.is_datetime64_any_dtype(df[column])):
            text = df[column].dropna().astype('string').str.strip()
            component = text.str.fullmatch(r'(?:\d{4}(?:[-/]\d{2})?|[A-Za-z]+[\s/-]+\d{4}|\d{4}[\s/-]+[A-Za-z]+|[Qq][1-4](?:[\s/-]+\d{4})?|\d{4}[\s/-]?[Qq][1-4])', na=False)
            if len(text) and component.mean() >= .8:
                inferred = infer_temporal_subtype(df[column])
                if inferred in ('Year', 'Year-Month', 'Quarter', 'Year-Quarter'):
                    item['variable_subtype'] = inferred
        if item['variable_type'] == 'Date / Time' and item['variable_subtype'] == 'Date / Time':
            item['variable_subtype'] = infer_variable_subtype(df[column], 'Date / Time')
    return state['variable_metadata']

def effective_data(df, metadata):
    data=df.copy()
    groups={k:[] for k in KINDS.values()}
    groups.update(temporal=[], time=[], duration=[], year=[], month=[], quarter=[], year_quarter=[])
    for c,m in metadata.items():
        t=m['variable_type']
        if t == 'Date / Time':
            groups['temporal'].append(c)
            subtype = m['variable_subtype']
            groups[subtype.lower().replace('-', '_') if subtype in ('Time', 'Duration', 'Year', 'Month', 'Quarter', 'Year-Quarter') else 'datetime'].append(c)
        else:
            groups[KINDS[t]].append(c)
        if t=='Numerical':
            data[c]=pd.to_numeric(data[c],errors='raise')
            if pd.api.types.is_bool_dtype(data[c]):
                data[c]=data[c].astype(float)
        elif t=='Date / Time':
            subtype = m['variable_subtype']
            data[c]=parse_temporal(data[c], infer_variable_subtype(data[c],t) if subtype=='Date / Time' else subtype)
        elif m['variable_subtype']=='Ordinal':
            data[c]=pd.Categorical(data[c],categories=m['order'],ordered=True)
        else:
            data[c]=pd.Categorical(data[c])
    return data,groups

def metadata_target_type(series, variable_type, subtype):
    from utils.predictive_routing import is_temporal_target
    if is_temporal_target(series, variable_type):
        return 'temporal'
    if variable_type=='Numerical':
        return 'continuous_numerical'  # Numerical counts remain regression targets, not class codes.
    if subtype=='Binary' and series.nunique(dropna=True)==2:
        return 'binary'
    return 'multiclass'


def encode_binary_target(series):
    """Encode two observed labels as 0/1 for F1, ROC and logistic scoring."""
    labels=list(series.dropna().unique())
    if len(labels)!=2:
        raise ValueError('Binary variables must contain exactly two unique non-missing values.')
    try:
        labels=sorted(labels)
    except TypeError:
        labels=sorted(labels,key=lambda value:(type(value).__name__,str(value)))
    encoded=series.map({value:i for i,value in enumerate(labels)})
    return encoded.astype(int), labels
