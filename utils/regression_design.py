"""Single reusable transformation pipeline for estimation and new observations."""
from itertools import product
import numpy as np
import pandas as pd


def specification(df, target, predictors, categorical=None, interactions=(), degrees=None):
    if not df.columns.is_unique or not predictors or target in predictors or len(set(predictors)) != len(predictors):
        raise ValueError('Choose a numeric response and distinct predictors with unique column names.')
    if target not in df or any(c not in df for c in predictors):
        raise ValueError('A selected variable is no longer available.')
    if not pd.api.types.is_numeric_dtype(df[target]) or pd.api.types.is_bool_dtype(df[target]):
        raise ValueError('Linear regression requires a numeric dependent variable.')
    categorical = list(categorical) if categorical is not None else [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    if not set(categorical).issubset(predictors):
        raise ValueError('Categorical variables must be selected predictors.')
    numeric = [c for c in predictors if c not in categorical]
    degrees = dict(degrees or {})
    if any(c not in numeric or d not in (1, 2, 3) for c, d in degrees.items()):
        raise ValueError('Polynomial degrees must be 1, 2 or 3 for numeric predictors only.')
    pairs = []
    for pair in interactions:
        if len(pair) != 2 or pair[0] == pair[1] or any(c not in predictors for c in pair):
            raise ValueError('Interactions require two distinct selected predictors.')
        pair = tuple(sorted(pair, key=predictors.index))
        if pair not in pairs:
            pairs.append(pair)
    data = df[[target] + predictors].copy()
    for c in [target] + numeric:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    valid = data.notna().all(axis=1).to_numpy()
    clean = data.loc[valid]
    if len(clean) < 3:
        raise ValueError('At least three complete observations are required.')
    if clean[target].nunique() < 2:
        raise ValueError('The outcome has no variation.')
    levels = {}
    for c in predictors:
        if clean[c].nunique() < 2:
            raise ValueError(f'Predictor {c} is constant or has only one observed category after cleaning.')
        if c in categorical:
            if isinstance(clean[c].dtype, pd.CategoricalDtype) and clean[c].cat.ordered:
                levels[c] = [v for v in clean[c].cat.categories if v in set(clean[c].dropna())]
            else:
                levels[c] = sorted(pd.unique(clean[c]), key=lambda v: (type(v).__name__, str(v)))
            if len(levels[c]) > 100:
                raise ValueError(f'{c} has over 100 observed levels; remove identifiers or reduce categories.')
    center = bool(pairs or any(d > 1 for d in degrees.values()))
    means = {c: float(clean[c].mean()) if center else 0.0 for c in numeric}
    blocks = []
    for c in predictors:
        blocks.append(('main', c))
        if c in numeric:
            blocks.extend(('power', c, d) for d in range(2, degrees.get(c, 1)+1))
    blocks.extend(('interaction', a, b) for a, b in pairs)
    spec = dict(target=target, predictors=list(predictors), categorical=categorical, numeric=numeric,
                levels=levels, means=means, centered=center, interactions=pairs, degrees=degrees, blocks=blocks)
    return spec, clean, valid


def label(block):
    if block[0] == 'interaction':
        return f'{block[1]} × {block[2]}'
    return str(block[1]) + ({2:'²', 3:'³'}[block[2]] if block[0] == 'power' else '')


def dependencies(block):
    if block[0] == 'interaction':
        return {('main', block[1]), ('main', block[2])}
    if block[0] == 'power':
        return {('main', block[1])} | {('power', block[1], d) for d in range(2, block[2])}
    return set()


def design(data, spec, active=None):
    """Construct model columns in one canonical order using fitted levels/means."""
    active = list(spec['blocks'] if active is None else active)
    widths = {c:len(spec['levels'][c])-1 if c in spec['categorical'] else 1 for c in spec['predictors']}
    estimated = 1 + sum(widths[b[1]] if b[0]=='main' else widths[b[1]]*widths[b[2]] if b[0]=='interaction' else 1 for b in active)
    if estimated > 300:
        raise ValueError('The model expands to more than 300 coefficients; reduce factors or interactions.')
    base, names = {}, {}
    for c in spec['predictors']:
        if c not in data:
            raise ValueError(f'Provide a value for {c}.')
        if c in spec['categorical']:
            if data[c].isna().any() or not data[c].isin(spec['levels'][c]).all():
                raise ValueError(f'{c} contains missing or previously unseen categories.')
            base[c] = [(data[c] == level).to_numpy(dtype=float) for level in spec['levels'][c][1:]]
            names[c] = [f'{c} = {level!r}' for level in spec['levels'][c][1:]]
        else:
            values = pd.to_numeric(data[c], errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Provide finite numeric values for {c}.')
            base[c] = [values - spec['means'][c]]
            names[c] = [f'{c} (centered)' if spec['centered'] else str(c)]
    columns, labels, indices = [np.ones(len(data))], ['Intercept'], {}
    with np.errstate(over='ignore', invalid='ignore'):
        for block in active:
            start = len(columns)
            c = block[1]
            if block[0] == 'main':
                columns.extend(base[c]); labels.extend(names[c])
            elif block[0] == 'power':
                columns.append(base[c][0] ** block[2]); labels.append(names[c][0] + {2:'²',3:'³'}[block[2]])
            else:
                d = block[2]
                for i, j in product(range(len(base[c])), range(len(base[d]))):
                    columns.append(base[c][i]*base[d][j]); labels.append(f'{names[c][i]} × {names[d][j]}')
            indices[block] = list(range(start, len(columns)))
    if len(columns) > 300:
        raise ValueError('The specified model expands to more than 300 coefficients; reduce factors or interactions.')
    x = np.column_stack(columns)
    if not np.isfinite(x).all() or len(set(labels)) != len(labels):
        raise ValueError('Terms overflow numerically or their labels collide. Rescale or rename variables.')
    return x, labels, indices
