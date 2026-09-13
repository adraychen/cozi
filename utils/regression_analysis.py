"""OLS calculations and guarded diagnostics for Relationship analysis."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan


def regression(df, target, predictors, categorical=None):
    if not predictors or target in predictors or len(set(predictors)) != len(predictors):
        raise ValueError('Choose an outcome and distinct predictors.')
    if not df.columns.is_unique:
        raise ValueError('Column names must be unique.')
    if not pd.api.types.is_numeric_dtype(df[target]) or pd.api.types.is_bool_dtype(df[target]):
        raise ValueError('Linear regression requires a numeric dependent variable.')
    if categorical is None:
        categorical = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    if not set(categorical).issubset(predictors):
        raise ValueError('Categorical variables must be selected predictors.')
    data = df[[target] + predictors].copy()
    for column in [target] + [c for c in predictors if c not in categorical]:
        data[column] = pd.to_numeric(data[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for column in categorical:
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)
    valid = data.notna().all(axis=1).to_numpy()
    clean = data.loc[valid]
    n = len(clean)
    if not n:
        raise ValueError('No complete observations remain for the selected variables.')
    encoded, terms, coding = [], [], []
    for column in predictors:
        if column not in categorical:
            encoded.append(clean[column].to_numpy(dtype=float))
            terms.append(column)
        else:
            levels = sorted(pd.unique(clean[column]), key=lambda v: (type(v).__name__, str(v)))
            if len(levels) < 2:
                raise ValueError(f'Categorical predictor {column} has fewer than two observed levels after cleaning.')
            if len(levels) > 100:
                raise ValueError(f'{column} has over 100 levels. Remove identifiers or reduce categories before fitting.')
            for level in levels[1:]:
                term = f'{column} [{level!r} vs {levels[0]!r}]'
                encoded.append((clean[column] == level).to_numpy(dtype=float))
                terms.append(term)
                coding.append({'Term': term, 'Predictor': column, 'Level': repr(level), 'Reference': repr(levels[0]),
                               'Count': int((clean[column] == level).sum()), 'Reference count': int((clean[column] == levels[0]).sum())})
    if len(set(['Intercept'] + terms)) != len(terms) + 1:
        raise ValueError('Encoded term names collide. Rename the affected columns.')
    k = len(terms) + 1
    if n <= k:
        raise ValueError(f'More than {k} complete rows are required to estimate residual variation.')
    x = np.column_stack([np.ones(n)] + encoded)
    y = clean[target].to_numpy(dtype=float)
    if np.linalg.matrix_rank(x) < k:
        raise ValueError('Predictors are constant, perfectly collinear, or numerically ill-conditioned. Remove redundant predictors or rescale them.')
    if np.ptp(y) == 0:
        raise ValueError('The outcome has no variation.')
    model = sm.OLS(y, x).fit()
    residual = model.resid
    # Suppress inference when residual error is indistinguishable from roundoff.
    perfect = np.linalg.norm(residual) <= 100 * np.finfo(float).eps * max(np.linalg.norm(y), 1.0)
    ci = model.conf_int()
    table = pd.DataFrame({'Term': ['Intercept'] + terms, 'Coefficient': model.params,
                          'Standard error': model.bse, '95% lower': ci[:, 0],
                          '95% upper': ci[:, 1], 'p-value': model.pvalues})
    notes = []
    if coding and any(min(item['Count'], item['Reference count']) < 5 for item in coding):
        notes.append('Some categorical levels have fewer than five complete observations. Their coefficient estimates may be unstable.')
    if n < 30:
        notes.append('Small sample: diagnostics and inference may be unreliable. Thirty rows is a warning threshold, not a validity requirement.')
    if model.df_resid < 8:
        notes.append('Few residual degrees of freedom: interpret uncertainty and diagnostics cautiously.')
    diagnostics = []
    mean_ci = np.full((n, 2), np.nan)
    cooks = np.full(n, np.nan)
    leverage = np.full(n, np.nan)
    dw = np.nan
    if perfect:
        notes.append('Perfect or near-perfect fit: residual-based tests, p-values, standard errors, and confidence intervals are unavailable.')
        table.loc[:, ['Standard error', '95% lower', '95% upper', 'p-value']] = np.nan
    else:
        mean_ci = model.get_prediction(x).conf_int()
        for name, minimum, fn in [
            ('Jarque–Bera', 3, lambda: jarque_bera(residual)[:2]),
            ('Omnibus', 8, lambda: omni_normtest(residual)),
            ('Breusch–Pagan (Koenker LM)', k + 1, lambda: het_breuschpagan(residual, x, robust=True)[:2]),
        ]:
            statistic, p = np.nan, np.nan
            if n < minimum:
                status = f'Unavailable: requires at least {minimum} complete rows'
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    statistic, p = map(float, fn())
                status = ('Unavailable: degenerate diagnostic' if not np.isfinite(p) else
                          'Review warning' if p < .05 else 'No strong evidence of violation')
            diagnostics.append({'Diagnostic': name, 'Statistic': statistic, 'p-value': p, 'Interpretation': status})
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        with np.errstate(divide='ignore', invalid='ignore'):
            cooks = influence.cooks_distance[0]
        dw = float(durbin_watson(residual))
        if np.any(~np.isfinite(cooks)) or np.any(leverage >= 1 - 1e-10):
            notes.append('Some observations have extreme leverage; their influence estimates may be undefined.')
    return {'coefficients': table, 'n': n, 'excluded': len(df) - n,
            'r2': model.rsquared, 'adjusted_r2': model.rsquared_adj,
            'actual': y, 'fitted': model.fittedvalues, 'clean': clean,
            'residual': residual, 'mean_ci': mean_ci, 'perfect_fit': perfect,
            'diagnostics': pd.DataFrame(diagnostics), 'warnings': notes,
            'cooks': cooks, 'leverage': leverage, 'dw': dw, 'k': k,
            'source_rows': np.flatnonzero(valid) + 1, 'coding': coding}
