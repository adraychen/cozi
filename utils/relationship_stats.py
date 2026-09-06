import numpy as np
import pandas as pd
from scipy import stats


def regression(df, target, predictors):
    if not predictors or target in predictors:
        raise ValueError('Choose an outcome and distinct predictors.')
    clean = df[[target] + predictors].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    x = np.column_stack([np.ones(len(clean)), clean[predictors].to_numpy(dtype=float)])
    y = clean[target].to_numpy(dtype=float)
    if len(clean) <= x.shape[1]:
        raise ValueError('More complete rows than fitted coefficients are required.')
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError('Predictors are constant or perfectly collinear. Remove redundant predictors.')
    if np.ptp(y) == 0:
        raise ValueError('The outcome has no variation.')
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residual = y - fitted
    dof = len(y) - x.shape[1]
    mse = float(residual @ residual / dof)
    se = np.sqrt(np.maximum(0, np.diag(np.linalg.pinv(x.T @ x)) * mse))
    critical = stats.t.ppf(.975, dof)
    p = np.full(len(beta), np.nan)
    np.divide(beta, se, out=p, where=se > 0)
    p = 2 * stats.t.sf(np.abs(p), dof)
    r2 = 1 - float(residual @ residual) / float(np.sum((y-y.mean())**2))
    table = pd.DataFrame({'Term': ['Intercept'] + predictors, 'Coefficient': beta, 'Standard error': se,
                          '95% lower': beta-critical*se, '95% upper': beta+critical*se, 'p-value': p})
    return {'coefficients': table, 'n': len(clean), 'excluded': len(df)-len(clean), 'r2': r2,
            'adjusted_r2': 1-(1-r2)*(len(y)-1)/dof, 'actual': y, 'fitted': fitted, 'clean': clean}


def group_anova(df, category, measure):
    clean = df[[category, measure]].copy()
    clean[measure] = pd.to_numeric(clean[measure], errors='coerce').replace([np.inf,-np.inf], np.nan)
    clean = clean.dropna()
    groups = [g[measure].to_numpy() for _,g in clean.groupby(category, observed=True)]
    table = clean.groupby(category, observed=True)[measure].agg(['count','mean','median','std','min','max']).reset_index()
    if len(groups) < 2 or any(len(g)<2 for g in groups):
        raise ValueError('ANOVA requires at least two groups, each with two complete observations.')
    if all(np.ptp(g)==0 for g in groups):
        raise ValueError('ANOVA cannot estimate within-group variation for these constant groups.')
    f,p = stats.f_oneway(*groups)
    return table, float(f), float(p), len(df)-len(clean)
