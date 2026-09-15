"""OLS calculations and guarded diagnostics for Relationship analysis."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan


def summarize(model, clean, df, valid, terms, coding):
    n = len(clean)
    x = model.model.exog
    y = model.model.endog
    k = x.shape[1]
    residual = model.resid
    # Suppress inference when residual error is indistinguishable from roundoff.
    perfect = np.linalg.norm(residual) <= 100 * np.finfo(float).eps * max(np.linalg.norm(y), 1.0)
    ci = model.conf_int()
    table = pd.DataFrame({'Term': ['Intercept'] + terms, 'Coefficient': model.params,
                          'Standard error': model.bse, 't-statistic': model.tvalues, '95% lower': ci[:, 0],
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
        table.loc[:, ['Standard error', 't-statistic', '95% lower', '95% upper', 'p-value']] = np.nan
    else:
        mean_ci = model.get_prediction(x).conf_int()
        for name, minimum, fn in [
            ('Jarque–Bera', 3, lambda: jarque_bera(residual)[:2]),
            ('Omnibus', 8, lambda: omni_normtest(residual)),
            ('Breusch–Pagan (Koenker LM)', n + 1 if k == 1 else k + 1, lambda: het_breuschpagan(residual, x, robust=True)[:2]),
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
            'source_rows': np.flatnonzero(valid) + 1, 'coding': coding, 'model': model}
