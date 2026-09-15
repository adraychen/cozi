
"""OLS fitting with hierarchical terms, selection, diagnostics and prediction."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.regression_design import specification, design, dependencies, label
from utils.regression_statistics import summarize


def near_perfect(model):
    return np.linalg.norm(model.resid) <= 100*np.finfo(float).eps*max(np.linalg.norm(model.model.endog), 1.0)


def term_tests(model, indices, active):
    protected = set().union(*(dependencies(b) for b in active)) if active else set()
    tests = []
    for block in active:
        p = np.nan if near_perfect(model) else float(model.f_test(np.eye(len(model.params))[indices[block]]).pvalue)
        tests.append({'block':block, 'Term':label(block), 'p-value':p, 'Hierarchy protected':block in protected})
    return tests


def regression(df, target, predictors, categorical=None, interactions=(), degrees=None,
               elimination=False, threshold=.05):
    if not 0 < threshold < 1:
        raise ValueError('The removal threshold must lie between zero and one.')
    spec, clean, valid = specification(df, target, predictors, categorical, interactions, degrees)
    active = list(spec['blocks'])
    history = []
    while True:
        x, names, indices = design(clean, spec, active)
        if len(clean) <= x.shape[1]:
            raise ValueError(f'More than {x.shape[1]} complete rows are required for this model, including all generated coefficients.')
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError('Predictors or generated terms are constant, perfectly collinear, or numerically singular. Remove redundant terms or rescale predictors.')
        model = sm.OLS(clean[target].to_numpy(dtype=float), x).fit()
        if not np.isfinite(model.params).all() or not np.isfinite(model.resid).all():
            raise ValueError('Model calculations are nonfinite. Rescale the numeric variables and retry.')
        if history and history[-1]['Adjusted R²'] is None:
            history[-1]['Adjusted R²'] = float(model.rsquared_adj)
        tests = term_tests(model, indices, active)
        eligible = [t for t in tests if not t['Hierarchy protected'] and np.isfinite(t['p-value']) and t['p-value'] > threshold]
        if not elimination or not eligible:
            break
        worst = max(eligible, key=lambda t:t['p-value'])
        active.remove(worst['block'])
        history.append({'Step':len(history)+1, 'Removed term':worst['Term'], 'p-value':worst['p-value'], 'Adjusted R²':None})
    coding = []
    for c in spec['categorical']:
        for level in spec['levels'][c][1:]:
            coding.append({'Term':f'{c} = {level!r}', 'Predictor':c, 'Level':repr(level),
                           'Reference':repr(spec['levels'][c][0]), 'Count':int((clean[c] == level).sum()),
                           'Reference count':int((clean[c] == spec['levels'][c][0]).sum())})
    result = summarize(model, clean, df, valid, names[1:], coding)
    table = result['coefficients']
    table['Standardized coefficient'] = np.nan
    complex_variables = {c for b in active if b[0] != 'main' for c in (b[1:3] if b[0]=='interaction' else [b[1]])}
    for c in spec['numeric']:
        block = ('main', c)
        if block in active and c not in complex_variables:
            i = indices[block][0]
            table.loc[i, 'Standardized coefficient'] = model.params[i]*clean[c].std(ddof=1)/clean[target].std(ddof=1)
    vif_rows = []
    for i in range(1, x.shape[1]):
        with np.errstate(divide='ignore', invalid='ignore'):
            vif = float(variance_inflation_factor(x, i))
        vif_rows.append({'Predictor':names[i], 'VIF':vif, 'Tolerance':1/vif,
                         'Assessment':'Serious concern' if vif > 10 else 'Potential concern' if vif >= 5 else 'Generally acceptable'})
    influence = model.get_influence()
    with np.errstate(divide='ignore', invalid='ignore'):
        studentized = np.full(len(clean), np.nan) if result['perfect_fit'] else influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks = result['cooks']
    flag = (np.abs(studentized) > 2) | (leverage > 2*x.shape[1]/len(clean)) | (cooks > 4/len(clean))
    if not result['perfect_fit']:
        flag |= ~np.isfinite(studentized) | ~np.isfinite(cooks)
    influences = pd.DataFrame({'Original data row':np.flatnonzero(valid)+1, 'Original index':[str(i) for i in clean.index],
                               'Studentized residual':studentized, 'Leverage':leverage, "Cook's distance":cooks,
                               'Flagged':flag})
    prediction = model.get_prediction(x).summary_frame(alpha=.05)
    if result['perfect_fit']:
        prediction.loc[:, [c for c in prediction if c != 'mean']] = np.nan
    result.update(spec=spec, active=active, indices=indices, names=names, design=x,
                  term_tests=pd.DataFrame(tests), history=pd.DataFrame(history), elimination=elimination,
                  threshold=threshold, original_n=len(df), vif=pd.DataFrame(vif_rows), influence=influences,
                  prediction_frame=prediction, rmse=float(np.sqrt(np.mean(model.resid**2))),
                  mae=float(np.mean(np.abs(model.resid))), residual_se=float(np.sqrt(model.mse_resid)),
                  model_df=float(model.df_model), residual_df=float(model.df_resid),
                  f_stat=np.nan if result['perfect_fit'] or not active else float(model.fvalue),
                  f_p=np.nan if result['perfect_fit'] or not active else float(model.f_pvalue))
    if elimination:
        result['warnings'].append('Selection was performed on these observations. Reported p-values and intervals are conditional on the selected model and do not account for selection uncertainty.')
    return result


def predict(result, data):
    """Accept original units/categories and reconstruct the fitted model exactly."""
    data = pd.DataFrame(data)
    x, _, _ = design(data, result['spec'], result['active'])
    predicted = result['model'].get_prediction(x).summary_frame(alpha=.05)
    if result['perfect_fit']:
        predicted.loc[:, [c for c in predicted if c != 'mean']] = np.nan
    return predicted
