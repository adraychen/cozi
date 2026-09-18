"""Classical binomial-logit estimation with the shared regression design."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.regression_design import specification, design, dependencies, label


def separation_status(x,y):
    """Linear feasibility tests for a separating direction, after column scaling."""
    scale=np.maximum(np.max(np.abs(x),axis=0),1.)
    signed=(2*np.asarray(y)-1)[:,None]*(x/scale)
    full=linprog(np.zeros(x.shape[1]),A_ub=-signed,b_ub=-np.ones(len(y)),bounds=[(None,None)]*x.shape[1],method='highs',options={'time_limit':5})
    if full.success: return 'Complete separation'
    if full.status!=2: return 'Separation check unavailable (solver limit or numerical failure)'
    quasi=linprog(-signed.sum(axis=0),A_ub=-signed,b_ub=np.zeros(len(y)),bounds=[(-1,1)]*x.shape[1],method='highs',options={'time_limit':5})
    if not quasi.success: return 'Separation check unavailable (solver limit or numerical failure)'
    if -quasi.fun>1e-7: return 'Quasi-complete separation'
    return 'No separating direction detected'


def fit_glm(x,y,maxiter=100,check_separation=True):
    if len(y)<=x.shape[1]:
        raise ValueError(f'Too few complete observations for {x.shape[1]} estimated parameters. Reduce model complexity.')
    if np.linalg.matrix_rank(x)<x.shape[1]:
        raise ValueError('The design matrix is singular: remove redundant predictors or terms, or rescale numeric values.')
    separation=separation_status(x,y) if check_separation else 'Not checked'
    if separation in ('Complete separation','Quasi-complete separation'):
        raise ValueError(separation+' detected. Classical maximum-likelihood estimates may be infinite. Revise the terms or review sparse categories; no reliable coefficient table can be reported.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            fitted=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=maxiter,tol=1e-9)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
            raise ValueError('Logistic estimation failed. Review separation, sparse groups and numerical scaling. '+str(exc)) from exc
    if not fitted.converged:
        raise ValueError('Logistic regression did not converge. Estimates are not reported; review separation, sparse groups, model complexity and scaling.')
    if any('separation' in str(w.message).lower() for w in caught):
        raise ValueError('Potential complete or quasi-complete separation detected during fitting. Estimates are not reliable; revise the specification.')
    if not np.isfinite(fitted.params).all() or not np.isfinite(fitted.bse).all() or not np.isfinite(fitted.cov_params()).all() or not np.isfinite(fitted.llf):
        raise ValueError('Logistic estimates or their uncertainty are nonfinite. Review separation, redundancy and scaling.')
    return fitted,separation


def logistic_regression(df,target,predictors,event,categorical=None,interactions=(),degrees=None,references=None,confidence=.95,elimination=False,continuous=None,maxiter=100):
    if not .5<confidence<1:
        raise ValueError('Confidence level must be greater than 50% and below 100%.')
    if target not in df or any(c not in df for c in predictors):
        raise ValueError('Select an available response and predictors.')
    for c in [target]+list(predictors):
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_complex_dtype(df[c]) or any(isinstance(v,(complex,np.complexfloating)) for v in df[c].dropna()):
            raise ValueError(f'{c}: date/time and complex-valued variables must be prepared before logistic regression.')
    categorical=list(categorical) if categorical is not None else [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    levels=list(df[target].dropna().unique())
    if len(levels)!=2 or event not in levels:
        raise ValueError('Choose a response with exactly two valid outcome levels and select its modeled event.')
    if any(isinstance(v,(int,float,np.number)) and not np.isfinite(v) for v in levels):
        raise ValueError('The binary response contains nonfinite outcome labels. Correct these values first.')
    for c in categorical or []:
        if c not in predictors: raise ValueError('Categorical variables must be selected predictors.')
        values=df[c].dropna()
        if pd.api.types.is_datetime64_any_dtype(values):
            raise ValueError(f'{c}: unprocessed date/time is not supported as a categorical predictor.')
        if any(not isinstance(v,(str,int,float,bool,np.generic)) for v in values):
            raise ValueError(f'{c}: unsupported complex/object values; use numeric measures or category labels.')
        text=values.map(lambda v: str(v))
        if len(text) and (text.str.len().median()>100 or text.str.split().map(len).median()>15):
            raise ValueError(f'{c}: free-text-like values are not supported; derive a meaningful categorical variable first.')
    encoded=df.copy()
    encoded[target]=df[target].map(lambda v:np.nan if pd.isna(v) else float(v==event)).astype(float)
    try:
        spec,clean,valid=specification(encoded,target,predictors,categorical,interactions,degrees)
    except ValueError as exc:
        if 'outcome has no variation' in str(exc):
            raise ValueError('Both outcomes must remain after missing/nonfinite rows are excluded.') from exc
        raise
    for c,ref in (references or {}).items():
        if c not in spec['categorical'] or ref not in spec['levels'][c]:
            raise ValueError(f'The reference category for {c} is unavailable in the complete cases.')
        spec['levels'][c]=[ref]+[v for v in spec['levels'][c] if v!=ref]
    y=clean[target].to_numpy(dtype=float)
    if len(np.unique(y))!=2: raise ValueError('Both outcomes must remain after missing/nonfinite rows are excluded.')
    active=list(spec['blocks']);history=[];selection_notes=[]
    x,names,indices=design(clean,spec,active)
    model,separation=fit_glm(x,y,maxiter)
    while elimination and active:
        protected=set().union(*(dependencies(b) for b in active))
        candidates=[]
        for block in active:
            if block in protected: continue
            reduced=[b for b in active if b!=block]
            xx,nn,ii=design(clean,spec,reduced)
            try:
                candidate,_=fit_glm(xx,y,maxiter,check_separation=False)
                candidates.append((candidate.aic,block,candidate,xx,nn,ii,reduced))
            except ValueError as exc:
                selection_notes.append(f'Could not evaluate removing {label(block)}: {exc}')
        if not candidates: break
        best=min(candidates,key=lambda v:v[0])
        if best[0]>=model.aic-1e-7: break
        history.append({'Step':len(history)+1,'Removed term':label(best[1]),'Previous AIC':float(model.aic),'Refitted AIC':float(best[0])})
        _,_,model,x,names,indices,active=best
    alpha=1-confidence
    ci=np.asarray(model.conf_int(alpha=alpha))
    display_names=list(names)
    for c in spec['categorical']:
        for level in spec['levels'][c][1:]:
            old=f'{c} = {level!r}'
            display_names=[name.replace(old,f'{c}: {level!r} vs {spec["levels"][c][0]!r}') for name in display_names]
    with np.errstate(over='ignore'):
        coefficients=pd.DataFrame({'Term':display_names,'Coefficient':model.params,'Std. Error':model.bse,'Wald Statistic (χ², df=1)':model.tvalues**2,'p-value':model.pvalues,'Coefficient CI Lower':ci[:,0],'Coefficient CI Upper':ci[:,1],'Odds Ratio':np.exp(model.params),'OR CI Lower':np.exp(ci[:,0]),'OR CI Upper':np.exp(ci[:,1])})
    protected=set().union(*(dependencies(b) for b in active)) if active else set()
    tests=[]
    for block in active:
        test=model.wald_test(np.eye(len(model.params))[indices[block]],scalar=True)
        tests.append({'Term':label(block),'Wald χ²':float(test.statistic),'df':len(indices[block]),'p-value':float(test.pvalue),'Hierarchy protected':block in protected})
    null=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial()).fit()
    lr=max(0.,2*(model.llf-null.llf));lr_df=x.shape[1]-1
    vif=[]
    for i in range(1,x.shape[1]):
        with np.errstate(divide='ignore',invalid='ignore'):
            value=float(variance_inflation_factor(x,i))
        vif.append({'Predictor':display_names[i],'VIF':value,'Tolerance':1/value,'Assessment':'Strong warning' if value>10 else 'Investigate' if value>=5 else 'Generally acceptable'})
    prediction=model.get_prediction(x).summary_frame(alpha=alpha)
    result=dict(spec=spec,clean=clean,valid=valid,active=active,model=model,design=x,names=display_names,indices=indices,
                event=event,non_event=next(v for v in levels if v!=event),confidence=confidence,
                n=len(clean),original_n=len(df),excluded=len(df)-len(clean),events=int(y.sum()),non_events=int(len(y)-y.sum()),k=x.shape[1],
                coefficients=coefficients,term_tests=pd.DataFrame(tests),history=pd.DataFrame(history),elimination=elimination,
                separation=separation,selection_notes=selection_notes,vif=pd.DataFrame(vif),prediction_frame=prediction,
                lr=lr,lr_df=lr_df,lr_p=float(chi2.sf(lr,lr_df)) if lr_df else np.nan,
                log_likelihood=float(model.llf),deviance=float(model.deviance),aic=float(model.aic),bic=float(model.bic_llf),
                pseudo_r2=float(1-model.llf/null.llf),brier=float(np.mean((y-model.fittedvalues)**2)),perfect_fit=False)
    from utils.logistic_diagnostics import add_diagnostics
    add_diagnostics(result,continuous if continuous is not None else spec['numeric'])
    return result


def predict_logistic(result,data):
    x,_,_=design(pd.DataFrame(data),result['spec'],result['active'])
    return result['model'].get_prediction(x).summary_frame(alpha=1-result['confidence'])
