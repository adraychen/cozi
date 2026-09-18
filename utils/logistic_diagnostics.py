"""Diagnostics for classical logistic regression, without normality tests."""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from utils.regression_design import dependencies


def linearity_checks(r,continuous):
    from utils.logistic_analysis import fit_glm
    rows=[];x=r['design'];spec=r['spec'];y=r['clean'][spec['target']].to_numpy(dtype=float)
    for c in continuous:
        if ('main',c) not in r['active']: continue
        row={'Predictor':c,'Assessment':'Not assessed','Evidence':'','Recommendation':'Review functional form visually','p-value':np.nan}
        if any(b[0]=='interaction' and c in b[1:] for b in r['active']):
            row['Evidence']='Retained interaction: a single logit slope is not assumed. Inspect conditional curves.'
        else:
            powers=[b[2] for b in r['active'] if b[0]=='power' and b[1]==c]
            if powers:
                row.update(Assessment='Nonlinear form specified',Evidence='Retained polynomial terms already model curvature; this is not proof of adequate form.',Recommendation='Inspect residuals and calibration')
            else:
                values=r['clean'][c].to_numpy(dtype=float)
                z=(values-values.mean())/values.std()
                xx=np.column_stack([x,z*z,z*z*z])
                try:
                    model,_=fit_glm(xx,y)
                    statistic=max(0.,2*(model.llf-r['model'].llf))
                    row.update({'p-value':float(chi2.sf(statistic,2)),'Evidence':'Nested LR test adding quadratic and cubic terms on the same cases; zero/negative values are supported.'})
                except ValueError as exc:
                    row['Evidence']='Nonlinearity diagnostic unavailable: '+str(exc)
        rows.append(row)
    available=[i for i,row in enumerate(rows) if np.isfinite(row['p-value'])]
    if available:
        adjusted=multipletests([rows[i]['p-value'] for i in available],method='holm')[1]
        for i,p in zip(available,adjusted):
            rows[i]['Holm-adjusted p-value']=float(p)
            rows[i]['Assessment']='Possible nonlinearity' if p<.05 else 'No strong nonlinearity detected'
            rows[i]['Recommendation']='Consider a polynomial term or other defensible transformation' if p<.05 else 'Inspect plots; a nonsignificant test does not establish linearity'
    return pd.DataFrame(rows,columns=['Predictor','Assessment','Evidence','Recommendation','p-value','Holm-adjusted p-value'])


def calibration_table(y,p,confidence=.95):
    frame=pd.DataFrame({'Outcome':np.asarray(y),'Probability':np.asarray(p)})
    if frame.Probability.nunique()<2:
        frame['Group']=0
    else:
        frame['Group']=pd.qcut(frame.Probability,q=min(10,max(2,len(frame)//20)),duplicates='drop')
        if frame.Group.isna().all(): frame['Group']=0
    grouped=frame.groupby('Group',observed=True).agg(Count=('Outcome','size'),Events=('Outcome','sum'),Predicted=('Probability','mean'),Observed=('Outcome','mean')).reset_index(drop=True)
    lo,hi=proportion_confint(grouped.Events,grouped.Count,alpha=1-confidence,method='wilson')
    grouped['Observed CI Lower']=lo;grouped['Observed CI Upper']=hi
    return grouped


def add_diagnostics(r,continuous):
    spec=r['spec'];clean=r['clean'];model=r['model'];n=r['n'];k=r['k'];y=clean[spec['target']].to_numpy(dtype=float)
    issues=[]
    def flag(level,issue,evidence,action):
        issues.append({'Severity':level,'Issue':issue,'Evidence':evidence,'Recommended action':action})
    if min(r['events'],r['non_events'])<10*k:
        flag('Review','Limited outcome information',f"Smaller outcome group: {min(r['events'],r['non_events'])}; estimated parameters including intercept: {k}.",'Consider a simpler model. The 10-per-parameter screen is a warning heuristic, not an acceptance rule.')
    if 'unavailable' in r['separation'].lower():
        flag('Review','Separation diagnostic unavailable',r['separation'],'Review sparse counts and coefficient stability; no conclusion about absence of separation.')
    scaled=np.std(r['design'],axis=0);scaled[0]=1
    if (np.abs(model.params)*scaled>20).any() or (model.bse*scaled>20).any():
        flag('Significant','Unstable coefficient estimates','Large coefficients or standard errors on the scaled design (>20).','Review scaling, sparse groups and near separation. Treat coefficient inference with caution.')
    if not np.isfinite(r['coefficients'].select_dtypes(include='number').to_numpy()).all():
        flag('Significant','Unbounded reported uncertainty','One or more odds ratios/confidence bounds exceed numerical limits.','Interpretation is unreliable; revise the model before using these estimates.')
    if ((model.fittedvalues<1e-8)|(model.fittedvalues>1-1e-8)).any():
        flag('Review','Extreme fitted probabilities','Some probabilities are within 1e-8 of zero or one.','Review near separation and influential cases; this alone does not prove separation.')
    if not r['vif'].empty and (r['vif'].VIF>=5).any():
        severe=(r['vif'].VIF>10).any()
        flag('Significant' if severe else 'Review','Multicollinearity',f"{int((r['vif'].VIF>=5).sum())} encoded terms have VIF ≥5.",'Review redundant measures; do not delete predictors solely from VIF.')
    sparse=[]
    for c in spec['categorical']:
        for level in spec['levels'][c]:
            outcomes=clean.loc[clean[c]==level,spec['target']]
            events=int(outcomes.sum());non_events=len(outcomes)-events
            reasons=[]
            if len(outcomes)<5 or len(outcomes)/n<.01: reasons.append('Rare category')
            if events==0: reasons.append('Zero events')
            if non_events==0: reasons.append('Zero non-events')
            if min(events,non_events)<5: reasons.append('Sparse outcome counts')
            sparse.append({'Predictor':c,'Category':repr(level),'Reference':level==spec['levels'][c][0],'Count':len(outcomes),'Events':events,'Non-events':non_events,'Flag':'; '.join(reasons)})
    r['sparse']=pd.DataFrame(sparse)
    if sparse and any(row['Flag'] for row in sparse):
        flag('Review','Sparse categorical levels','Category-by-outcome table contains rare or sparse cells.','Review the data and scientific meaning before combining categories; none are combined automatically.')
    r['linearity']=linearity_checks(r,continuous)
    if not r['linearity'].empty:
        if (r['linearity'].Assessment=='Possible nonlinearity').any():
            flag('Review','Possible logit nonlinearity','At least one Holm-adjusted nested LR test has p < 0.05.','Review the indicated predictors and consider an appropriate nonlinear specification.')
        if (r['linearity'].Assessment=='Not assessed').any():
            flag('Review','Functional form requires review','At least one logit-linearity diagnostic is unavailable or conditional on interactions.','Inspect conditional probability curves and residual patterns.')
    influence=pd.DataFrame({'Original data row':np.flatnonzero(r['valid'])+1,'Original index':[str(v) for v in clean.index],
                            'Predicted Probability':model.fittedvalues,'Deviance Residual':model.resid_deviance,'Pearson Residual':model.resid_pearson})
    try:
        inf=model.get_influence()
        influence['Leverage']=inf.hat_matrix_diag
        influence["Cook's Distance"]=inf.cooks_distance[0]
        dfbetas=np.asarray(inf.dfbetas)
        influence['Max absolute DFBETA']=np.max(np.abs(dfbetas),axis=1)
        r['dfbetas']=pd.DataFrame(dfbetas,columns=r['names'])
        r['dfbetas'].insert(0,'Original data row',np.flatnonzero(r['valid'])+1)
        if not np.isfinite(influence.select_dtypes(include='number').to_numpy()).all(): raise ValueError('Nonfinite influence measures')
        influence['Flag']=(np.abs(influence['Deviance Residual'])>3)|(np.abs(influence['Pearson Residual'])>3)|(influence.Leverage>2*k/n)|(influence["Cook's Distance"]>4/n)|(influence['Max absolute DFBETA']>2/np.sqrt(n))
        if influence.Flag.any():
            flag('Review','Potentially influential observations',f'{int(influence.Flag.sum())} rows exceed one or more displayed review thresholds.','Inspect the original rows and sensitivity of conclusions; no rows are removed.')
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        influence['Flag']=np.abs(influence['Deviance Residual'])>3
        r['dfbetas']=pd.DataFrame()
        flag('Review','Influence diagnostic unavailable',str(exc),'Review residuals and numerical stability.')
    r['influence']=influence
    r['calibration']=calibration_table(y,model.fittedvalues,r['confidence'])
    if (r['calibration'].Count<20).any():
        flag('Review','Calibration precision','Some calibration groups have fewer than 20 observations.','Interpret the calibration plot cautiously; it describes the fitting sample.')
    calibration=r['calibration']
    disagreement=(calibration.Count>=20) & ((calibration.Predicted<calibration['Observed CI Lower']) | (calibration.Predicted>calibration['Observed CI Upper']))
    if int(disagreement.sum())>=2:
        flag('Review','Grouped calibration disagreement',f'{int(disagreement.sum())} groups with at least 20 observations have predicted means outside the observed-proportion intervals.','Inspect the calibration plot and model specification. This is an exploratory in-sample screen, not a formal goodness-of-fit test; groups and comparisons are dependent.')
    if r['selection_notes']:
        flag('Review','Incomplete AIC comparisons','; '.join(dict.fromkeys(r['selection_notes'])),'Review unsuccessful model reductions; the search is not exhaustive.')
    r['issues']=pd.DataFrame(issues,columns=['Severity','Issue','Evidence','Recommended action'])
    r['adequacy']='Significant model concerns' if any(i['Severity']=='Significant' for i in issues) else 'Review recommended' if issues else 'No major issues detected'
