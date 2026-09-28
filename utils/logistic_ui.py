"""Seven-tab classical logistic regression interface."""
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import presentation
from utils.regression_controls import render_term_checkboxes
from utils.regression_reporting import format_table,p_text,selected_terms_table
from utils.regression_design import label
from utils.logistic_analysis import logistic_regression,predict_logistic
from utils.regression_effect_plots import adjusted_curves,effect_figure,retained_interactions,retained_polynomials


def fit_cached(df,target,predictors,event,cats,interactions,degrees,references,confidence,elimination,continuous):
    selected=df[[target]+predictors]
    digest=hashlib.sha256(pd.util.hash_pandas_object(selected,index=True).values.tobytes())
    digest.update(repr((list(selected.columns),list(map(str,selected.dtypes)),target,predictors,event,cats,interactions,degrees,references,confidence,elimination,continuous)).encode())
    key=digest.hexdigest()
    cached=st.session_state.get('rel_logistic_fit')
    if cached is None or cached[0]!=key:
        fitted=logistic_regression(df,target,predictors,event,cats,interactions,degrees,references,confidence,elimination,continuous)
        st.session_state['rel_logistic_fit']=(key,fitted)
    return key,st.session_state['rel_logistic_fit'][1]


def render_logistic(df,target,predictors,cats,metadata):
    st.markdown('##### Binary logistic regression')
    nums=[c for c in predictors if c not in cats]
    event=st.selectbox('Modeled event / positive outcome',[None]+list(df[target].dropna().unique()),format_func=lambda v:'Choose the modeled event' if v is None else repr(v),key='rel_log_event_'+str(target))
    confidence=st.number_input('Confidence level (%)',min_value=51.,max_value=99.9,value=95.,step=1.,key='rel_log_confidence')/100
    references={}
    with st.expander('Categorical reference categories',expanded=bool(cats)):
        for c in cats:
            values=list(df[c].dropna().unique())
            if len(values)>100:
                st.error(f'{c} has more than 100 categories. Free text and high-cardinality identifiers are not supported; revise the variable selection.');return
            if any(not isinstance(v,(str,int,float,bool,np.generic)) for v in values):
                st.error(f'{c} contains unsupported complex/object values. Use numeric measures or categorical labels.');return
            if isinstance(df[c].dtype,pd.CategoricalDtype) and df[c].cat.ordered:
                values=[v for v in df[c].cat.categories if v in values]
            if not values:
                st.error(f'{c} has no non-missing categories.');return
            references[c]=st.selectbox('Reference category: '+str(c),values,format_func=repr,key='rel_log_ref_'+str(c))
        st.caption('Categorical and ordinal variables use k − 1 indicators. Ordinal levels are not assumed equally spaced. Date/time and unsupported free text must be prepared before use.')
    with st.expander('Model terms and elimination'):
        interactions,degrees=render_term_checkboxes(predictors,nums)
        mode=st.selectbox('Model-building method',['No elimination','Backward elimination — AIC, hierarchy preserved'],key='rel_log_elimination')
        st.caption('Fit the specified model first. Optional AIC elimination removes whole terms only when AIC decreases; required main effects and lower powers are retained. No train/test split, tuning or cross-validation is performed.')
    selected=df[[target]+predictors].copy()
    for c in nums: selected[c]=pd.to_numeric(selected[c],errors='coerce')
    selected=selected.replace([np.inf,-np.inf],np.nan)
    complete=selected.notna().all(axis=1)
    excluded=len(df)-int(complete.sum())
    st.caption(f'Complete-case analysis: {len(df):,} total rows; {int(complete.sum()):,} included; {excluded:,} excluded for missing/nonfinite values ({100*excluded/len(df) if len(df) else 0:.1f}%). The sample stays fixed during elimination.')
    if event is None:
        st.info('Choose the modeled event explicitly before fitting.');return
    st.write(f'Modeled event: {target} = {event!r}')
    if cats:
        with st.expander('Pre-fit category/outcome counts'):
            rows=[]
            for c in cats:
                for value in selected.loc[complete,c].unique():
                    outcomes=selected.loc[complete & (selected[c]==value),target]
                    count=int((outcomes==event).sum())
                    rows.append({'Predictor':c,'Category':repr(value),'Events':count,'Non-events':len(outcomes)-count})
            presentation.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.caption('Zero outcome cells can indicate separation; these counts remain available even when reliable estimation is impossible.')
    continuous=[c for c in nums if metadata.get(c,{}).get('variable_subtype','Continuous')=='Continuous']
    try:
        key,r=fit_cached(df,target,predictors,event,cats,interactions,degrees,references,confidence,mode!='No elimination',continuous)
        render_results(r,key)
    except (ValueError,TypeError,KeyError,np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
        st.error('Cannot report a reliable logistic model: '+str(exc))
        st.warning('Model Adequacy: Significant model concerns. Revise the specification or review the data. No alternative model is fitted automatically.')


def render_results(r,key):
    spec=r['spec'];confidence=100*r['confidence'];model=r['model'];event=f'{spec["target"]} = {r["event"]!r}'
    if r['adequacy']=='Significant model concerns':
        st.warning('Significant model concerns: '+ '; '.join(r['issues'].loc[r['issues'].Severity=='Significant','Issue'])+'. Review Model Adequacy before interpreting estimates.')
    tabs=st.tabs(['Model Overview','Predictor Analysis','Diagnostics','Multicollinearity & Influence','Visual Analysis','Prediction','Regression Findings'])
    with tabs[0]:
        st.markdown('**Model summary**')
        summary=[('Response / Dependent Variable',str(spec['target'])),('Modeled event',repr(r['event'])),('Non-event',repr(r['non_event'])),('Predictors / Independent Variables',', '.join(map(str,spec['predictors']))),('Original rows',str(r['original_n'])),('Used rows',str(r['n'])),('Excluded missing/nonfinite rows',str(r['excluded'])),('Percentage excluded',f"{100*r['excluded']/r['original_n']:.1f}%"),('Events',str(r['events'])),('Non-events',str(r['non_events'])),('Estimated parameters (including intercept)',str(r['k'])),('Confidence level',f'{confidence:g}%'),('Model-building method','Backward elimination — AIC' if r['elimination'] else 'No elimination')]
        presentation.dataframe(pd.DataFrame(summary,columns=['Model detail','Value']))
        st.markdown('**Fitted logistic regression equation**')
        eta=f'{model.params[0]:.5g}'+''.join(f' {b:+.5g} × {name}' for name,b in zip(r['names'][1:],model.params[1:]))
        st.write('log(p / (1 − p)) = '+eta)
        st.write('p = 1 / (1 + exp(−η)), where η is the right-hand side above.')
        st.caption('p is the probability of '+event+'. Category contrasts represent indicators (1 for that category, 0 otherwise). Interactions and polynomial terms describe conditional associations on the log-odds scale.')
        st.markdown('**Model performance**')
        stats=[('Log likelihood',r['log_likelihood']),('Deviance',r['deviance']),('AIC',r['aic']),('BIC (log-likelihood definition)',r['bic']),('McFadden Pseudo-R²',r['pseudo_r2']),('Likelihood-ratio χ²',r['lr']),('Likelihood-ratio degrees of freedom',r['lr_df']),('Likelihood-ratio p-value',p_text(r['lr_p'])),('Brier score (in-sample)',r['brier'])]
        presentation.dataframe(pd.DataFrame([(name,f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v)) for name,v in stats],columns=['Measure','Value']))
        st.caption('The LR test compares with an intercept-only model on the same rows. McFadden Pseudo-R² is not percentage of variance explained and is not directly comparable with ordinary R². All fit/calibration measures describe these analysis observations; significance does not establish practical importance.')
        st.markdown('**Selected terms**');presentation.dataframe(selected_terms_table(r))
        if spec['categorical']:
            presentation.dataframe(pd.DataFrame([{'Predictor':c,'Reference category':repr(spec['levels'][c][0]),'Observed levels':len(spec['levels'][c])} for c in spec['categorical']]))
        with st.expander('Elimination history'):
            if r['history'].empty: st.write('No terms were removed.')
            else: presentation.dataframe(format_table(r['history']),hide_index=True,use_container_width=True)
            if r['elimination']: st.caption('Selection uncertainty is not included in the reported p-values or intervals. AIC selection is exploratory, not proof of scientific importance.')
    with tabs[1]:
        st.caption(f'Wald statistics use χ² with one degree of freedom per coefficient. All confidence intervals use {confidence:g}%. Odds ratios multiply odds of {event}, not probabilities; the intercept OR represents baseline odds.')
        presentation.dataframe(format_table(r['coefficients']),hide_index=True,use_container_width=True)
        if not r['term_tests'].empty:
            st.markdown('**Joint term tests**');presentation.dataframe(format_table(r['term_tests']),hide_index=True,use_container_width=True)
            st.caption('Multi-level factors and their interactions are tested jointly. Individual main-effect odds ratios in interaction/polynomial models are conditional at centered zero and reference categories; use probability curves for the overall relationship.')
        st.download_button('Download logistic coefficients (CSV)',r['coefficients'].to_csv(index=False),'logistic_coefficients.csv',key='rel_log_coeff_download')
    with tabs[2]:
        st.write('Convergence: achieved. '+r['separation']+'.')
        presentation.dataframe(pd.DataFrame({'Measure':['Events','Non-events','Parameters including intercept','Smaller outcome count per parameter'],'Value':[r['events'],r['non_events'],r['k'],min(r['events'],r['non_events'])/r['k']]}))
        st.caption('Limited outcome information triggers a review warning, not a hard events-per-variable rule. Independence depends on study design; repeated or clustered observations require a suitable method.')
        st.markdown('**Linearity of the logit**')
        if r['linearity'].empty: st.info('No retained Continuous predictor requires a separate logit-linearity check.')
        else: presentation.dataframe(format_table(r['linearity']),hide_index=True,use_container_width=True)
        st.caption('For eligible additive continuous predictors, compare the fitted model with additional quadratic/cubic terms using a nested LR test and Holm adjustment. This supports zero/negative values. A nonsignificant result does not prove linearity; tests may have low power. Existing nonlinear/interaction terms require visual review. No transformations are applied automatically.')
        if not r['sparse'].empty:
            st.markdown('**Category-by-outcome counts**');presentation.dataframe(r['sparse'],hide_index=True,use_container_width=True)
        for residual in ['Deviance Residual','Pearson Residual']:
            fig=px.scatter(r['influence'],x='Predicted Probability',y=residual,hover_data=['Original data row','Original index'])
            for cutoff in [-3,0,3]: fig.add_hline(y=cutoff,line_dash='dash')
            presentation.plotly_chart(fig,use_container_width=True)
        st.caption('Residual magnitude >3 is a review heuristic. Binary logistic residuals are not normally distributed; their branching patterns alone do not establish misspecification.')
    with tabs[3]:
        st.markdown('**Multicollinearity**')
        if r['vif'].empty: st.info('The retained model is intercept-only; VIF is not applicable.')
        else: presentation.dataframe(format_table(r['vif']),hide_index=True,use_container_width=True)
        st.caption('VIF <5: generally acceptable; 5–10: investigate; >10: strong warning. No variables are automatically removed based on VIF.')
        st.markdown('**Influential observations**')
        inf=r['influence']
        presentation.dataframe(format_table(inf),hide_index=True,use_container_width=True)
        st.caption('Review thresholds: |residual| >3, leverage >2k/n, Cook’s distance >4/n, or |DFBETA| >2/√n. k includes the intercept. These are heuristics; inspect cases and sensitivity rather than deleting observations automatically.')
        st.download_button('Download observation diagnostics (CSV)',inf.to_csv(index=False),'logistic_observations.csv',key='rel_log_inf_download')
        if not r['dfbetas'].empty:
            with st.expander('Coefficient influence (DFBETAs)'):
                presentation.dataframe(format_table(r['dfbetas']),hide_index=True,use_container_width=True)
    with tabs[4]:
        render_visuals(r,key)
    with tabs[5]:
        render_prediction(r,key)
    with tabs[6]:
        render_findings(r,key)


def draw_effect(r,xvar,moderator,title,key):
    levels=None
    if moderator in r['spec']['categorical'] and len(r['spec']['levels'][moderator])>8:
        options=r['spec']['levels'][moderator]
        levels=st.multiselect('Moderator categories to display',options,default=options[:8],key=key+'_levels')
        if not levels: st.info('Select at least one moderator category.');return
    curves,settings=adjusted_curves(r,xvar,moderator,levels,prediction_function=predict_logistic)
    fig=effect_figure(r,xvar,curves,title,confidence=r['confidence'])
    fig.update_yaxes(title_text=f"Probability of {r['spec']['target']} = {r['event']!r}",range=[0,1])
    presentation.plotly_chart(fig,use_container_width=True,key=key)
    if not settings.empty: presentation.dataframe(settings,hide_index=True,use_container_width=True)
    st.caption(f"Adjusted event probabilities with pointwise {100*r['confidence']:g}% confidence intervals. Other numerical predictors use complete-case means and categories use reference levels, as shown. Numeric moderators use P25/P50/P75. Curves stay within observed predictor ranges (within category when applicable); sparse multivariable combinations can still be poorly supported. Category connections are visual guides.")


def render_visuals(r,key):
    st.markdown('**Calibration — analysis sample**')
    cal=r['calibration']
    fig=go.Figure(go.Scatter(x=cal.Predicted,y=cal.Observed,mode='lines+markers',name='Observed vs predicted',error_y=dict(type='data',symmetric=False,array=cal['Observed CI Upper']-cal.Observed,arrayminus=cal.Observed-cal['Observed CI Lower'])))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash'),name='Perfect agreement'))
    fig.update_layout(xaxis_title='Mean predicted event probability',yaxis_title='Observed event proportion')
    fig.update_xaxes(range=[0,1]);fig.update_yaxes(range=[0,1]);presentation.plotly_chart(fig,use_container_width=True)
    presentation.dataframe(format_table(cal),hide_index=True,use_container_width=True)
    st.caption(f"In-sample, quantile-grouped calibration with {100*r['confidence']:g}% Wilson intervals for observed proportions. Ties can reduce the number of groups. Brier score: {r['brier']:.4g}. Grouping and limited sample size affect appearance; this is not independent validation. An in-sample calibration intercept/slope is omitted because it can be ideal by construction.")
    st.markdown('**Odds ratios**')
    table=r['coefficients'].iloc[1:]
    finite=np.isfinite(table[['Odds Ratio','OR CI Lower','OR CI Upper']]).all(axis=1) if len(table) else []
    table=table.loc[finite] if len(table) else table
    if not table.empty:
        fig=go.Figure(go.Scatter(x=table['Odds Ratio'],y=table.Term,mode='markers',error_x=dict(type='data',symmetric=False,array=table['OR CI Upper']-table['Odds Ratio'],arrayminus=table['Odds Ratio']-table['OR CI Lower'])))
        fig.add_vline(x=1,line_dash='dash');fig.update_xaxes(type='log',title_text=f'Odds ratio ({100*r["confidence"]:g}% CI, log scale)');presentation.plotly_chart(fig,use_container_width=True)
    interactions=retained_interactions(r);polynomials=retained_polynomials(r)
    main=[c for c in r['spec']['predictors'] if ('main',c) in r['active']]
    if main:
        with st.expander('Adjusted probability curves',expanded=not bool(interactions or polynomials)):
            c=st.selectbox('Predictor for probability curve',main,key='rel_log_curve_'+key[:12])
            draw_effect(r,c,None,'Adjusted probability: '+str(c),'rel_log_main_'+key[:12])
    if interactions:
        st.markdown('**Interaction plots**')
        b=st.selectbox('Retained interaction to plot',interactions,format_func=label,key='rel_log_interaction_'+key[:12])
        numeric=[c for c in b[1:] if c in r['spec']['numeric']]
        xvar=numeric[0] if len(numeric)==1 else st.selectbox('Interaction horizontal axis',list(b[1:]),key='rel_log_axis_'+str(b))
        moderator=b[2] if xvar==b[1] else b[1]
        draw_effect(r,xvar,moderator,'Interaction: '+label(b),'rel_log_int_plot_'+key[:12])
    if polynomials:
        st.markdown('**Polynomial plots**')
        xvar=st.selectbox('Predictor with retained polynomial terms',polynomials,key='rel_log_poly_'+key[:12])
        moderators=list(dict.fromkeys(b[2] if b[1]==xvar else b[1] for b in interactions if xvar in b[1:]))
        moderator=st.selectbox('Interacting predictor for separate curves',moderators,key='rel_log_poly_mod_'+key[:12]) if moderators else None
        draw_effect(r,xvar,moderator,'Polynomial probability: '+str(xvar),'rel_log_poly_plot_'+key[:12])
    with st.expander('Discrimination Analysis (optional)'):
        if st.checkbox('Show ROC curve and AUC',key='rel_log_roc_'+key[:12]):
            from sklearn.metrics import roc_curve,roc_auc_score
            y=r['clean'][r['spec']['target']]
            fpr,tpr,_=roc_curve(y,r['model'].fittedvalues)
            fig=px.line(x=fpr,y=tpr,labels={'x':'False-positive rate','y':'True-positive rate'})
            presentation.plotly_chart(fig,use_container_width=True)
            st.write(f'In-sample ROC-AUC: {roc_auc_score(y,r["model"].fittedvalues):.3f}')
            st.caption('Discrimination does not establish calibration or statistical adequacy. No threshold is optimized and probabilities are not converted into classifications.')


def render_prediction(r,key):
    confidence=100*r['confidence']
    st.markdown('**Fitted event probabilities**')
    pred=r['prediction_frame'][['mean','mean_ci_lower','mean_ci_upper']].copy()
    pred.columns=['Event probability',f'{confidence:g}% CI lower',f'{confidence:g}% CI upper']
    pred.insert(0,'Original data row',np.flatnonzero(r['valid'])+1)
    pred.insert(1,'Original index',[str(v) for v in r['clean'].index])
    presentation.dataframe(pred,hide_index=True,use_container_width=True)
    st.caption('Only complete analysis rows have fitted probabilities. Excluded rows are not silently imputed. Intervals quantify uncertainty in the estimated event probability, not an individual 0/1 outcome.')
    st.download_button('Download fitted probabilities (CSV)',pred.to_csv(index=False),'logistic_probabilities.csv',key='rel_log_prob_download')
    st.markdown('**New observation**')
    values={};outside=[]
    for c in r['spec']['predictors']:
        if c in r['spec']['categorical']:
            values[c]=st.selectbox(str(c),r['spec']['levels'][c],format_func=repr,key='rel_log_new_'+key[:12]+'_'+str(c))
        else:
            values[c]=st.number_input(str(c),value=float(r['clean'][c].mean()),key='rel_log_new_'+key[:12]+'_'+str(c))
            if not r['clean'][c].min()<=values[c]<=r['clean'][c].max(): outside.append(str(c))
    if st.button('Predict event probability',key='rel_log_predict_'+key[:12]):
        if outside: st.warning('Outside the observed predictor range: '+', '.join(outside)+'. This is extrapolation.')
        predicted=predict_logistic(r,pd.DataFrame([values])).iloc[0]
        st.metric(f"Probability of {r['spec']['target']} = {r['event']!r}",f"{predicted['mean']:.4f}")
        st.write(f"{confidence:g}% confidence interval: {predicted['mean_ci_lower']:.4f} to {predicted['mean_ci_upper']:.4f}")
        st.caption('Prediction uses the retained model and original fitted centers/reference categories. Confidence intervals assume the model is appropriate and do not account for selection uncertainty.')


def render_findings(r,key):
    event=f'{r["spec"]["target"]} = {r["event"]!r}'
    st.markdown('**Model Adequacy**')
    concerns=st.multiselect('Concerns identified during visual or study-design review',['Calibration disagreement','Systematic residual pattern','Dependence or clustering between observations'],key='rel_log_review_'+key[:12])
    status=r['adequacy']
    if concerns and status=='No major issues detected': status='Review recommended'
    if 'Dependence or clustering between observations' in concerns: status='Significant model concerns'
    st.write(status)
    if not r['issues'].empty: presentation.dataframe(r['issues'],hide_index=True,use_container_width=True)
    if concerns: st.warning('User-identified concerns: '+', '.join(concerns)+'. Review specification and study design before interpreting inference.')
    st.caption('This summary reflects available diagnostics, not a pass/fail certificate. Independence and visual calibration require user review; no arbitrary adequacy score is calculated.')
    if status=='Significant model concerns':
        st.warning('The data may not be adequately represented by classical binary logistic regression. Review the listed issues and revise the model. If prediction is the primary goal, consider Predictive Analytics; no alternative model is fitted automatically.')
    lines=[f'This model estimates the probability of {event}.']
    if r['lr_df']:
        lines.append(f"The overall likelihood-ratio test (χ²={r['lr']:.4g}, df={r['lr_df']}, p={p_text(r['lr_p'])}) "+('provides evidence of better fit than the intercept-only model.' if r['lr_p']<.05 else 'does not provide strong evidence of better fit than the intercept-only model.'))
    else: lines.append('The retained model is intercept-only; the overall likelihood-ratio comparison is not applicable.')
    complex_variables={c for b in r['active'] if b[0]!='main' for c in (b[1:] if b[0]=='interaction' else [b[1]])}
    for c in r['spec']['predictors']:
        b=('main',c)
        if b not in r['active']: continue
        if c in complex_variables:
            lines.append(f'{c} participates in retained interactions or polynomial terms. Its association depends on predictor values; inspect conditional probability plots rather than interpreting a constant overall odds ratio.')
            continue
        for i in r['indices'][b]:
            row=r['coefficients'].iloc[i]
            if not np.isfinite([row['Odds Ratio'],row['OR CI Lower'],row['OR CI Upper']]).all():
                lines.append(f"{row['Term']} has unbounded numerical odds-ratio uncertainty; do not interpret a finite effect size until the specification is revised.")
                continue
            direction='higher' if row['Odds Ratio']>1 else 'lower' if row['Odds Ratio']<1 else 'unchanged'
            uncertainty='The confidence interval includes 1, so the effect direction is uncertain at this confidence level.' if row['OR CI Lower']<=1<=row['OR CI Upper'] else 'The confidence interval excludes 1 at the selected confidence level.'
            lines.append(f"{row['Term']}: the point estimate indicates {direction} odds of {event}. {uncertainty}")
            comparison='relative to its displayed reference category' if c in r['spec']['categorical'] else 'for a one-unit increase'
            lines.append(f"After adjusting for the other model variables, {row['Term']} has an estimated odds ratio of {row['Odds Ratio']:.4g} {comparison} for {event} ({100*r['confidence']:g}% CI {row['OR CI Lower']:.4g}–{row['OR CI Upper']:.4g}; p={p_text(row['p-value'])}). This is an association in odds, not a probability ratio or causal effect.")
    lines.append(f"McFadden Pseudo-R²={r['pseudo_r2']:.4g}; this is not the percentage of variance explained. The in-sample Brier score is {r['brier']:.4g}; assess it alongside the calibration plot.")
    if r['elimination']: lines.append('AIC selection used the same observations. Reported inference is conditional on the selected model.')
    for line in lines: st.write(line)
    st.download_button('Download Regression Findings','\n\n'.join([status]+lines+concerns+[row['Issue']+': '+row['Evidence']+' '+row['Recommended action'] for row in r['issues'].to_dict('records')]),'logistic_findings.txt',key='rel_log_findings_download')
