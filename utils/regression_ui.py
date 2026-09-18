"""Interactive specification and reporting for exploratory OLS regression."""
import hashlib
from itertools import combinations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.regression_analysis import regression, predict
from utils.regression_design import label
from utils.regression_reporting import format_table, findings, equation, p_text
from utils.regression_plots import actual_fitted, fitted_curve, coefficient_plot, partial_plot
from utils.regression_validation_ui import render_validation
from utils.regression_effect_plots import render_effect_plots


def fit_cached(df, target, predictors, categorical, interactions, degrees, elimination, threshold):
    # Session-local cache retains one fit; every selected data value and model option participates.
    frame = df[[target]+predictors]
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes())
    digest.update(repr((list(frame.columns),list(map(str,frame.dtypes)),target,predictors,categorical,interactions,degrees,elimination,threshold)).encode())
    key = digest.hexdigest()
    cached = st.session_state.get('regression_cached_fit')
    if cached is None or cached[0] != key:
        result = regression(df,target,predictors,categorical,interactions,degrees,elimination,threshold)
        st.session_state['regression_cached_fit'] = (key,result)
    return key, st.session_state['regression_cached_fit'][1]


def render_regression(df, numerical, categorical):
    st.markdown('##### Linear regression')
    responses = [c for c in numerical if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not responses:
        st.info('Select a numeric dependent variable.'); return
    if st.session_state.get('rel_target_False') not in responses:
        st.session_state.pop('rel_target_False',None)
    target = st.selectbox('Target / dependent variable (Y)',responses,key='rel_target_False')
    options = [c for c in dict.fromkeys(numerical+categorical) if c != target]
    if 'rel_predictors' in st.session_state:
        st.session_state['rel_predictors'] = [c for c in st.session_state['rel_predictors'] if c in options]
    defaults = {} if 'rel_predictors' in st.session_state else {'default':options[:1]}
    predictors = st.multiselect('Predictors / independent variables (X)',options,key='rel_predictors',**defaults)
    st.caption('Use meaningful measures rather than numeric identifiers. Ordered ratings assume equally spaced scores. Categorical predictors use reference-level 0/1 indicators.')
    if not predictors:
        st.info('Select at least one independent variable.'); return
    cats = [c for c in predictors if c in categorical]
    nums = [c for c in predictors if c not in cats]
    with st.expander('Model terms and elimination'):
        pairs = list(combinations(predictors,2))
        if 'reg_interactions' in st.session_state:
            st.session_state['reg_interactions'] = [p for p in st.session_state['reg_interactions'] if p in pairs]
        interactions = st.multiselect('Interaction terms',pairs,format_func=lambda pair:' × '.join(map(str,pair)),key='reg_interactions')
        degrees = {c:st.selectbox(f'Polynomial degree: {c}',[1,2,3],key=f'reg_degree_{c}') for c in nums}
        mode = st.selectbox('Model-building method',['No elimination','Backward Elimination — Hierarchy Preserved'],key='reg_elimination')
        threshold = st.number_input('Removal p-value threshold',min_value=.001,max_value=.99,value=.05,step=.01,key='reg_threshold',disabled=mode=='No elimination')
        st.caption('Interactions are explicit. Hierarchy retains main effects and lower polynomial powers. Categorical factors and interactions are tested jointly as whole terms. Continuous predictors are mean-centered when interactions or powers are selected; Y remains in original units.')
    try:
        key, result = fit_cached(df,target,predictors,cats,interactions,degrees,mode!='No elimination',threshold)
        render_results(result,key)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError, FloatingPointError) as exc:
        st.error(f'Cannot fit this regression: {exc}')


def render_results(r,key):
    spec = r['spec']; model = r['model']
    tabs = st.tabs(['Model Overview','Model Performance','Predictor Analysis','Diagnostics',
                    'Multicollinearity & Influence','Visual Analysis','Prediction','Regression Findings'])
    with tabs[0]:
        st.write(f"Response: {spec['target']}; specified predictors: {', '.join(map(str,spec['predictors']))}.")
        st.write(f"Original rows: {r['original_n']:,}; used: {r['n']:,}; excluded missing/nonfinite rows: {r['excluded']:,}.")
        st.write(f"Fitted terms: {len(r['active'])}; fitted coefficients including intercept: {r['k']}.")
        st.write('Retained terms: ' + (', '.join(label(b) for b in r['active']) or 'Intercept only'))
        st.write('Categorical predictors: ' + (', '.join(map(str,spec['categorical'])) or 'None'))
        if r['coding']:
            st.dataframe(pd.DataFrame(r['coding']),hide_index=True,use_container_width=True)
        st.write('Selected interactions: ' + (', '.join(' × '.join(map(str,p)) for p in spec['interactions']) or 'None'))
        st.write('Selected polynomial degrees: ' + (', '.join(f'{c}: {d}' for c,d in spec['degrees'].items() if d>1) or 'None'))
        if spec['centered']:
            st.write('Centering applied: ' + '; '.join(f'{c} (centered) = {c} − {mean:.6g}' for c,mean in spec['means'].items()))
        else:
            st.write('Centering: not applied.')
        st.write(equation(r))
        st.caption('Category terms are indicators (1 for that level, otherwise 0). Main effects in interaction/polynomial models are conditional at centered zero and reference categories.')
        st.write('Elimination: ' + ('Backward Elimination — Hierarchy Preserved' if r['elimination'] else 'No elimination'))
        if r['history'].empty:
            st.write('No terms were removed.')
        else:
            st.dataframe(format_table(r['history']),hide_index=True,use_container_width=True)
            st.caption('Removal p-values are joint partial F-tests for the removed blocks before removal. Adjusted R² is for the refitted model after each removal. The complete-case sample stays fixed throughout.')
        for note in r['warnings']:
            st.warning(note)
    with tabs[1]:
        a,b,c = st.columns(3)
        a.metric('Adjusted R²',f"{r['adjusted_r2']:.3f}",help='R² adjusted for fitted model size; emphasized for multiple regression.')
        b.metric('R²',f"{r['r2']:.3f}")
        c.metric('Residual standard error',f"{r['residual_se']:.4g}")
        stats = {'RMSE':r['rmse'],'MAE':r['mae'],'Observations':r['n'],'Model terms':len(r['active']),
                 'Model degrees of freedom':r['model_df'],'Residual degrees of freedom':r['residual_df'],'F-statistic':r['f_stat'],'p-value':r['f_p']}
        st.dataframe(format_table(pd.DataFrame([stats])),hide_index=True,use_container_width=True)
        st.caption('Descriptive fit on the analyzed observations. RMSE = √mean(residual²); MAE = mean(|residual|); residual standard error = √(residual sum of squares / residual df). Error measures use response units. Overall F tests all non-intercept coefficients jointly. Inference uses α = 0.05.')
    with tabs[2]:
        st.dataframe(format_table(r['coefficients']),hide_index=True,use_container_width=True)
        st.caption('Standardized coefficients = coefficient × SD(X)/SD(Y), only for ordinary numeric main effects without retained interactions or powers. N/A for other terms. Interactions describe changing associations; powers jointly describe a curve, not constant slopes.')
        if not r['term_tests'].empty:
            st.dataframe(format_table(r['term_tests'].drop(columns='block')),hide_index=True,use_container_width=True)
            st.caption('Joint term tests keep categorical levels together. Hierarchy-protected terms cannot be removed while their dependent higher-order terms remain.')
        st.download_button('Download regression coefficients (CSV)',r['coefficients'].to_csv(index=False),file_name='regression_coefficients.csv',key='reg_coeff_download')
    with tabs[3]:
        render_validation(r)
        visual_concerns = st.multiselect('Patterns observed when reviewing the plots',
            ['Systematic residual pattern','Funnel-shaped residual spread','Large Q–Q departures'],key='reg_visual_'+key[:12])
    with tabs[4]:
        st.markdown('**Multicollinearity**')
        if r['vif'].empty:
            st.info('No predictor terms remain for VIF analysis.')
        else:
            display = r['vif'].copy()
            display['VIF'] = display['VIF'].map(lambda v:f'{v:.2f}')
            display['Tolerance'] = display['Tolerance'].map(lambda v:f'{v:.3f}')
            st.dataframe(display,hide_index=True,use_container_width=True)
        st.caption('VIF < 5: generally acceptable; 5–10: potential concern; >10: serious concern. Tolerance = 1/VIF. These are guidelines for clearly labeled encoded terms. Interaction/polynomial terms can be correlated even after centering.')
        if len(spec['numeric'])>=2:
            corr = r['clean'][spec['numeric']].corr()
            st.plotly_chart(px.imshow(corr,zmin=-1,zmax=1,text_auto='.2f',color_continuous_scale='RdBu_r',title='Original numeric predictor correlations'),use_container_width=True)
            st.caption('Original selected numeric predictors on the same complete cases. Categorical codes and generated terms are excluded.')
        else:
            st.info('A numeric predictor correlation matrix requires at least two numeric predictors.')
        st.markdown('**Influential observations**')
        inf = r['influence']
        st.write(f"{int(inf['Flagged'].sum())} observations flagged for review. None removed.")
        st.caption('Internally studentized residuals: |r| > 2 merits review, >3 is potentially extreme. Leverage > 2k/n and Cook’s distance > 4/n are heuristics; k includes the intercept. Undefined measures also warrant review.')
        ranked = inf.sort_values("Cook's distance",ascending=False,na_position='last').head(20)
        st.dataframe(format_table(ranked),hide_index=True,use_container_width=True)
        st.write('Highest-leverage observations')
        st.dataframe(format_table(inf.sort_values('Leverage',ascending=False).head(10)),hide_index=True,use_container_width=True)
        for measure, cutoff in [("Cook's distance",4/r['n']),('Leverage',2*r['k']/r['n']),('Studentized residual',2)]:
            fig = px.scatter(inf,x='Original data row',y=measure,hover_data=['Original index'],title=measure)
            fig.add_hline(y=cutoff,line_dash='dash')
            if measure=='Studentized residual':
                fig.add_hline(y=-2,line_dash='dash'); fig.add_hline(y=3,line_dash='dot'); fig.add_hline(y=-3,line_dash='dot')
            st.plotly_chart(fig,use_container_width=True)
        st.download_button('Download all observation diagnostics (CSV)',inf.to_csv(index=False),file_name='regression_influence.csv',key='reg_inf_download')
    with tabs[5]:
        render_effect_plots(r,key)
        if len(spec['predictors'])==1 and not spec['categorical'] and not any(b[0]=='power' for b in r['active']):
            st.plotly_chart(fitted_curve(r),use_container_width=True)
            st.caption('Curve covers the observed predictor range. Blue confidence band: uncertainty around the mean response. Orange prediction band: uncertainty for an individual future observation. Bands rely on the regression assumptions; they are not guarantees.')
        st.plotly_chart(actual_fitted(r),use_container_width=True)
        st.caption('Actual versus fitted values describe fit on these observations; the reference line represents perfect agreement.')
        if r['active']:
            st.plotly_chart(coefficient_plot(r),use_container_width=True)
        eligible = [c for c in spec['numeric'] if ('main',c) in r['active'] and not any(b[0]=='interaction' and c in b[1:] for b in r['active'])]
        if eligible:
            c = st.selectbox('Continuous predictor for partial residual plot',eligible,key='reg_partial_'+key[:12])
            st.plotly_chart(partial_plot(r,c),use_container_width=True)
            st.caption('Residual plus this predictor’s fitted component, including retained powers. This checks form after accounting for the other fitted components.')
        else:
            st.info('A partial residual plot is not shown when no continuous main effect without interactions remains; its interpretation would be ambiguous.')
    with tabs[6]:
        prediction_ui(r,key)
    with tabs[7]:
        lines = findings(r,st.session_state.get('rel_reg_ordered',False),visual_concerns)
        for text in lines:
            st.write(text)
        st.download_button('Download Regression Findings', '\n\n'.join(lines),file_name='regression_findings.txt',key='reg_find_download')


def prediction_ui(r,key):
    spec = r['spec']
    st.caption('Enter original predictor values. Centering, indicators, powers and interactions are reconstructed automatically. Values are constrained to observed marginal numeric ranges; unusual combinations can still be poorly supported.')
    values = {}
    with st.form('reg_predict_'+key[:12]):
        for c in spec['predictors']:
            if c in spec['categorical']:
                values[c] = st.selectbox(str(c),spec['levels'][c],key=f'pred_{key[:12]}_{c}')
            else:
                values[c] = st.number_input(str(c),min_value=float(r['clean'][c].min()),max_value=float(r['clean'][c].max()),value=float(r['clean'][c].mean()),key=f'pred_{key[:12]}_{c}')
        submitted = st.form_submit_button('Predict response')
    if submitted:
        try:
            result = predict(r,pd.DataFrame([values])).iloc[0]
            st.metric(f"Predicted {spec['target']}",f"{result['mean']:.5g}")
            if r['perfect_fit']:
                st.info('Intervals are unavailable for a perfect or near-perfect fit.')
            else:
                st.write(f"95% confidence interval for mean response: {result['mean_ci_lower']:.5g} to {result['mean_ci_upper']:.5g}")
                st.write(f"95% prediction interval for an individual future observation: {result['obs_ci_lower']:.5g} to {result['obs_ci_upper']:.5g}")
            st.caption('Intervals assume an appropriate model and independent errors with constant variance; the prediction interval is not a guarantee. Selection uncertainty is not included after elimination.')
        except (ValueError,TypeError,np.linalg.LinAlgError) as exc:
            st.error(str(exc))
