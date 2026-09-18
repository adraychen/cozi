"""Interactive specification and reporting for exploratory OLS regression."""
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.regression_analysis import regression, predict
from utils.regression_reporting import format_table, findings, equation, p_text, selected_terms_table
from utils.regression_plots import actual_fitted, fitted_curve, coefficient_plot, partial_plot
from utils.regression_validation_ui import render_validation
from utils.regression_effect_plots import render_effect_plots
from utils.regression_controls import render_checkbox_specification, render_term_checkboxes


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
    target, predictors = render_checkbox_specification(responses,numerical,categorical)
    st.caption('Use meaningful measures rather than numeric identifiers. Ordered ratings assume equally spaced scores. Categorical predictors use reference-level 0/1 indicators.')
    if not predictors:
        st.info('Select at least one independent variable.'); return
    cats = [c for c in predictors if c in categorical]
    nums = [c for c in predictors if c not in cats]
    with st.expander('Model terms and elimination'):
        interactions, degrees = render_term_checkboxes(predictors,nums)
        mode = st.selectbox('Model-building method',['No elimination','Backward Elimination — Hierarchy Preserved'],key='reg_elimination')
        threshold = st.number_input('Removal p-value threshold',min_value=.001,max_value=.99,value=.05,step=.01,key='reg_threshold',disabled=mode=='No elimination')
        st.caption('Interactions are explicit. Hierarchy retains main effects and lower polynomial powers. Categorical factors and interactions are tested jointly as whole terms. Continuous predictors are mean-centered when interactions or powers are selected; Y remains in original units.')
    try:
        key, result = fit_cached(df,target,predictors,cats,interactions,degrees,mode!='No elimination',threshold)
        render_results(result,key)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError, FloatingPointError) as exc:
        st.error(f'Cannot fit this regression: {exc}')


def render_results(r,key):
    spec = r['spec']
    tabs = st.tabs(['Model Overview','Predictor Analysis','Diagnostics',
                    'Multicollinearity & Influence','Visual Analysis','Prediction','Regression Findings'])
    with tabs[0]:
        st.markdown('**Model summary**')
        summary = [('Response / Dependent Variable',str(spec['target'])),
                   ('Predictors / Independent Variables',', '.join(map(str,spec['predictors']))),
                   ('Original rows',str(r['original_n'])),('Used rows',str(r['n'])),
                   ('Excluded missing/nonfinite rows',str(r['excluded'])),
                   ('Model-building method','Backward elimination' if r['elimination'] else 'No elimination')]
        st.table(pd.DataFrame(summary,columns=['Model detail','Value']))
        st.markdown('**Fitted regression equation**')
        st.write(equation(r))
        st.markdown('**Model performance**')
        a,b,c = st.columns(3)
        a.metric('Adjusted R²',f"{r['adjusted_r2']:.3f}",help='R² adjusted for fitted model size.')
        b.metric('R²',f"{r['r2']:.3f}")
        c.metric('Residual standard error',f"{r['residual_se']:.4g}")
        stats = [('RMSE',f"{r['rmse']:.4g}"),('MAE',f"{r['mae']:.4g}"),
                 ('Model degrees of freedom',str(r['model_df'])),('Residual degrees of freedom',str(r['residual_df'])),
                 ('Overall F-statistic',f"{r['f_stat']:.4g}" if np.isfinite(r['f_stat']) else 'N/A'),('Overall F-test p-value',p_text(r['f_p']))]
        st.table(pd.DataFrame(stats,columns=['Performance measure','Value']))
        st.caption('Fit on the analyzed observations. RMSE = √mean(residual²); MAE = mean(|residual|); residual standard error = √(residual sum of squares / residual df). Error measures use response units. Overall F tests all non-intercept coefficients jointly. Inference uses α = 0.05.')
        st.markdown('**Selected terms**')
        st.table(selected_terms_table(r))
        if not r['active']:
            st.info('The final model contains the intercept only.')
        if r['coding']:
            st.markdown('**Categorical reference coding**')
            st.table(pd.DataFrame(r['coding']))
        st.caption('Category terms are indicators (1 for that level, otherwise 0). Main effects in interaction/polynomial models are conditional at centered zero and reference categories.')
        with st.expander('Elimination history'):
            if r['history'].empty:
                st.write('No terms were removed.')
            else:
                st.dataframe(format_table(r['history']),hide_index=True,use_container_width=True)
                st.caption('Removal p-values are joint partial F-tests before removal. Adjusted R² describes the refitted model after each removal. The complete-case sample stays fixed throughout.')
        for note in r['warnings']:
            st.warning(note)
    with tabs[1]:
        st.dataframe(format_table(r['coefficients']),hide_index=True,use_container_width=True)
        st.caption('Standardized coefficients = coefficient × SD(X)/SD(Y), only for ordinary numeric main effects without retained interactions or powers. N/A for other terms. Interactions describe changing associations; powers jointly describe a curve, not constant slopes.')
        if not r['term_tests'].empty:
            st.dataframe(format_table(r['term_tests'].drop(columns='block')),hide_index=True,use_container_width=True)
            st.caption('Joint term tests keep categorical levels together. Hierarchy-protected terms cannot be removed while their dependent higher-order terms remain.')
        st.download_button('Download regression coefficients (CSV)',r['coefficients'].to_csv(index=False),file_name='regression_coefficients.csv',key='reg_coeff_download')
    with tabs[2]:
        render_validation(r)
        visual_concerns = st.multiselect('Patterns observed when reviewing the plots',
            ['Systematic residual pattern','Funnel-shaped residual spread','Large Q–Q departures'],key='reg_visual_'+key[:12])
    with tabs[3]:
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
    with tabs[4]:
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
    with tabs[5]:
        prediction_ui(r,key)
    with tabs[6]:
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
