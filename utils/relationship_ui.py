import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.anova_ui import render_anova
from utils.regression_analysis import regression
from utils.regression_validation_ui import render_validation
from utils.contingency_ui import render_contingency


def show_regression(result):
    st.write(f"Complete rows: {result['n']:,}; excluded: {result['excluded']:,}")
    a,b = st.columns(2)
    a.metric('R² (in-sample)', f"{result['r2']:.3f}")
    b.metric('Adjusted R²', f"{result['adjusted_r2']:.3f}")
    st.dataframe(result['coefficients'], hide_index=True, use_container_width=True)
    st.caption('Ordinary least squares. Coefficient intervals and p-values assume a correctly specified linear model, independent errors, constant error variance, and approximately normal errors for small-sample inference. Association does not imply causation. In-sample fit is not predictive validation.')


def render_relationship_analysis(df, numerical, categorical, dates):
    methods = []
    if len(numerical)>=2:
        methods.append('Numerical–numerical: correlation and regression')
    if categorical and numerical:
        methods.append('Categorical–numerical: group summaries and ANOVA')
    if len(categorical)>=2:
        methods.append('Categorical–categorical: cross-tabulation and chi-square')
    if dates and numerical:
        methods.append('Time–numerical: trend analysis')
    if len(numerical)>=3:
        methods.append('Multiple variables: multivariable regression')
    if not methods:
        return
    st.markdown('##### Relationship analysis')
    method = st.selectbox('Analysis method', methods, key='rel_method')
    try:
        if method.startswith('Categorical–categorical'):
            render_contingency(df, categorical, key_prefix='rel_ct')
        elif method.startswith('Categorical–numerical'):
            cat = st.selectbox('Grouping variable', categorical, key='rel_group')
            num = st.selectbox('Numerical outcome', numerical, key='rel_group_y')
            render_anova(df, cat, num)
        elif method.startswith('Time–numerical'):
            date = st.selectbox('Time variable', dates, key='rel_date')
            num = st.selectbox('Measure', numerical, key='rel_time_y')
            freq = st.selectbox('Time period', ['Daily','Weekly','Monthly','Quarterly','Yearly'], index=2, key='rel_period')
            agg = st.selectbox('Aggregation', ['Mean','Sum'], key='rel_agg')
            clean = pd.DataFrame({'Date': pd.to_datetime(df[date],errors='coerce'), 'Value': pd.to_numeric(df[num],errors='coerce').replace([np.inf,-np.inf],np.nan)}).dropna()
            if clean.empty:
                raise ValueError('No valid date and measure pairs.')
            period = clean.Date.dt.to_period({'Daily':'D','Weekly':'W-SUN','Monthly':'M','Quarterly':'Q','Yearly':'Y'}[freq])
            grouped = clean.groupby(period).Value
            values = grouped.mean() if agg=='Mean' else grouped.sum(min_count=1)
            if len(values)<3:
                raise ValueError('At least three observed periods are required for trend estimation.')
            elapsed = (values.index.to_timestamp()-values.index.to_timestamp()[0]).total_seconds()/86400
            trend = regression(pd.DataFrame({'Value':values.to_numpy(),'Elapsed days':elapsed}), 'Value',['Elapsed days'])
            chart = pd.DataFrame({'Date':values.index.to_timestamp(),'Observed':values.to_numpy(),'Linear trend':trend['fitted']})
            st.plotly_chart(px.line(chart,x='Date',y=['Observed','Linear trend']),use_container_width=True)
            slope = trend['coefficients'].iloc[1]['Coefficient']
            st.write(f'Linear trend: {slope:.4g} units per elapsed day; R² = {trend["r2"]:.3f}.')
            st.dataframe(chart,hide_index=True,use_container_width=True)
            st.caption(f'Excluded {len(df)-len(clean):,} invalid/missing pairs. Missing periods are omitted, not filled with zero; elapsed calendar time is used. Trend is descriptive, not a forecast or significance test: seasonality and autocorrelation are not modelled. Boundary periods may be incomplete.')
        else:
            multiple = method.startswith('Multiple')
            target = st.selectbox('Outcome (Y)', numerical, key='rel_target_'+str(multiple))
            options = [c for c in numerical if c!=target]
            if multiple:
                predictors = st.multiselect('Predictors (X)', options, default=options[:2], key='rel_predictors')
                if len(predictors)<2:
                    st.info('Select at least two numerical predictors.')
                    return
            else:
                predictors = [st.selectbox('Predictor (X)', options, key='rel_x')]
            result = regression(df,target,predictors)
            if multiple:
                chart = pd.DataFrame({'Observed': result['actual'], 'Fitted':result['fitted']})
                st.plotly_chart(px.scatter(chart,x='Observed',y='Fitted'),use_container_width=True)
            else:
                x = result['clean'][predictors[0]].to_numpy()
                fig = px.scatter(x=x,y=result['actual'],labels={'x':predictors[0],'y':target})
                order = np.argsort(x)
                if not result['perfect_fit']:
                    fig.add_scatter(x=x[order], y=result['mean_ci'][order, 0], mode='lines', line=dict(width=0), showlegend=False)
                    fig.add_scatter(x=x[order], y=result['mean_ci'][order, 1], mode='lines', line=dict(width=0), fill='tonexty', name='95% CI for mean response')
                fig.add_scatter(x=x[order],y=result['fitted'][order],mode='lines',name='OLS fit')
                st.plotly_chart(fig,use_container_width=True)
                from scipy.stats import pearsonr, spearmanr
                r,p = pearsonr(x,result['actual'])
                rho,_ = spearmanr(x,result['actual'])
                st.write(f'Pearson r = {r:.4f} (p = {p:.4g}); Spearman ρ = {rho:.4f}.')
            show_regression(result)
            terms = result['coefficients']
            equation = f"{target} = {terms.iloc[0]['Coefficient']:.4g}"
            for _, row in terms.iloc[1:].iterrows():
                equation += f" {row['Coefficient']:+.4g} × {row['Term']}"
            st.write(equation)
            for _, row in terms.iloc[1:].iterrows():
                st.write(f"A one-unit increase in {row['Term']} is associated with an estimated {row['Coefficient']:.4g}-unit change in {target}" + (', holding the other selected predictors constant.' if multiple else '.'))
            render_validation(result)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        st.info(str(exc))
