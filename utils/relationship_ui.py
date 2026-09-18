import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.anova_ui import render_anova
from utils.regression_analysis import regression
from utils.regression_router import render_regression, response_candidates
from utils.contingency_ui import render_contingency
from utils.correlation_ui import render_correlation


def render_relationship_analysis(df, numerical, categorical, dates):
    methods = []
    if response_candidates(df,numerical,categorical,st.session_state.get('variable_metadata',{})) and len(numerical) + len(categorical) >= 2:
        methods.append('Regression')
    if len(numerical) >= 2:
        methods.append('Correlation')
    if categorical and numerical:
        methods.append('Categorical–numerical: group summaries and ANOVA')
    if len(categorical)>=2:
        methods.append('Categorical–categorical: cross-tabulation and chi-square')
    if dates and numerical:
        methods.append('Time–numerical: trend analysis')
    if not methods:
        return
    st.markdown('##### Relationship analysis')
    if st.session_state.get('rel_method') not in methods:
        st.session_state.pop('rel_method',None)
    method = st.selectbox('Analysis method', methods, key='rel_method')
    try:
        if method == 'Correlation':
            render_correlation(df, numerical)
        elif method.startswith('Categorical–categorical'):
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
            render_regression(df, numerical, categorical)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        st.info(str(exc))
