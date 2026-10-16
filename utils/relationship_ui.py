from utils.numerical_formatting import format_statistic as fmt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import presentation
from utils.regression_analysis import regression
from utils.regression_router import render_regression, response_candidates
from utils.contingency_ui import render_contingency
from utils.correlation_ui import render_correlation
from utils.temporal import period_frequencies, period_values


def render_relationship_analysis(df, numerical, categorical, dates):
    metadata = st.session_state.get('variable_metadata', {})
    dates = [date for date in dates if period_frequencies(metadata.get(date, {}).get('variable_subtype')) and metadata.get(date, {}).get('variable_subtype') != 'Year-Quarter']
    methods = []
    if response_candidates(df,numerical,categorical,st.session_state.get('variable_metadata',{})) and len(numerical) + len(categorical) >= 2:
        methods.append('Regression')
    if len(numerical) >= 2:
        methods.append('Correlation')
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
        elif method.startswith('Time–numerical'):
            date = st.selectbox('Time variable', dates, key='rel_date')
            num = st.selectbox('Measure', numerical, key='rel_time_y')
            subtype = metadata.get(date, {}).get('variable_subtype')
            frequencies = period_frequencies(subtype)
            if st.session_state.get('rel_period') not in frequencies:
                st.session_state.pop('rel_period', None)
            freq = st.selectbox('Time period', frequencies, index=frequencies.index('Monthly') if 'Monthly' in frequencies else 0, key='rel_period')
            agg = st.selectbox('Aggregation', ['Mean','Sum'], key='rel_agg')
            clean = pd.DataFrame({'Period': period_values(df[date], freq, subtype), 'Value': pd.to_numeric(df[num],errors='coerce').replace([np.inf,-np.inf],np.nan)}).dropna()
            if clean.empty:
                raise ValueError('No valid date and measure pairs.')
            grouped = clean.groupby('Period').Value
            values = grouped.mean() if agg=='Mean' else grouped.sum(min_count=1)
            if len(values)<3:
                raise ValueError('At least three observed periods are required for trend estimation.')
            annual = subtype == 'Year'
            elapsed = values.index.year-values.index.year[0] if annual else (values.index.to_timestamp()-values.index.to_timestamp()[0]).total_seconds()/86400
            elapsed_label = 'Elapsed years' if annual else 'Elapsed days'
            trend = regression(pd.DataFrame({'Value':values.to_numpy(),elapsed_label:elapsed}), 'Value',[elapsed_label])
            axis = 'Year' if annual else 'Date'
            chart = pd.DataFrame({axis:values.index.year if annual else values.index.to_timestamp(),'Observed':values.to_numpy(),'Linear trend':trend['fitted']})
            presentation.plotly_chart(px.line(chart,x=axis,y=['Observed','Linear trend']),use_container_width=True)
            slope = trend['coefficients'].iloc[1]['Coefficient']
            unit = 'year' if annual else 'day'
            st.write(f'Linear trend: {slope:.4g} units per elapsed {unit}; R² = {fmt(trend['r2'], 'r2')}.')
            presentation.dataframe(chart,hide_index=True,use_container_width=True)
            st.caption(f'Excluded {len(df)-len(clean):,} invalid/missing pairs. Missing periods are omitted, not filled with zero; elapsed calendar time is used. Trend is descriptive, not a forecast or significance test: seasonality and autocorrelation are not modelled. Boundary periods may be incomplete.')
        else:
            render_regression(df, numerical, categorical)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        st.info(str(exc))
