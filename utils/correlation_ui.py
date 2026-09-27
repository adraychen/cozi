"""Correlation analysis, separate from regression."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import presentation
from scipy import stats


def correlation_data(df, columns, method):
    if method not in ('Pearson', 'Spearman'):
        raise ValueError('Choose Pearson or Spearman.')
    if len(columns) < 2 or len(set(columns)) != len(columns):
        raise ValueError('Select at least two distinct numeric variables.')
    clean = df[columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid = clean.notna().astype(int)
    counts = valid.T @ valid
    matrix = clean.corr(method=method.lower(), min_periods=3)
    return clean, matrix, counts


def render_correlation(df, numerical):
    st.markdown('##### Correlation')
    method = st.selectbox('Correlation method', ['Pearson', 'Spearman'], key='rel_corr_method')
    clean, matrix, counts = correlation_data(df, numerical, method)
    st.caption('Uses the numeric columns selected above; categorical and date columns are not included. Pearson measures linear association; Spearman measures monotonic association using ranks. Correlation does not imply causation.')
    st.caption('Missing and infinite values are excluded separately for each pair. At least three usable pairs and variation in both variables are required; undefined correlations are shown as blank/NaN, not zero.')
    if len(numerical) == 2:
        a, b = numerical
        pairs = clean.dropna()
        st.write(f'Usable pairs: {len(pairs):,}; excluded: {len(df)-len(pairs):,}.')
        if not np.isfinite(matrix.loc[a, b]):
            st.warning('Correlation is unavailable: insufficient usable pairs or a constant variable.')
            return
        coefficient, p = (stats.pearsonr if method == 'Pearson' else stats.spearmanr)(pairs[a], pairs[b])
        left, right = st.columns(2)
        left.metric(f'{method} correlation', f'{coefficient:.4f}')
        right.metric('p-value (two-sided)', f'{p:.4g}')
        presentation.plotly_chart(px.scatter(pairs, x=a, y=b), use_container_width=True)
        st.caption('The p-value assumes independent observations. Pearson inference assumes an appropriate bivariate-normal model; Spearman uses an asymptotic approximation that can be unreliable in small samples. Inspect the scatter plot for outliers and nonlinear patterns.')
    else:
        fig = px.imshow(matrix, zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                        text_auto='.2f', title=f'{method} correlation matrix', aspect='auto')
        fig.update_traces(customdata=counts.to_numpy(), hovertemplate='%{y} / %{x}<br>Correlation: %{z:.3f}<br>Usable pairs: %{customdata}<extra></extra>')
        presentation.plotly_chart(fig, use_container_width=True)
        presentation.dataframe(matrix, use_container_width=True)
        if matrix.isna().any().any():
            st.warning('Some correlations are undefined because of insufficient pairs or constant values.')
        st.caption('Each matrix cell uses its own complete pairs, so sample sizes may differ. The matrix shows coefficients, not significance tests.')
        with st.expander('Usable observations per pair'):
            presentation.dataframe(counts, use_container_width=True)
