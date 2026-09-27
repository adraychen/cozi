"""Visual diagnostics complement statistical tests; none certify assumptions."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import presentation
from scipy import stats
from utils.regression_reporting import format_table
from statsmodels.nonparametric.smoothers_lowess import lowess



def render_validation(result):
    for note in result['warnings']:
        st.warning(note)
    diagnostic = result['diagnostics']
    if not diagnostic.empty and diagnostic['Interpretation'].eq('Review warning').any():
        flagged = diagnostic.loc[diagnostic['Interpretation'].eq('Review warning'), 'Diagnostic'].astype(str).tolist()
        st.warning('Diagnostic checks to review (p < 0.05): ' + ', '.join(flagged) + '. These results suggest possible departures from the tested assumptions; review the corresponding plots and study design.')
    st.caption('Numeric identifiers are not meaningful measures. For ordered ratings, regression treats adjacent score differences as equally spaced.')
    with st.expander('Model validation', expanded=False):
        if result['perfect_fit']:
            st.info('Residual diagnostics are unavailable for a perfect or near-perfect fit. Check for duplicated variables or an outcome calculated directly from its predictors.')
            return
        residual = result['residual']
        st.markdown('**Linearity and constant error variance**')
        fig = px.scatter(x=result['fitted'], y=residual, labels={'x': 'Fitted outcome', 'y': 'Residual'})
        if len(np.unique(result['fitted'])) >= 3:
            smooth = lowess(residual, result['fitted'], frac=.6, return_sorted=True)
            fig.add_scatter(x=smooth[:, 0], y=smooth[:, 1], mode='lines', name='Smooth trend')
        fig.add_hline(y=0, line_dash='dash')
        presentation.plotly_chart(fig, use_container_width=True)
        st.write('Curvature suggests a missing nonlinear relationship. A funnel shape suggests changing error variance. In multiple regression, also consider relationships with individual predictors.')
        presentation.dataframe(format_table(diagnostic), hide_index=True, use_container_width=True)
        st.caption('Breusch–Pagan uses the Koenker LM variant. Unequal variance can invalidate conventional standard errors: consider robust standard errors or a revised model before relying on the reported intervals and p-values. This test can be unreliable in small samples.')
        st.markdown('**Normality of residuals**')
        a, b = st.columns(2)
        with a:
            presentation.plotly_chart(px.histogram(x=residual, labels={'x': 'Residual'}, title='Residual histogram'), use_container_width=True)
        with b:
            (theoretical, observed), (slope, intercept, _) = stats.probplot(residual, dist='norm')
            qq = px.scatter(x=theoretical, y=observed, labels={'x': 'Theoretical normal quantile', 'y': 'Ordered residual'}, title='Normal Q–Q plot')
            qq.add_scatter(x=theoretical, y=intercept + slope * theoretical, mode='lines', name='Normal reference')
            presentation.plotly_chart(qq, use_container_width=True)
        st.write('Skewness, heavy tails, and departures from the reference line warrant review. Normality matters for exact small-sample inference; non-normality alone does not bias OLS coefficients when the conditional mean model is correct.')
        st.caption('Jarque–Bera uses a large-sample approximation. Omnibus requires at least 8 observations and is less reliable below 20. Neither test validates the model by itself.')
        st.markdown('**Independence and observation order**')
        st.write('Independence depends on study design. Repeated measurements from one customer, clustered observations, or time dependence may require another model. Constant variance does not imply independence.')
        ordered = st.checkbox('The original row order represents a meaningful observation sequence', key='rel_reg_ordered')
        if ordered:
            st.metric('Durbin–Watson statistic', f"{result['dw']:.3f}")
            st.caption('Near 2 suggests little lag-one residual autocorrelation; toward 0 suggests positive and toward 4 negative autocorrelation. This is not a pass/fail rule or proof of independence. Rows retain their original order; excluded rows create gaps, and the statistic uses consecutive retained observations.')
            fig = px.scatter(x=result['source_rows'], y=residual, labels={'x': 'Original data row (1-based, excluding header)', 'y': 'Residual'})
            fig.add_hline(y=0, line_dash='dash')
            presentation.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Durbin–Watson and the sequence plot are hidden until meaningful row order is confirmed. Independence remains unverified.')
        st.markdown('**Influential observations**')
        influence = pd.DataFrame({'Original data row': result['source_rows'], 'Residual': residual,
                                  'Leverage': result['leverage'], "Cook’s distance": result['cooks']})
        flagged = (result['cooks'] > 4 / result['n']) | (result['leverage'] > 2 * result['k'] / result['n']) | ~np.isfinite(result['cooks'])
        st.write(f'{int(flagged.sum())} observations flagged for review; none removed automatically.')
        st.caption('Cook’s distance > 4/n or leverage > 2k/n are screening heuristics, not formal rejection rules. Here k includes the intercept. Undefined influence estimates are also flagged.')
        if flagged.any():
            presentation.dataframe(influence.loc[flagged], hide_index=True, use_container_width=True)
