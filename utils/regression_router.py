"""One Regression entry point, with model family controlled by effective metadata."""
import pandas as pd
import streamlit as st
from utils.regression_controls import render_checkbox_specification


def response_candidates(df,numerical,categorical,metadata=None):
    metadata=metadata or {}
    numeric=[c for c in numerical if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    binary=[c for c in categorical if df[c].nunique(dropna=True)==2 and metadata.get(c,{}).get('variable_subtype','Binary')=='Binary']
    return numeric+binary


def render_regression(df,numerical,categorical):
    metadata=st.session_state.get('variable_metadata',{})
    responses=response_candidates(df,numerical,categorical,metadata)
    if not responses:
        st.info('Regression requires a Numerical or Categorical → Binary response. Update Column Overview if needed.')
        return
    st.markdown('##### Regression')
    st.caption('Response classification selects the model: Numerical → linear regression; Categorical → Binary → binary logistic regression. Numeric 0/1 stays numerical unless you change its classification in Column Overview.')
    target,predictors=render_checkbox_specification(responses,numerical,categorical)
    if not predictors:
        st.info('Select at least one independent variable.');return
    if target in numerical:
        from utils.regression_ui import render_regression as render_linear
        render_linear(df,numerical,categorical,selection=(target,predictors))
    else:
        from utils.logistic_ui import render_logistic
        render_logistic(df,target,predictors,[c for c in predictors if c in categorical],metadata)
