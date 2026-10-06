"""Shared explanations and guides for the workspace."""
import streamlit as st


def render_help():
    st.subheader('Restructure data')
    with st.expander('Aggregate', expanded=True):
        st.write('Group rows and calculate summaries.')
    with st.expander('Pivot', expanded=True):
        st.write('Turn values into columns.')
    with st.expander('Unpivot', expanded=True):
        st.write('Turn columns into values.')
