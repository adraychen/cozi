"""Shared explanations and guides for the workspace."""
import streamlit as st


FEATURE_HELP = {
    'Aggregate': 'Group rows and calculate summaries.',
    'Pivot': 'Turn values into columns.',
    'Unpivot': 'Turn columns into values.',
}


def render_help():
    query = st.text_input('Search features', placeholder='Enter a feature name…',
                          key='help_feature_search').strip().casefold()
    features = [name for name in sorted(FEATURE_HELP, key=str.casefold)
                if query in name.casefold()]
    if not features:
        st.write('No matching features found.')
    for name in features:
        with st.expander(name, expanded=True):
            st.write(FEATURE_HELP[name])
