"""Shared explanations and guides for the workspace."""
import streamlit as st


FEATURE_HELP = {
    '95% Confidence Interval (Mean)': "Estimates the population mean using Student's t distribution. This method assumes independent observations and approximately normal data or a sufficiently large sample. Missing and infinite values are excluded. At least two valid values are required.",
    'Aggregate': 'Group rows and calculate summaries.',
    'Coefficient of Variation': 'Expresses the sample standard deviation as a percentage of the mean. It is calculated by dividing the sample standard deviation by the mean and multiplying by 100. The app shows it only when the mean is positive. It is meaningful for ratio-scale data with a true zero and can be unstable when the mean is close to zero.',
    'Pivot': 'Turn values into columns.',
    'Standard Error (Mean)': 'Measures the uncertainty in the estimated mean. It is calculated by dividing the sample standard deviation by the square root of the number of valid observations. Missing and infinite values are excluded. At least two valid values are required.',
    'Unpivot': 'Turn columns into values.',
}


def render_help():
    query = st.text_input('Search features', placeholder='Enter a feature name…',
                          label_visibility='collapsed', icon=':material/search:',
                          key='help_feature_search').strip().casefold()
    features = [name for name in sorted(FEATURE_HELP, key=str.casefold)
                if query in name.casefold()]
    with st.container(border=True):
        if not features:
            st.write('No matching features found.')
        for index, name in enumerate(features):
            if index:
                st.divider()
            st.markdown('**'+name+'**')
            st.write(FEATURE_HELP[name])
