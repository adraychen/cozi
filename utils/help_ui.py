"""Shared explanations and guides for the workspace."""
import streamlit as st


FEATURE_HELP = {
    '95% Confidence Interval (Mean)': "Estimates the population mean using Student's t distribution. This method assumes independent observations and approximately normal data or a sufficiently large sample. Missing and infinite values are excluded. At least two valid values are required.",
    'Aggregate': 'Group rows and calculate summaries.',
    'Coefficient of Variation': 'Expresses the sample standard deviation as a percentage of the mean. The app shows it only when the mean is positive. It is meaningful for ratio-scale data with a true zero and can be unstable when the mean is close to zero.',
    "Effect Size (Cohen's f²)": (
        "Cohen's f² measures the magnitude of differences among group means relative to within-group variability. "
        "A variance-weighted version compatible with Welch's ANOVA is used to accommodate unequal group variances.\n\n"
        "Larger values indicate greater separation of group means. The effect size is not a percentage of explained variation, "
        "and its practical importance depends on the analytical context."
    ),
    'Games–Howell Pairwise Comparisons': """**Games–Howell Pairwise Comparisons** identifies which specific group means differ significantly after a One-way ANOVA. It is particularly suitable when groups have unequal variances or different sample sizes.

The method compares every pair of groups and adjusts for multiple comparisons to reduce the risk of false-positive findings.

**How to interpret the results:**

- **Mean Difference:** The difference between two group means (first group minus second group). A positive value indicates that the first group has a higher mean.
- **Adjusted p-value:** Indicates whether the difference is statistically significant after accounting for multiple comparisons. A value below the selected significance level (typically 0.05) indicates a statistically significant difference.
- **95% Confidence Interval:** Provides a range of plausible values for the difference between the two population means, adjusted for multiple comparisons. If the interval excludes zero, the difference is statistically significant at the corresponding confidence level.
- **Pairwise Comparison Plot:** Displays mean differences and their confidence intervals. The vertical reference line at zero represents no difference between the two group means.

**Important:** A statistically significant difference does not necessarily mean the difference is practically important. Consider the magnitude of the mean difference and its relevance to the analytical context.
""",
    'Mean Absolute Error (MAE)': 'Measures the average absolute difference between observed and predicted response values. It is expressed in response units. A residual is an observed value minus its predicted value.',
    'Pivot': 'Turn values into columns.',
    'Residual Standard Error': 'Estimates the standard deviation of regression errors, expressed in response units. Residual degrees of freedom account for the number of fitted model parameters. A residual is an observed value minus its predicted value.',
    'Root Mean Squared Error (RMSE)': 'Measures the size of prediction errors, giving larger errors more weight than MAE. It is expressed in response units. A residual is an observed value minus its predicted value.',
    'Standard Error (Mean)': 'Measures the uncertainty in the estimated mean. In the formula, n is the number of valid observations. Missing and infinite values are excluded. At least two valid values are required.',
    'Unpivot': 'Turn columns into values.',
    "Welch's One-way ANOVA": "Welch's One-way ANOVA compares a numerical response across independent groups and allows unequal variances. Identifiers are not meaningful outcomes; ordered ratings assume equally spaced scores.",
}

FEATURE_FORMULAS = {
    'Coefficient of Variation': r'\text{Coefficient of variation} = \frac{\text{Sample standard deviation}}{\text{Mean}} \times 100\%',
    'Mean Absolute Error (MAE)': r'\text{MAE} = \operatorname{mean}(|\text{residual}|)',
    'Residual Standard Error': r'\text{Residual standard error} = \sqrt{\frac{\text{Residual sum of squares}}{\text{Residual degrees of freedom}}}',
    'Root Mean Squared Error (RMSE)': r'\text{RMSE} = \sqrt{\operatorname{mean}(\text{residual}^{2})}',
    'Standard Error (Mean)': r'\text{Standard error} = \frac{\text{Sample standard deviation}}{\sqrt{n}}',
}


def render_help():
    query = st.text_input('Search features',
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
            if name in FEATURE_FORMULAS:
                st.latex(FEATURE_FORMULAS[name])
            st.write(FEATURE_HELP[name])
