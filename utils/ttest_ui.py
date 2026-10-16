from utils.numerical_formatting import DESCRIPTIVE
from utils.report_capture import reportable, offer_result
"""Comparison t-test configuration and tabbed statistical reporting."""
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import presentation
from scipy import stats
from utils.charts import make_histogram_boxplot, CHART_COLOR, _base_layout, _style_axes
from utils.regression_reporting import format_table
from utils.ttest_analysis import numerical_candidates, prepare_ttest, run_ttest, TTestInputError
from utils.ttest_reporting import result_table, interpretation, export_result

MODES = {'One-Sample t-Test': 'one', 'Two Independent Samples t-Test': 'independent', 'Paired Samples t-Test': 'paired'}
ALTERNATIVES = {'Difference != 0 (two-sided)': 'two-sided', 'Difference > 0 (greater)': 'greater', 'Difference < 0 (less)': 'less'}


def choose(label, options, key, format_func=str):
    """Drop stale state when the outer Comparison selection or dataset changes."""
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    return st.selectbox(label, options, key=key, format_func=format_func)


def render_ttest(df, numerical, categorical, guided=False):
    try:
        _render_ttest(df, numerical, categorical, guided=guided)
    except TTestInputError as exc:
        presentation.info(str(exc))
    except Exception:
        logging.getLogger(__name__).exception('Unable to render t-test')
        presentation.error('Unable to calculate this t-test. Review the selected variables, usable sample sizes, and measurement scale.')


@reportable
def _render_ttest(df, numerical, categorical, guided=False):
    numbers = numerical_candidates(df, numerical, st.session_state.get('variable_metadata', {}))
    if not numbers:
        presentation.info('Select at least one quantitative Numerical variable in Comparison. Review Variable Type/Subtype in Column Overview; automatically classified identifier-like names are excluded.')
        return
    categories = [c for c in categorical if c in df and st.session_state.get('variable_metadata', {}).get(c, {}).get('variable_type', 'Categorical') == 'Categorical']
    presentation.markdown('##### t-Test')
    presentation.caption('Choose the study design explicitly. Two separate columns are not automatically paired. Use meaningful measurements; identifier-like names require a deliberate Numerical override in Column Overview.')
    modes = list(MODES)
    if guided:
        modes = (['Two Independent Samples t-Test'] if categories else
                 ['One-Sample t-Test'] if len(numbers) == 1 else
                 ['Two Independent Samples t-Test', 'Paired Samples t-Test'])
    mode_label = choose('Test type', modes, 'gs_tt_mode')
    mode = MODES[mode_label]
    second = group = levels = None
    null_mean = 0.
    equal_var = False
    structure = 'One column' if mode == 'one' else 'Matched columns'
    if mode == 'independent':
        structures = []
        if categories:
            structures.append('Stacked format')
        if len(numbers) >= 2:
            structures.append('Separate-column format')
        if not structures:
            presentation.info('Select a numerical response and a categorical group, or two numerical sample columns in Comparison.')
            return
        structure = choose('Data structure', structures, 'gs_tt_structure')
    if mode == 'paired' and len(numbers) < 2:
        presentation.info('Select two numerical variables in Comparison for matched observations, one pair per row.')
        return
    first = choose('Numerical response' if mode == 'one' or structure == 'Stacked format' else 'Sample 1 / Variable 1', numbers, 'gs_tt_first')
    if mode == 'one':
        null_mean = st.number_input('Hypothesized population mean', value=0., key='gs_tt_null')
    elif structure == 'Stacked format':
        group = choose('Categorical grouping variable', categories, 'gs_tt_group')
        observed = list(df[group].dropna().unique())
        if len(observed) < 2:
            presentation.info('The grouping variable needs at least two observed groups. Choose another grouping variable.')
            return
        level1 = choose('Group 1', observed, 'gs_tt_level1')
        level2 = choose('Group 2', [v for v in observed if v != level1], 'gs_tt_level2')
        levels = [level1, level2]
    elif mode != 'one':
        second = choose('Sample 2 / Variable 2', [c for c in numbers if c != first], 'gs_tt_second')
    if mode == 'independent':
        equal_var = st.checkbox('Assume equal variances', value=False, key='gs_tt_equal')
        presentation.caption('Off: Welch. On: pooled Student t-test. Method selection is explicit and never determined by a preliminary variance test.')
    alt_label = choose('Alternative hypothesis', list(ALTERNATIVES), 'gs_tt_alternative')
    confidence = st.number_input('Confidence level (%)', min_value=80., max_value=99.9, value=95., step=1., key='gs_tt_confidence') / 100
    presentation.caption(f'Significance level alpha = {1-confidence:.3g}. Choose a directional hypothesis before reviewing the results. One-sided intervals have one infinite bound.')
    if mode == 'paired':
        presentation.info('Confirm your study uses matched observations in the same row. Observations are related within each pair and must be independent between pairs. Normality concerns the paired differences.')
    else:
        presentation.info('Observations must be independent within each sample and, for two samples, between samples. Independence cannot be verified from the dataset alone. Use Paired Samples t-Test for matched data.')
    prepared = prepare_ttest(df, mode, first, second, group, levels)
    result, error = None, None
    try:
        result = run_ttest(prepared, null_mean, ALTERNATIVES[alt_label], confidence, equal_var)
    except TTestInputError as exc:
        error = str(exc)
    difference = (f'{first} - hypothesized mean ({null_mean:g})' if mode == 'one' else
                  f'{group} = {levels[0]} - {group} = {levels[1]}' if levels is not None else f'{first} - {second}')
    presentation.markdown(f'**Difference = {difference}**')
    from utils.report_capture import begin_result
    begin_result()
    tabs = st.tabs(['Test Configuration', 'Descriptive Statistics', 'Diagnostics', 't-Test Results', 'Interpretation'])
    with tabs[0]:
        method = ("Pooled Two-Sample t-Test" if equal_var else "Welch's Two-Sample t-Test") if mode == 'independent' else mode_label
        config = [('Test type', mode_label), ('Method', method), ('Data structure', structure),
                  ('Difference', difference), ('Null hypothesis', 'Population difference = 0'),
                  ('Alternative hypothesis', alt_label), ('Confidence level', f'{confidence:.1%}'),
                  ('Significance level', f'{1-confidence:.3g}')]
        if mode == 'one':
            config.append(('Hypothesized mean', str(null_mean)))
        if mode == 'independent':
            config.append(('Equal variance assumption', 'Yes' if equal_var else 'No'))
        presentation.table(pd.DataFrame(config, columns=['Test detail', 'Value']))
        presentation.table(prepared['accounting'])
        if structure == 'Separate-column format':
            presentation.caption('Missing/nonfinite values are excluded separately in each sample; row positions do not establish pairing.')
    with tabs[1]:
        presentation.dataframe(format_table(prepared['descriptive'], DESCRIPTIVE), result_types=DESCRIPTIVE, hide_index=True, use_container_width=True)
        if mode == 'paired':
            presentation.caption('Original-variable summaries use the same complete pairs as the difference-based test.')
        st.download_button('Download descriptive statistics (CSV)', prepared['descriptive'].to_csv(index=False).encode('utf-8-sig'),
                           file_name='ttest_descriptive.csv', mime='text/csv', key='gs_tt_descriptive_download')
    with tabs[2]:
        presentation.write('Inference about the mean requires an appropriate sampling design. t-tests are often robust to moderate non-normality with adequate sample sizes, but strong skewness, extreme values, and very small samples can impair reliability. Graphical diagnostics do not certify assumptions.')
        if mode == 'paired':
            presentation.caption('All diagnostics below use Variable 1 - Variable 2 for complete pairs, not the original variables separately.')
        presentation.dataframe(format_table(prepared['diagnostics'], {'N':'count', 'IQR outlier flags':'count'}), hide_index=True, use_container_width=True)
        for note in prepared['warnings']:
            presentation.warning(note)
        if not prepared['outliers'].empty:
            presentation.markdown('**Potential outliers (retained)**')
            presentation.dataframe(prepared['outliers'], hide_index=True, use_container_width=True)
        samples = prepared['diagnostic_samples']
        index = choose('Diagnostic sample', list(range(len(samples))), 'gs_tt_diagnostic', format_func=lambda i: samples[i][0])
        label, values = samples[index]
        presentation.markdown(f'**{label}**')
        if len(values):
            presentation.plotly_chart(make_histogram_boxplot(values), use_container_width=True, key='gs_tt_histogram')
        if len(values) >= 3 and values.nunique() > 1:
            (x, y), (slope, intercept, _) = stats.probplot(values, dist='norm')
            fig = go.Figure(go.Scatter(x=x, y=y, mode='markers', marker_color=CHART_COLOR, name=label))
            fig.add_scatter(x=x, y=intercept+slope*x, mode='lines', name='Normal reference')
            fig.update_layout(**_base_layout(320), xaxis_title='Theoretical normal quantile', yaxis_title=str(label))
            presentation.plotly_chart(_style_axes(fig), use_container_width=True, key='gs_tt_qq')
        else:
            presentation.info('A useful Q-Q plot needs at least three observations and nonconstant values.')
    with tabs[3]:
        if error:
            presentation.warning(error)
        else:
            presentation.markdown(f'**{result["method"]}**')
            presentation.table(result_table(result))
            presentation.caption('Effect-size formula: ' + result['effect_formula'])
            if mode == 'independent' and not equal_var:
                presentation.caption("For Welch, Cohen's d uses a pooled SD only as a descriptive standardizer; the test, standard error and confidence interval do not assume equal variances. Under unequal variances, interpret this standardizer with care.")
            st.download_button('Download t-test results (CSV)', export_result(result), file_name='ttest_results.csv', mime='text/csv', key='gs_tt_result_download')
    with tabs[4]:
        if result:
            presentation.write(interpretation(result))
            presentation.caption('No causal conclusion follows from the test alone. Repeated exploratory comparisons increase false-positive risk. No automatic multiplicity adjustment is applied.')
        else:
            presentation.info('Interpretation is available when the selected data support the test. Review the validation message in t-Test Results.')
    if result is not None:
        offer_result('t-Test', mode_label+' of '+str(first),
                     {'first': first, 'second': second, 'group': group, 'levels': levels},
                     {'method': mode_label, 'confidence': confidence, 'alternative': alt_label,
                      'equal_variances': equal_var, 'null_mean': null_mean}, result,
                     prepared.get('diagnostics'), prepared.get('warnings', []) +
                     ['Independence and the chosen study design must be justified.'], key='ttest')
