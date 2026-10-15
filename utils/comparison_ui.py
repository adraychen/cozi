"""Group and time comparisons, using existing statistical implementations."""
import streamlit as st
from utils.anova_ui import render_anova
from utils.factorial_anova_ui import render_factorial_anova
from utils.ttest_analysis import numerical_candidates
from utils.ttest_ui import render_ttest
from utils.group_statistics_ui import render_group_statistics
from utils.temporal import temporal_columns, period_frequencies
from utils.period_comparison import interpret_time

ANALYSES = ['One-way ANOVA', 'Segment Comparison', 'Period-over-Period Comparison', 't-Test', 'Factorial ANOVA']


def available_comparisons(df, selected, metadata=None):
    """Match the full selection to supported analysis roles, not a subset."""
    n, c, t = (len(selected[k]) for k in ('numerical', 'categorical', 'datetime'))
    quantitative = numerical_candidates(df, selected['numerical'], metadata)
    methods = []
    if n == 1 and len(quantitative) == 1 and c == 1 and t == 0:
        methods.append('One-way ANOVA')
    if n == 1 and len(quantitative) == 1 and c >= 2 and t == 0:
        methods.append('Factorial ANOVA')
    if n <= 1 and c >= 1 and t == 0:
        methods.append('Segment Comparison')
    if n <= 1 and t == 1:
        date = selected['datetime'][0]
        try:
            interpret_time(df[date], (metadata or {}).get(date, {}).get('variable_subtype'))
        except (ValueError, TypeError):
            pass
        else:
            methods.append('Period-over-Period Comparison')
    if t == 0 and len(quantitative) == n:
        if (n in (1, 2) and c == 0) or (n == 1 and c == 1 and df[selected['categorical'][0]].nunique(dropna=True) == 2):
            methods.append('t-Test')
    return methods


def render_comparison(df, col_types):
    st.markdown('#### Comparison')
    st.caption('Compare groups or periods to identify differences in your data.')
    selected = {}
    def set_group(columns, value):
        for column in columns:
            st.session_state[f'gs_cmp_variable_{column}'] = value

    for kind, label in [('numerical', 'Numerical'), ('categorical', 'Categorical'), ('datetime', 'Temporal')]:
        columns = temporal_columns(col_types) if kind == 'datetime' else col_types[kind]
        selected[kind] = []
        if not columns:
            continue
        st.markdown(f'**{label}**')
        left, right = st.columns(2)
        left.button('Select all', key=f'gs_cmp_all_{kind}', on_click=set_group, args=(columns, True))
        right.button('Clear all', key=f'gs_cmp_clear_{kind}', on_click=set_group, args=(columns, False))
        for start in range(0, len(columns), 4):
            cells = st.columns(4)
            for cell, column in zip(cells, columns[start:start + 4]):
                if cell.checkbox(column, key=f'gs_cmp_variable_{column}'):
                    selected[kind].append(column)
    from utils.anova_effects import prune_effect_state
    for prefix in ('fac', 'rel_anova'):
        prune_effect_state(st.session_state, selected['categorical'], prefix)
    methods = available_comparisons(df, selected, st.session_state.get('variable_metadata', {}))
    st.caption(f"Selected: {len(selected['numerical'])} numerical · {len(selected['categorical'])} categorical · {len(selected['datetime'])} temporal. Available analyses match this selection.")
    with st.expander('Which variables should I select?'):
        st.markdown(
            '- **One-way ANOVA:** one numerical response and one categorical factor.\n'
            '- **Factorial ANOVA:** one quantitative numerical response and two or more categorical factors.\n'
            '- **t-Test:** one quantitative numerical variable for a one-sample test; one numerical variable and one categorical variable with exactly two observed groups for an independent-samples test; or two numerical variables for separate samples or paired measurements. Choose the study design explicitly.\n'
            '- **Segment Comparison:** one or more categorical variables, with zero or one numerical measure. Without a measure, compare row counts.\n'
            '- **Period-over-Period Comparison:** exactly one temporal variable, zero or one numerical measure, and optional categorical segments.\n\n'
            'Select temporal variables only for period comparisons. Eligibility does not establish that statistical assumptions are met; each analysis performs further validation. Identifier-like variables are excluded from quantitative t-test and factorial response choices unless explicitly classified as Numerical.'
        )
    if not methods:
        st.session_state.pop('gs_cmp_analysis', None)
        if len(selected['datetime']) == 1 and len(selected['numerical']) <= 1:
            date = selected['datetime'][0]
            try:
                interpret_time(df[date], st.session_state.get('variable_metadata', {}).get(date, {}).get('variable_subtype'))
            except (ValueError, TypeError) as exc:
                st.info(str(exc))
        st.info('No comparison matches the full selection. Select numerical and categorical variables according to the guide above. For ANOVA, keep exactly one numerical response; for a two-column t-test, keep two numerical variables and no categorical or temporal variables.')
        return
    if st.session_state.get('gs_cmp_analysis') == 'Group Comparison (ANOVA)':
        st.session_state['gs_cmp_analysis'] = 'One-way ANOVA'
    if st.session_state.get('gs_cmp_analysis') not in methods:
        st.session_state.pop('gs_cmp_analysis', None)
    st.markdown('##### Comparison analysis')
    analysis = st.selectbox('Analysis method', methods, key='gs_cmp_analysis')
    col_types = selected
    context = (st.session_state.get('file_id'), tuple(selected['numerical']),
               tuple(selected['categorical']), tuple(selected['datetime']))
    if st.session_state.get('gs_cmp_selection_context') != context:
        st.session_state['gs_cmp_selection_context'] = context
        st.session_state['fac_x'] = selected['categorical'][:]
        for prefix in ('gs_cmp_segment', 'gs_cmp_period'):
            st.session_state[prefix + '_dimensions'] = selected['categorical'][:]
            st.session_state[prefix + '_measure_optional'] = selected['numerical'][0] if len(selected['numerical']) == 1 else None
    if analysis == 't-Test':
        render_ttest(df, col_types['numerical'], col_types['categorical'], guided=True)
    elif analysis == 'Factorial ANOVA':
        render_factorial_anova(df, col_types['numerical'], col_types['categorical'])
    elif analysis=='One-way ANOVA':
        st.caption('One-way ANOVA: one numerical response variable and one categorical factor.')
        if not col_types['categorical'] or not col_types['numerical']:
            st.info('One-way ANOVA requires a categorical grouping variable and a numerical outcome. Set their Variable Types in Column Overview.')
            return
        category = col_types['categorical'][0]
        outcome = col_types['numerical'][0]
        st.write('**Grouping variable:**', str(category))
        st.write('**Numerical outcome:**', str(outcome))
        try:
            render_anova(df,category,outcome)
        except (ValueError,TypeError) as exc:
            st.warning('Unable to calculate One-way ANOVA: '+str(exc))
    else:
        render_group_statistics(df,col_types,analysis=analysis,key_prefix='gs_cmp_segment' if analysis=='Segment Comparison' else 'gs_cmp_period',show_heading=False)
