"""Group and time comparisons, using existing statistical implementations."""
import streamlit as st
from utils.anova_ui import render_anova
from utils.factorial_anova_ui import render_factorial_anova
from utils.ttest_analysis import numerical_candidates
from utils.ttest_ui import render_ttest
from utils.group_statistics_ui import render_group_statistics

ANALYSES = ['One-way ANOVA', 'Segment Comparison', 'Period-over-Period Comparison', 't-Test', 'Factorial ANOVA']


def render_comparison(df, col_types):
    st.markdown('#### Comparison')
    st.caption('Compare groups or periods to identify differences in your data.')
    selected = {}
    def set_group(columns, value):
        for column in columns:
            st.session_state[f'gs_cmp_variable_{column}'] = value

    for kind, label in [('numerical', 'Numerical'), ('categorical', 'Categorical'), ('datetime', 'Temporal')]:
        columns = col_types[kind]
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
    methods = []
    if selected['numerical'] and selected['categorical']:
        methods.append(ANALYSES[0])
        methods.append(ANALYSES[4])
    if selected['categorical']:
        methods.append(ANALYSES[1])
    if selected['datetime']:
        methods.append(ANALYSES[2])
    if numerical_candidates(df, selected['numerical'], st.session_state.get('variable_metadata', {})):
        methods.append(ANALYSES[3])
    if not methods:
        st.info('Select numerical and categorical variables for ANOVA, a quantitative numerical variable for t-Test, categorical variables for segment comparison, or a Temporal variable for period comparison. Comparisons can use row counts without a numerical measure.')
        return
    if st.session_state.get('gs_cmp_analysis') == 'Group Comparison (ANOVA)':
        st.session_state['gs_cmp_analysis'] = 'One-way ANOVA'
    if st.session_state.get('gs_cmp_analysis') not in methods:
        st.session_state.pop('gs_cmp_analysis', None)
    st.markdown('##### Comparison analysis')
    analysis = st.selectbox('Analysis method', methods, key='gs_cmp_analysis')
    col_types = selected
    if analysis == 't-Test':
        render_ttest(df, col_types['numerical'], col_types['categorical'])
    elif analysis == 'Factorial ANOVA':
        render_factorial_anova(df, col_types['numerical'], col_types['categorical'])
    elif analysis=='One-way ANOVA':
        st.caption('One-way ANOVA: one numerical response variable and one categorical factor.')
        if not col_types['categorical'] or not col_types['numerical']:
            st.info('One-way ANOVA requires a categorical grouping variable and a numerical outcome. Set their Variable Types in Column Overview.')
            return
        category=st.selectbox('Grouping variable',col_types['categorical'],key='gs_cmp_group')
        outcome=st.selectbox('Numerical outcome',col_types['numerical'],key='gs_cmp_outcome')
        try:
            render_anova(df,category,outcome)
        except (ValueError,TypeError) as exc:
            st.warning('Unable to calculate One-way ANOVA: '+str(exc))
    else:
        render_group_statistics(df,col_types,analysis=analysis,key_prefix='gs_cmp_segment' if analysis=='Segment Comparison' else 'gs_cmp_period',show_heading=False)
