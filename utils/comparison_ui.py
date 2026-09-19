"""Group and time comparisons, using existing statistical implementations."""
import streamlit as st
from utils.anova_ui import render_anova
from utils.group_statistics_ui import render_group_statistics

ANALYSES = ['Group Comparison (ANOVA)', 'Segment Comparison', 'Period-over-Period Comparison']


def render_comparison(df, col_types):
    st.markdown('#### Comparison')
    st.caption('Compare group means with ANOVA, segments against a reference, or changes across calendar periods.')
    analysis=st.selectbox('Comparison analysis',ANALYSES,key='gs_cmp_analysis')
    if analysis=='Group Comparison (ANOVA)':
        if not col_types['categorical'] or not col_types['numerical']:
            st.info('ANOVA requires a categorical grouping variable and a numerical outcome. Set their Variable Types in Column Overview.')
            return
        category=st.selectbox('Grouping variable',col_types['categorical'],key='gs_cmp_group')
        outcome=st.selectbox('Numerical outcome',col_types['numerical'],key='gs_cmp_outcome')
        try:
            render_anova(df,category,outcome)
        except (ValueError,TypeError) as exc:
            st.warning('Unable to calculate ANOVA: '+str(exc))
    else:
        render_group_statistics(df,col_types,analysis=analysis,key_prefix='gs_cmp_segment' if analysis=='Segment Comparison' else 'gs_cmp_period',show_heading=False)
