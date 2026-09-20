import unittest
from streamlit.testing.v1 import AppTest

class ComparisonUITests(unittest.TestCase):
    def test_three_analyses_and_independent_controls(self):
        app=AppTest.from_string('''
import pandas as pd
import streamlit as st
from utils.group_statistics_ui import render_group_statistics
from utils.comparison_ui import render_comparison
from utils.relationship_ui import render_relationship_analysis
st.session_state['file_id']='comparison_fixture'
df=pd.DataFrame({'v':[1.,3.,5.,2.,4.,8.],'g':['A','A','A','B','B','B'],'date':pd.to_datetime(['2024-01-01','2024-02-01','2024-03-01']*2)})
types={'numerical':['v'],'categorical':['g'],'datetime':['date']}
a,b,c=st.tabs(['Group Statistics','Relationship','Comparison'])
with a: render_group_statistics(df,types)
with b: render_relationship_analysis(df,['v'],['g'],['date'])
with c: render_comparison(df,types)
''').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='gs_analysis').options,['Group Summary','Running & Cumulative Statistics','Rolling Averages','Contribution to Total'])
        self.assertFalse(any('ANOVA' in option for option in app.selectbox(key='rel_method').options))
        self.assertFalse(any(x.key == 'gs_cmp_analysis' for x in app.selectbox))
        for kind in ['numerical', 'categorical', 'datetime']:
            app.button(key='gs_cmp_all_' + kind).click().run(timeout=40)
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options,['Group Comparison (ANOVA)','Segment Comparison','Period-over-Period Comparison','t-Test'])
        app.selectbox(key='gs_cmp_analysis').select('Segment Comparison').run(timeout=40)
        app.multiselect(key='gs_cmp_segment_dimensions').set_value(['g']).run(timeout=40)
        app.selectbox(key='gs_cmp_segment_measure_optional').select('v').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertFalse(app.checkbox(key='gs_dimensions_column_g').value)
        self.assertFalse(app.checkbox(key='gs_measure_column_v').value)
        self.assertTrue(any('Difference %' in table.value.columns for table in app.dataframe))
        app.selectbox(key='gs_cmp_analysis').select('Period-over-Period Comparison').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(any('Previous' in table.value.columns for table in app.dataframe))
        app.selectbox(key='gs_cmp_analysis').select('Group Comparison (ANOVA)').run(timeout=40)
        app.checkbox(key='rel_anova_independent').check().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(any('Welch one-way ANOVA' in m.value for m in app.markdown))
        self.assertEqual([t.label for t in app.tabs][-5:], ['Model Overview', 'Group Analysis', 'Diagnostics', 'Visual Analysis', 'ANOVA Findings'])
        app.button(key='gs_cmp_clear_numerical').click().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options, ['Segment Comparison', 'Period-over-Period Comparison'])
        self.assertEqual(app.selectbox(key='gs_cmp_segment_measure_optional').options, ['None (categorical data / row counts)'])
        app.button(key='gs_cmp_clear_categorical').click().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options, ['Period-over-Period Comparison'])

    def test_missing_required_types_and_stale_group_selection(self):
        app=AppTest.from_string('''
import streamlit as st
import pandas as pd
from utils.group_statistics_ui import render_group_statistics
from utils.comparison_ui import render_comparison
if 'fixture_init' not in st.session_state:
    st.session_state['gs_analysis']='Segment Comparison'
    st.session_state['fixture_init']=True
df=pd.DataFrame({'g':['A','B','A']})
types={'numerical':[],'categorical':['g'],'datetime':[]}
render_group_statistics(df,types)
render_comparison(df,types)
''').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='gs_analysis').value,'Group Summary')
        self.assertTrue(any('Select numerical and categorical' in message.value for message in app.info))
        app.checkbox(key='gs_cmp_variable_g').check().run(timeout=30)
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options, ['Segment Comparison'])
        app.button(key='gs_cmp_clear_categorical').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(any(x.key == 'gs_cmp_analysis' for x in app.selectbox))

if __name__=='__main__': unittest.main()
