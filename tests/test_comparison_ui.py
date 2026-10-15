import unittest
from streamlit.testing.v1 import AppTest

class ComparisonUITests(unittest.TestCase):
    def test_no_default_and_switching_ttest_does_not_show_anova(self):
        app = AppTest.from_string("""
import pandas as pd
import streamlit as st
from utils.comparison_ui import render_comparison
st.session_state['file_id'] = 'method-selection'
df = pd.DataFrame({'Height':[1.,3.,5.,2.,4.,8.], 'Fertilizer':['A']*3+['B']*3})
render_comparison(df, {'numerical':['Height'], 'categorical':['Fertilizer'], 'datetime':[]})
""").run(timeout=40)
        app.button(key='gs_cmp_all_numerical').click().run(timeout=40)
        self.assertIsNone(app.selectbox(key='gs_cmp_analysis').value)
        self.assertFalse(app.tabs)
        self.assertFalse(app.dataframe)
        app.button(key='gs_cmp_all_categorical').click().run(timeout=40)
        self.assertIsNone(app.selectbox(key='gs_cmp_analysis').value)
        self.assertFalse(app.tabs)
        self.assertFalse(app.dataframe)
        app.selectbox(key='gs_cmp_analysis').select('One-way ANOVA').run(timeout=40)
        self.assertEqual([tab.label for tab in app.tabs], ['Overview', 'Group Comparisons', 'Diagnostics', 'ANOVA Findings'])
        app.selectbox(key='gs_cmp_analysis').select('t-Test').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn('t-Test Results', [tab.label for tab in app.tabs])
        self.assertNotIn('Overview', [tab.label for tab in app.tabs])
        self.assertFalse(any('One-way ANOVA:' in caption.value for caption in app.caption))
        self.assertFalse(any('Grouping variable:' in text.value for text in app.markdown))
        self.assertTrue(any('Welch' in text.value for text in app.markdown))
        app.selectbox(key='gs_cmp_analysis').select('One-way ANOVA').run(timeout=40)
        self.assertNotIn('t-Test Results', [tab.label for tab in app.tabs])
        self.assertIn('Overview', [tab.label for tab in app.tabs])
        app.button(key='gs_cmp_clear_numerical').click().run(timeout=40)
        self.assertIsNone(app.selectbox(key='gs_cmp_analysis').value)
        self.assertFalse(app.tabs)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.exception)

    def test_selection_drives_eligibility(self):
        import pandas as pd
        from utils.comparison_ui import available_comparisons
        df = pd.DataFrame({'y':[1.,2,3,4], 'z':[2.,4,6,8], 'w':[3.,5,7,9],
                           'a':['A','A','B','B'], 'b':['A','B','C','C'], 'c':['X','Y','X','Y'],
                           'date':pd.date_range('2024-01-01', periods=4)})
        cases = [
            (['y','z','w'], ['a','b','c'], [], []),
            (['y'], ['a','b','c'], [], ['Factorial ANOVA','Segment Comparison']),
            (['y'], ['a'], [], ['One-way ANOVA','Segment Comparison','t-Test']),
            (['y'], ['b'], [], ['One-way ANOVA','Segment Comparison']),
            (['y','z'], [], [], ['t-Test']),
            (['y'], [], [], ['t-Test']),
            ([], ['a','b'], [], ['Segment Comparison']),
            (['y'], ['a'], ['date'], ['Period-over-Period Comparison']),
            (['y','z'], ['a'], [], []),
        ]
        for numerical, categorical, temporal, expected in cases:
            with self.subTest(numerical=numerical, categorical=categorical, temporal=temporal):
                self.assertEqual(available_comparisons(df, dict(numerical=numerical, categorical=categorical, datetime=temporal)), expected)

    def test_excess_selection_guidance_and_recovery(self):
        app = AppTest.from_string('''
from tests.test_factorial_anova import fixture
from utils.comparison_ui import render_comparison
df = fixture()
df['Z'] = df.Y + 1
df['W'] = df.Y + 2
render_comparison(df, {'numerical':['Y','Z','W'], 'categorical':['A','B','D'], 'datetime':[]})
''').run(timeout=30)
        app.button(key='gs_cmp_all_numerical').click().run()
        app.button(key='gs_cmp_all_categorical').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(any(s.key == 'gs_cmp_analysis' for s in app.selectbox))
        self.assertTrue(any('No comparison matches' in m.value for m in app.info))
        app.checkbox(key='gs_cmp_variable_Z').uncheck().run()
        app.checkbox(key='gs_cmp_variable_W').uncheck().run()
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options, ['Factorial ANOVA','Segment Comparison'])
        self.assertIsNone(app.selectbox(key='gs_cmp_analysis').value)
        app.selectbox(key='gs_cmp_analysis').select('Factorial ANOVA').run(timeout=40)
        self.assertEqual(app.multiselect(key='fac_x').value, ['A','B','D'])
        self.assertFalse(app.exception)

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
        for kind in ['numerical', 'categorical']:
            app.button(key='gs_cmp_all_' + kind).click().run(timeout=40)
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options,['One-way ANOVA','Segment Comparison','t-Test'])
        app.selectbox(key='gs_cmp_analysis').select('Segment Comparison').run(timeout=40)
        app.multiselect(key='gs_cmp_segment_dimensions').set_value(['g']).run(timeout=40)
        app.selectbox(key='gs_cmp_segment_measure_optional').select('v').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertFalse(app.checkbox(key='gs_dimensions_column_g').value)
        self.assertFalse(app.checkbox(key='gs_measure_column_v').value)
        self.assertTrue(any('Difference %' in table.value.columns for table in app.dataframe))
        app.button(key='gs_cmp_all_datetime').click().run(timeout=40)
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options, ['Period-over-Period Comparison'])
        self.assertFalse(app.exception)
        self.assertIsNone(app.selectbox(key='gs_cmp_analysis').value)
        app.selectbox(key='gs_cmp_analysis').select('Period-over-Period Comparison').run(timeout=40)
        self.assertTrue(any('Previous' in table.value.columns for table in app.dataframe))
        app.button(key='gs_cmp_clear_datetime').click().run(timeout=40)
        app.selectbox(key='gs_cmp_analysis').select('One-way ANOVA').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(any('Welch One-way ANOVA' in m.value for m in app.markdown))
        self.assertEqual([t.label for t in app.tabs][-4:], ['Overview', 'Group Comparisons', 'Diagnostics', 'ANOVA Findings'])
        app.button(key='gs_cmp_clear_numerical').click().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='gs_cmp_analysis').options, ['Segment Comparison'])
        self.assertIsNone(app.selectbox(key='gs_cmp_analysis').value)
        app.selectbox(key='gs_cmp_analysis').select('Segment Comparison').run(timeout=40)
        self.assertEqual(app.selectbox(key='gs_cmp_segment_measure_optional').options, ['None (categorical data / row counts)'])
        app.button(key='gs_cmp_all_datetime').click().run(timeout=40)
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
