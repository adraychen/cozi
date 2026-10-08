import json
import unittest
from streamlit.testing.v1 import AppTest
from utils.visual_analysis import available_charts

FIXTURE = '''
import pandas as pd
import streamlit as st
from utils.visual_analysis import render_visual_analysis
from utils.group_statistics_ui import render_group_statistics
frame=pd.DataFrame({'a':[1.,2.,4.,8.],'b':[2.,5.,3.,7.],'size':[3.,4.,5.,6.],
                    'g':['A','A','B','B'],'h':['X','Y','X','Y'],'date':pd.date_range('2024-01-01',periods=4)})
types={'numerical':['a','b','size'],'categorical':['g','h'],'datetime':['date']}
render_group_statistics(frame,types)
render_visual_analysis(frame,types)
'''


class EDALayoutTests(unittest.TestCase):
    def test_multiple_numerical_distributions_have_independent_panels(self):
        app = AppTest.from_string('''
import numpy as np
import pandas as pd
from utils.visual_analysis import render_visual_analysis
frame = pd.DataFrame({'a':[1.,2.,None,4.], 'b':[None,5.,6.,np.inf], 'empty':[None,np.inf,None,None]})
render_visual_analysis(frame, {'numerical':['a','b','empty'], 'categorical':[], 'datetime':[]})
''').run()
        app.button(key='va_select_numerical_all').click().run()
        for chart, trace in [('Histogram','histogram'), ('Dot plot','scatter'), ('Box plot','box')]:
            with self.subTest(chart=chart):
                app.button(key='va_graph_button_'+chart).click().run()
                self.assertFalse(app.exception)
                plots = app.get('plotly_chart')
                self.assertEqual(len(plots), 2)
                self.assertEqual([json.loads(p.proto.spec)['data'][0]['type'] for p in plots], [trace,trace])
                self.assertTrue(any('No finite numerical values' in i.value for i in app.info))
                self.assertFalse(any(c.key.startswith('va_univariate_column_') for c in app.checkbox))

    def test_group_exclusivity_and_multiple_dimensions(self):
        app=AppTest.from_string(FIXTURE).run(timeout=45)
        app.checkbox(key='gs_measure_column_a').check().run(timeout=45)
        app.checkbox(key='gs_measure_column_b').check().run(timeout=45)
        self.assertFalse(app.checkbox(key='gs_measure_column_a').value)
        self.assertTrue(app.checkbox(key='gs_measure_column_b').value)
        app.button(key='gs_dimensions_all').click().run(timeout=45)
        self.assertTrue(app.checkbox(key='gs_dimensions_column_g').value)
        self.assertTrue(app.checkbox(key='gs_dimensions_column_h').value)
        app.selectbox(key='gs_analysis').select('Rolling Averages').run(timeout=45)
        self.assertTrue(any('Select one Date or DateTime' in i.value for i in app.info))
        app.checkbox(key='gs_temporal_column_date').check().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('Rolling average' in table.value.columns for table in app.dataframe))
        self.assertFalse(app.checkbox(key='va_select_numerical_column_b').value)

    def test_requested_chart_rendering_and_role_changes(self):
        app=AppTest.from_string(FIXTURE).run(timeout=45)
        app.button(key='va_select_numerical_all').click().run(timeout=45)
        app.button(key='va_select_categorical_all').click().run(timeout=45)
        charts={'Pie chart':'pie','Bar chart':'bar','Stacked-bar chart':'bar',
                'Scatter plot':'scatter','Bubble plot':'scatter','Heatmap':'heatmap','Pair plot':'splom'}
        for chart,trace_type in charts.items():
            with self.subTest(chart=chart):
                app.button(key='va_graph_button_'+chart).click().run(timeout=45)
                self.assertFalse(app.exception)
                plot=app.get('plotly_chart')[-1]
                self.assertEqual(json.loads(plot.proto.spec)['data'][0]['type'],trace_type)
        app.button(key='va_graph_button_Scatter plot').click().run(timeout=45)
        app.checkbox(key='va_scatter_x_column_size').check().run(timeout=45)
        self.assertFalse(app.checkbox(key='va_scatter_x_column_a').value)
        self.assertFalse(app.exception)
        app.button(key='va_select_numerical_clear').click().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertNotIn('Pair plot',[b.label for b in app.button])
        app.button(key='va_select_categorical_clear').click().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('Select a numeric or categorical' in i.value for i in app.info))

    def test_pie_counts_sums_and_negative_validation(self):
        app=AppTest.from_string('''
import pandas as pd
from utils.visual_analysis import render_visual_analysis
render_visual_analysis(pd.DataFrame({'v':[2.,4.,-1.],'g':['A','A','B']}), {'numerical':['v'],'categorical':['g'],'datetime':[]})
''').run(timeout=45)
        app.checkbox(key='va_select_categorical_column_g').check().run(timeout=45)
        app.button(key='va_graph_button_Pie chart').click().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual(sorted(app.dataframe[0].value.Value.tolist()),[1,2])
        app.checkbox(key='va_select_numerical_column_v').check().run(timeout=45)
        app.checkbox(key='va_bar_measure_column_v').check().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('nonnegative category totals' in i.value for i in app.info))
        self.assertEqual(len(app.get('plotly_chart')),0)

    def test_chart_availability(self):
        for cats in ([], ['g']):
            choices = available_charts(['a','b'], cats, [])
            for chart in ('Histogram','Dot plot','Box plot'):
                self.assertIn(chart, choices)
        self.assertIn('Pie chart',available_charts([],['g'],[]))
        self.assertIn('Heatmap',available_charts(['a','b'],[],[]))
        self.assertIn('Bubble plot',available_charts(['a','b','size'],[],[]))
        self.assertNotIn('Bubble plot',available_charts(['a','b'],[],[]))

if __name__=='__main__': unittest.main()
