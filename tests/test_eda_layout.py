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
    def test_histogram_bins_include_both_extremes(self):
        from utils.visual_analysis import histogram_bins
        for values in ([1.,2.,4.], [2.,2.,2.], [-4.,-2.,-1.]):
            bins = histogram_bins(values, 30)
            self.assertLess(bins['start'], min(values))
            self.assertGreater(bins['end'], max(values))
            self.assertAlmostEqual((bins['end']-bins['start'])/bins['size'], 30)

    def test_histogram_grid_wraps_two_panels_per_row(self):
        for count in (2,3,5):
            with self.subTest(variables=count):
                app = AppTest.from_string(f'''
import pandas as pd
from utils.visual_analysis import render_visual_analysis
frame = pd.DataFrame({{f'v{{i}}':[1.,2.,3.] for i in range({count})}})
render_visual_analysis(frame, {{'numerical':list(frame.columns), 'categorical':[], 'datetime':[]}})
''').run()
                app.button(key='va_select_numerical_all').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('plotly_chart')), 1)
                spec = json.loads(app.get('plotly_chart')[0].proto.spec)
                self.assertEqual(len(spec['data']), count)
                self.assertEqual(len({t['marker']['color'] for t in spec['data']}), 1)
                self.assertEqual(len(app.slider), 0)
                app.button(key='va_graph_button_Box plot').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('plotly_chart')), 1)
                boxes = json.loads(app.get('plotly_chart')[0].proto.spec)
                self.assertEqual(len(boxes['data']), count)
                self.assertTrue(all(t['type'] == 'box' for t in boxes['data']))
                self.assertEqual(len({t['marker']['color'] for t in boxes['data']}), 1)
                self.assertEqual([a['text'] for a in boxes['layout']['annotations']], [f'v{i}' for i in range(count)])
                self.assertEqual(boxes['layout']['yaxis']['domain'], boxes['layout']['yaxis2']['domain'])
                if count > 2:
                    self.assertLess(boxes['layout']['yaxis3']['domain'][1], boxes['layout']['yaxis']['domain'][0])
                self.assertEqual(len(spec['layout']['sliders']), count)
                for index, slider in enumerate(spec['layout']['sliders']):
                    suffix = str(index+1) if index else ''
                    self.assertGreater(slider['y'], spec['layout']['yaxis'+suffix]['domain'][1])
                    domain = spec['layout']['xaxis'+suffix]['domain']
                    self.assertLess(slider['len'], domain[1]-domain[0])
                self.assertEqual([a['text'] for a in spec['layout']['annotations']], [f'v{i}' for i in range(count)])
                self.assertEqual(spec['layout']['yaxis']['domain'], spec['layout']['yaxis2']['domain'])
                if count > 2:
                    self.assertLess(spec['layout']['yaxis3']['domain'][1], spec['layout']['yaxis']['domain'][0])
                    gap = spec['layout']['yaxis']['domain'][0]-spec['layout']['yaxis3']['domain'][1]
                    plot_height = spec['layout']['height']-spec['layout']['margin']['t']-spec['layout']['margin']['b']
                    self.assertGreaterEqual(gap*plot_height, 149)
                self.assertGreaterEqual(spec['layout']['height'], 330*((count+1)//2))
                app.button(key='va_graph_button_Dot plot').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('plotly_chart')), 1)
                dots = json.loads(app.get('plotly_chart')[0].proto.spec)
                self.assertEqual(len(dots['data']), count)
                self.assertEqual(len({t['marker']['color'] for t in dots['data']}), 1)
                self.assertTrue(all(t['type'] == 'scatter' for t in dots['data']))
                self.assertEqual([a['text'] for a in dots['layout']['annotations']], [f'v{i}' for i in range(count)])
                self.assertEqual(dots['layout']['yaxis']['domain'], dots['layout']['yaxis2']['domain'])
                if count > 2:
                    self.assertLess(dots['layout']['yaxis3']['domain'][1], dots['layout']['yaxis']['domain'][0])
                self.assertEqual(len(app.slider), 0)

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
                if chart in ('Histogram', 'Dot plot', 'Box plot'):
                    self.assertEqual(len(plots), 1)
                    spec = json.loads(plots[0].proto.spec)
                    self.assertEqual([t['type'] for t in spec['data']], [trace,trace])
                    self.assertEqual(len({t['marker']['color'] for t in spec['data']}), 1)
                    self.assertEqual([t['xaxis'] for t in spec['data']], ['x','x2'])
                    self.assertLess(spec['layout']['yaxis3']['domain'][1], spec['layout']['yaxis']['domain'][0])
                    self.assertFalse(spec['layout']['xaxis4']['visible'])
                    if chart == 'Histogram':
                        self.assertEqual(len(app.slider), 0)
                        self.assertEqual(len(spec['layout']['sliders']), 2)
                        import plotly.io as pio
                        changed = pio.from_json(plots[0].proto.spec)
                        step = spec['layout']['sliders'][0]['steps'][5]
                        self.assertEqual(step['method'], 'restyle')
                        self.assertEqual(step['args'][1], [0])
                        changed.data[0].update(step['args'][0])
                        self.assertEqual([t.nbinsx for t in changed.data], [10,30])
                    else:
                        self.assertEqual(len(app.slider), 0)
                else:
                    self.assertEqual(len(plots), 2)
                    self.assertEqual([json.loads(p.proto.spec)['data'][0]['type'] for p in plots], [trace,trace])
                self.assertTrue(any('No finite numerical values' in i.value for i in app.info))
                self.assertFalse(any(c.key.startswith('va_univariate_column_') for c in app.checkbox))

    def test_histogram_by_category_uses_shared_bins_colors_and_one_chart(self):
        app = AppTest.from_string('''
import pandas as pd
from utils.visual_analysis import render_visual_analysis
frame = pd.DataFrame({'value':[1.,2.,3.,4.,5.,6.,None], 'group':['A','A','B','B','C','C','A']})
render_visual_analysis(frame, {'numerical':['value'], 'categorical':['group'], 'datetime':[]})
''').run()
        app.checkbox(key='va_select_numerical_column_value').check().run()
        app.checkbox(key='va_select_categorical_column_group').check().run()
        app.button(key='va_graph_button_Histogram').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('plotly_chart')), 1)
        spec = json.loads(app.get('plotly_chart')[0].proto.spec)
        self.assertEqual([t['name'] for t in spec['data']], ['A','B','C'])
        self.assertEqual(len({t['marker']['color'] for t in spec['data']}), 3)
        self.assertEqual(len({str(t['xbins']) for t in spec['data']}), 1)
        self.assertEqual(spec['layout']['barmode'], 'overlay')
        self.assertEqual(len(app.slider), 1)

    def test_categorical_dot_plot_keeps_duplicate_values_visible(self):
        import pandas as pd
        from utils.visual_analysis import dot_points
        dots, sampled = dot_points(pd.DataFrame({'v':[1.,1.,1.,2.,None], 'g':['A','B','C',None,'A']}), 'v','g')
        self.assertFalse(sampled)
        self.assertEqual(dots['Stack'].tolist(), [1,2,3])
        self.assertEqual(dots['Category'].tolist(), ['A','B','C'])
        app = AppTest.from_string('''
import pandas as pd
from utils.visual_analysis import render_visual_analysis
frame = pd.DataFrame({'v':[1.,1.,1.,2.,None], 'g':['A','B','C',None,'A']})
render_visual_analysis(frame, {'numerical':['v'], 'categorical':['g'], 'datetime':[]})
''').run()
        app.checkbox(key='va_select_numerical_column_v').check().run()
        app.checkbox(key='va_select_categorical_column_g').check().run()
        app.button(key='va_graph_button_Dot plot').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('plotly_chart')), 1)
        spec = json.loads(app.get('plotly_chart')[0].proto.spec)
        self.assertEqual([t['name'] for t in spec['data']], ['A','B','C'])
        self.assertEqual(len({t['marker']['color'] for t in spec['data']}), 3)
        self.assertEqual(len(app.slider), 0)

    def test_boxplot_panels_preserve_categorical_grouping(self):
        app = AppTest.from_string('''
import pandas as pd
from utils.visual_analysis import render_visual_analysis
frame = pd.DataFrame({'a':[1.,2.,3.,4.], 'b':[4.,3.,2.,1.], 'g':['A','A','B',None]})
render_visual_analysis(frame, {'numerical':['a','b'], 'categorical':['g'], 'datetime':[]})
''').run()
        app.button(key='va_select_numerical_all').click().run()
        app.checkbox(key='va_select_categorical_column_g').check().run()
        app.button(key='va_graph_button_Box plot').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('plotly_chart')), 1)
        spec = json.loads(app.get('plotly_chart')[0].proto.spec)
        self.assertEqual([t['x'] for t in spec['data']], [['A','A','B'],['A','A','B']])
        self.assertEqual(spec['layout']['xaxis']['title']['text'], 'g')
        self.assertEqual(spec['layout']['xaxis2']['title']['text'], 'g')
        self.assertEqual(len({t['marker']['color'] for t in spec['data']}), 1)

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
        self.assertNotIn('Line chart', available_charts(['a','b','c'], [], []))
        self.assertNotIn('Line chart', available_charts(['a','b','c'], [], ['date']))
        self.assertIn('Line chart', available_charts(['a'], [], ['date']))
        self.assertIn('Line chart', available_charts(['a','b'], [], []))
        self.assertIn('Line chart', available_charts(['a'], ['stage'], [], ['stage']))
        self.assertNotIn('Line chart', available_charts(['a'], ['group'], []))
        for cats in ([], ['g']):
            choices = available_charts(['a','b'], cats, [])
            for chart in ('Histogram','Dot plot','Box plot'):
                self.assertIn(chart, choices)
        self.assertIn('Pie chart',available_charts([],['g'],[]))
        self.assertIn('Heatmap',available_charts(['a','b'],[],[]))
        self.assertIn('Bubble plot',available_charts(['a','b','size'],[],[]))
        self.assertNotIn('Bubble plot',available_charts(['a','b'],[],[]))

    def test_numeric_line_requires_confirmation_but_temporal_does_not(self):
        app = AppTest.from_string(FIXTURE.replace('render_group_statistics(frame,types)', '')).run()
        app.checkbox(key='va_select_numerical_column_a').check().run()
        app.checkbox(key='va_select_numerical_column_b').check().run()
        app.button(key='va_graph_button_Line chart').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('plotly_chart')), 0)
        app.checkbox(key='va_line_order_confirm_a').check().run()
        self.assertEqual(len(app.get('plotly_chart')), 1)
        app.checkbox(key='va_select_numerical_column_size').check().run()
        self.assertNotIn('Line chart', [b.label for b in app.button])
        app.button(key='va_select_numerical_clear').click().run()
        app.checkbox(key='va_select_numerical_column_a').check().run()
        app.checkbox(key='va_select_datetime_column_date').check().run()
        app.button(key='va_graph_button_Line chart').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('plotly_chart')), 1)
        self.assertFalse(any(c.key.startswith('va_line_order_confirm_') for c in app.checkbox))
        spec = json.loads(app.get('plotly_chart')[0].proto.spec)
        self.assertEqual(spec['layout']['xaxis']['title']['text'], 'date')

    def test_ordinal_line_uses_defined_order_instead_of_alphabetical_order(self):
        app = AppTest.from_string('''
import pandas as pd
from utils.visual_analysis import render_visual_analysis
frame = pd.DataFrame({'value':[2.,1.,3.,4.], 'stage':pd.Categorical(['Middle','Start','End','End'], categories=['Start','Middle','End'], ordered=True)})
render_visual_analysis(frame, {'numerical':['value'], 'categorical':['stage'], 'datetime':[]})
''').run()
        app.checkbox(key='va_select_numerical_column_value').check().run()
        app.checkbox(key='va_select_categorical_column_stage').check().run()
        app.button(key='va_graph_button_Line chart').click().run()
        self.assertFalse(app.exception)
        spec = json.loads(app.get('plotly_chart')[0].proto.spec)
        self.assertEqual(spec['data'][0]['x'], ['Start','Middle','End'])
        self.assertFalse(any(c.key.startswith('va_line_order_confirm_') for c in app.checkbox))

if __name__=='__main__': unittest.main()
