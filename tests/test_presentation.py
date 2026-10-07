import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import presentation


class PresentationTests(unittest.TestCase):
    def test_saved_charts_with_plotly_typed_arrays_render_and_keep_dimensions(self):
        import plotly.io as pio
        figures = [
            go.Figure(go.Scatter(x=np.arange(20), y=np.arange(20))),
            go.Figure(go.Bar(x=np.arange(20), y=np.arange(20))),
            go.Figure(go.Bar(x=np.arange(20), y=np.arange(20), orientation='h')),
            go.Figure(go.Heatmap(z=np.eye(20))),
        ]
        for figure in figures:
            with self.subTest(trace=figure.data[0].type):
                restored = pio.from_json(figure.to_json())
                self.assertEqual(presentation.chart_size(restored), presentation.chart_size(figure))
                original = restored.to_json()
                with patch.object(presentation.st, 'plotly_chart') as renderer:
                    presentation.plotly_chart(restored, key='report_chart')
                    self.assertEqual(renderer.call_args.kwargs['key'], 'report_chart')
                self.assertEqual(restored.to_json(), original)

    def test_tables_fit_content_without_fixed_width_caps_or_column_squeezing(self):
        for count in (2, 16, 200):
            frame = pd.DataFrame({f'Column {i}': ['Long content ' * 20] for i in range(count)})
            config = {'Column 0': {'width': 80, 'help': 'Keep this help'}, 'Column 1': None}
            with patch.object(presentation.st, 'dataframe') as renderer:
                presentation.dataframe(frame, width=1200, column_config=config)
                self.assertEqual(renderer.call_args.kwargs['width'], 'content')
                actual = renderer.call_args.kwargs['column_config']
                self.assertIsNone(actual['Column 0']['width'])
                self.assertEqual(actual['Column 0']['help'], 'Keep this help')
                self.assertIsNone(actual['Column 1'])
                self.assertEqual(config['Column 0']['width'], 80)
        with patch.object(presentation.st, 'dataframe') as renderer:
            presentation.table(frame)
            self.assertEqual(renderer.call_args.kwargs['width'], 'content')
            self.assertTrue(renderer.call_args.kwargs['hide_index'])

    def test_charts_size_by_structure_not_scatter_sample_count(self):
        simple = px.bar(x=['A','B'], y=[1,2])
        busy = px.bar(x=[f'Category {i}' for i in range(20)], y=list(range(20)))
        self.assertLess(presentation.chart_size(simple)[0], presentation.chart_size(busy)[0])
        horizontal = px.bar(y=[f'Long category {i}' for i in range(20)], x=list(range(20)),orientation='h')
        self.assertGreater(presentation.chart_size(horizontal)[1],presentation.chart_size(simple)[1])
        self.assertEqual(presentation.chart_size(px.scatter(x=[1,2],y=[2,3])),
                         presentation.chart_size(px.scatter(x=np.arange(10000),y=np.arange(10000))))
        panels=make_subplots(rows=1,cols=2)
        panels.add_trace(go.Scatter(x=[1,2],y=[3,4]),row=1,col=1)
        panels.add_trace(go.Scatter(x=[1,2],y=[3,4]),row=1,col=2)
        self.assertGreater(presentation.chart_size(panels)[0],presentation.chart_size(simple)[0])
        self.assertGreater(presentation.chart_size(px.imshow(np.eye(20)))[0],presentation.chart_size(px.imshow(np.eye(2)))[0])
        self.assertLess(presentation.chart_size(px.box(y=np.arange(1000)))[1],900)
        self.assertGreaterEqual(presentation.chart_size(px.pie(values=[1,2],names=['A','B']))[0],560)

    def test_rendering_preserves_data_styles_keys_and_figures(self):
        frame=pd.DataFrame({'a':[1,2],'b':[3,4]})
        styled=frame.style.format(precision=2)
        with patch.object(presentation.st,'dataframe') as renderer:
            presentation.dataframe(styled,hide_index=True,key='table',use_container_width=True,height=300)
            self.assertIs(renderer.call_args.args[0],styled)
            self.assertEqual(renderer.call_args.kwargs['key'],'table')
            self.assertEqual(renderer.call_args.kwargs['width'],'content')
            self.assertNotIn('use_container_width',renderer.call_args.kwargs)
            self.assertLess(renderer.call_args.kwargs['height'],300)
        figure=px.bar(frame,x='a',y='b')
        original=figure.to_json()
        with patch.object(presentation.st,'plotly_chart') as renderer:
            presentation.plotly_chart(figure,key='chart',use_container_width=True)
            self.assertEqual(figure.to_json(),original)
            self.assertEqual(renderer.call_args.kwargs['key'],'chart')
            self.assertEqual(renderer.call_args.args[0].data[0].y.tolist(),[3,4])


if __name__=='__main__':
    unittest.main()
