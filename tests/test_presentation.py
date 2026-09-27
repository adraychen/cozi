import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import presentation


class PresentationTests(unittest.TestCase):
    def test_tables_account_for_columns_text_styles_and_hidden_columns(self):
        small = pd.DataFrame({'Level': ['A','B'], 'Mean': [1.2,3.4]})
        wide = pd.DataFrame({f'Measure {i}': [1,2] for i in range(6)})
        prose = pd.DataFrame({'Check': ['Independence'], 'Explanation': ['Study design and repeated observations need review. '*4]})
        compact = presentation.table_width(small, hide_index=True)
        self.assertLess(compact, presentation.table_width(wide, hide_index=True))
        self.assertLess(compact, presentation.table_width(prose, hide_index=True))
        self.assertEqual(compact, presentation.table_width(small.style.format(precision=2), hide_index=True))
        self.assertEqual(compact, presentation.table_width(wide, hide_index=True, column_order=['Measure 0','Measure 1']))
        self.assertLess(presentation.table_width(wide,column_config={'Measure 0':None}), presentation.table_width(wide))
        self.assertLessEqual(presentation.table_width(pd.DataFrame(np.ones((10,200)))),1200)

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
            self.assertIsInstance(renderer.call_args.kwargs['width'],int)
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
