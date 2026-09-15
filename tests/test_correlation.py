import unittest
import numpy as np
import pandas as pd
from utils.correlation_ui import correlation_data


class CorrelationTests(unittest.TestCase):
    def test_pairwise_missing_and_constant(self):
        data = pd.DataFrame({'a':[1.,2,3,4], 'b':[2.,4,6,np.inf], 'c':[1.,1,1,1]})
        _, matrix, counts = correlation_data(data, list(data), 'Pearson')
        self.assertAlmostEqual(matrix.loc['a','b'], 1)
        self.assertEqual(counts.loc['a','b'], 3)
        self.assertTrue(np.isnan(matrix.loc['a','c']))

    def test_spearman_and_insufficient_pairs(self):
        data = pd.DataFrame({'a':[1.,2,3,4], 'b':[1.,4,9,16], 'c':[1.,2,np.nan,np.nan]})
        _, matrix, _ = correlation_data(data, list(data), 'Spearman')
        self.assertAlmostEqual(matrix.loc['a','b'], 1)
        self.assertTrue(np.isnan(matrix.loc['a','c']))

    def test_ui_separation_and_matrix(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import pandas as pd
import streamlit as st
from utils.relationship_ui import render_relationship_analysis
df = pd.DataFrame({'a':[1.,2,3,4,5], 'b':[2.,5,4,8,7], 'c':[9.,7,8,5,3]})
cols = st.multiselect('Columns', list(df), default=['a','b'], key='cols')
render_relationship_analysis(df, cols, [], [])
''').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(any('Pearson' in str(item.value) for item in app.markdown))
        app.selectbox(key='rel_method').select('Correlation').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label == 'Pearson correlation' for m in app.metric))
        self.assertFalse(any(s.label.startswith('Target /') for s in app.selectbox))
        app.multiselect(key='cols').set_value(['a','b','c']).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value.shape, (3,3))
        self.assertTrue(app.get('plotly_chart'))
        app.selectbox(key='rel_corr_method').select('Spearman').run(timeout=30)
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
