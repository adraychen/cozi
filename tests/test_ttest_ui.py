import unittest
from streamlit.testing.v1 import AppTest

FIXTURE = '''
import pandas as pd
import numpy as np
import streamlit as st
from utils.comparison_ui import render_comparison
from utils.variable_metadata import infer_metadata, apply_user_override, effective_data
raw = pd.DataFrame({'a':[1.,3.,6.,2.,8.,12.,np.nan,4.,6.],
                    'b':[2.,2.,3.,5.,9.,13.,7.,np.nan,2.],
                    'g':['A','A','A','B','B','B','C','C','C'],
                    'constant':[1.]*9, 'binary':[0,1,0,1,0,1,0,1,0], 'CustomerID':list(range(9))})
metadata = infer_metadata(raw)
metadata = apply_user_override(metadata,'binary',raw['binary'],'Categorical','Binary')
st.session_state['variable_metadata'] = metadata
st.session_state['file_id'] = 'ttest_ui'
data, types = effective_data(raw, metadata)
render_comparison(data, types)
'''


class TTestUITests(unittest.TestCase):
    def app(self):
        app = AppTest.from_string(FIXTURE).run(timeout=45)
        app.checkbox(key='gs_cmp_variable_a').check().run(timeout=45)
        app.selectbox(key='gs_cmp_analysis').select('t-Test').run(timeout=45)
        return app

    def assert_healthy(self, app):
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

    def test_modes_charts_and_changing_inputs(self):
        app = self.app()
        self.assert_healthy(app)
        self.assertEqual([t.label for t in app.tabs], ['Test Configuration','Descriptive Statistics','Diagnostics','t-Test Results','Interpretation'])
        self.assertEqual(len(app.get('plotly_chart')),2)
        app.number_input(key='gs_tt_null').set_value(2.).run(timeout=45)
        app.selectbox(key='gs_tt_alternative').select('Difference > 0 (greater)').run(timeout=45)
        self.assertTrue(any('hypothesized mean (2)' in m.value for m in app.markdown))
        self.assertTrue(any('+infinity' in str(t.value) for t in app.table))
        app.selectbox(key='gs_tt_mode').select('Paired Samples t-Test').run(timeout=45)
        self.assertTrue(any('Select two numerical' in i.value for i in app.info))
        app.checkbox(key='gs_cmp_variable_b').check().run(timeout=45)
        self.assert_healthy(app)
        self.assertEqual(app.selectbox(key='gs_tt_second').options,['b'])
        self.assertTrue(any('Paired differences' in str(d.value) for d in app.dataframe))
        app.selectbox(key='gs_tt_mode').select('Two Independent Samples t-Test').run(timeout=45)
        self.assert_healthy(app)
        self.assertFalse(app.checkbox(key='gs_tt_equal').value)
        self.assertTrue(any('Welch' in m.value for m in app.markdown))
        app.checkbox(key='gs_tt_equal').check().run(timeout=45)
        self.assertTrue(any('Pooled Two-Sample' in m.value for m in app.markdown))
        # Selecting the first variable formerly used for the second resets that role safely.
        app.selectbox(key='gs_tt_first').select('b').run(timeout=45)
        self.assertEqual(app.selectbox(key='gs_tt_second').value,'a')
        self.assert_healthy(app)
        app.checkbox(key='gs_cmp_variable_b').uncheck().run(timeout=45)
        self.assert_healthy(app)
        self.assertTrue(any('Select a numerical response' in i.value for i in app.info))

    def test_stacked_groups_and_semantic_exclusions(self):
        app = self.app()
        app.button(key='gs_cmp_all_numerical').click().run(timeout=45)
        app.button(key='gs_cmp_all_categorical').click().run(timeout=45)
        app.selectbox(key='gs_tt_mode').select('Two Independent Samples t-Test').run(timeout=45)
        app.selectbox(key='gs_tt_structure').select('Stacked format').run(timeout=45)
        self.assert_healthy(app)
        self.assertNotIn('binary', app.selectbox(key='gs_tt_first').options)
        self.assertNotIn('CustomerID', app.selectbox(key='gs_tt_first').options)
        self.assertEqual(app.selectbox(key='gs_tt_level1').options,['A','B','C'])
        app.selectbox(key='gs_tt_level1').select('C').run(timeout=45)
        app.selectbox(key='gs_tt_level2').select('B').run(timeout=45)
        self.assertTrue(any('g = C - g = B' in m.value for m in app.markdown))
        app.selectbox(key='gs_tt_group').select('binary').run(timeout=45)
        self.assert_healthy(app)
        self.assertEqual(app.selectbox(key='gs_tt_level1').options,['0','1'])
        app.checkbox(key='gs_cmp_variable_g').uncheck().run(timeout=45)
        self.assert_healthy(app)
        app.button(key='gs_cmp_clear_categorical').click().run(timeout=45)
        self.assert_healthy(app)
        self.assertEqual(app.selectbox(key='gs_tt_structure').value,'Separate-column format')

    def test_constant_and_missing_selection_no_raw_errors(self):
        app = self.app()
        app.checkbox(key='gs_cmp_variable_constant').check().run(timeout=45)
        app.selectbox(key='gs_tt_first').select('constant').run(timeout=45)
        self.assert_healthy(app)
        self.assertTrue(any('variation must be positive' in w.value for w in app.warning))
        app.button(key='gs_cmp_clear_numerical').click().run(timeout=45)
        self.assert_healthy(app)
        self.assertFalse(any(x.key=='gs_cmp_analysis' for x in app.selectbox))

    def test_binary_and_identifier_alone_do_not_enable_ttest(self):
        app = AppTest.from_string(FIXTURE).run(timeout=45)
        app.checkbox(key='gs_cmp_variable_binary').check().run(timeout=45)
        self.assertNotIn('t-Test',app.selectbox(key='gs_cmp_analysis').options)
        app.checkbox(key='gs_cmp_variable_CustomerID').check().run(timeout=45)
        self.assertNotIn('t-Test',app.selectbox(key='gs_cmp_analysis').options)
        self.assert_healthy(app)


if __name__=='__main__': unittest.main()
