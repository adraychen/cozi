import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from utils.regression_controls import widget_key, change_selection

class CheckboxControls(unittest.TestCase):
    def test_missing_widget_callback_preserves_state(self):
        for state in [{}, {'rel_predictors': ['x']}, {'rel_polynomial_terms': [('x', 2)]}]:
            for state_key, option, powers in [('rel_predictors', 'x', False), ('rel_polynomial_terms', ('x', 3), True)]:
                with self.subTest(state=state, option=option):
                    before = dict(state)
                    with patch('utils.regression_controls.st.session_state', state):
                        change_selection(state_key, option, 'missing_widget', powers)
                    self.assertEqual(state, before)

    def test_classification_override_with_regression_controls(self):
        app=AppTest.from_string("""
import pandas as pd
import streamlit as st
from utils.variable_metadata_ui import render_column_overview
from utils.regression_controls import render_checkbox_specification, render_term_checkboxes
raw=pd.DataFrame({'y':[1,2,3,4], 'flag':[0,1,0,1]})
data,groups=render_column_overview(raw,'one')
target,predictors=render_checkbox_specification(groups['numerical'],groups['numerical'],groups['categorical'])
render_term_checkboxes(predictors,[c for c in predictors if c in groups['numerical']])
""").run(timeout=30)
        self.assertFalse(app.exception)
        app.checkbox(key=widget_key('power',('flag',3))).check().run()
        app.selectbox(key='vm_column').select('flag').run()
        next(s for s in app.selectbox if s.label=='Variable Type').select('Categorical').run()
        next(s for s in app.selectbox if s.label=='Variable Subtype').select('Nominal').run()
        app.button(key='vm_apply').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['variable_metadata']['flag']['variable_subtype'],'Nominal')
        self.assertEqual(app.session_state['rel_polynomial_terms'],[])
        app.checkbox(key=widget_key('predictor','flag')).uncheck().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['rel_predictors'],[])

    def test_target_hierarchy_and_pruning(self):
        app=AppTest.from_string('''
from utils.regression_controls import render_checkbox_specification,render_term_checkboxes
import streamlit as st
target,predictors=render_checkbox_specification(['y','x','z'],['y','x','z'],['region'])
interactions,degrees=render_term_checkboxes(predictors,[c for c in predictors if c!='region'])
st.session_state['test_spec']=(target,predictors,interactions,degrees)
''').run()
        self.assertFalse(app.exception)
        app.checkbox(key=widget_key('target','x')).check().run()
        self.assertEqual(app.session_state['test_spec'][0],'x')
        self.assertEqual(sum(app.checkbox(key=widget_key('target',c)).value for c in ['y','x','z']),1)
        app.checkbox(key=widget_key('target','x')).uncheck().run()
        self.assertTrue(app.checkbox(key=widget_key('target','x')).value)
        self.assertNotIn('x',app.session_state['test_spec'][1])
        app.checkbox(key=widget_key('predictor','z')).check().run()
        app.checkbox(key=widget_key('predictor','region')).check().run()
        app.checkbox(key=widget_key('interaction',('z','region'))).check().run()
        app.checkbox(key=widget_key('power',('z',3))).check().run()
        self.assertTrue(app.checkbox(key=widget_key('power',('z',2))).value)
        self.assertTrue(app.checkbox(key=widget_key('power',('z',2))).disabled)
        self.assertEqual(app.session_state['test_spec'][3]['z'],3)
        app.checkbox(key=widget_key('power',('z',3))).uncheck().run()
        self.assertFalse(app.checkbox(key=widget_key('power',('z',2))).disabled)
        app.button(key='rel_poly_all').click().run()
        self.assertTrue(app.checkbox(key=widget_key('power',('z',3))).value)
        app.button(key='rel_poly_clear').click().run()
        self.assertFalse(app.checkbox(key=widget_key('power',('z',2))).value)
        app.checkbox(key=widget_key('predictor','z')).uncheck().run()
        self.assertEqual(app.session_state['test_spec'][2],[])
        self.assertNotIn('z',app.session_state['test_spec'][3])
        app.run()
        self.assertEqual(app.session_state['test_spec'][1],['region'])
        self.assertFalse(app.exception)
        app.button(key='rel_predictor_all').click().run()
        self.assertEqual(app.session_state['test_spec'][1],['y','z','region'])
        self.assertNotIn('x',app.session_state['test_spec'][1])
        app.button(key='rel_interaction_all').click().run()
        self.assertEqual(len(app.session_state['test_spec'][2]),3)
        app.button(key='rel_interaction_clear').click().run()
        self.assertEqual(app.session_state['test_spec'][2],[])
        app.button(key='rel_predictor_clear').click().run()
        self.assertEqual(app.session_state['test_spec'][1],[])
        self.assertEqual(app.session_state['test_spec'][2],[])
        self.assertTrue(any('Response / dependent variable (Y)' in m.value for m in app.markdown))
        self.assertFalse(app.exception)

if __name__=='__main__': unittest.main()
