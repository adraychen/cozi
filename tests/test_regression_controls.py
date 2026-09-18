import unittest
from streamlit.testing.v1 import AppTest
from utils.regression_controls import widget_key

class CheckboxControls(unittest.TestCase):
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
