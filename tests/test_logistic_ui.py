import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.regression_router import response_candidates
from utils.regression_controls import widget_key

class LogisticUI(unittest.TestCase):
    def test_metadata_routing_candidates(self):
        d=pd.DataFrame({'count':[0,1,0],'binary':[0,1,0],'nominal':['A','B','A']})
        metadata={'binary':{'variable_subtype':'Binary'},'nominal':{'variable_subtype':'Nominal'}}
        self.assertEqual(response_candidates(d,['count'],['binary','nominal'],metadata),['count','binary'])

    def test_full_logistic_ui_and_switch_to_linear(self):
        app=AppTest.from_string('''
import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import expit
from utils.relationship_ui import render_relationship_analysis
rng=np.random.default_rng(400)
x=rng.normal(size=220)
df=pd.DataFrame({'x':x,'g':['A','B']*110,'y':np.where(rng.binomial(1,expit(.4+.7*x)),'Yes','No')})
st.session_state['variable_metadata']={'x':{'variable_subtype':'Continuous'},'y':{'variable_subtype':'Binary'},'g':{'variable_subtype':'Binary'}}
if 'rel_target_False' not in st.session_state: st.session_state['rel_target_False']='y'
render_relationship_analysis(df,['x'],['y','g'],[])
''').run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='rel_method').value,'Regression')
        self.assertTrue(any('Choose the modeled event' in m.value for m in app.info))
        app.selectbox(key='rel_log_event_y').select('Yes').run(timeout=45)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual([t.label for t in app.tabs],['Model Overview','Predictor Analysis','Diagnostics','Multicollinearity & Influence','Visual Analysis','Prediction','Regression Findings'])
        app.checkbox(key=widget_key('predictor','g')).check().run(timeout=45)
        app.selectbox(key='rel_log_ref_g').select('B').run(timeout=45)
        app.checkbox(key=widget_key('interaction',('x','g'))).check().run(timeout=45)
        app.checkbox(key=widget_key('power',('x',3))).check().run(timeout=45)
        app.number_input(key='rel_log_confidence').set_value(90.).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        r=app.session_state['rel_logistic_fit'][1]
        self.assertEqual(r['spec']['levels']['g'][0],'B')
        self.assertEqual(r['confidence'],.9)
        self.assertIn(('power','x',2),r['active'])
        from utils import logistic_ui
        with patch.object(logistic_ui,'logistic_regression',side_effect=AssertionError('Unexpected refit')):
            next(b for b in app.button if b.label=='Predict event probability').click().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('Probability of' in m.label for m in app.metric))
        app.checkbox(key=widget_key('target','x')).check().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('Linear regression' in m.value for m in app.markdown))
        self.assertFalse(any(s.label=='Modeled event / positive outcome' for s in app.selectbox))

    def test_categorical_only_and_separation_message(self):
        app=AppTest.from_string('''
import pandas as pd
from utils.relationship_ui import render_relationship_analysis
df=pd.DataFrame({'y':['Yes','No']*20,'group':['A','B']*20})
render_relationship_analysis(df,[],['y','group'],[])
''').run(timeout=30)
        self.assertEqual(app.selectbox(key='rel_method').value,'Regression')
        app.selectbox(key='rel_log_event_y').select('Yes').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('separation' in e.value.lower() for e in app.error))
        self.assertFalse(app.tabs)

if __name__=='__main__': unittest.main()
