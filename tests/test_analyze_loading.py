"""Guard against hidden-panel work and unsolicited external calls."""
import subprocess
import sys
import unittest
from streamlit.testing.v1 import AppTest

SCRIPT = """
import builtins
from unittest.mock import patch
import pandas as pd
import streamlit as st
from utils.enterprise_ui import preserve_page_state
from utils.dataset_views import render_analyze
preserve_page_state()
st.session_state['file_id'] = 'loading-test'
frame = pd.DataFrame({'measure':range(20), 'other':[v*2 for v in range(20)], 'group':['A','B']*10})
def ai(**kwargs):
    st.session_state['ai_calls'] = st.session_state.get('ai_calls', 0) + 1
    return 'Test interpretation'
def group(*args, **kwargs):
    st.session_state['group_calls'] = st.session_state.get('group_calls', 0) + 1
    st.write('Group panel rendered')
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name in ('utils.modelling', 'shap'):
        raise AssertionError('Heavy model library loaded before Run/results')
    return original_import(name, *args, **kwargs)
with patch('utils.dataset_views.interpret_single_column', side_effect=ai), patch('utils.group_statistics_ui.render_group_statistics', side_effect=group), patch('builtins.__import__', side_effect=guarded_import):
    render_analyze(frame, lambda *args: None)
"""

class AnalyzeLoadingTests(unittest.TestCase):
    def test_only_open_panel_runs_and_ai_requires_explicit_action(self):
        app = AppTest.from_string(SCRIPT)
        app.session_state['view_analyze_tabs'] = 'Predictive Analytics'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotIn('ai_calls', app.session_state)
        self.assertNotIn('group_calls', app.session_state)
        self.assertFalse(any(item.key.startswith('pa_target_check_') for item in app.checkbox))
        self.assertFalse(any(item.key == 'pa_target_select' for item in app.selectbox))
        self.assertEqual([tab.label for tab in app.tabs if tab.label in ('Exploratory Data Analysis', 'Predictive Analytics')], ['Exploratory Data Analysis', 'Predictive Analytics'])
        generate = next(button for button in app.button if button.label == 'Generate interpretation')
        generate.click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['ai_calls'], 1)
        app.run(timeout=60)
        self.assertEqual(app.session_state['ai_calls'], 1)
        app.session_state['view_analyze_tabs'] = 'Group Statistics'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['group_calls'], 1)
        app.session_state['view_analyze_tabs'] = 'Descriptive Statistics'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['group_calls'], 1)
        self.assertEqual(app.session_state['ai_calls'], 1)
        app.session_state['view_analytics_section'] = 'Predictive Analytics'
        app.run(timeout=60)
        self.assertTrue(any(item.key.startswith('pa_target_check_') for item in app.checkbox))
        self.assertFalse(any(tab.label == 'Descriptive Statistics' for tab in app.tabs))
        app.button(key='pa_select_all_Numerical').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key='pa_pred_other').value)
        app.session_state['view_analytics_section'] = 'Exploratory Data Analysis'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['ai_calls'], 1)
        app.session_state['view_analytics_section'] = 'Predictive Analytics'
        app.run(timeout=60)
        self.assertTrue(app.checkbox(key='pa_pred_other').value)
        self.assertFalse(app.exception)

    def test_lightweight_configuration_does_not_import_training_libraries(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; import utils.modelling_config; assert 'shap' not in sys.modules; assert 'sklearn' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == '__main__':
    unittest.main()
