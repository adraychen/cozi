"""Regression coverage for ephemeral factorial actions in the real workspace."""
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.dataset_workspace import dataset_state_key, add_dataset, activate, save_active


ACTIONS = ('fac_open_diagnostics', 'fac_apply_model', 'fac_action_cell_coverage',
           'fac_action_estimability', 'fac_action_hc3', 'fac_action_refine_confirm', 'fac_action_ai_generate',
           'fac_action_random_apply', 'fac_action_design_add_group', 'fac_action_design_confirm_example')

SCRIPT = '''
import streamlit as st
from unittest.mock import patch
from tests.test_factorial_guided_workflow import design
from utils.dataset_workspace import add_dataset, activate, render_workspace
from utils.factorial_anova_ui import render_factorial_anova
if 'ws_datasets' not in st.session_state:
    first = add_dataset(st.session_state, design(), 'First', 'Test')
    second = add_dataset(st.session_state, design(), 'Second', 'Test')
    st.session_state.pop('ws_pending', None)
    activate(st.session_state, first)
    st.session_state['ws_workflow'] = 'Analyze'
    st.session_state['ws_datasets'][second]['workflow'] = 'Analyze'
def analyze(frame, feedback):
    st.session_state.setdefault('fac_interaction_mode', 'Select interactions manually')
    render_factorial_anova(frame, ['Y'], ['A','B'])
with patch('utils.dataset_views.render_analyze', side_effect=analyze):
    render_workspace([], lambda: None)
'''


class FactorialWorkspaceActionsTests(unittest.TestCase):
    def test_actions_excluded_and_old_saved_actions_ignored(self):
        state = {}
        first = add_dataset(state, pd.DataFrame({'Y': [1, 2]}), 'First', 'Test')
        activate(state, first)
        for key in ACTIONS:
            self.assertFalse(dataset_state_key(key), key)
            state[key] = False
        state.update(fac_study_answer="I'm not sure.", fac_results_tab='Diagnostics',
                     fac_diagnostic_focus='coverage', fac_interactions=[('A', 'B')],
                     fac_interaction_mode='Select interactions manually')
        save_active(state)
        saved = state['ws_datasets'][first]['state']
        self.assertTrue(all(key not in saved for key in ACTIONS))
        self.assertEqual(saved['fac_results_tab'], 'Diagnostics')
        second = add_dataset(state, pd.DataFrame({'Y': [3, 4]}), 'Second', 'Test')
        activate(state, second)
        # A project saved before the fix may still contain button values.
        state['ws_datasets'][first]['state'].update({key: True for key in ACTIONS})
        activate(state, first)
        self.assertTrue(all(key not in state for key in ACTIONS))
        self.assertEqual(state['fac_study_answer'], "I'm not sure.")
        self.assertEqual(state['fac_interaction_mode'], 'Select interactions manually')

    def test_workspace_reruns_navigation_model_apply_and_dataset_roundtrip(self):
        app = AppTest.from_string(SCRIPT).run(timeout=60)
        self.assertFalse(app.exception)
        # The original failure occurs on the next workspace rerun, even without a click.
        app.run(timeout=60)
        self.assertFalse(app.exception)
        app.button(key='fac_open_diagnostics').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['fac_results_tab'], 'Diagnostics')
        app.button(key='fac_action_cell_coverage').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['fac_diagnostic_focus'], 'coverage')
        app.button(key='fac_action_hc3').click().run(timeout=60)
        self.assertEqual(app.session_state['fac_results_tab'], 'Model Overview')
        app.multiselect(key='fac_interaction_choices').set_value([('A', 'B')]).run(timeout=60)
        app.run(timeout=60)
        self.assertFalse(app.exception)
        for name in ('Second', 'First'):
            app.session_state['ws_dataset_tabs'] = name
            app.run(timeout=60)
            self.assertFalse(app.exception)
        self.assertEqual(app.multiselect(key='fac_interaction_choices').value, [('A', 'B')])
        active = app.session_state['ws_active']
        saved = app.session_state['ws_datasets'][active]['state']
        self.assertTrue(all(key not in saved for key in ACTIONS))


if __name__ == '__main__':
    unittest.main()
