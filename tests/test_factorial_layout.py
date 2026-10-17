"""Automatic factorial results and clarification above the results navigation."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from utils.factorial_interactions_ui import ALL_INTERACTIONS, NO_INTERACTION
from utils.factorial_study_design import factor_key, SEPARATE, UNSURE
from tests.test_factorial_study_design import SCRIPT as MIXED_SCRIPT
from tests.test_factorial_interactions import SCRIPT as FIXED_SCRIPT


class FactorialLayoutTests(unittest.TestCase):
    def test_one_dropdown_defaults_all_and_updates_without_apply(self):
        app = AppTest.from_string(FIXED_SCRIPT).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(any(r.key == 'fac_interaction_mode' for r in app.radio))
        self.assertEqual(app.multiselect(key='fac_interaction_choices').value, [ALL_INTERACTIONS])
        self.assertTrue(any(e.label == 'Which interaction effects should I select?' for e in app.expander))
        app.button(key='report_add_factorial').click().run(timeout=60)
        saved = deepcopy(app.session_state['cozy_report']['blocks'])
        app.multiselect(key='fac_interaction_choices').set_value([NO_INTERACTION]).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['report_latest']['settings']['interactions'], [])
        self.assertFalse(any(b.key == 'fac_apply_model' for b in app.button))
        app.multiselect(key='fac_interaction_choices').set_value([('A','B'),('B','C')]).run(timeout=60)
        self.assertEqual(app.session_state['report_latest']['settings']['interactions'], [['A','B'],['B','C']])
        self.assertEqual(app.session_state['cozy_report']['blocks'], saved)

    def test_all_and_none_are_exclusive_with_specific_pairs(self):
        app = AppTest.from_string(FIXED_SCRIPT).run(timeout=60)
        app.multiselect(key='fac_interaction_choices').set_value([ALL_INTERACTIONS, ('A','B')]).run(timeout=60)
        self.assertEqual(app.multiselect(key='fac_interaction_choices').value, [('A','B')])
        app.multiselect(key='fac_interaction_choices').set_value([('A','B'), NO_INTERACTION]).run(timeout=60)
        self.assertEqual(app.multiselect(key='fac_interaction_choices').value, [NO_INTERACTION])
        app.multiselect(key='fac_interaction_choices').set_value([NO_INTERACTION, ALL_INTERACTIONS]).run(timeout=60)
        self.assertEqual(app.multiselect(key='fac_interaction_choices').value, [ALL_INTERACTIONS])

    def test_study_design_is_above_tabs_and_confirmation_generates_results(self):
        with patch('utils.anova_effects_ui._fit') as fitting:
            app = AppTest.from_string(MIXED_SCRIPT).run(timeout=60)
            fitting.assert_not_called()
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs)
        self.assertEqual(app.expander[-1].label, 'Review advised')
        app.button(key='fac_action_design_confirm_' + factor_key('Operator')).click().run(timeout=60)
        app.radio(key='fac_design_collection').set_value(SEPARATE).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.tabs[0].label, 'Model Overview')
        self.assertTrue(any(e.label == 'Study Design' for e in app.expander))
        self.assertFalse(any('assignment' in s.label for s in app.tabs[1].selectbox))
        self.assertTrue(any('Model Estimability' in e.label for e in app.tabs[1].expander))
        self.assertIn('report_latest', app.session_state)
        # The clarification container precedes the tab row in the rendered tree.
        elements = list(app)
        panel = next(i for i, e in enumerate(elements) if e.type == 'expander' and e.label == 'Study Design')
        tab = next(i for i, e in enumerate(elements) if e.type == 'tab' and e.label == 'Model Overview')
        self.assertLess(panel, tab)
        app.multiselect(key='fac_interaction_choices').set_value([NO_INTERACTION]).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(app.tabs)
        self.assertEqual(app.session_state['report_latest']['settings']['interactions'], [])
        self.assertTrue(app.session_state['report_latest']['settings']['study_design']['assignments'][0]['confirmed'])
        app.radio(key='fac_design_collection').set_value(UNSURE).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs)
        self.assertNotIn('report_latest', app.session_state)
        self.assertTrue(any(e.label == 'Review advised' for e in app.expander))


if __name__ == '__main__':
    unittest.main()
