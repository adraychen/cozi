import unittest
from streamlit.testing.v1 import AppTest
from utils.predictive_controls import target_key


class TargetCheckboxTests(unittest.TestCase):
    def test_groups_and_single_target_switching(self):
        app = AppTest.from_string('''
from utils.predictive_controls import render_target_selector
render_target_selector(['amount', 'group', 'date'], {
    'numerical': ['amount'], 'categorical': ['group'], 'datetime': ['date']})
''').run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.markdown],
                         ['**Numerical**', '**Categorical**', '**Temporal**'])
        self.assertEqual(app.session_state['pa_target_select'], 'amount')
        for name in ['group', 'date', 'amount']:
            app.checkbox(key=target_key(name)).check().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['pa_target_select'], name)
            self.assertEqual(sum(c.value for c in app.checkbox), 1)
        app.checkbox(key=target_key('amount')).uncheck().run()
        self.assertTrue(app.checkbox(key=target_key('amount')).value)

    def test_existing_target_is_preserved(self):
        app = AppTest.from_string('''
from utils.predictive_controls import render_target_selector
render_target_selector(['amount', 'group'], {
    'numerical': ['amount'], 'categorical': ['group'], 'datetime': []})
''')
        app.session_state['pa_target_select'] = 'group'
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key=target_key('group')).value)
