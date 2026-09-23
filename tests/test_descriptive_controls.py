import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest
from utils.descriptive_controls import select_descriptive_variable


class DescriptiveControlsTests(unittest.TestCase):
    def test_callback_after_widget_state_is_cleared(self):
        state = {}
        cell = MagicMock()
        with patch('utils.descriptive_controls.st.session_state', state), \
                patch('utils.descriptive_controls.st.markdown'), \
                patch('utils.descriptive_controls.st.columns', return_value=[cell] * 4):
            select_descriptive_variable({'numerical': ['sepal_length'], 'categorical': []})
            callback = cell.checkbox.call_args.kwargs['on_change']
            del state['dist_profile_column_sepal_length']
            before = dict(state)
            callback('sepal_length')
            self.assertEqual(state, before)
            state.clear()
            callback('sepal_length')
            self.assertEqual(state, {})

    def test_selection_across_groups_and_deselection(self):
        app = AppTest.from_string('''
from utils.descriptive_controls import select_descriptive_variable
select_descriptive_variable({'numerical': ['sepal_length'],
                             'categorical': ['species'], 'temporal': ['date']})
''').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key='dist_profile_column_sepal_length').value)
        for column in ['species', 'date', 'sepal_length']:
            app.checkbox(key=f'dist_profile_column_{column}').check().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['dist_profile_selected'], column)
            self.assertEqual(sum(widget.value for widget in app.checkbox), 1)
        app.checkbox(key='dist_profile_column_sepal_length').uncheck().run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state['dist_profile_selected'])
        self.assertFalse(any(widget.value for widget in app.checkbox))


if __name__ == '__main__':
    unittest.main()
