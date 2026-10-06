import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class HelpTests(unittest.TestCase):
    def test_alphabetical_list_and_feature_search(self):
        with patch('utils.help_ui.FEATURE_HELP', {
            'Unpivot': 'Turn columns into values.',
            'Aggregate': 'Group rows and calculate summaries.',
            'Pivot': 'Turn values into columns.',
        }):
            app = AppTest.from_string('from utils.help_ui import render_help\nrender_help()').run()
            self.assertFalse(app.exception)
            self.assertEqual([e.label for e in app.expander], ['Aggregate', 'Pivot', 'Unpivot'])
            app.text_input(key='help_feature_search').set_value('  PIVOT ').run()
            self.assertEqual([e.label for e in app.expander], ['Pivot', 'Unpivot'])
            app.text_input(key='help_feature_search').set_value('unknown').run()
            self.assertEqual(len(app.expander), 0)
            self.assertTrue(any(m.value == 'No matching features found.' for m in app.markdown))
            app.text_input(key='help_feature_search').set_value('').run()
            self.assertEqual([e.label for e in app.expander], ['Aggregate', 'Pivot', 'Unpivot'])
