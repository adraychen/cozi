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
            def feature_names():
                return [m.value.strip('*') for m in app.markdown if m.value.startswith('**')]
            self.assertEqual(len(app.expander), 0)
            self.assertEqual(feature_names(), ['Aggregate', 'Pivot', 'Unpivot'])
            self.assertEqual(app.text_input(key='help_feature_search').proto.icon, ':material/search:')
            self.assertEqual(app.text_input(key='help_feature_search').proto.label_visibility.value, 2)
            app.text_input(key='help_feature_search').set_value('  PIVOT ').run()
            self.assertEqual(feature_names(), ['Pivot', 'Unpivot'])
            app.text_input(key='help_feature_search').set_value('unknown').run()
            self.assertEqual(len(app.expander), 0)
            self.assertTrue(any(m.value == 'No matching features found.' for m in app.markdown))
            app.text_input(key='help_feature_search').set_value('').run()
            self.assertEqual(feature_names(), ['Aggregate', 'Pivot', 'Unpivot'])

    def test_games_howell_help_is_searchable(self):
        app = AppTest.from_string('from utils.help_ui import render_help\nrender_help()').run()
        app.text_input(key='help_feature_search').set_value('Games–Howell').run()
        self.assertFalse(app.exception)
        text = '\n'.join(item.value for item in app.markdown)
        self.assertIn('Games–Howell Pairwise Comparisons', text)
        self.assertIn('**Adjusted p-value:**', text)
        self.assertIn('**95% Confidence Interval:**', text)
        self.assertIn('practically important', text)
