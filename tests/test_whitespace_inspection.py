import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.quality import leading_trailing_whitespace_pct, check_categorical_quality
from utils.dataset_workspace import add_dataset
from test_dataset_workspace import FIXTURE


class WhitespaceInspectionTests(unittest.TestCase):
    def test_representative_values_and_text_only_denominator(self):
        values = ['New York', ' New York', 'New York ', ' New York ',
                  'Los Angeles', 'Hong Kong', '\tToronto', 'Toronto\t',
                  None, float('nan'), ' 3000 ', 3000, True, pd.Timestamp('2026-01-01'), pd.NA]
        series = pd.Series(values, dtype=object)
        original = series.copy(deep=True)
        self.assertAlmostEqual(leading_trailing_whitespace_pct(series), 6 / 9 * 100)
        pd.testing.assert_series_equal(series, original)

    def test_example_and_supported_string_storage(self):
        values = ['Toronto', ' Toronto', 'Toronto ', '  Toronto  ', None]
        for dtype in [object, 'str', 'string', 'category']:
            with self.subTest(dtype=dtype):
                self.assertEqual(leading_trailing_whitespace_pct(pd.Series(values, dtype=dtype)), 75.0)

    def test_zero_and_other_whitespace(self):
        for values in [[], [None, pd.NA, float('nan')], [1, False], ['New York', 'John Smith', 'Hong Kong', '']]:
            self.assertEqual(leading_trailing_whitespace_pct(pd.Series(values, dtype=object)), 0.0)
        self.assertEqual(leading_trailing_whitespace_pct(pd.Series(['\u00a0A', 'A\n', ' A ', 'A\tB'])), 75.0)

    def test_inspect_order_format_special_characters_and_source_preservation(self):
        frame = pd.DataFrame({'City': pd.Series(['Toronto', ' Toronto', 'Toronto ', '  Toronto  ', None], dtype=object),
                              'Clean': pd.Series(['Hong Kong', 'John Smith', 'New York', 'A!', None], dtype=object)})
        original = frame.copy(deep=True)
        before = check_categorical_quality(frame, list(frame))
        app = AppTest.from_string(FIXTURE)
        state = {}
        source_id = add_dataset(state, frame, 'Whitespace sample', 'Test data')
        for key, value in state.items():
            app.session_state[key] = value
        app.run(timeout=60)
        self.assertFalse(app.exception)
        tables = [element.value for element in app.dataframe
                  if list(element.value.columns) == ['Statistic', 'Value']
                  and 'Leading/Trailing Whitespace' in element.value['Statistic'].tolist()]
        self.assertEqual(len(tables), 2)
        for table, expected in zip(tables, ['75.0%', '0.0%']):
            labels = table['Statistic'].tolist()
            self.assertEqual(labels[labels.index('Special Characters') + 1], 'Leading/Trailing Whitespace')
            values = table.set_index('Statistic')['Value']
            self.assertEqual(values['Leading/Trailing Whitespace'], expected)
            # Inspect's existing conversion to string storage is unchanged.
            self.assertEqual(values['Special Characters'], 'No')
        self.assertEqual(before['Special Characters Found'].tolist(), ['No', 'Yes'])
        pd.testing.assert_frame_equal(app.session_state['ws_datasets'][source_id]['frame'], original)
        pd.testing.assert_frame_equal(check_categorical_quality(frame, list(frame)), before)


if __name__ == '__main__':
    unittest.main()
