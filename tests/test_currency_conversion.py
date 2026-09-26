import copy
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.currency_conversion import convert_currency_to_numeric, CurrencyConversionError, detect_currency_columns
from utils.dataset_workspace import add_dataset
from utils.variable_metadata import infer_metadata
from test_dataset_workspace import FIXTURE


class CurrencyConversionTests(unittest.TestCase):
    def test_smart_detection_does_not_guess_plain_text_or_modify_data(self):
        frame = pd.DataFrame({'Money': ['$3000', 'Unknown'], 'Negative': ['$(300)', None],
                              'Name': ['Toronto', 'New York'], 'ID': ['001', '002'],
                              'Number': [3, 4], 'Note': ['Budget $300 approved', 'None']})
        original = frame.copy()
        self.assertEqual(detect_currency_columns(frame), ['Money', 'Negative'])
        pd.testing.assert_frame_equal(frame, original)

    def test_accounting_parentheses(self):
        frame = pd.DataFrame({'Amount': ['$(3000)', '($3,000.00)', '(3000)', ' £(1,250) ', None]})
        result = convert_currency_to_numeric(frame, 'Amount')
        self.assertEqual(str(result.Amount.dtype), 'Int64')
        self.assertEqual(result.Amount.iloc[:4].tolist(), [-3000, -3000, -3000, -1250])
        self.assertTrue(pd.isna(result.Amount.iloc[4]))
        self.assertEqual(frame.Amount.iloc[0], '$(3000)')
        result = convert_currency_to_numeric(pd.DataFrame({'Amount': ['€(3,000.50)', '(¥250.25)']}), 'Amount')
        self.assertEqual(str(result.Amount.dtype), 'Float64')
        self.assertEqual(result.Amount.tolist(), [-3000.5, -250.25])
        for value in ['$(3000', '$3000)', '$((3000))', '$(-3000)', '(-$3000)', '$(+3000)', '$($3000)', '($-)']:
            with self.subTest(value=value), self.assertRaises(CurrencyConversionError):
                convert_currency_to_numeric(pd.DataFrame({'Amount': ['$3000', value]}), 'Amount')

    def test_supported_values_decimals_and_missing(self):
        values = ['$3000', '$3,000.00', ' $3000 ', '£3000', '¥3000', '€3,000.50',
                  '$-', '£-', '¥-', '€-', '-$300', '$-300', None, float('nan'), '', '  ']
        frame = pd.DataFrame({'Amount': values})
        original = frame.copy()
        result = convert_currency_to_numeric(frame, 'Amount')
        self.assertEqual(str(result.Amount.dtype), 'Float64')
        self.assertEqual(result.Amount.iloc[:12].tolist(), [3000, 3000, 3000, 3000, 3000, 3000.5, 0, 0, 0, 0, -300, -300])
        self.assertTrue(result.Amount.iloc[12:].isna().all())
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(infer_metadata(result)['Amount']['variable_subtype'], 'Continuous')

    def test_nullable_whole_numbers_and_existing_numeric(self):
        indian = convert_currency_to_numeric(pd.DataFrame({'Amount': ['$1,02,667.50']}), 'Amount')
        self.assertEqual(indian.Amount.iloc[0], 102667.5)
        for values in [['$3,000.00', '$2,000.00', '$-', None], [3000, None, 2500], [None, '']]:
            result = convert_currency_to_numeric(pd.DataFrame({'Amount': values}), 'Amount')
            self.assertEqual(str(result.Amount.dtype), 'Int64')
            metadata = infer_metadata(result)['Amount']
            self.assertEqual(metadata['variable_type'], 'Numerical')
            self.assertEqual(metadata['variable_subtype'], 'Discrete')

    def test_atomic_failure_and_counts(self):
        frame = pd.DataFrame({'Amount': ['$3,000', 'Unknown', 'TBD', 'Unknown', '$2,500', '$N/A', 'None']})
        original = frame.copy()
        with self.assertRaises(CurrencyConversionError) as caught:
            convert_currency_to_numeric(frame, 'Amount')
        counts = caught.exception.invalid.set_index('Value')['Count'].to_dict()
        self.assertEqual(counts, {"'Unknown'": 2, "'TBD'": 1, "'$N/A'": 1, "'None'": 1})
        pd.testing.assert_frame_equal(frame, original)

    def test_malformed_and_out_of_range_values_rejected(self):
        for value in ['3,00', '$3 000', 'New York', '$$300', 'NaN', 'Infinity', '$9223372036854775808', True]:
            with self.subTest(value=value), self.assertRaises(CurrencyConversionError):
                convert_currency_to_numeric(pd.DataFrame({'Amount': [value]}), 'Amount')

    def make_app(self, frame):
        app = AppTest.from_string(FIXTURE)
        state = {}
        source = add_dataset(state, frame, 'Currency source', 'Test data')
        for key, value in state.items():
            app.session_state[key] = value
        app.run(timeout=60)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Convert Currency to Numeric').run(timeout=60)
        return app, source

    def test_ui_preview_create_and_metadata(self):
        frame = pd.DataFrame({'Amount': ['$3,000.00', '$-', None, '£125.50', '$(3000)']})
        app, source = self.make_app(frame)
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        preview = app.session_state['tf_result']['frame'].copy()
        self.assertEqual(str(preview.Amount.dtype), 'Float64')
        self.assertEqual(preview.Amount.iloc[-1], -3000)
        pd.testing.assert_frame_equal(app.session_state['ws_datasets'][source]['frame'], frame)
        app.button(key='tf_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        pd.testing.assert_frame_equal(app.session_state['df'], preview)
        metadata = app.session_state['variable_metadata']['Amount']
        self.assertEqual(metadata['variable_type'], 'Numerical')
        self.assertEqual(metadata['variable_subtype'], 'Continuous')
        self.assertIn('Amount', app.session_state['col_types']['numerical'])

    def test_ui_failure_preserves_metadata_and_blocks_apply(self):
        frame = pd.DataFrame({'Amount': ['$3,000', 'Unknown', 'Unknown', 'TBD']})
        app, source = self.make_app(frame)
        metadata = copy.deepcopy(app.session_state['variable_metadata'])
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any('Conversion incomplete' in e.value for e in app.error))
        self.assertNotIn('tf_result', app.session_state)
        self.assertFalse(any(b.key in ('tf_create', 'tf_apply') for b in app.button))
        tables = [d.value for d in app.dataframe if list(d.value.columns) == ['Column', 'Value', 'Count']]
        self.assertEqual(tables[0]['Count'].tolist(), [2, 1])
        self.assertEqual(app.session_state['variable_metadata'], metadata)
        pd.testing.assert_frame_equal(app.session_state['ws_datasets'][source]['frame'], frame)

    def test_all_columns_conversion_and_atomic_failure(self):
        frame = pd.DataFrame({'Amount': ['$(3000)', '$-'], 'All columns': ['£125.50', None], 'Count': [1, 2], 'Name': ['Toronto', 'New York']})
        app, source = self.make_app(frame)
        app.selectbox(key='tf_currency_column').select(0)
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        result = app.session_state['tf_result']['frame']
        self.assertEqual(str(result.Amount.dtype), 'Int64')
        self.assertEqual(str(result['All columns'].dtype), 'Float64')
        self.assertEqual(result.Amount.tolist(), [-3000, 0])
        pd.testing.assert_series_equal(result.Count, frame.Count)
        pd.testing.assert_series_equal(result.Name, frame.Name)
        app.button(key='tf_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        pd.testing.assert_frame_equal(app.session_state['df'], result)
        pd.testing.assert_frame_equal(app.session_state['ws_datasets'][source]['frame'], frame)
        frame = pd.DataFrame({'Amount': ['$3000', '$-'], 'Name': ['Unknown', 'Unknown'], 'Other': ['TBD', '$20']})
        app, source = self.make_app(frame)
        app.selectbox(key='tf_currency_column').select(0)
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotIn('tf_result', app.session_state)
        details = [d.value for d in app.dataframe if list(d.value.columns) == ['Column', 'Value', 'Count']][0]
        self.assertEqual(details.Column.tolist(), ['Other'])
        self.assertEqual(details.Count.tolist(), [1])
        pd.testing.assert_frame_equal(app.session_state['ws_datasets'][source]['frame'], frame)
        app.multiselect(key='tf_currency_targets').set_value(['Amount'])
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['tf_result']['frame'].Amount.tolist(), [3000, 0])
        pd.testing.assert_series_equal(app.session_state['tf_result']['frame'].Other, frame.Other)


if __name__ == '__main__':
    unittest.main()
