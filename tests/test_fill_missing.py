import unittest
import numpy as np
import pandas as pd
from utils.table_editing import fill_missing_values, fill_value_types


class FillMissingTests(unittest.TestCase):
    def test_fill_only_missing_and_preserve_source_and_types(self):
        cases = [(pd.Series(['', ' ', None, 'A'], dtype='string'), 'Unknown', 'Text'),
                 (pd.Series([1,None,3], dtype='Int64'), '0', 'Number'),
                 (pd.Series([True,None,False], dtype='boolean'), 'False', 'True/False'),
                 (pd.Series([1.,np.nan,3.]), '2.5', 'Number'),
                 (pd.Series(pd.Categorical(['A',None,'B'])), 'Unknown', 'Text'),
                 (pd.Series(pd.to_datetime(['2024-01-01',None])), '2024-02-01', 'Date / Time'),
                 (pd.Series(pd.to_timedelta(['1 day',None])), '2 hours', 'Duration')]
        for series, value, kind in cases:
            with self.subTest(kind=kind, dtype=series.dtype):
                frame = pd.DataFrame({'x':series})
                original = frame.copy(deep=True)
                result, count = fill_missing_values(frame, 'x', value, kind)
                self.assertEqual(count,1)
                self.assertFalse(result.x.isna().any())
                self.assertEqual(result.x[series.notna()].tolist(), series.dropna().tolist())
                pd.testing.assert_frame_equal(frame, original)
                if kind != 'Text' or not isinstance(series.dtype, pd.CategoricalDtype):
                    self.assertEqual(result.x.dtype,series.dtype)

    def test_invalid_values_and_noop(self):
        frame = pd.DataFrame({'x':pd.Series([1,None],dtype='Int8')})
        for value in ['1.5','500','nan','inf','hello']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                fill_missing_values(frame,'x',value,'Number')
        with self.assertRaisesRegex(ValueError,'compatible'):
            fill_missing_values(frame,'x','Unknown','Text')
        frame = pd.DataFrame({'x':['',' ','A']})
        result, count = fill_missing_values(frame,'x','Unknown','Text')
        self.assertEqual(count,0)
        pd.testing.assert_frame_equal(frame,result)

    def test_timezone_and_nullable_storage(self):
        frame = pd.DataFrame({'x':pd.to_datetime(['2024-01-01',None], utc=True)})
        with self.assertRaisesRegex(ValueError,'timezone'):
            fill_missing_values(frame,'x','2024-02-01','Date / Time')
        result,count = fill_missing_values(frame,'x','2024-02-01T00:00:00Z','Date / Time')
        self.assertEqual(count,1)
        self.assertEqual(result.x.dtype, frame.x.dtype)


class FillMissingUITests(unittest.TestCase):
    def test_group_navigation_preview_apply_and_create(self):
        from test_editing_workspace import EditingWorkspaceTests
        from test_dataset_workspace import select_operation
        frame = pd.DataFrame({'amount':pd.Series([1,None,3],dtype='Int64'), 'label':['A','',' ']})
        app, source = EditingWorkspaceTests().app(frame)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        from utils.transform_ui import OPERATION_GROUPS, operation_menu_key
        self.assertEqual([app.selectbox(key=operation_menu_key(group)).label
                          for group in OPERATION_GROUPS], list(OPERATION_GROUPS))
        select_operation(app, 'Aggregate')
        self.assertIsNone(app.selectbox(key='tf_menu_combine_datasets').value)
        self.assertEqual(app.selectbox(key='tf_menu_restructure_data').options,
                         ['Choose operation…', 'Aggregate — Group rows and calculate summaries', 'Pivot — Turn values into columns', 'Unpivot — Turn columns into rows'])
        select_operation(app,'Fill missing values')
        self.assertEqual(app.selectbox(key='tf_fill_type').options,['Number'])
        app.text_input(key='tf_fill_value_Number').set_value('1.5')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertTrue(app.error)
        self.assertNotIn('tf_result', app.session_state)
        app.text_input(key='tf_fill_value_Number').set_value('0')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertEqual(app.session_state['tf_result']['report']['Filled cells'],1)
        app.button(key='tf_apply').click().run(timeout=60)
        self.assertEqual(app.selectbox(key='tf_menu_restructure_data').value, None)
        self.assertEqual(app.selectbox(key='tf_menu_clean_values').value,'Fill missing values')
        select_operation(app, 'Rename column')
        self.assertIsNone(app.selectbox(key='tf_menu_clean_values').value)
        app.selectbox(key='tf_menu_manage_columns').select(None).run(timeout=60)
        self.assertTrue(all(app.selectbox(key=operation_menu_key(group)).value is None
                            for group in OPERATION_GROUPS))
        self.assertNotIn('tf_result',app.session_state)
        self.assertIn('tf_draft',app.session_state)
        app.button(key='tf_draft_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['df'].amount.tolist(),[1,0,3])
        pd.testing.assert_frame_equal(app.session_state['ws_datasets'][source]['frame'],frame)


if __name__ == '__main__':
    unittest.main()
