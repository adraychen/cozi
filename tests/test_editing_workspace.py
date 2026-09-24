import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.dataset_workspace import add_dataset
from test_dataset_workspace import FIXTURE


class EditingWorkspaceTests(unittest.TestCase):
    def app(self, frame):
        app = AppTest.from_string(FIXTURE)
        initial = {}
        source_id = add_dataset(initial, frame, 'Source', 'Test data')
        for key, value in initial.items():
            app.session_state[key] = value
        app.run(timeout=60)
        self.assertFalse(app.exception)
        return app, source_id

    def test_inspect_cleanup_locations_results_and_switching(self):
        app, source_id = self.app(pd.DataFrame({'a': [1, 1, 2], 'copy': [1, 1, 2], 'b': [10, 10, 20]}))
        expanders = [e.label for e in app.expander]
        self.assertLess(expanders.index('Column Overview'), expanders.index('Row Overview'))
        self.assertLess(expanders.index('Row Overview'), expanders.index('Data Quality Summary'))
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(metrics['Number of rows'], '3')
        self.assertEqual(metrics['Duplicated rows'], '1')
        app.button(key='view_remove_duplicate_columns').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(list(app.session_state['df']), ['a', 'b'])
        app.button(key='view_remove_duplicate_rows').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['df']), 2)
        self.assertEqual(app.session_state['ws_workflow'], 'Inspect')
        original = app.session_state['ws_datasets'][source_id]['frame']
        self.assertEqual(original.shape, (3, 3))
        self.assertEqual(len(app.session_state['ws_datasets']), 3)
        app.session_state['ws_dataset_tabs'] = 'Source'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['ws_datasets']), 3)
        self.assertTrue(any(b.key == 'view_remove_duplicate_columns' for b in app.button))

    def test_transform_column_operations_preview_validation_and_save(self):
        app, source_id = self.app(pd.DataFrame({'a': [2., 4.], 'b': [1., 2.]}))
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        self.assertNotIn('Remove duplicated columns', app.radio(key='tf_operation').options)
        app.radio(key='tf_operation').set_value('Rename column').run(timeout=60)
        app.text_input(key='tf_rename_name').set_value('b')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertTrue(app.error)
        self.assertNotIn('tf_result', app.session_state)
        app.text_input(key='tf_rename_name').set_value('renamed')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertEqual(list(app.session_state['tf_result']['frame']), ['renamed', 'b'])
        app.button(key='tf_create').click().run(timeout=60)
        self.assertEqual(list(app.session_state['df']), ['renamed', 'b'])
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Add column').run(timeout=60)
        app.text_input(key='tf_add_name').set_value('total')
        app.selectbox(key='tf_add_method').select('Calculation').run(timeout=60)
        app.selectbox(key='tf_add_left').select('renamed')
        app.selectbox(key='tf_add_right').select('b')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertEqual(app.session_state['tf_result']['frame'].total.tolist(), [3., 6.])
        app.button(key='tf_create').click().run(timeout=60)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Remove column').run(timeout=60)
        app.multiselect(key='tf_remove_columns').set_value(['renamed', 'b', 'total'])
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertTrue(app.error)
        app.multiselect(key='tf_remove_columns').set_value(['renamed', 'b'])
        app.button(key='tf_preview').click().run(timeout=60)
        app.button(key='tf_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(list(app.session_state['df']), ['total'])
        self.assertEqual(list(app.session_state['ws_datasets'][source_id]['frame']), ['a', 'b'])

    def test_value_cleaning_preview_and_create(self):
        frame = pd.DataFrame({'Sales': ['$3,000', '$1,500', '$-', None]})
        app, source_id = self.app(frame)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Replace Values').run(timeout=60)
        app.text_input(key='tf_replace_find_Text').set_value('$-')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(pd.isna(app.session_state['tf_result']['frame'].Sales.iloc[2]))
        app.button(key='tf_create').click().run(timeout=60)
        self.assertEqual(app.session_state['ws_workflow'], 'Inspect')
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Remove Characters').run(timeout=60)
        app.text_input(key='tf_remove_characters').set_value('$')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertEqual(app.session_state['tf_result']['frame'].Sales.iloc[0], '3,000')
        app.text_input(key='tf_remove_characters').set_value('$,').run(timeout=60)
        self.assertNotIn('tf_result', app.session_state)
        app.button(key='tf_preview').click().run(timeout=60)
        app.button(key='tf_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['df'].Sales.iloc[:2].tolist(), ['3000', '1500'])
        self.assertTrue(app.session_state['df'].Sales.iloc[2:].isna().all())
        pd.testing.assert_frame_equal(app.session_state['ws_datasets'][source_id]['frame'], frame)
        self.assertEqual(len(app.session_state['ws_datasets']), 3)

    def test_duplicate_checks_use_raw_values_before_temporal_conversion(self):
        frame = pd.DataFrame({'date': ['2024-01-01'] * 8 + ['invalid-one', 'invalid-two'],
                              'other_date': ['2024-01-01'] * 8 + ['invalid-three', 'invalid-four']})
        app, _ = self.app(frame)
        self.assertEqual({m.label: m.value for m in app.metric}['Duplicated rows'], '7')
        self.assertFalse(any(b.key == 'view_remove_duplicate_columns' for b in app.button))
        app.button(key='view_remove_duplicate_rows').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['df'].date.tolist(), ['2024-01-01', 'invalid-one', 'invalid-two'])


if __name__ == '__main__':
    unittest.main()
