import unittest
import io
from unittest.mock import patch, Mock
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.dataset_workspace import add_dataset, activate, save_active


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = f'''
from unittest.mock import patch
import runpy
with patch('utils.database.record_visit'), patch('utils.database.get_stats', return_value={{'visits': 0, 'likes': 0, 'dislikes': 0}}), patch('utils.ai_interpreter.interpret_dataset_overview', return_value='Test summary'), patch('utils.ai_interpreter.interpret_single_column', return_value='Test column summary'):
    runpy.run_path({str(ROOT / 'app.py')!r}, run_name='__main__')
'''


class WorkspaceTests(unittest.TestCase):
    def test_created_dataset_ignores_previous_workflow_browser_state(self):
        from streamlit.proto.WidgetStates_pb2 import WidgetStates
        app = AppTest.from_string(FIXTURE)
        state = {}
        source = add_dataset(state, pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), 'Source', 'Test')
        for key, value in state.items():
            app.session_state[key] = value
        app.run(timeout=60)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        mapper = app._session_state._state._key_id_mapper
        old_key = f'ws_workflow_{source}' if f'ws_workflow_{source}' in app.session_state else 'ws_workflow'
        old_id = mapper.get_id_from_key(old_key)
        new_state = {'ws_datasets': app.session_state['ws_datasets']}
        add_dataset(new_state, pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
                    'Transformed', 'Transform', parents=[source])
        app.session_state['ws_pending'] = new_state['ws_pending']
        app.run(timeout=60)
        self.assertEqual(app.session_state['ws_workflow'], 'Inspect')
        # The browser still submits the source tab selection with the next input.
        event = WidgetStates()
        workflow = event.widgets.add()
        workflow.id = old_id
        workflow.string_value = 'Transform'
        column = event.widgets.add()
        column.id = mapper.get_id_from_key('vm_column')
        column.string_value = 'b'
        app._run(widget_state=event, timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['ws_workflow'], 'Inspect')
        self.assertEqual(app.selectbox(key='vm_column').value, 'b')
        app.session_state['ws_dataset_tabs'] = 'Source'
        app.run(timeout=60)
        self.assertEqual(app.session_state['ws_workflow'], 'Transform')
        app.session_state['ws_dataset_tabs'] = 'Transformed'
        app.run(timeout=60)
        self.assertEqual(app.session_state['ws_workflow'], 'Inspect')
        self.assertEqual(app.selectbox(key='vm_column').value, 'b')

    def test_transform_switch_from_browser_widget_event(self):
        from streamlit.proto.WidgetStates_pb2 import WidgetStates
        app = AppTest.from_string(FIXTURE)
        state = {}
        add_dataset(state, pd.DataFrame({'Amount': ['$3000', '$-']}), 'Source', 'Test')
        for key, value in state.items():
            app.session_state[key] = value
        app.run(timeout=60)
        for selected in ['Transform', 'Inspect', 'Transform']:
            event = WidgetStates()
            widget = event.widgets.add()
            widget.id = app._session_state._state._key_id_mapper.get_id_from_key('ws_workflow_' + app.session_state['ws_active'])
            widget.string_value = selected
            app._run(widget_state=event, timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['ws_workflow'], selected)
            if selected == 'Transform':
                self.assertTrue(app.radio(key='tf_operation'))

    def test_add_dataset_does_not_wait_for_ai_and_generation_is_explicit(self):
        script = "import streamlit as st\n" + FIXTURE.replace('utils.ai_interpreter.interpret_dataset_overview', 'utils.dataset_views.interpret_dataset_overview').replace(
            "return_value='Test summary'",
            "side_effect=lambda **kwargs: (st.session_state.__setitem__('test_ai_calls', st.session_state.get('test_ai_calls', 0) + 1) or 'Test summary')")
        app = AppTest.from_string(script)
        state = {}
        add_dataset(state, pd.DataFrame({'Amount': [1, 2, 3]}), 'First', 'Test')
        for key, value in state.items():
            app.session_state[key] = value
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotIn('test_ai_calls', app.session_state)
        app.button(key='view_generate_ai').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['test_ai_calls'], 1)
        self.assertEqual(app.session_state['ai_summary'], 'Test summary')
        app.run(timeout=60)
        self.assertEqual(app.session_state['test_ai_calls'], 1)
        app.button(key='view_generate_ai').click().run(timeout=60)
        self.assertEqual(app.session_state['test_ai_calls'], 2)
        next_state = {'ws_datasets': app.session_state['ws_datasets']}
        add_dataset(next_state, pd.DataFrame({'Amount': [4, 5]}), 'Second', 'Test')
        for key, value in next_state.items():
            app.session_state[key] = value
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['test_ai_calls'], 2)
        self.assertNotIn('ai_summary', app.session_state)

    def test_excel_sheet_discovery_cached_by_content(self):
        from utils.dataset_workspace import uploaded_sheet_names
        uploaded_sheet_names.clear()
        with patch('utils.dataset_workspace.pd.ExcelFile') as reader:
            reader.return_value.__enter__.return_value.sheet_names = ['Sheet1']
            self.assertEqual(uploaded_sheet_names(b'workbook-one'), ['Sheet1'])
            self.assertEqual(uploaded_sheet_names(b'workbook-one'), ['Sheet1'])
            self.assertEqual(reader.call_count, 1)
            uploaded_sheet_names(b'workbook-two')
            self.assertEqual(reader.call_count, 2)
        uploaded_sheet_names.clear()

    def test_join_validation_readable_counts(self):
        app = AppTest.from_string('''
from utils.transform_ui import render_validation
render_validation({'Output rows': 1418, 'Matched left rows': 1418,
    'Unmatched left rows': 0, 'Unmatched right rows': 2,
    'Left rows with duplicate keys': 0, 'Right rows with duplicate keys': 3,
    'Left rows with missing keys': 0, 'Right rows with missing keys': 1,
    'Many-to-many matches': False}, 'Join')
''').run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ['1,418', '1,418'])
        self.assertEqual(app.dataframe[0].value['Right dataset'].tolist(), ['2', '3', '1'])
        self.assertTrue(any('**Many-to-many matches:** No' == m.value for m in app.markdown))
        self.assertFalse(app.get('json'))

    def test_csv_download_contains_full_raw_dataset(self):
        from utils.dataset_views import dataset_csv
        frame = pd.DataFrame({'id': range(150), 'label': ['Montréal, "East"'] * 150,
                              'raw_date': ['not converted'] * 150})
        result = pd.read_csv(io.BytesIO(dataset_csv(frame)))
        pd.testing.assert_frame_equal(result, frame)

    def test_registry_and_isolated_state(self):
        state = {}
        frame = pd.DataFrame({'x': [1, 2]})
        a = add_dataset(state, frame, 'Data', 'test')
        b = add_dataset(state, frame, 'Data', 'test')
        self.assertEqual(state['ws_datasets'][b]['name'], 'Data (2)')
        activate(state, a)
        state['variable_metadata'] = {'x': {'source': 'Manual'}}
        state['pa_results'] = {'score': 1}
        state['vm_apply'] = True
        state['va_graph_button_Pie chart'] = True
        state['dist_profile_selected'] = 'x'
        activate(state, b)
        self.assertNotIn('variable_metadata', state)
        self.assertNotIn('pa_results', state)
        state['dist_profile_selected'] = None
        activate(state, a)
        self.assertEqual(state['pa_results']['score'], 1)
        self.assertEqual(state['dist_profile_selected'], 'x')
        self.assertNotIn('vm_apply', state)
        self.assertNotIn('va_graph_button_Pie chart', state)
        frame.loc[0, 'x'] = 999
        self.assertEqual(state['df'].x.iloc[0], 1)

    def app(self):
        return AppTest.from_string(FIXTURE).run(timeout=60)

    def add_manual(self, app, name, csv):
        app.session_state['ws_dataset_tabs'] = '＋ Add datasets'
        app.run(timeout=60)
        app.button(key='manual_open').click().run(timeout=60)
        app.radio(key='manual_mode').set_value('CSV text').run(timeout=60)
        app.text_area(key='manual_csv').set_value(csv)
        app.text_input(key='manual_dataset_name').set_value(name)
        app.button(key='manual_submit').click().run(timeout=60)
        self.assertFalse(app.exception)

    def test_manual_add_switch_inspect_and_transform_creation(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any(t.label == '＋ Add datasets' for t in app.tabs))
        self.add_manual(app, 'Sales', 'id,amount\n1,20\n2,30')
        self.assertEqual(len(app.session_state['ws_datasets']), 1)
        self.assertEqual([e.label for e in app.expander][:3], ['Dataset Preview', 'Column Overview', 'Edit variable classification'])
        app.selectbox(key='vm_column').select('id').run(timeout=60)
        app.selectbox(key='vm_0_0_type').select('Categorical').run(timeout=60)
        app.button(key='vm_apply').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.add_manual(app, 'Customers', 'id,region\n1,East\n3,West')
        app.session_state['ws_dataset_tabs'] = 'Sales'
        app.run(timeout=60)
        self.assertEqual(app.session_state['variable_metadata']['id']['variable_type'], 'Categorical')
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['tf_result']['frame']), 2)
        app.button(key='tf_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['ws_datasets']), 3)
        self.assertEqual(app.session_state['ws_workflow'], 'Inspect')
        created = app.session_state['ws_datasets'][app.session_state['ws_active']]
        self.assertEqual(created['source'], 'Join')
        self.assertTrue(pd.isna(created['frame'].region.iloc[1]))
        self.assertTrue(any(b.proto.label == 'Download dataset (CSV)' for b in app.get('download_button')))

    def test_analyze_and_preview_invalidation(self):
        app = self.app()
        self.add_manual(app, 'Data', 'group,value\nA,1\nA,2\nB,3\nB,4')
        app.session_state['ws_workflow'] = 'Analyze'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Aggregate').run(timeout=60)
        app.multiselect(key='tf_groups').set_value(['group'])
        app.multiselect(key='tf_measures').set_value(['value']).run(timeout=60)
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['tf_result']['frame']['value_sum'].tolist(), [3, 7])
        app.multiselect(key='tf_functions').set_value(['Mean']).run(timeout=60)
        self.assertNotIn('tf_result', app.session_state)
        self.assertFalse(any(b.key == 'tf_create' for b in app.button))

    def test_multi_upload_excel_sheets_and_same_name(self):
        app = self.app()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame({'x': [1]}).to_excel(writer, sheet_name='First', index=False)
            pd.DataFrame({'x': [2]}).to_excel(writer, sheet_name='Second', index=False)
        app.file_uploader[0].set_value([
            ('data.csv', b'x\n3', 'text/csv'),
            ('data.csv', b'x\n4', 'text/csv'),
            ('book.xlsx', buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        ]).run(timeout=60)
        app.multiselect[0].set_value(['First', 'Second']).run(timeout=60)
        app.button(key='ws_import').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['ws_workflow'], 'Inspect')
        self.assertTrue({'Dataset Preview', 'Column Overview', 'Row Overview', 'Data Quality Summary'}.issubset({e.label for e in app.expander}))
        entries = list(app.session_state['ws_datasets'].values())
        self.assertEqual([d['name'] for d in entries], ['data.csv', 'data.csv (2)', 'book.xlsx · First', 'book.xlsx · Second'])
        self.assertEqual([d['frame'].x.iloc[0] for d in entries], [3, 4, 1, 2])
        app.session_state['ws_dataset_tabs'] = '＋ Add datasets'
        app.run(timeout=60)
        self.assertFalse(app.file_uploader[0].value)

    def test_uploaded_text_and_dates_render_inspect_without_tab_roundtrip(self):
        app = self.app()
        frame = pd.DataFrame({'City': [' New York ', 'Toronto'] * 350,
                              'Date': ['2024-01-01', '2024-02-01'] * 350,
                              'Amount': ['$3000', '$-'] * 350})
        app.file_uploader[0].set_value([('data.csv', frame.to_csv(index=False).encode(), 'text/csv')]).run(timeout=60)
        app.button(key='ws_import').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue({'Dataset Preview', 'Column Overview', 'Row Overview', 'Data Quality Summary'}.issubset({e.label for e in app.expander}))
        self.assertEqual(app.session_state['variable_metadata']['Date']['variable_type'], 'Date / Time')
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(app.radio(key='tf_operation'))
        app.session_state['ws_workflow'] = 'Inspect'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any(e.label == 'Row Overview' for e in app.expander))

    def test_failed_import_and_sample_add_preserve_existing(self):
        app = self.app()
        self.add_manual(app, 'Existing', 'x\n1\n2')
        app.session_state['ws_dataset_tabs'] = '＋ Add datasets'
        app.run(timeout=60)
        app.file_uploader[0].set_value([('valid.csv', b'x\n4', 'text/csv'), ('empty.csv', b'', 'text/csv')]).run(timeout=60)
        app.button(key='ws_import').click().run(timeout=60)
        self.assertEqual(len(app.session_state['ws_datasets']), 1)
        self.assertTrue(app.error)
        response = Mock(text='x,y\n2,3\n4,5')
        with patch('requests.get', return_value=response):
            app.button(key='ws_sample_iris.csv').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['ws_datasets']), 2)
        self.assertEqual(app.session_state['ws_dataset_tabs'], 'iris.csv')

    def test_append_pivot_unpivot_forms_and_empty_result(self):
        app = self.app()
        self.add_manual(app, 'Monthly', 'id,Jan,Feb\n1,20,30\n2,40,50')
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Unpivot').run(timeout=60)
        app.multiselect(key='tf_unpivot_ids').set_value(['id'])
        app.multiselect(key='tf_unpivot_values').set_value(['Jan', 'Feb'])
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        app.button(key='tf_create').click().run(timeout=60)
        self.assertEqual(len(app.session_state['df']), 4)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Pivot').run(timeout=60)
        app.multiselect(key='tf_pivot_rows').set_value(['id'])
        app.selectbox(key='tf_pivot_column').select('variable')
        app.selectbox(key='tf_pivot_value').select('value')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['tf_result']['frame'].Jan.tolist(), [20, 40])
        app.button(key='tf_create').click().run(timeout=60)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        app.radio(key='tf_operation').set_value('Append').run(timeout=60)
        original = next(iter(app.session_state['ws_datasets']))
        app.multiselect(key='tf_append_inputs').set_value([original])
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['tf_result']['frame']), 4)
        app.button(key='tf_create').click().run(timeout=60)
        self.assertEqual(len(app.session_state['ws_datasets']), 4)


if __name__ == '__main__':
    unittest.main()
