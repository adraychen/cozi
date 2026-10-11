import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.dataset_workspace import add_dataset, activate, save_active
from utils.project_file import dumps, loads, project_payload, replace_project, fingerprint, remove_dataset, prepare_download


class ProjectFileTests(unittest.TestCase):
    def test_complete_round_trip_and_independent_projects(self):
        state = {}
        frame = pd.DataFrame({'month': pd.Series([1, 2, None], dtype='Int64'),
                              'quarter': pd.period_range('2025Q1', periods=3, freq='Q'),
                              'date': pd.date_range('2025-01-01', periods=3)})
        identity = add_dataset(state, frame, 'Original', 'Manual input')
        activate(state, identity)
        state['variable_metadata'] = {'month': {'variable_subtype': 'Month', 'source': 'User override'}}
        state['tf_draft'] = {'frame': frame.copy(), 'steps': [{'operation': 'Add column'}]}
        state['cozy_report'] = {'blocks': [{'text': 'My report', 'chart': {'png': b'chart-bytes', 'plotly': '{}'}}]}
        before = fingerprint(state)
        self.assertEqual(before, fingerprint(state))
        restored = {'df': pd.DataFrame({'old': [9]}), 'ws_datasets': {}, 'report_text_old': 'stale'}
        replace_project(restored, loads(dumps(project_payload(state))))
        assert_frame_equal(restored['df'], frame)
        self.assertEqual(restored['variable_metadata'], state['variable_metadata'])
        assert_frame_equal(restored['tf_draft']['frame'], frame)
        self.assertEqual(restored['cozy_report'], state['cozy_report'])
        self.assertNotIn('report_text_old', restored)
        self.assertEqual(restored['project_saved_hash'], fingerprint(restored))
        restored['tf_draft']['steps'].append({'operation': 'Rename'})
        prepare_download(restored)
        saved = loads(restored['project_download']['content'])
        self.assertEqual(len(saved['datasets'][identity]['state']['tf_draft']['steps']), 2)
        restored['variable_metadata']['month']['variable_subtype'] = 'Year'
        self.assertNotEqual(restored['project_saved_hash'], fingerprint(restored))
        replace_project(restored)
        self.assertFalse(restored['ws_datasets'])
        self.assertNotIn('df', restored)
        self.assertNotIn('cozy_report', restored)

    def test_bad_files_and_versions_leave_current_state_untouched(self):
        with self.assertRaises(ValueError): loads(b'not a project')
        payload = dict(version=999, name='Future', datasets={}, active=None, report=None, latest=None)
        with self.assertRaisesRegex(ValueError, 'version'): loads(dumps(payload))

    def test_remove_source_preserves_derived_dataset_and_report(self):
        state = {}
        first = add_dataset(state, pd.DataFrame({'a': [1]}), 'Source', 'Manual')
        second = add_dataset(state, pd.DataFrame({'a': [2]}), 'Derived', 'Transform', parents=[first])
        activate(state, first)
        state['cozy_report'] = {'text': 'Snapshot'}
        remove_dataset(state, first)
        self.assertEqual(state['ws_active'], second)
        self.assertEqual(state['cozy_report']['text'], 'Snapshot')
        remove_dataset(state, second)
        self.assertNotIn('df', state)

    def test_menu_create_confirmation_and_app_without_votes(self):
        from streamlit.testing.v1 import AppTest
        from pathlib import Path
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.metric)
        self.assertFalse(any(b.label in ('👍', '👎') for b in app.button))
        app.session_state['project_name'] = 'Changed'
        app.run(timeout=60)
        app.button(key='project_create').click().run(timeout=60)
        self.assertTrue(any('unsaved changes' in w.value for w in app.warning))
        app.button(key='project_confirm').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['project_name'], 'Untitled Project')
        source = {'project_name': 'Reopened'}
        identity = add_dataset(source, pd.DataFrame({'month': pd.Series([1, 2], dtype='Int64')}), 'Monthly data', 'Manual')
        activate(source, identity)
        source['variable_metadata'] = {'month': dict(variable_type='Date / Time', variable_subtype='Month', source='User override', order=[])}
        source['variable_dataset_id'] = identity
        source['cozy_report'] = {'metadata': {}, 'blocks': []}
        content = dumps(project_payload(source))
        app.button(key='project_open').click().run(timeout=60)
        app.file_uploader(key='project_upload').set_value(('saved.cozy', content, 'application/octet-stream')).run(timeout=60)
        app.button(key='project_open_confirm').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['project_name'], 'Reopened')
        assert_frame_equal(app.session_state['df'], source['df'])
        self.assertEqual(app.session_state['variable_metadata']['month']['source'], 'User override')
        app.button(key='project_save').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(loads(app.session_state['project_download']['content'])['name'], 'Reopened')
        app.button(key='project_cancel').click().run(timeout=60)
        app.button(key='ws_remove_' + identity).click().run(timeout=60)
        self.assertIn(identity, app.session_state['ws_datasets'])
        app.button(key='ws_remove_cancel').click().run(timeout=60)
        self.assertIn(identity, app.session_state['ws_datasets'])
        app.button(key='ws_remove_' + identity).click().run(timeout=60)
        app.button(key='ws_remove_confirm').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state['ws_datasets'])
        app.button(key='project_close').click().run(timeout=60)
        self.assertTrue(any('unsaved changes' in w.value for w in app.warning))
        app.button(key='project_cancel').click().run(timeout=60)
        self.assertEqual(app.session_state['project_name'], 'Reopened')
        app.button(key='project_close').click().run(timeout=60)
        app.button(key='project_save_before').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertIn('cozy_report', app.session_state)
        app.button(key='project_cancel').click().run(timeout=60)
        app.button(key='project_close').click().run(timeout=60)
        app.button(key='project_confirm').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['project_name'], 'Untitled Project')
        self.assertNotIn('cozy_report', app.session_state)

    def test_save_before_close_preserves_download_then_resets_project(self):
        from unittest.mock import patch
        from utils.project_file import saved_project_download
        state = {'project_name': 'Work'}
        identity = add_dataset(state, pd.DataFrame({'value': [1, 2]}), 'Data', 'Manual')
        activate(state, identity)
        state['project_pending'] = ('close', None)
        state['project_dialog'] = 'save_before'
        prepare_download(state)
        download = state['project_download']
        with patch('streamlit.session_state', state):
            saved_project_download(download['hash'])
        self.assertEqual(state['project_name'], 'Untitled Project')
        self.assertTrue(state['project_saved_message'])
        self.assertFalse(state['ws_datasets'])
        self.assertIn(identity, loads(download['content'])['datasets'])

