import unittest
import pandas as pd
from utils.variable_metadata import infer_metadata, apply_user_override, effective_data
from utils.transformation_metadata import inherit_metadata
from utils.dataset_workspace import add_dataset, activate
from utils.table_transforms import join_tables, append_tables, aggregate_table


class TransformationMetadataTests(unittest.TestCase):
    def test_app_inspect_override_survives_chained_transform_and_analysis(self):
        from streamlit.testing.v1 import AppTest
        from tests.test_dataset_workspace import FIXTURE, select_operation
        app = AppTest.from_string(FIXTURE)
        state = {}
        add_dataset(state, pd.DataFrame({'year': [2020, 2021, 2022], 'value': [10, 15, 30]}), 'Source', 'Test')
        for key, value in state.items():
            app.session_state[key] = value
        app.run(timeout=60)
        app.selectbox(key='vm_column').select('year').run(timeout=60)
        next(s for s in app.selectbox if s.label == 'Variable Type').select('Categorical').run(timeout=60)
        app.button(key='vm_apply').click().run(timeout=60)
        next(s for s in app.selectbox if s.label == 'Variable Type').select('Date / Time').run(timeout=60)
        next(s for s in app.selectbox if s.label == 'Variable Subtype').select('Year').run(timeout=60)
        app.button(key='vm_apply').click().run(timeout=60)
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        select_operation(app, 'Rename column')
        app.text_input(key='tf_rename_name').set_value('annual')
        app.button(key='tf_preview').click().run(timeout=60)
        app.button(key='tf_apply').click().run(timeout=60)
        select_operation(app, 'Trim Spaces')
        app.button(key='tf_preview').click().run(timeout=60)
        app.button(key='tf_apply').click().run(timeout=60)
        app.button(key='tf_draft_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['variable_metadata']['annual']['variable_subtype'], 'Year')
        self.assertEqual(app.session_state['variable_metadata']['annual']['source'], 'User override')
        app.session_state['ws_workflow'] = 'Analyze'
        app.session_state['view_analyze_tabs'] = 'Comparison'
        app.run(timeout=60)
        app.checkbox(key='gs_cmp_variable_annual').check().run(timeout=60)
        app.checkbox(key='gs_cmp_variable_value').check().run(timeout=60)
        app.selectbox(key='gs_cmp_analysis').select('Period-over-Period Comparison').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='gs_cmp_period_frequency').options, ['Yearly'])

    def setUp(self):
        self.frame = pd.DataFrame({'year': [2020, 2021, 2022], 'value': [10, 15, 30]})
        self.metadata = apply_user_override(infer_metadata(self.frame), 'year', self.frame.year, 'Date / Time', 'Year')

    def derive(self, result, config, left=None, metadata=None, registry=None):
        return inherit_metadata(result, self.frame if left is None else left,
                                self.metadata if metadata is None else metadata, config, registry or {})

    def test_rename_then_clean_and_dataset_switch_preserve_year(self):
        renamed = self.frame.rename(columns={'year': 'annual'})
        metadata, warnings = self.derive(renamed, {'operation': 'Rename column', 'column': 'year', 'name': 'annual'})
        cleaned = renamed.copy()
        cleaned['value'] *= 2
        metadata, warnings = self.derive(cleaned, {'operation': 'Replace Values'}, renamed, metadata)
        self.assertFalse(warnings)
        state = {}
        source = add_dataset(state, self.frame, 'Source', 'Test', metadata=self.metadata)
        activate(state, source)
        output = add_dataset(state, cleaned, 'Result', 'Transform', parents=[source], metadata=metadata)
        activate(state, output)
        self.assertEqual(state['variable_dataset_id'], output)
        self.assertEqual(state['variable_metadata']['annual']['source'], 'User override')
        self.assertEqual(effective_data(cleaned, state['variable_metadata'])[1]['year'], ['annual'])
        state['variable_metadata']['annual']['variable_subtype'] = 'Month'
        activate(state, source)
        self.assertEqual(state['variable_metadata']['year']['variable_subtype'], 'Year')

    def test_invalid_values_fall_back_with_warning(self):
        invalid = self.frame.copy()
        invalid['year'] = ['not a year'] * 3
        metadata, warnings = self.derive(invalid, {'operation': 'Replace Values'})
        self.assertEqual(metadata['year']['source'], 'Automatic')
        self.assertEqual(len(warnings), 1)

    def test_copy_preserves_but_calculation_and_aggregate_are_inferred(self):
        copied = self.frame.assign(copy=self.frame.year)
        metadata, _ = self.derive(copied, {'operation': 'Add column', 'method': 'Copy existing column', 'source': 'year', 'name': 'copy'})
        self.assertEqual(metadata['copy']['variable_subtype'], 'Year')
        metadata, _ = self.derive(copied, {'operation': 'Add column', 'method': 'Calculation', 'name': 'copy'})
        self.assertEqual(metadata['copy']['source'], 'Automatic')
        result = aggregate_table(self.frame, ['year'], [{'column': 'value', 'function': 'Sum', 'name': 'total'}])
        metadata, _ = self.derive(result, {'operation': 'Aggregate', 'groups': ['year']})
        self.assertEqual(metadata['year']['variable_subtype'], 'Year')
        self.assertEqual(metadata['total']['source'], 'Automatic')

    def test_join_suffixes_and_shared_keys(self):
        right = self.frame.copy()
        registry = {'right': {'frame': right, 'state': {'variable_metadata': self.metadata}}}
        result, _ = join_tables(self.frame, right, ['value'], ['value'])
        metadata, warnings = self.derive(result, {'operation': 'Join', 'right': 'right', 'left_keys': ['value'], 'right_keys': ['value']}, registry=registry)
        self.assertFalse(warnings)
        self.assertEqual(metadata['year_left']['variable_subtype'], 'Year')
        self.assertEqual(metadata['year_right']['variable_subtype'], 'Year')
        result, _ = join_tables(self.frame, right, ['year'], ['year'])
        metadata, warnings = self.derive(result, {'operation': 'Join', 'right': 'right', 'left_keys': ['year'], 'right_keys': ['year']}, registry=registry)
        self.assertFalse(warnings)
        self.assertEqual(metadata['year']['variable_subtype'], 'Year')

    def test_restructured_and_converted_columns_do_not_inherit_old_roles(self):
        pivoted = self.frame.rename(columns={'value': 'new_measure'})
        metadata, _ = self.derive(pivoted, {'operation': 'Pivot', 'rows': ['year']})
        self.assertEqual(metadata['year']['variable_subtype'], 'Year')
        self.assertEqual(metadata['new_measure']['source'], 'Automatic')
        unpivoted = self.frame.assign(variable='value')
        metadata, _ = self.derive(unpivoted, {'operation': 'Unpivot', 'identifiers': ['year']})
        self.assertEqual(metadata['year']['variable_subtype'], 'Year')
        self.assertEqual(metadata['variable']['source'], 'Automatic')
        metadata, warnings = self.derive(self.frame, {'operation': 'Convert Currency to Numeric', 'columns': ['year']})
        self.assertFalse(warnings)
        self.assertEqual(metadata['year']['variable_type'], 'Numerical')

    def test_append_valid_override_and_conflicting_overrides(self):
        right = self.frame.copy()
        result, _ = append_tables([self.frame, right], ['left', 'right'])
        other = infer_metadata(right)
        registry = {'right': {'frame': right, 'state': {'variable_metadata': other}}}
        config = {'operation': 'Append', 'inputs': ['right'], 'source_column': None}
        metadata, warnings = self.derive(result, config, registry=registry)
        self.assertFalse(warnings)
        self.assertEqual(metadata['year']['variable_subtype'], 'Year')
        other['year'] = dict(variable_type='Categorical', variable_subtype='Nominal', order=[], source='User override')
        metadata, warnings = self.derive(result, config, registry=registry)
        self.assertEqual(metadata['year']['source'], 'Automatic')
        self.assertIn('conflicting', warnings[0])
