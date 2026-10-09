import io
import unittest
from copy import deepcopy
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest
from utils.date_components import combine_date_components, component_metadata, component_preview, DateComponentError, STRICT, KEEP_MISSING
from utils.table_editing import add_column
from utils.transformation_metadata import inherit_metadata
from utils.variable_metadata import infer_metadata, effective_data
from utils.temporal_validation import assess_temporal_selection
from utils.period_comparison import interpret_time, build_timeline, calculate
from utils.dataset_views import dataset_csv


class DateComponentTests(unittest.TestCase):
    def test_month_formats_source_preservation_and_chronological_sorting(self):
        frame = pd.DataFrame({'year': ['2026', 2025, 2025., ' 2026 '], 'month': [' jan ', 'November', '12', '02']}, index=[9, 2, 7, 5])
        original = frame.copy(deep=True)
        result = add_column(frame, 'Month-Year', method='Combine Date Components', combination='Year + Month', year='year', month='month')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['Month-Year']))
        self.assertEqual(result['Month-Year'].dt.strftime('%Y-%m').tolist(), ['2026-01', '2025-11', '2025-12', '2026-02'])
        self.assertEqual(result.sort_values('Month-Year')['Month-Year'].dt.strftime('%Y-%m').tolist(), ['2025-11', '2025-12', '2026-01', '2026-02'])
        assert_frame_equal(frame, original)
        assert_frame_equal(result[frame.columns], original)

    def test_numeric_and_abbreviated_months(self):
        frame = pd.DataFrame({'y': [2026] * 6, 'm': [1, 2., '03', ' APR ', 'may', 'June']})
        values, issues = combine_date_components(frame, 'Year + Month', 'y', 'm')
        self.assertFalse(len(issues))
        self.assertEqual(values.dt.month.tolist(), [1, 2, 3, 4, 5, 6])

    def test_invalid_missing_components_block_or_remain_missing_explicitly(self):
        frame = pd.DataFrame({'y': [2026, 2026, 2026, None, 2025, True, 26], 'm': ['January', 13, 'Unknown', 'March', None, 1, 1]})
        with self.assertRaises(DateComponentError) as error:
            combine_date_components(frame, 'Year + Month', 'y', 'm')
        self.assertEqual(len(error.exception.issues), 6)
        self.assertIn('Missing year', error.exception.issues.Reason.tolist())
        self.assertIn('Invalid month', error.exception.issues.Reason.tolist())
        values, issues = combine_date_components(frame, 'Year + Month', 'y', 'm', missing_policy=KEEP_MISSING)
        self.assertEqual(values.notna().sum(), 1)
        self.assertEqual(values.iloc[0], pd.Timestamp('2026-01-01'))
        self.assertEqual(len(issues), 6)

    def test_complete_dates_leap_year_and_invalid_days(self):
        frame = pd.DataFrame({'y': [2024, 2025, 2026, 2026], 'm': [2, 2, 1, 4], 'd': [29, 29, '15', 31]})
        values, issues = combine_date_components(frame, 'Year + Month + Day', 'y', 'm', day='d', missing_policy=KEEP_MISSING)
        self.assertEqual(values.iloc[0], pd.Timestamp('2024-02-29'))
        self.assertEqual(values.iloc[2], pd.Timestamp('2026-01-15'))
        self.assertTrue(pd.isna(values.iloc[1]))
        self.assertTrue(pd.isna(values.iloc[3]))
        self.assertEqual(len(issues), 2)

    def test_quarters_metadata_and_reusable_transformation(self):
        frame = pd.DataFrame({'y': [2025, 2026], 'q': [' q4 ', 1]})
        config = {'operation': 'Add column', 'method': 'Combine Date Components', 'combination': 'Year + Quarter',
                  'name': 'Reporting Quarter', 'year': 'y', 'quarter': 'q', 'display_format': '[Q]Q YYYY'}
        result = add_column(frame, config['name'], **{k: v for k, v in config.items() if k not in ('operation', 'name')})
        metadata, warnings = inherit_metadata(result, frame, infer_metadata(frame), config, {})
        self.assertFalse(warnings)
        self.assertEqual(metadata['Reporting Quarter']['variable_subtype'], 'Year-Quarter')
        self.assertEqual(component_preview(result, metadata)['Reporting Quarter'].tolist(), ['Q4 2025', 'Q1 2026'])
        data, groups = effective_data(result, metadata)
        self.assertEqual(groups['year_quarter'], ['Reporting Quarter'])
        self.assertEqual(data['Reporting Quarter'].iloc[0], pd.Period('2025Q4'))

    def test_overwrite_duplicate_sources_bad_format_and_missing_source_rejected(self):
        frame = pd.DataFrame({'y': [2026], 'm': [1]})
        with self.assertRaisesRegex(ValueError, 'already exists'):
            add_column(frame, 'm', method='Combine Date Components', combination='Year + Month', year='y', month='m')
        with self.assertRaisesRegex(ValueError, 'distinct'):
            combine_date_components(frame, 'Year + Month', 'y', 'y')
        with self.assertRaisesRegex(ValueError, 'existing column'):
            combine_date_components(frame, 'Year + Month', 'absent', 'm')
        with self.assertRaisesRegex(ValueError, 'display format'):
            combine_date_components(frame, 'Year + Month', 'y', 'm', display_format='unsupported')

    def test_metadata_precision_survives_rename_preview_and_csv(self):
        frame = pd.DataFrame({'Year': [2025, 2026], 'Month': ['December', 'Jan']})
        config = {'operation': 'Add column', 'method': 'Combine Date Components', 'combination': 'Year + Month',
                  'name': 'Month-Year', 'year': 'Year', 'month': 'Month', 'display_format': 'MMM YYYY'}
        result = add_column(frame, 'Month-Year', **{k: v for k, v in config.items() if k not in ('name', 'operation')})
        metadata, _ = inherit_metadata(result, frame, infer_metadata(frame), config, {})
        self.assertEqual(component_preview(result, metadata)['Month-Year'].tolist(), ['Dec 2025', 'Jan 2026'])
        renamed = result.rename(columns={'Month-Year': 'Period'})
        renamed_metadata, _ = inherit_metadata(renamed, result, metadata, {'operation': 'Rename column', 'column': 'Month-Year', 'name': 'Period'}, {})
        self.assertEqual(renamed_metadata['Period']['variable_subtype'], 'Year-Month')
        self.assertEqual(renamed_metadata['Period']['display_format'], 'MMM YYYY')
        exported = pd.read_csv(io.BytesIO(dataset_csv(renamed, renamed_metadata)))
        self.assertEqual(exported.Period.tolist(), ['2025-12', '2026-01'])

    def test_repeated_transactions_are_preserved_and_correct_period_changes(self):
        frame = pd.DataFrame({'Year': [2025, 2025, 2026, 2026, 2026],
                              'Month': ['November', 'December', 'January', 'January', 'February'],
                              'Sales': [120000, 135000, 100000, 42000, 138000]})
        result = add_column(frame, 'Month-Year', method='Combine Date Components', combination='Year + Month', year='Year', month='Month')
        self.assertEqual(len(result), 5)
        plan = build_timeline(result, 'Sales', [], interpret_time(result['Month-Year'], 'Year-Month'), 'Monthly', 'Sum')
        table = calculate(plan['timeline'], plan['keys'])
        self.assertEqual(table.Period.tolist(), ['Nov 2025', 'Dec 2025', 'Jan 2026', 'Feb 2026'])
        self.assertEqual(table.Value.tolist(), [120000, 135000, 142000, 138000])
        self.assertEqual(table.Change.iloc[2], 7000)
        self.assertAlmostEqual(table['Change %'].iloc[2], 7000 / 135000 * 100)


class TemporalGuidanceTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({'Year': [2025, 2025, 2026, 2026], 'Month': ['January', 'February'] * 2,
                             'Sales': [100, 120, 135, 142]})

    def test_single_year_needs_no_warning(self):
        frame = self.fixture().iloc[:2]
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Supported')
        self.assertEqual(assessment.years, [2025])
        self.assertFalse(assessment.message)

    def test_multiple_years_block_chronology_but_allow_explicit_seasonality(self):
        frame = self.fixture()
        original = frame.copy(deep=True)
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Blocked')
        self.assertEqual(assessment.year_column, 'Year')
        self.assertEqual(assessment.repeated_months, [1, 2])
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame), objective='Monthly seasonal profile (pool years)')
        self.assertEqual(assessment.status, 'Warning')
        self.assertIn('intentionally', assessment.message)
        axis = interpret_time(frame.Month, 'Month')
        with self.assertRaisesRegex(ValueError, 'multiple years'):
            build_timeline(frame, 'Sales', [], axis, 'Monthly', 'Sum')
        planned = build_timeline(frame, 'Sales', [], axis, 'Monthly', 'Sum', objective='Monthly seasonal profile (pool years)')
        self.assertEqual(planned['timeline'].Value.tolist(), [235, 262])
        assert_frame_equal(frame, original)

    def test_existing_month_year_is_verified_not_just_named(self):
        frame = self.fixture()
        frame['Period'] = ['2025-01', '2025-02', '2026-01', '2026-02']
        frame['Month-Year'] = ['2020-01', '2020-02'] * 2
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.existing_columns, ['Period'])

    def test_dated_column_proves_multiple_years_when_no_year_component_exists(self):
        frame = self.fixture().drop(columns='Year')
        frame['Date'] = ['2025-01-15', '2025-02-15', '2026-01-15', '2026-02-15']
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Blocked')
        self.assertEqual(assessment.existing_columns, ['Date'])

    def test_no_year_and_names_without_valid_values_are_not_guessed(self):
        frame = self.fixture().drop(columns='Year')
        frame['Fiscal Year'] = ['unknown'] * 4
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Warning')
        self.assertFalse(assessment.year_columns)
        self.assertIn('unavailable', assessment.message)

    def test_categorical_override_is_not_recommended_as_a_temporal_variable(self):
        frame = self.fixture()
        frame['Period'] = ['2025-01', '2025-02', '2026-01', '2026-02']
        metadata = infer_metadata(frame)
        metadata['Period'] = dict(variable_type='Categorical', variable_subtype='Nominal', order=[], source='User override')
        assessment = assess_temporal_selection(frame, 'Month', metadata)
        self.assertFalse(assessment.existing_columns)

    def test_conflicting_date_sources_offer_explicit_selection(self):
        frame = self.fixture().drop(columns='Year')
        frame['Reporting Date'] = ['2025-01-01', '2025-02-01', '2026-01-01', '2026-02-01']
        frame['Event Date'] = ['2024-01-01', '2024-02-01', '2025-01-01', '2025-02-01']
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Blocked')
        self.assertEqual(set(assessment.existing_columns), {'Reporting Date', 'Event Date'})
        self.assertIn('disagree', assessment.message)

    def test_ambiguous_year_columns_need_clarification(self):
        frame = self.fixture()
        frame['Fiscal Year'] = [2026] * 4
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Blocked')
        self.assertIsNone(assessment.year_column)
        self.assertIn('disagree', assessment.message)
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame), year_column='Year')
        self.assertEqual(assessment.year_column, 'Year')
        self.assertEqual(assessment.years, [2025, 2026])

    def test_multiple_years_without_repeated_months_still_need_dated_order(self):
        frame = pd.DataFrame({'Year': [2025, 2026], 'Month': ['December', 'January']})
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Blocked')
        self.assertFalse(assessment.repeated_months)

    def test_relevant_segments_and_incomplete_year_information(self):
        frame = self.fixture()
        frame['Segment'] = ['A', 'A', 'B', 'B']
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame), dimensions=['Segment'])
        self.assertEqual(assessment.status, 'Supported')
        frame.loc[0, 'Year'] = None
        assessment = assess_temporal_selection(frame, 'Month', infer_metadata(frame))
        self.assertEqual(assessment.status, 'Blocked')
        self.assertEqual(assessment.missing_year_rows, 1)


class DateComponentWorkspaceTests(unittest.TestCase):
    def app(self, text):
        from tests.test_dataset_workspace import FIXTURE, WorkspaceTests
        app = AppTest.from_string(FIXTURE).run(timeout=60)
        WorkspaceTests().add_manual(app, 'Components', text)
        return app

    def comparison(self, app):
        app.session_state['ws_workflow'] = 'Analyze'
        app.session_state['view_analyze_tabs'] = 'Comparison'
        app.run(timeout=60)
        app.checkbox(key='gs_cmp_variable_Month').check().run(timeout=60)
        app.checkbox(key='gs_cmp_variable_Sales').check().run(timeout=60)

    def test_guided_creation_return_and_dated_analysis(self):
        app = self.app('Year,Month,Sales\n2025,November,120000\n2025,December,135000\n2026,January,142000\n2026,February,138000')
        source_id = app.session_state['ws_active']
        original = app.session_state['df'].copy(deep=True)
        self.comparison(app)
        self.assertFalse(app.exception)
        self.assertFalse(any('Change %' in table.value for table in app.dataframe))
        app.button(key='temporal_create_button_temporal_guidance_gs_cmp_period_Month').click().run(timeout=60)
        self.assertEqual(app.session_state['ws_workflow'], 'Transform')
        self.assertEqual(app.selectbox(key='tf_add_method').value, 'Combine Date Components')
        self.assertEqual(app.selectbox(key='tf_add_year').value, 'Year')
        self.assertEqual(app.selectbox(key='tf_add_month').value, 'Month')
        self.assertEqual(app.text_input(key='tf_add_name').value, 'Month-Year')
        self.assertNotIn('Month-Year', app.session_state['df'])
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['tf_result']['metadata']['Month-Year']['variable_subtype'], 'Year-Month')
        preview = next(table.value for table in app.dataframe if 'Month-Year' in table.value)
        self.assertEqual(preview['Month-Year'].tolist(), ['2025-11', '2025-12', '2026-01', '2026-02'])
        app.button(key='tf_create').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state['ws_active'], source_id)
        assert_frame_equal(app.session_state['ws_datasets'][source_id]['frame'], original)
        app.button(key='temporal_resume_button').click().run(timeout=60)
        self.assertEqual(app.session_state['ws_workflow'], 'Analyze')
        self.assertFalse(any('Change %' in table.value for table in app.dataframe))
        app.button(key='temporal_select_button_return').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key='gs_cmp_variable_Month-Year').value)
        self.assertFalse(app.checkbox(key='gs_cmp_variable_Month').value)
        self.assertTrue(app.checkbox(key='gs_cmp_variable_Sales').value)
        table = next(table.value for table in app.dataframe if 'Change %' in table.value)
        self.assertEqual(table.Period.tolist(), ['Nov 2025', 'Dec 2025', 'Jan 2026', 'Feb 2026'])
        self.assertEqual(table.Change.iloc[2], 7000)

    def test_single_year_month_comparison_has_no_multi_year_warning(self):
        app = self.app('Year,Month,Sales\n2026,January,100\n2026,February,120')
        self.comparison(app)
        self.assertFalse(app.exception)
        self.assertFalse(any('multiple years' in warning.value or 'No unambiguous year' in warning.value for warning in app.warning))
        self.assertFalse(any(select.label == 'Analysis objective' for select in app.selectbox))
        table = next(table.value for table in app.dataframe if 'Change %' in table.value)
        self.assertEqual(table['Change %'].iloc[-1], 20)

    def test_ambiguous_year_choice_and_related_group_statistics_guidance(self):
        app = self.app('Year,Fiscal Year,Month,Sales\n2025,2026,January,100\n2026,2027,January,135')
        self.comparison(app)
        self.assertFalse(app.exception)
        year_key = 'temporal_guidance_gs_cmp_period_Month_year'
        self.assertEqual(app.selectbox(key=year_key).options, ['Choose the Year column…', 'Year', 'Fiscal Year'])
        self.assertFalse(any('Change %' in table.value for table in app.dataframe))
        app.selectbox(key=year_key).select('Year').run(timeout=60)
        app.button(key='temporal_create_button_temporal_guidance_gs_cmp_period_Month').click().run(timeout=60)
        self.assertEqual(app.selectbox(key='tf_add_year').value, 'Year')
        self.assertFalse(app.exception)
        app.session_state['ws_workflow'] = 'Analyze'
        app.session_state['view_analyze_tabs'] = 'Group Statistics'
        app.run(timeout=60)
        app.checkbox(key='gs_temporal_column_Month').check().run(timeout=60)
        app.checkbox(key='gs_measure_column_Sales').check().run(timeout=60)
        app.selectbox(key='gs_analysis').select('Running & Cumulative Statistics').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any('year sources disagree' in warning.value.lower() for warning in app.warning))
        self.assertFalse(any('Cumulative sum' in table.value for table in app.dataframe))

    def test_existing_variable_and_intentional_seasonal_profile(self):
        app = self.app('Year,Month,Sales,Period\n2025,January,100,2025-01\n2025,February,120,2025-02\n2026,January,135,2026-01\n2026,February,142,2026-02')
        self.comparison(app)
        self.assertFalse(app.exception)
        self.assertFalse(any(button.label == 'Create Month-Year Variable' for button in app.button))
        self.assertTrue(app.button(key='temporal_select_button_temporal_guidance_gs_cmp_period_Month_0'))
        objective_key = 'temporal_guidance_gs_cmp_period_Month_objective'
        app.selectbox(key=objective_key).select('Monthly seasonal profile (pool years)').run(timeout=60)
        table = next(table.value for table in app.dataframe if 'Change %' in table.value)
        self.assertEqual(table.Value.tolist(), [235, 262])
        self.assertIn('pooled', app.session_state['report_latest']['interpretation'])
        self.assertEqual(app.session_state['report_latest']['settings']['objective'], 'Monthly seasonal profile (pool years)')
        app.selectbox(key=objective_key).select('Chronological reporting periods').run(timeout=60)
        app.button(key='temporal_select_button_temporal_guidance_gs_cmp_period_Month_0').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key='gs_cmp_variable_Period').value)
        self.assertFalse(app.checkbox(key='gs_cmp_variable_Month').value)
        table = next(table.value for table in app.dataframe if 'Change %' in table.value)
        self.assertEqual(len(table), 14)
        self.assertEqual(table.Value.dropna().tolist(), [100, 120, 135, 142])

    def test_invalid_preview_explicit_missing_policy_and_draft_discard(self):
        from tests.test_dataset_workspace import select_operation
        app = self.app('Year,Month,Sales\n2026,Jan,10\n2026,Unknown,20\n2026,13,30')
        app.session_state['ws_workflow'] = 'Transform'
        app.run(timeout=60)
        select_operation(app, 'Add column')
        app.selectbox(key='tf_add_method').select('Combine Date Components').run(timeout=60)
        app.text_input(key='tf_add_name').set_value('Month-Year')
        app.selectbox(key='tf_add_year').select('Year')
        app.selectbox(key='tf_add_month').select('Month')
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertTrue(app.error)
        self.assertNotIn('tf_result', app.session_state)
        self.assertFalse(any(button.key == 'tf_create' for button in app.button))
        app.selectbox(key='tf_add_missing_policy').select(KEEP_MISSING).run(timeout=60)
        app.button(key='tf_preview').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['tf_result']['frame']['Month-Year'].isna().sum(), 2)
        app.button(key='tf_apply').click().run(timeout=60)
        self.assertEqual(len(app.session_state['tf_draft']['steps']), 1)
        self.assertNotIn('Month-Year', app.session_state['df'])
        app.button(key='tf_draft_discard').click().run(timeout=60)
        self.assertNotIn('tf_draft', app.session_state)
        self.assertNotIn('Month-Year', app.session_state['df'])
