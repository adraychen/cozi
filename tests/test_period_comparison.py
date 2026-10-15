from copy import deepcopy
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest
from utils.period_comparison import interpret_time, frequency_rules, build_timeline, calculate, interpret_result
from utils.period_comparison_ui import comparison_charts
from utils.variable_metadata import infer_metadata, effective_data


class PeriodComparisonTests(unittest.TestCase):
    def run_result(self, periods, values, frequency='Monthly', subtype=None, zero=False, statistic='Sum', **kwargs):
        frame = pd.DataFrame({'time': periods, 'metric': values})
        axis = interpret_time(frame.time, subtype)
        plan = build_timeline(frame, 'metric', [], axis, frequency, statistic, **kwargs)
        return calculate(plan['timeline'], plan['keys'], zero), plan

    def test_month_labels_order_and_six_observed_range_gaps(self):
        result, plan = self.run_result(['October', 'January', 'July', 'April'], [40, 10, 30, 20])
        self.assertEqual(result.Period.tolist(), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October'])
        self.assertEqual(plan['gaps'].Period.tolist(), ['February', 'March', 'May', 'June', 'August', 'September'])
        self.assertEqual(result.Value.notna().sum(), 4)
        self.assertTrue(result.loc[~result['Missing period'], 'Previous'].isna().all())
        self.assertNotIn('0001', result.to_csv())
        self.assertIn('nonconsecutive', interpret_result(result, 'Monthly'))

    def test_abbreviations_preserved_and_same_months_pooled(self):
        result, plan = self.run_result(['Feb', 'Jan', 'JAN'], [5, 10, 20])
        self.assertEqual(result.Period.tolist(), ['Jan', 'Feb'])
        self.assertEqual(result.Value.tolist(), [30, 5])
        self.assertAlmostEqual(result['Change %'].iloc[-1], -100 * 25 / 30)
        self.assertIn('pooled', ' '.join(plan['warnings']))

    def test_year_month_forms_and_december_january_transition(self):
        for labels in [['2025-12', '2026-01'], ['December 2025', 'Jan 2026'], ['2025 Dec', '2026 January']]:
            with self.subTest(labels=labels):
                result, plan = self.run_result(labels, [10, 15])
                self.assertEqual(plan['axis'].kind, 'Year-Month')
                self.assertEqual(result.Period.tolist(), ['Dec 2025', 'Jan 2026'])
                self.assertEqual(result['Change %'].iloc[-1], 50)
                self.assertTrue(plan['gaps'].empty)

    def test_quarter_labels_and_year_quarters(self):
        for labels, subtype, expected in [(['Q4', 'Q1'], None, ['Q1', 'Q2', 'Q3', 'Q4']),
                                          (['2025-Q4', '2026-Q1'], None, ['Q4 2025', 'Q1 2026']),
                                          (['Q4 2025', 'Q1 2026'], None, ['Q4 2025', 'Q1 2026'])]:
            with self.subTest(labels=labels):
                result, _ = self.run_result(labels, [10, 20], 'Quarterly', subtype)
                self.assertEqual(result.Period.tolist(), expected)
        frame = pd.DataFrame({'quarter': ['2026-Q1', '2026-Q2'], 'v': [1, 2]})
        metadata = infer_metadata(frame)
        data, groups = effective_data(frame, metadata)
        self.assertEqual(metadata['quarter']['variable_subtype'], 'Year-Quarter')
        self.assertEqual(groups['year_quarter'], ['quarter'])
        self.assertEqual(interpret_time(data.quarter, 'Year-Quarter').values.iloc[0], pd.Period('2026Q1'))

    def test_year_only_and_numeric_year_column(self):
        for labels in [['2024', '2025'], [2024, 2025]]:
            result, _ = self.run_result(labels, [10, 20], 'Yearly')
            self.assertEqual(result.Period.tolist(), ['2024', '2025'])
            self.assertEqual(result['Change %'].iloc[-1], 100)
        metadata = infer_metadata(pd.DataFrame({'Year': [2024, 2025], 'id': [2024, 2025]}))
        self.assertEqual(metadata['Year']['variable_subtype'], 'Year')
        self.assertEqual(metadata['id']['variable_type'], 'Numerical')

    def test_frequency_states_and_unavailable_execution(self):
        cases = [('Month', ['Jan', 'Feb'], ['Monthly', 'Quarterly']), ('Quarter', ['Q1', 'Q2'], ['Quarterly']),
                 ('Year', [2024, 2025], ['Yearly']), ('Year-Month', ['2024-01', '2024-02'], ['Monthly', 'Quarterly', 'Yearly']),
                 ('Year-Quarter', ['2024-Q1', '2024-Q2'], ['Quarterly', 'Yearly'])]
        for subtype, values, supported in cases:
            axis = interpret_time(pd.Series(values), subtype)
            rules = frequency_rules(axis)
            self.assertEqual([k for k, v in rules.items() if v[0] != 'Unavailable'], supported)
            if subtype == 'Month':
                self.assertEqual(rules['Quarterly'][0], 'Warning')
                with self.assertRaises(ValueError):
                    build_timeline(pd.DataFrame({'t': values, 'v': [1, 2]}), 'v', [], axis, 'Yearly', 'Sum')
            self.assertTrue(all(status == 'Unavailable' for status, _ in frequency_rules(axis, 'Period-end value').values()))

    def test_quarterly_clarification_and_partial_aggregation(self):
        with self.assertRaisesRegex(ValueError, 'Clarify'):
            self.run_result(['January', 'April'], [10, 20], 'Quarterly')
        result, plan = self.run_result(['January', 'April'], [10, 20], 'Quarterly', month_reporting='Monthly observations')
        self.assertEqual(result['Change %'].iloc[-1], 100)
        self.assertIn('partial-quarter', ' '.join(plan['warnings']))
        result, plan = self.run_result(['January', 'April'], [10, 20], 'Quarterly', month_reporting='Quarterly reports labelled by month')
        self.assertEqual(result.Period.tolist(), ['Q1', 'Q2'])
        with self.assertRaisesRegex(ValueError, 'Multiple month'):
            self.run_result(['January', 'February'], [10, 20], 'Quarterly', month_reporting='Quarterly reports labelled by month')
        result, _ = self.run_result(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], [10, 20, 30, 20, 40, 60], 'Quarterly', month_reporting='Monthly observations')
        self.assertEqual(result.Value.tolist(), [60, 120])

    def test_missing_periods_blank_values_actual_zero_separate(self):
        result, plan = self.run_result(['January', 'March', 'April', 'May'], [10, None, 0, 20], zero=True)
        self.assertEqual(plan['gaps'].Period.tolist(), ['February'])
        self.assertEqual(result.loc[result.Period.eq('February'), 'Value'].iloc[0], 0)
        self.assertTrue(pd.isna(result.loc[result.Period.eq('March'), 'Value'].iloc[0]))
        self.assertEqual(result.loc[result.Period.eq('April'), 'Value'].iloc[0], 0)
        self.assertEqual(result['Inserted zero'].sum(), 1)
        self.assertTrue(pd.isna(result.loc[result.Period.eq('May'), 'Change %'].iloc[0]))
        self.assertEqual(result.loc[result.Period.eq('May'), 'Change'].iloc[0], 20)
        self.assertEqual(result.loc[result.Period.eq('February'), 'Change %'].iloc[0], -100)

    def test_two_periods_single_period_and_empty(self):
        result, _ = self.run_result(['Jan'], [5])
        self.assertTrue(result['Change'].isna().all())
        self.assertIn('only one', interpret_result(result, 'Monthly'))
        result, _ = self.run_result(['Jan', 'Feb'], [5, 10])
        self.assertEqual(result['Change %'].iloc[-1], 100)
        with self.assertRaisesRegex(ValueError, 'No valid'):
            self.run_result([], [])

    def test_negative_and_zero_baselines(self):
        result, _ = self.run_result(['Jan', 'Feb', 'Mar', 'Apr'], [-10, -5, 0, 10])
        self.assertEqual(result['Change %'].iloc[1], 50)
        self.assertEqual(result['Change %'].iloc[2], 100)
        self.assertTrue(pd.isna(result['Change %'].iloc[3]))
        self.assertEqual(result['Change'].iloc[3], 10)
        self.assertFalse(np.isinf(result['Change %'].to_numpy()).any())

    def test_segments_have_independent_observed_ranges_and_gaps(self):
        df = pd.DataFrame({'t': ['Jan', 'Mar', 'Feb', 'Mar'], 'v': [10, 30, None, 5], 'g': ['A', 'A', 'B', 'B']})
        plan = build_timeline(df, 'v', ['g'], interpret_time(df.t), 'Monthly', 'Sum')
        result = calculate(plan['timeline'], plan['keys'], True)
        self.assertEqual(len(plan['gaps']), 1)
        self.assertEqual(result[result.Segment.eq('g: A')].Value.tolist(), [10, 0, 30])
        b = result[result.Segment.eq('g: B')]
        self.assertEqual(b.Period.tolist(), ['Feb', 'Mar'])
        self.assertTrue(pd.isna(b.Value.iloc[0]))
        self.assertTrue(pd.isna(b.Previous.iloc[-1]))

    def test_aggregation_and_source_immutability(self):
        df = pd.DataFrame({'t': ['2024-01-02', '2024-01-31', '2024-02-01'], 'v': [10., None, 20.]})
        original = df.copy(deep=True)
        for statistic, expected in [('Sum', 10), ('Mean', 10), ('Row count', 2), ('Non-missing count', 1)]:
            plan = build_timeline(df, 'v', [], interpret_time(df.t), 'Monthly', statistic)
            self.assertEqual(plan['timeline'].Value.iloc[0], expected)
        plan = build_timeline(df, 'v', [], interpret_time(df.t), 'Monthly', 'Period-end value')
        self.assertTrue(pd.isna(plan['timeline'].Value.iloc[0]))
        assert_frame_equal(df, original)
        conflicting = pd.DataFrame({'t': ['2024-01-31'] * 2, 'v': [10, 20]})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            build_timeline(conflicting, 'v', [], interpret_time(conflicting.t), 'Monthly', 'Period-end value')

    def test_year_over_year_strict_baseline(self):
        df = pd.DataFrame({'t': ['2024-01', '2025-01'], 'v': [10, 15]})
        plan = build_timeline(df, 'v', [], interpret_time(df.t), 'Monthly', 'Sum')
        result = calculate(plan['timeline'], plan['keys'], lag=12)
        self.assertEqual(result['Change %'].iloc[-1], 50)
        self.assertEqual(result['Previous period'].iloc[-1], 'Jan 2024')

    def test_invalid_mixed_formats_and_large_timeline_guard(self):
        with self.assertRaises(ValueError):
            interpret_time(pd.Series(['Jan', 'Q2', '2026', 'invalid']))
        with self.assertRaises(ValueError):
            interpret_time(pd.Series(['08:00', '09:00']))
        result, plan = self.run_result(['2024-01-01'] * 4 + ['invalid'], [1, 2, 3, 4, 100])
        self.assertEqual(plan['invalid'], 1)
        self.assertEqual(result.Value.iloc[0], 10)
        with self.assertRaisesRegex(ValueError, 'Too many'):
            self.large_segments()
        with self.assertRaisesRegex(ValueError, 'different temporal precision'):
            interpret_time(pd.Series(['2026-01', '2026-02-01', '2026-03-01', '2026-04-01', '2026-05-01']))

    def large_segments(self):
        df = pd.DataFrame({'t': ['1000', '9999'] * 30, 'v': [1] * 60, 'g': np.repeat(range(30), 2)})
        return build_timeline(df, 'v', ['g'], interpret_time(df.t), 'Yearly', 'Sum')

    def test_charts_gap_labels_values_and_percentage_signs(self):
        result, _ = self.run_result(['Jan', 'Feb', 'Apr'], [10, 20, 5])
        actual, changes = comparison_charts(result, 'Metric', 'Monthly')
        self.assertFalse(actual.data[0].connectgaps)
        self.assertEqual(list(actual.data[0].y), [10, 20, None, 5])
        self.assertEqual(list(actual.data[0].x), ['Jan', 'Feb', 'March', 'Apr'])
        self.assertEqual(changes.data[0].y[1], 100)
        self.assertEqual(changes.layout.shapes[0].y0, 0)
        self.assertIn('N/A', list(changes.data[1].text))
        result, _ = self.run_result(['Jan', 'Feb', 'Mar'], [20, 10, 15])
        _, changes = comparison_charts(result, 'Metric', 'Monthly')
        self.assertEqual(list(changes.data[0].y), [None, -50, 50])

    def test_charts_nullable_numeric_missing_changes(self):
        result, _ = self.run_result(['Jan', 'Feb', 'Mar'], pd.Series([0, 10, 20], dtype='Int64'))
        for column in ['Value', 'Previous', 'Change', 'Change %']:
            result[column] = result[column].astype('Float64')
        original = result.copy(deep=True)
        actual, changes = comparison_charts(result, 'Metric', 'Monthly')
        self.assertEqual(list(actual.data[0].y), [0, 10, 20])
        self.assertEqual(list(changes.data[0].y), [None, None, 100])
        self.assertEqual(changes.data[0].customdata[0][1], 'N/A')
        self.assertEqual(changes.data[0].customdata[2][1], 10)
        self.assertEqual(list(changes.data[1].text), ['N/A', 'N/A'])
        assert_frame_equal(result, original)


UI_FIXTURE = '''
import pandas as pd
import streamlit as st
from utils.group_statistics_ui import render_group_statistics
st.session_state['file_id'] = 'period_fixture'
df = pd.DataFrame({'month':['January','April','July','October'], 'value':[100., 80., 120., 60.]})
if st.session_state.get('complete_fixture', False):
    df = pd.DataFrame({'month':['January','February','March'], 'value':[100., 110., 120.]})
st.session_state['variable_metadata'] = {'month':{'variable_subtype':'Month'}}
st.session_state.setdefault('period_measure_optional', 'value')
render_group_statistics(df, {'numerical':['value'],'categorical':[],'datetime':[], 'temporal':['month']},
                        analysis='Period-over-Period Comparison', key_prefix='period', show_heading=False)
'''


class PeriodComparisonUITests(unittest.TestCase):
    def test_workspace_quarter_and_abbreviated_month_classification(self):
        from tests.test_dataset_workspace import FIXTURE, WorkspaceTests
        helper = WorkspaceTests()
        app = AppTest.from_string(FIXTURE).run(timeout=60)
        helper.add_manual(app, 'Quarter Reports', 'quarter,value\n2025-Q4,10\n2026-Q1,15')
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['variable_metadata']['quarter']['variable_subtype'], 'Year-Quarter')
        app.session_state['ws_workflow'] = 'Analyze'
        app.session_state['view_analyze_tabs'] = 'Comparison'
        app.run(timeout=60)
        app.checkbox(key='gs_cmp_variable_quarter').check().run(timeout=60)
        app.checkbox(key='gs_cmp_variable_value').check().run(timeout=60)
        app.selectbox(key='gs_cmp_analysis').select('Period-over-Period Comparison').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='gs_cmp_period_frequency').options, ['Quarterly', 'Yearly'])
        table = next(item.value for item in app.dataframe if 'Change %' in item.value)
        self.assertEqual(table.Period.tolist(), ['Q4 2025', 'Q1 2026'])
        self.assertEqual(table['Change %'].iloc[-1], 50)
        app.session_state['ws_dataset_tabs'] = '＋ Add datasets'
        app.run(timeout=60)
        helper.add_manual(app, 'Monthly Reports', 'month,value\nJan,10\nApr,20\nJul,30\nOct,40')
        app.session_state['ws_workflow'] = 'Analyze'
        app.session_state['view_analyze_tabs'] = 'Comparison'
        app.run(timeout=60)
        app.checkbox(key='gs_cmp_variable_month').check().run(timeout=60)
        app.checkbox(key='gs_cmp_variable_value').check().run(timeout=60)
        app.selectbox(key='gs_cmp_analysis').select('Period-over-Period Comparison').run(timeout=60)
        self.assertFalse(app.exception)
        table = next(item.value for item in app.dataframe if 'Change %' in item.value)
        self.assertEqual(table.Period.iloc[0], 'Jan')
        self.assertEqual(table.Period.iloc[-1], 'Oct')
        self.assertEqual(table['Missing period'].sum(), 6)

    def test_missing_policy_charts_table_and_frequency_revalidation(self):
        app = AppTest.from_string(UI_FIXTURE).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='period_frequency').options, ['Monthly', 'Quarterly'])
        self.assertEqual(app.radio(key='period_missing_policy').value, 'Keep missing periods empty (recommended)')
        self.assertTrue(any('6 monthly periods' in item.value for item in app.warning))
        self.assertEqual(len(app.get('plotly_chart')), 2)
        table = next(item.value for item in app.dataframe if 'Change %' in item.value)
        self.assertEqual(table['Missing period'].sum(), 6)
        app.radio(key='period_missing_policy').set_value('Treat missing periods as zero').run(timeout=60)
        table = next(item.value for item in app.dataframe if 'Change %' in item.value)
        self.assertEqual(table['Inserted zero'].sum(), 6)
        self.assertEqual(table['Value'].isna().sum(), 0)
        app.selectbox(key='period_frequency').select('Quarterly').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any('Choose the reporting meaning' in item.value for item in app.info))
        self.assertFalse(any('Change %' in item.value for item in app.dataframe))
        app.selectbox(key='period_month_reporting').select('Quarterly reports labelled by month').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.radio)
        table = next(item.value for item in app.dataframe if 'Change %' in item.value)
        self.assertEqual(table.Period.tolist(), ['Q1', 'Q2', 'Q3', 'Q4'])
        self.assertEqual(table['Change %'].iloc[1], -20)
        self.assertEqual(app.session_state['report_latest']['settings']['frequency'], 'Quarterly')

    def test_no_gap_question_and_stale_zero_policy_ignored(self):
        app = AppTest.from_string(UI_FIXTURE).run(timeout=60)
        app.radio(key='period_missing_policy').set_value('Treat missing periods as zero').run(timeout=60)
        app.session_state['complete_fixture'] = True
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.radio)
        table = next(item.value for item in app.dataframe if 'Change %' in item.value)
        self.assertFalse(table['Inserted zero'].any())
        self.assertEqual(table['Change %'].iloc[1], 10)

    def test_report_preserves_all_output_when_settings_change(self):
        app = AppTest.from_string(UI_FIXTURE).run(timeout=60)
        app.button(key='report_add_period').click().run(timeout=60)
        report = app.session_state['cozy_report']
        snapshot = deepcopy(report['blocks'][0]['snapshot'])
        self.assertEqual(len(snapshot['charts']), 2)
        self.assertEqual(snapshot['variables']['temporal_column'], 'month')
        self.assertEqual(snapshot['settings']['frequency'], 'Monthly')
        self.assertEqual(snapshot['results']['Missing periods'], 6)
        self.assertIn('nonconsecutive', snapshot['interpretation'])
        self.assertEqual(len(snapshot['tables']), 1)
        self.assertEqual(snapshot['tables'][0]['columns'][0], 'Period')
        app.radio(key='period_missing_policy').set_value('Treat missing periods as zero').run(timeout=60)
        self.assertEqual(app.session_state['cozy_report']['blocks'][0]['snapshot'], snapshot)
        self.assertEqual(app.session_state['report_latest']['results']['Inserted zeros'], 6)
        self.assertNotEqual(app.session_state['report_latest']['charts'], snapshot['charts'])
        self.assertTrue(app.session_state['report_latest']['warnings'])
