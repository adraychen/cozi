import unittest
import pandas as pd
from utils.temporal import parse_temporal, temporal_value
from utils.variable_metadata import infer_metadata, apply_user_override, effective_data, validate_type_subtype
from utils.variable_overview_display import value_details


class TemporalComponentTests(unittest.TestCase):
    def test_month_details_refresh_existing_automatic_date_classification(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import calendar
import pandas as pd
import streamlit as st
from utils.variable_metadata_ui import render_column_overview
raw = pd.DataFrame({'Month Name': list(calendar.month_name)[1:] * 2})
if 'variable_metadata' not in st.session_state:
    st.session_state['variable_dataset_id'] = 'months'
    st.session_state['variable_metadata'] = {'Month Name': dict(variable_type='Date / Time', variable_subtype='Date', source='Automatic', order=[])}
render_column_overview(raw, 'months')
''').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['variable_metadata']['Month Name']['variable_subtype'], 'Month')
        details = next(e for e in app.expander if e.label == 'Unique value details')
        self.assertEqual(details.dataframe[0].value['Value'].tolist(),
                         ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        self.assertEqual(details.dataframe[0].value['Detail'].unique().tolist(), ['Value'])

    def test_month_names_are_components_and_details_list_unique_months(self):
        from utils.temporal import infer_temporal_subtype
        from utils.variable_metadata import ensure_metadata
        raw = pd.DataFrame({'Month Name': ['December', 'jan', ' January ', 'September', None]})
        self.assertEqual(infer_temporal_subtype(raw['Month Name']), 'Month')
        metadata = infer_metadata(raw)
        self.assertEqual(metadata['Month Name']['variable_subtype'], 'Month')
        heading, details = value_details(raw['Month Name'], metadata['Month Name'])
        self.assertEqual(heading, 'All unique months in calendar order')
        self.assertEqual(details, [('Value', 'January'), ('Value', 'September'), ('Value', 'December')])
        self.assertTrue(parse_temporal(raw['Month Name'], 'DateTime').isna().all())
        self.assertEqual(infer_temporal_subtype(pd.Series(['January 1, 2024', 'December 2, 2024'])), 'Date')
        state = {'variable_dataset_id': 'data', 'variable_metadata': {
            'Month Name': dict(variable_type='Date / Time', variable_subtype='Date', source='Automatic', order=[])}}
        ensure_metadata(state, raw, 'data')
        self.assertEqual(state['variable_metadata']['Month Name']['variable_subtype'], 'Month')
        override = apply_user_override(metadata, 'Month Name', raw['Month Name'], 'Categorical', 'Nominal')
        state['variable_metadata'] = override
        ensure_metadata(state, raw, 'data')
        self.assertEqual(state['variable_metadata']['Month Name']['variable_type'], 'Categorical')

    def test_inspect_year_override_is_available_in_relationship_and_comparison(self):
        from streamlit.testing.v1 import AppTest
        from unittest.mock import patch
        from tests.test_dataset_workspace import FIXTURE, WorkspaceTests
        helper = WorkspaceTests()
        with patch('utils.dataset_views.render_predictive'):
            app = AppTest.from_string(FIXTURE).run(timeout=60)
            helper.add_manual(app, 'Annual', 'year,value\n2020,10\n2021,15\n2022,30')
            app.selectbox(key='vm_column').select('year').run(timeout=60)
            next(s for s in app.selectbox if s.label == 'Variable Type').select('Categorical').run(timeout=60)
            app.button(key='vm_apply').click().run(timeout=60)
            next(s for s in app.selectbox if s.label == 'Variable Type').select('Date / Time').run(timeout=60)
            next(s for s in app.selectbox if s.label == 'Variable Subtype').select('Year').run(timeout=60)
            app.button(key='vm_apply').click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['variable_metadata']['year']['variable_subtype'], 'Year')
            app.session_state['ws_workflow'] = 'Analyze'
            app.session_state['view_analyze_tabs'] = 'Relationship'
            app.run(timeout=60)
            app.checkbox(key='mc_year').check().run(timeout=60)
            app.checkbox(key='mc_value').check().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='rel_period').options, ['Yearly'])
            trend = next(d.value for d in app.dataframe if 'Linear trend' in d.value)
            self.assertEqual(trend['Year'].tolist(), [2020,2021,2022])
            app.session_state['view_analyze_tabs'] = 'Comparison'
            app.run(timeout=60)
            app.checkbox(key='gs_cmp_variable_year').check().run(timeout=60)
            app.checkbox(key='gs_cmp_variable_value').check().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='gs_cmp_period_frequency').options, ['Yearly'])
            table = next(d.value for d in app.dataframe if 'Change %' in d.value)
            self.assertEqual(table.Period.tolist(), ['2020','2021','2022'])
            self.assertAlmostEqual(table['Change %'].iloc[-1], 100)
            self.assertEqual(app.session_state['col_types']['year'], ['year'])
            self.assertNotIn('year', app.session_state['col_types']['categorical'])
            app.session_state['view_analyze_tabs'] = 'Group Statistics'
            app.run(timeout=60)
            app.checkbox(key='gs_temporal_column_year').check().run(timeout=60)
            app.checkbox(key='gs_measure_column_value').check().run(timeout=60)
            app.selectbox(key='gs_analysis').select('Running & Cumulative Statistics').run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='gs_frequency').options, ['Yearly'])
            table = next(d.value for d in app.dataframe if 'Cumulative sum' in d.value)
            self.assertEqual(table.Period.tolist(), ['2020', '2021', '2022'])
            self.assertEqual(table['Cumulative sum'].tolist(), [10, 25, 55])

    def test_year_and_month_parsing(self):
        years = parse_temporal(pd.Series([2024, 2025., '2026', None, '26', '2026-01-01', 2024.5, True]), 'Year')
        self.assertEqual(years.dropna().tolist(), [2024,2025,2026])
        months = parse_temporal(pd.Series(['January','jan',' SEPT ', 12, 1., '02', None, 0, 13, 1.5, '2026-09', True]), 'Month')
        self.assertEqual(months.dropna().tolist(), [1,1,9,12,1,2])
        self.assertEqual(temporal_value(9, 'Month'), 'September')
        self.assertEqual(temporal_value(2024, 'Year'), '2024')

    def test_year_month_and_no_silent_truncation(self):
        values = parse_temporal(pd.Series(['2026-09','2024/02',None,'2026-13','2026-09-01','September',202609]), 'Year-Month')
        self.assertEqual(values.dropna().tolist(), [pd.Timestamp('2026-09-01'), pd.Timestamp('2024-02-01')])
        self.assertEqual(temporal_value(values.iloc[0], 'Year-Month'), '2026-09')
        for subtype, values in [('Year', [2024.5, 12]), ('Month',[0,13]), ('Year-Month',['2026-09-30'])]:
            with self.assertRaises(ValueError):
                validate_type_subtype(pd.Series(values), 'Date / Time', subtype)

    def test_metadata_and_calendar_routing_preserve_original(self):
        raw = pd.DataFrame({'year':[2024,2025], 'month':['Jan','February'], 'period':['2024-01','2025-02']})
        original = raw.copy(deep=True)
        metadata = infer_metadata(raw)
        self.assertEqual(metadata['year']['variable_subtype'], 'Year')
        for column, subtype in [('year','Year'),('month','Month'),('period','Year-Month')]:
            metadata = apply_user_override(metadata, column, raw[column], 'Date / Time', subtype)
        data, groups = effective_data(raw, metadata)
        self.assertEqual(groups['year'], ['year'])
        self.assertEqual(groups['month'], ['month'])
        self.assertEqual(groups['datetime'], ['period'])
        self.assertEqual(groups['temporal'], ['year','month','period'])
        self.assertEqual(data.month.tolist(), [1,2])
        self.assertEqual(value_details(raw.period, metadata['period'])[1][0][1], '2024-01')
        pd.testing.assert_frame_equal(raw, original)

    def test_editor_and_component_views(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import pandas as pd
import streamlit as st
from utils.variable_metadata_ui import render_column_overview
from utils.temporal_ui import render_noncalendar_temporal
raw = pd.DataFrame({'year':[2024,2025], 'month':['Jan','Feb'], 'period':['2024-01','2025-02']})
data, groups = render_column_overview(raw, 'components')
for c in groups['temporal']:
    subtype = st.session_state['variable_metadata'][c]['variable_subtype']
    if subtype in ('Year','Month','Year-Month'):
        render_noncalendar_temporal(data[c], subtype, c)
''').run(timeout=40)
        for column, subtype in [('year','Year'),('month','Month'),('period','Year-Month')]:
            app.selectbox(key='vm_column').select(column).run()
            next(s for s in app.selectbox if s.label == 'Variable Type').select('Date / Time').run()
            next(s for s in app.selectbox if s.label == 'Variable Subtype').select(subtype).run()
            if not app.button(key='vm_apply').disabled:
                app.button(key='vm_apply').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.session_state['variable_metadata'][column]['variable_subtype'], subtype)
        self.assertEqual(len(app.get('plotly_chart')),3)

    def test_year_month_period_controls(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import pandas as pd
import streamlit as st
from utils.group_statistics_ui import render_group_statistics
st.session_state['variable_metadata'] = {'period':{'variable_subtype':'Year-Month'}}
st.session_state['period_frequency'] = 'Daily'
df = pd.DataFrame({'period':pd.to_datetime(['2024-01-01','2024-02-01']), 'value':[1,2]})
render_group_statistics(df, {'numerical':['value'],'categorical':[],'datetime':['period']}, analysis='Period-over-Period Comparison', key_prefix='period')
''').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='period_frequency').options, ['Monthly','Quarterly','Yearly'])


if __name__ == '__main__':
    unittest.main()
