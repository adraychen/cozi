import unittest
import pandas as pd
from utils.temporal import parse_temporal, temporal_value
from utils.variable_metadata import infer_metadata, apply_user_override, effective_data, validate_type_subtype
from utils.variable_overview_display import value_details


class TemporalComponentTests(unittest.TestCase):
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
        self.assertEqual(metadata['year']['variable_type'], 'Numerical')
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
