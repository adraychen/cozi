import datetime as dt
import unittest
import pandas as pd
from utils.temporal import infer_temporal_subtype, parse_temporal
from utils.variable_metadata import infer_metadata, effective_data, apply_user_override, ensure_metadata, validate_type_subtype, get_allowed_subtypes
from utils.variable_overview_display import value_details


class TemporalSubtypeTests(unittest.TestCase):
    def test_inference(self):
        cases = [(['2024-01-01','2024-01-02'],'Date'),
                 (['2024-01-01 00:00:00','2024-01-02 00:00:00'],'DateTime'),
                 (['2024-01-01 12:30:00','2024-01-02 18:00:00'],'DateTime'),
                 (['08:30','17:45'],'Time'), (['8:30 AM','11:00 PM'],'Time'),
                 ([dt.time(9,10),dt.time(13,20)],'Time'),
                 (['1 day','2 hours'],'Duration'), (['PT2H','P1DT3H'],'Duration'),
                 (pd.to_timedelta(['1 day','3 days']),'Duration'),
                 (pd.date_range('2024-01-01',periods=2),'Date')]
        for values, expected in cases:
            with self.subTest(values=values):
                s=pd.Series(values)
                self.assertEqual(infer_temporal_subtype(s),expected)
                m=infer_metadata(pd.DataFrame({'x':s}))['x']
                self.assertEqual((m['variable_type'],m['variable_subtype']),('Date / Time',expected))
        self.assertIsNone(infer_temporal_subtype(pd.Series([0,1,2])))
        self.assertIsNone(infer_temporal_subtype(pd.Series(['red','blue'])))

    def test_parsing_and_routing(self):
        raw=pd.DataFrame({'date':['2024-01-01','2024-01-02'], 'stamp':['2024-01-01T10:00','2024-01-02T12:00'],
                          'clock':['08:15:30.123','23:59'], 'elapsed':['1 day','2 hours']})
        metadata=infer_metadata(raw)
        data,groups=effective_data(raw,metadata)
        self.assertEqual(groups['datetime'],['date','stamp'])
        self.assertEqual(groups['time'],['clock'])
        self.assertEqual(groups['duration'],['elapsed'])
        self.assertEqual(groups['temporal'],list(raw))
        self.assertEqual(data.clock.iloc[0],pd.Timedelta(hours=8,minutes=15,seconds=30,milliseconds=123))
        self.assertEqual(data.elapsed.iloc[0],pd.Timedelta(days=1))
        self.assertEqual(raw.clock.iloc[0],'08:15:30.123')
        heading,values=value_details(raw.clock,metadata['clock'])
        self.assertEqual(heading,'Time range')
        self.assertTrue(values[0][1].startswith('08:15'))

    def test_overrides_and_validation(self):
        self.assertEqual(get_allowed_subtypes('Date / Time'),['Date','Time','DateTime','Duration','Year','Month','Year-Month'])
        raw=pd.DataFrame({'x':['08:30','09:45']})
        m=apply_user_override(infer_metadata(raw),'x',raw.x,'Date / Time','Duration')
        data,groups=effective_data(raw,m)
        self.assertEqual(data.x.iloc[0],pd.Timedelta(hours=8,minutes=30))
        self.assertEqual(groups['datetime'],[])
        self.assertEqual(groups['duration'],['x'])
        for values,subtype in [([1,2],'Duration'),(['08:30','09:45'],'DateTime'),
                               (['2024-01-01','2024-01-02'],'Time'),
                               (['2024-01-01 12:30','2024-01-02 18:00'],'Date'),
                               (['2024-01-01']*9+['2024-01-02 01:00'],'Date'),
                               (['25:00','09:99'],'Time')]:
            with self.subTest(subtype=subtype,values=values):
                with self.assertRaises(ValueError):
                    validate_type_subtype(pd.Series(values),'Date / Time',subtype)
        validate_type_subtype(pd.Series(['08:30']*4+['bad']),'Date / Time','Time')
        self.assertEqual(parse_temporal(pd.Series(['08:30']*4+['bad']),'Time').isna().sum(),1)

    def test_legacy_metadata_and_timezone(self):
        df=pd.DataFrame({'date':['2024-01-01','2024-01-02']})
        state={'variable_dataset_id':'same','variable_metadata':{'date':{'variable_type':'Date / Time','variable_subtype':'Date / Time','source':'User override','order':[]}}}
        m=ensure_metadata(state,df,'same')
        self.assertEqual(m['date']['variable_subtype'],'Date')
        self.assertEqual(m['date']['source'],'User override')
        s=parse_temporal(pd.Series(['2024-01-01T12:00:00-05:00','2024-01-01T18:00:00Z']),'DateTime')
        self.assertEqual(s.iloc[0],pd.Timestamp('2024-01-01 17:00'))


class TemporalSubtypeUITests(unittest.TestCase):
    def test_editor_and_manual_override(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_string('''
import pandas as pd
import streamlit as st
from utils.variable_metadata_ui import render_column_overview
from utils.descriptive_controls import select_descriptive_variable
from utils.temporal_ui import render_noncalendar_temporal
raw=pd.DataFrame({'clock':['08:30','11:45','14:00'],'date':['2024-01-01','2024-01-02','2024-01-03']})
data,groups=render_column_overview(raw,'temporal-test')
column=select_descriptive_variable(groups)
if column and st.session_state['variable_metadata'][column]['variable_subtype'] in ('Time','Duration'):
    render_noncalendar_temporal(data[column],st.session_state['variable_metadata'][column]['variable_subtype'],'fixture')
''').run(timeout=40)
        self.assertFalse(app.exception)
        sub=next(s for s in app.selectbox if s.label=='Variable Subtype')
        self.assertEqual(sub.options,['Date','Time','DateTime','Duration','Year','Month','Year-Month'])
        self.assertEqual(sub.value,'Time')
        sub.select('Duration').run(timeout=40)
        app.button(key='vm_apply').click().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['variable_metadata']['clock']['variable_subtype'],'Duration')
        self.assertTrue(len(app.get('plotly_chart')))

if __name__=='__main__': unittest.main()
