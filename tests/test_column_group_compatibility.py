from pathlib import Path
from unittest.mock import patch
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils import database, variable_metadata_ui


class ColumnGroupCompatibilityTests(unittest.TestCase):
    def test_full_app_with_legacy_column_overview(self):
        original = variable_metadata_ui.render_column_overview
        def legacy_overview(*args, **kwargs):
            data, groups = original(*args, **kwargs)
            groups.pop('temporal', None)
            return data, groups
        with patch.object(database,'record_visit'), patch.object(database,'get_stats',return_value={'visits':0,'likes':0,'dislikes':0}), patch.object(variable_metadata_ui,'render_column_overview',side_effect=legacy_overview):
            for frame in [pd.DataFrame({'value':[1.,2.,3.]}),
                          pd.DataFrame({'value':[1.,2.,3.], 'date':pd.date_range('2024-01-01',periods=3),
                                        'clock':['08:00','09:00','10:00'],'duration':['1 hour','2 hours','3 hours']})]:
                with self.subTest(columns=list(frame)):
                    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'))
                    app.session_state['df']=frame
                    app.session_state['file_id']='legacy-groups'
                    app.run(timeout=90)
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state['col_types']['temporal'],[c for c in ['date','clock','duration'] if c in frame])

    def test_column_overview_with_legacy_effective_data(self):
        original=variable_metadata_ui.effective_data
        def legacy_data(*args, **kwargs):
            data, groups=original(*args, **kwargs)
            groups.pop('temporal',None)
            return data,groups
        with patch.object(variable_metadata_ui,'effective_data',side_effect=legacy_data):
            app=AppTest.from_string('''
import pandas as pd
from utils.variable_metadata_ui import render_column_overview
from utils.descriptive_controls import select_descriptive_variable
frame=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=3)})
data,groups=render_column_overview(frame,'legacy')
assert groups['temporal']==['date']
select_descriptive_variable({'numerical':[],'categorical':[],'datetime':['date']})
''').run(timeout=45)
            self.assertFalse(app.exception)
            self.assertTrue(app.checkbox(key='dist_profile_column_date').value)

if __name__=='__main__':unittest.main()
