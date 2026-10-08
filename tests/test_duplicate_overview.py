import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest


FIXTURE = '''
import streamlit as st
from utils.duplicate_overview import render_row_overview
render_row_overview(st.session_state['frame'], 'Heights')
'''


class DuplicateOverviewTests(unittest.TestCase):
    def heights(self):
        return pd.DataFrame([
            [78.5, 67, 69, 'female'], [75.5, 66.5, 65.5, 'female'],
            [75, 58.5, 62.5, 'female'], [74, 68, 73, 'male'],
            [74, 62, 67, 'female'], [73, 65, 70.2, 'male'],
            [72, 65, 65, 'female'], [72, 63, 70.5, 'male'],
            [72, 63, 69, 'male'], [72, 63, 63, 'female'],
        ], columns=['Father_height', 'Mother_height', 'Child_height', 'gender'],
            index=[3, 7, 20, 25, 37, 52, 88, 93, 97, 99])

    def app(self, frame):
        app = AppTest.from_string(FIXTURE)
        app.session_state['frame'] = frame
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_screenshot_rows_are_unique_when_no_earlier_matches_exist(self):
        app = self.app(self.heights())
        self.assertEqual(app.metric[1].value, '0')
        self.assertEqual(len(app.dataframe), 0)

    def test_preview_shows_earlier_matches_even_with_repeated_indices(self):
        heights = self.heights()
        frame = pd.concat([heights, heights])
        app = self.app(frame)
        self.assertEqual(app.metric[1].value, '10')
        pd.testing.assert_frame_equal(app.dataframe[0].value, heights)
        pd.testing.assert_frame_equal(app.dataframe[1].value, heights)

    def test_preview_matches_source_beyond_first_100_rows_and_missing_values(self):
        frame = pd.DataFrame({'x': list(range(150)) + [None, None, 140],
                              'y': ['a'] * 150 + ['b', 'b', 'a']})
        app = self.app(frame)
        self.assertEqual(app.metric[1].value, '2')
        pd.testing.assert_frame_equal(app.dataframe[0].value, frame.iloc[[140, 150]])
        pd.testing.assert_frame_equal(app.dataframe[1].value, frame.iloc[[151, 152]])


if __name__ == '__main__':
    unittest.main()
