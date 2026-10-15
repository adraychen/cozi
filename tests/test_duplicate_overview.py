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
        table = app.dataframe[0].value
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(table['Row index'].tolist(), [i for i in heights.index for _ in range(2)])
        self.assertEqual(table['Row status'].tolist(), ['Earlier matching', 'Duplicate'] * len(heights))
        pd.testing.assert_frame_equal(table[heights.columns], frame.iloc[[i for n in range(10) for i in (n, n + 10)]].reset_index(drop=True))

    def test_preview_matches_source_beyond_first_100_rows_and_missing_values(self):
        frame = pd.DataFrame({'x': list(range(150)) + [None, None, 140],
                              'y': ['a'] * 150 + ['b', 'b', 'a']})
        app = self.app(frame)
        self.assertEqual(app.metric[1].value, '2')
        table = app.dataframe[0].value
        self.assertEqual(table['Row index'].tolist(), [140, 152, 150, 151])
        self.assertEqual(table['Row status'].tolist(), ['Earlier matching', 'Duplicate'] * 2)
        pd.testing.assert_frame_equal(table[frame.columns], frame.iloc[[140, 152, 150, 151]].reset_index(drop=True))

    def test_all_duplicates_are_shown_and_metadata_names_do_not_overwrite_data(self):
        frame = pd.DataFrame({'Row index': ['a'] * 105, 'Row status': [1] * 105})
        app = self.app(frame)
        table = app.dataframe[0].value
        self.assertEqual(len(table), 105)
        self.assertEqual(table['Row status (overview)'].tolist(), ['Earlier matching'] + ['Duplicate'] * 104)
        pd.testing.assert_frame_equal(table[frame.columns], frame)

    def test_name_is_hidden_until_removal_requested_and_cancel_hides_it(self):
        app = self.app(pd.DataFrame({'x': [1, 1]}))
        self.assertEqual(len(app.text_input), 0)
        app.button(key='view_remove_duplicate_rows').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input[0].value, 'Heights · Unique rows')
        app.text_input[0].set_value('   ').run()
        self.assertTrue(app.button(key='view_duplicate_rows_confirm').disabled)
        app.button(key='view_duplicate_rows_cancel').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.text_input), 0)

    def test_confirm_creates_clean_dataset_with_source_preserved(self):
        from unittest.mock import patch
        frame = pd.DataFrame({'x': [1, 1, 2]}, index=[5, 5, 9])
        app = self.app(frame)
        app.button(key='view_remove_duplicate_rows').click().run()
        app.text_input[0].set_value('Clean heights').run()
        with patch('utils.duplicate_overview._create_cleaned') as create:
            app.button(key='view_duplicate_rows_confirm').click().run()
        self.assertFalse(app.exception)
        create.assert_called_once()
        pd.testing.assert_frame_equal(create.call_args.args[0], pd.DataFrame({'x': [1, 2]}))
        self.assertEqual(create.call_args.args[1], 'Clean heights')
        pd.testing.assert_frame_equal(app.session_state['frame'], frame)


if __name__ == '__main__':
    unittest.main()
