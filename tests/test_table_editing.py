import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.table_editing import (
    rename_column, remove_columns, add_column, duplicate_columns,
    remove_duplicate_columns, remove_duplicate_rows,
)


class TableEditingTests(unittest.TestCase):
    def test_rename_remove_validation_and_source_preservation(self):
        frame = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[8, 6])
        original = frame.copy(deep=True)
        self.assertEqual(list(rename_column(frame, 'a', ' renamed ')), ['renamed', 'b'])
        self.assertEqual(list(remove_columns(frame, ['a'])), ['b'])
        for name in ('', ' ', 'b', 'a'):
            with self.assertRaises(ValueError):
                rename_column(frame, 'a', name)
        for columns in ([], ['absent'], ['a', 'b'], ['a', 'a']):
            with self.assertRaises(ValueError):
                remove_columns(frame, columns)
        assert_frame_equal(frame, original)

    def test_add_column_methods_and_index_alignment(self):
        frame = pd.DataFrame({'a': [2., None, 8.], 'b': [1., 2., 4.]}, index=[8, 3, 5])
        original = frame.copy(deep=True)
        self.assertTrue(add_column(frame, 'blank').blank.isna().all())
        self.assertEqual(add_column(frame, 'text', 'Constant value', value='001').text.tolist(), ['001'] * 3)
        self.assertEqual(add_column(frame, 'number', 'Constant value', value='2.5', value_type='Number').number.tolist(), [2.5] * 3)
        self.assertEqual(add_column(frame, 'flag', 'Constant value', value='False', value_type='True/False').flag.tolist(), [False] * 3)
        self.assertTrue(add_column(frame, 'copy', 'Copy existing column', source='a')['copy'].equals(frame.a))
        self.assertEqual(add_column(frame, 'row', 'Row number').row.tolist(), [1, 2, 3])
        result = add_column(frame, 'ratio', 'Calculation', left='a', right='b', calculation='Divide')
        self.assertEqual(result.ratio.iloc[0], 2)
        self.assertTrue(pd.isna(result.ratio.iloc[1]))
        self.assertEqual(result.ratio.iloc[2], 2)
        self.assertEqual(result.index.tolist(), [8, 3, 5])
        assert_frame_equal(frame, original)

    def test_add_validation(self):
        frame = pd.DataFrame({'a': [0, 1], 'text': ['x', 'y']})
        for name in ('', 'a'):
            with self.assertRaises(ValueError):
                add_column(frame, name)
        for value in ('abc', 'inf'):
            with self.assertRaises(ValueError):
                add_column(frame, 'n', 'Constant value', value=value, value_type='Number')
        with self.assertRaisesRegex(ValueError, 'denominator'):
            add_column(frame, 'ratio', 'Calculation', left='a', right='a', calculation='Divide')
        with self.assertRaises(ValueError):
            add_column(frame, 'sum', 'Calculation', left='a', right='text', calculation='Add')

    def test_duplicate_columns_compare_all_rows_and_keep_first(self):
        base = list(range(40))
        changed = base.copy()
        changed[-1] = 100
        frame = pd.DataFrame({'first': base, 'copy': base, 'different': changed,
                              'float': [float(v) for v in base], 'text': [str(v) for v in base]})
        self.assertEqual(duplicate_columns(frame), {'copy': 'first'})
        result, removed = remove_duplicate_columns(frame)
        self.assertEqual(removed, {'copy': 'first'})
        self.assertEqual(list(result), ['first', 'different', 'float', 'text'])
        self.assertEqual(len(frame.columns), 5)
        nulls = pd.DataFrame({'a': [0., None], 'b': [-0., None], 'c': [None, 0.], 'd': [None, None], 'e': [None, None]})
        self.assertEqual(duplicate_columns(nulls), {'b': 'a', 'e': 'd'})
        self.assertEqual(duplicate_columns(frame.iloc[:0]), {})

    def test_duplicate_rows_count_extra_copies_include_missing_and_ignore_index(self):
        frame = pd.DataFrame({'a': [1., 1., 1., None, None], 'b': ['x', 'x', 'x', 'y', 'y']}, index=[5, 4, 3, 2, 1])
        result = remove_duplicate_rows(frame)
        self.assertEqual(int(frame.duplicated().sum()), 3)
        assert_frame_equal(result, frame.iloc[[0, 3]].reset_index(drop=True))
        self.assertEqual(len(frame), 5)
        self.assertTrue(remove_duplicate_rows(frame.iloc[:0]).empty)


if __name__ == '__main__':
    unittest.main()
