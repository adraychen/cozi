import unittest
import pandas as pd
from utils.table_editing import remove_characters, replace_values, trim_spaces


class ValueCleaningTests(unittest.TestCase):
    def test_trim_edges_preserves_internal_spaces(self):
        frame = pd.DataFrame({'x': pd.Series(['  New  York  ', '\tJune\u00a0', None, 12, '   '], dtype=object)})
        for mode, expected in [('Leading and trailing', 'New  York'), ('Leading only', 'New  York  '), ('Trailing only', '  New  York')]:
            result, count = trim_spaces(frame, 'x', mode)
            self.assertEqual(result.x.iloc[0], expected)
            self.assertEqual(count, 3)
            self.assertTrue(pd.isna(result.x.iloc[2]))
            self.assertEqual(result.x.iloc[3], 12)
            self.assertEqual(result.x.iloc[4], '')
        self.assertEqual(frame.x.iloc[0], '  New  York  ')

    def test_sales_examples_and_source_preserved(self):
        frame = pd.DataFrame({'Sales': ['$3,000', '$1,500', '$250', '$8,000']})
        original = frame.copy()
        for characters, expected in [('$', ['3,000', '1,500', '250', '8,000']),
                                     (',', ['$3000', '$1500', '$250', '$8000']),
                                     ('$,', ['3000', '1500', '250', '8000'])]:
            result, count = remove_characters(frame, 'Sales', characters)
            self.assertEqual(result.Sales.tolist(), expected)
            self.assertEqual(count, 3 if characters == ',' else 4)
        pd.testing.assert_frame_equal(frame, original)

    def test_literal_characters_and_nontext(self):
        frame = pd.DataFrame({'x': pd.Series(['[a].* $', 30, True, None, ''], dtype=object)})
        result, count = remove_characters(frame, 'x', '[].* $')
        self.assertEqual(result.x.iloc[:3].tolist(), ['a', 30, True])
        self.assertTrue(pd.isna(result.x.iloc[3]))
        self.assertEqual(result.x.iloc[4], '')
        self.assertEqual(count, 1)

    def test_exact_replacement_and_missing(self):
        frame = pd.DataFrame({'x': ['$-', '$-extra', ' $-', None, 'None']})
        result, count = replace_values(frame, 'x', '$-')
        self.assertEqual(count, 1)
        self.assertTrue(pd.isna(result.x.iloc[0]))
        self.assertTrue(pd.isna(result.x.iloc[3]))
        self.assertEqual(result.x.iloc[[1, 2, 4]].tolist(), ['$-extra', ' $-', 'None'])
        self.assertEqual(frame.x.iloc[0], '$-')

    def test_typed_matches(self):
        frame = pd.DataFrame({'x': pd.Series(['1', 1, True, None], dtype=object)})
        for kind, value, position in [('Text', '1', 0), ('Number', '1', 1), ('True/False', 'True', 2)]:
            result, count = replace_values(frame, 'x', value, 'changed', kind, 'Text')
            self.assertEqual(count, 1)
            self.assertEqual(result.x.iloc[position], 'changed')
        result, _ = replace_values(pd.DataFrame({'x': ['1', '2']}), 'x', '1', '12', replacement_type='Number')
        self.assertEqual(result.x.iloc[0], 12)

    def test_categorical_and_large_integer(self):
        frame = pd.DataFrame({'x': pd.Categorical(['$a', 'b', None])})
        result, count = remove_characters(frame, 'x', '$')
        self.assertEqual(count, 1)
        self.assertEqual(result.x.iloc[0], 'a')
        result, count = replace_values(frame, 'x', '$a', 'new', replacement_type='Text')
        self.assertEqual(count, 1)
        self.assertEqual(result.x.iloc[0], 'new')
        frame = pd.DataFrame({'x': [2**60, 2**60 + 1]})
        result, _ = replace_values(frame, 'x', str(2**60), find_type='Number')
        self.assertTrue(pd.isna(result.x.iloc[0]))
        self.assertEqual(result.x.iloc[1], 2**60 + 1)

    def test_optional_surrounding_whitespace_matching(self):
        frame = pd.DataFrame({'x': [' June ', '\tJune\u00a0', 'June sale', 'june', None]})
        result, count = replace_values(frame, 'x', 'June', 'june', replacement_type='Text', ignore_surrounding_spaces=True)
        self.assertEqual(count, 2)
        self.assertEqual(result.x.iloc[:4].tolist(), ['june', 'june', 'June sale', 'june'])
        self.assertTrue(pd.isna(result.x.iloc[4]))
        _, count = replace_values(frame, 'x', 'June', 'june', replacement_type='Text')
        self.assertEqual(count, 0)
        frame = pd.DataFrame({'x': [' -   ', ' $-   ']})
        result, count = replace_values(frame, 'x', '-', '0', replacement_type='Number', ignore_surrounding_spaces=True)
        self.assertEqual(count, 1)
        self.assertEqual(result.x.iloc[0], 0)
        self.assertEqual(result.x.iloc[1], ' $-   ')

    def test_invalid_inputs(self):
        frame = pd.DataFrame({'x': ['a']})
        for column, characters in [('missing', '$'), ('x', '')]:
            with self.assertRaises(ValueError):
                remove_characters(frame, column, characters)
        with self.assertRaises(ValueError):
            replace_values(frame, 'x', 'nan', find_type='Number')


if __name__ == '__main__':
    unittest.main()
