import unittest
from unittest.mock import patch
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.table_transforms import join_tables, append_tables, aggregate_table, pivot_table, unpivot_table


class TableTransformTests(unittest.TestCase):
    def test_join_missing_keys_duplicates_and_reports(self):
        left = pd.DataFrame({'id': [1., 1., 2., None], 'x': [10, 11, 12, 13]})
        right = pd.DataFrame({'id': [1., 1., 3., None], 'y': [20, 21, 22, 23]})
        original = left.copy(deep=True)
        result, report = join_tables(left, right, ['id'], ['id'])
        self.assertEqual(len(result), 6)
        self.assertEqual(report['Matched left rows'], 2)
        self.assertEqual(report['Unmatched left rows'], 2)
        self.assertTrue(report['Many-to-many matches'])
        self.assertTrue(result.loc[result.id.isna(), 'y'].isna().all())
        matched, _ = join_tables(left, right, ['id'], ['id'], match_missing=True)
        self.assertEqual(matched.loc[matched.id.isna(), 'y'].iloc[0], 23)
        for how, size in [('inner', 4), ('right', 6), ('outer', 8)]:
            with self.subTest(how=how):
                out, _ = join_tables(left, right, ['id'], ['id'], how)
                self.assertEqual(len(out), size)
        assert_frame_equal(left, original)

    def test_join_composite_different_names_and_empty(self):
        l = pd.DataFrame({'a': [1, 1], 'b': ['x', 'y']})
        r = pd.DataFrame({'c': [1], 'd': ['x'], 'v': [2]})
        result, _ = join_tables(l, r, ['a', 'b'], ['c', 'd'])
        self.assertEqual(result.v.notna().sum(), 1)
        result, _ = join_tables(l.iloc[:0], r, ['a', 'b'], ['c', 'd'], 'outer')
        self.assertEqual(len(result), 1)
        with self.assertRaises(ValueError):
            join_tables(l, r, ['a'], ['d'])

    def test_explosive_join_blocked_before_materialization(self):
        frame = pd.DataFrame({'id': [1] * 1100})
        with self.assertRaisesRegex(ValueError, 'workspace limit'):
            join_tables(frame, frame, ['id'], ['id'])

    def test_append_alignment_and_source(self):
        a = pd.DataFrame({'x': [1], 'y': ['a']})
        b = pd.DataFrame({'z': [True], 'x': [2]})
        result, warnings = append_tables([a, b], ['A', 'B'], source_column='source')
        self.assertEqual(result.x.tolist(), [1, 2])
        self.assertEqual(result.source.tolist(), ['A', 'B'])
        self.assertTrue(pd.isna(result.y.iloc[1]))
        common, _ = append_tables([a, b], ['A', 'B'], columns='common')
        self.assertEqual(list(common), ['x'])
        with self.assertRaises(ValueError):
            append_tables([a, b], ['A', 'B'], source_column='x')
        _, warnings = append_tables([a, pd.DataFrame({'x': ['001']})], ['A', 'B'])
        self.assertTrue(warnings)

    def test_aggregate_count_semantics_and_missing_groups(self):
        frame = pd.DataFrame({'g': ['A', 'A', None], 'v': [1., None, None]})
        specs = [{'column': 'v', 'function': fn, 'name': name} for fn, name in
                 [('Sum', 'total'), ('Count rows', 'rows'), ('Count nonmissing', 'valid')]]
        result = aggregate_table(frame, ['g'], specs)
        self.assertEqual(result['rows'].tolist(), [2, 1])
        self.assertEqual(result.valid.tolist(), [1, 0])
        self.assertTrue(pd.isna(result.total.iloc[1]))
        self.assertEqual(len(aggregate_table(frame, ['g'], specs, False)), 1)
        all_rows = aggregate_table(frame, [], specs)
        self.assertEqual(all_rows['rows'].iloc[0], 3)
        self.assertEqual(all_rows['total'].iloc[0], 1)
        with self.assertRaises(ValueError):
            aggregate_table(frame, [], [{'column': 'g', 'function': 'Sum', 'name': 'sum'}])

    def test_pivot_repeated_pairs_and_unpivot_roundtrip(self):
        frame = pd.DataFrame({'id': [1, 1, 2], 'month': ['Jan', 'Feb', 'Jan'], 'v': [2., 3., 4.]})
        wide = pivot_table(frame, ['id'], 'month', 'v')
        long = unpivot_table(wide, ['id'], ['Jan', 'Feb'], 'month', 'v', True)
        assert_frame_equal(long.sort_values(['id', 'month']).reset_index(drop=True), frame.sort_values(['id', 'month']).reset_index(drop=True))
        duplicate = pd.concat([frame, frame.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            pivot_table(duplicate, ['id'], 'month', 'v')
        self.assertEqual(pivot_table(duplicate, ['id'], 'month', 'v', 'Sum').Jan.iloc[0], 4)

    def test_unpivot_names_missing_and_overlapping_roles(self):
        frame = pd.DataFrame({'id': [1, 2], 'value': [10., None], 'other': [20, 30]})
        result = unpivot_table(frame, ['id'], ['value', 'other'])
        self.assertEqual(len(result), 4)
        self.assertEqual(len(unpivot_table(frame, ['id'], ['value', 'other'], drop_missing=True)), 3)
        for ids, cols, variable, value in [(['id'], ['id'], 'variable', 'value'), (['id'], ['other'], 'id', 'value'), ([], ['other'], 'same', 'same')]:
            with self.assertRaises(ValueError):
                unpivot_table(frame, ids, cols, variable, value)


if __name__ == '__main__':
    unittest.main()
