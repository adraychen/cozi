import unittest
import pandas as pd
from utils.group_statistics import contingency, summary, temporal


class CategoricalStatisticsTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([(r, c) for r, c, n in [('A', 'X', 20), ('A', 'Y', 10), ('B', 'X', 10), ('B', 'Y', 20)] for _ in range(n)], columns=['row', 'column'])

    def test_known_pearson_result(self):
        result = contingency(self.fixture(), 'row', 'column')
        self.assertEqual(result['n'], 60)
        self.assertTrue(result['expected'].eq(15).all().all())
        self.assertAlmostEqual(result['chi2'], 20/3)
        self.assertAlmostEqual(result['p'], 0.009823274507519247)
        self.assertEqual(result['dof'], 1)
        self.assertAlmostEqual(result['residual'].iloc[0, 0], 5/(15**0.5))

    def test_missing_and_unused_categories(self):
        df = self.fixture()
        df.loc[len(df)] = [None, 'X']
        df['row'] = pd.Categorical(df['row'], categories=['A', 'B', 'Unused'])
        result = contingency(df, 'row', 'column')
        self.assertEqual(result['excluded'], 1)
        self.assertEqual(result['observed'].shape, (2, 2))
        self.assertEqual(contingency(df, 'row', 'column', True)['n'], 61)

    def test_degenerate_and_empty(self):
        result = contingency(pd.DataFrame({'a': ['x'], 'b': ['y']}), 'a', 'b')
        self.assertIsNone(result['p'])
        with self.assertRaises(ValueError):
            contingency(pd.DataFrame({'a': [None], 'b': ['y']}), 'a', 'b')
        with self.assertRaises(ValueError):
            contingency(self.fixture(), 'row', 'row')

    def test_optional_numeric_counts(self):
        df = self.fixture()
        df['number'] = None
        counts, _ = summary(df, None, ['row'], 'Row count')
        self.assertEqual(counts.Value.tolist(), [30, 30])
        counts, _ = summary(df, 'number', ['row'], 'Row count')
        self.assertEqual(counts.Value.tolist(), [30, 30])
        counts, _ = summary(df, 'number', ['row'], 'Non-missing count')
        self.assertEqual(counts.Value.tolist(), [0, 0])

    def test_sparse_and_time_counts(self):
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['u', 'v'], 'date': ['2024-01-01', '2024-02-01']})
        self.assertEqual(contingency(df, 'a', 'b')['low_expected'], 4)
        counts, _ = temporal(df, None, [], 'date', 'Monthly', 'Row count', 'Running & Cumulative Statistics')
        self.assertEqual(counts['Cumulative sum'].tolist(), [1, 2])


if __name__ == '__main__':
    unittest.main()
