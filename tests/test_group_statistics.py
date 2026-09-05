import unittest
import pandas as pd
from utils.group_statistics import summary, compare, contribution, temporal


class GroupStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'sales': [10., 20., 40., 30.], 'region': ['A', 'A', 'A', 'B'],
                                'product': ['X', 'Y', 'X', 'X'],
                                'date': ['2024-01-02', '2024-01-03', '2024-03-01', '2024-01-01']})

    def test_summary_and_reference(self):
        result, keys = summary(self.df, 'sales', ['region'], 'Sum')
        self.assertEqual(result.Value.tolist(), [70., 30.])
        self.assertAlmostEqual(compare(result, 30.)['Difference %'].iloc[0], 400/3)
        self.assertTrue(compare(result, 0)['Difference %'].isna().all())

    def test_contributions(self):
        result, keys = summary(self.df, 'sales', ['region', 'product'], 'Sum')
        self.assertAlmostEqual(contribution(result, keys)['Contribution %'].sum(), 100)
        parent = contribution(result, keys, True)
        self.assertTrue(parent.groupby(keys[0])['Contribution %'].sum().eq(100).all())

    def test_calendar_gap(self):
        result, keys = temporal(self.df, 'sales', ['region'], 'date', 'Monthly', 'Sum', 'Period-over-Period Comparison')
        a = result[result[keys[0]] == 'A']
        self.assertEqual(len(a), 3)
        self.assertTrue(pd.isna(a.iloc[2]['Previous']))

    def test_rolling_and_running(self):
        args = (self.df, 'sales', ['region'], 'date', 'Monthly', 'Sum')
        result, keys = temporal(*args, 'Rolling Averages', window=2, zero_fill=True)
        self.assertEqual(result[result[keys[0]] == 'A']['Rolling average'].iloc[-1], 20)
        result, keys = temporal(*args, 'Running & Cumulative Statistics')
        self.assertEqual(result[result[keys[0]] == 'A']['Cumulative sum'].iloc[-1], 70)

    def test_missing_and_collision(self):
        df = pd.DataFrame({'Value': [None, 5.], 'Segment': [None, '(Missing)']})
        result, keys = summary(df, 'Value', ['Segment'], 'Sum')
        self.assertEqual(len(result), 2)
        self.assertEqual(result.Value.isna().sum(), 1)

    def test_year_comparison(self):
        df = pd.DataFrame({'v': [10., 15.], 'd': ['2023-01-01', '2024-01-01']})
        result, _ = temporal(df, 'v', [], 'd', 'Monthly', 'Sum', 'Period-over-Period Comparison', lag=12)
        self.assertEqual(result['Change %'].iloc[-1], 50)


if __name__ == '__main__':
    unittest.main()
