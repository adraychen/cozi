import unittest
from html.parser import HTMLParser
import pandas as pd
from utils.descriptive_statistics_display import STATISTIC_HELP, statistics_html
from utils.exploration import numerical_stats, detect_outliers


class DetailParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details = []
        self.summaries = []

    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            self.details.append(dict(attrs))
        if tag == 'summary':
            self.summaries.append(dict(attrs))


class DescriptiveDisplayTests(unittest.TestCase):
    def test_all_reported_statistics_have_closed_clickable_help(self):
        sample = pd.Series([1., 2., 3., 4., 50.])
        stats = numerical_stats(sample)
        outliers, _ = detect_outliers(sample)
        stats.update({'Outliers Found (IQR)': outliers['count'], 'Outlier %': outliers['pct']})
        self.assertEqual(set(stats), set(STATISTIC_HELP))
        parser = DetailParser()
        parser.feed(statistics_html(stats))
        self.assertEqual(len(parser.details), 15)
        self.assertTrue(all('open' not in attrs for attrs in parser.details))
        self.assertEqual([a['aria-label'] for a in parser.summaries],
                         ['How to interpret ' + name for name in stats])

    def test_untrusted_names_and_values_are_escaped(self):
        html = statistics_html({'<script>alert(1)</script>': '<img src=x onerror=alert(1)>', 'Mean': '1.00'})
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('&lt;img', html)
        self.assertIn('<td>1.00</td>', html)

    def test_kurtosis_help_matches_excess_calculation(self):
        stats = numerical_stats(pd.Series([1., 2., 3., 4., 5.]))
        self.assertEqual(stats['Kurtosis'], '-1.20')
        self.assertIn('excess kurtosis', STATISTIC_HELP['Kurtosis'])
        self.assertIn('0, not 3', STATISTIC_HELP['Kurtosis'])


if __name__ == '__main__':
    unittest.main()
