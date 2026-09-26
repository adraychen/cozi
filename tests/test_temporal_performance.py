import unittest
from unittest.mock import patch
import pandas as pd
from utils import temporal
from utils.detector import detect_data_types
from utils.variable_metadata import infer_metadata, infer_variable_subtype


class TemporalPerformanceTests(unittest.TestCase):
    def test_repeated_text_parsed_once_with_full_row_weighting(self):
        series = pd.Series(['2024-01-01'] * 90 + ['not a date'] * 10 + [None])
        with patch('utils.temporal._calendar', wraps=temporal._calendar) as parser:
            result = temporal.parse_temporal(series, 'Date')
            self.assertEqual(parser.call_count, 2)
        self.assertEqual(result.notna().sum(), 90)
        self.assertEqual(temporal.infer_temporal_subtype(series), 'Date')

    def test_metadata_matches_existing_rules(self):
        frame = pd.DataFrame({'date': ['2024-01-01', '2024-02-01'],
                              'time': ['08:00', '09:00'], 'duration': ['2 hours', '3 hours'],
                              'text': ['Hong Kong', 'New York'], 'amount': ['$3000', '$-'],
                              'number': [1, 2], 'boolean': [True, False]})
        groups = detect_data_types(frame)
        metadata = infer_metadata(frame)
        for label, key in [('Numerical', 'numerical'), ('Categorical', 'categorical'), ('Date / Time', 'datetime')]:
            for column in groups[key]:
                self.assertEqual(metadata[column]['variable_type'], label)
                self.assertEqual(metadata[column]['variable_subtype'], infer_variable_subtype(frame[column], label))


if __name__ == '__main__':
    unittest.main()
