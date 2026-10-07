import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.modelling_config import detect_target_type
from utils.variable_metadata import infer_metadata
from utils.temporal_target_guidance import guidance_context, fallback_guidance, validate_guidance, AVAILABLE_ANALYSES
from utils.predictive_routing import validate_standard_target
from utils.predictive_controls import target_key


class TemporalRoutingTests(unittest.TestCase):
    def test_standard_routes(self):
        from utils.dataset_workspace import dataset_state_key
        self.assertFalse(dataset_state_key('pa_temporal_choose_target'))
        self.assertEqual(detect_target_type(pd.Series([0, 1]), 'Numerical', 'Discrete'), 'continuous_numerical')
        self.assertEqual(detect_target_type(pd.Series([0, 1]), 'Numerical'), 'continuous_numerical')
        self.assertEqual(detect_target_type(pd.Series(['Yes', 'No']), 'Categorical', 'Binary'), 'binary')
        self.assertEqual(detect_target_type(pd.Series(['A', 'B', 'C']), 'Categorical', 'Nominal'), 'multiclass')

    def test_temporal_routes_and_raw_validation(self):
        for series in [pd.Series(pd.date_range('2025-01-01', periods=3)),
                       pd.Series(['2025-01-01', '2025-01-02']),
                       pd.Series(pd.to_timedelta([1, 2], unit='days'))]:
            with self.subTest(dtype=series.dtype):
                before = series.copy()
                self.assertEqual(detect_target_type(series, 'Date / Time', 'Date'), 'temporal')
                with self.assertRaisesRegex(ValueError, 'Temporal targets'):
                    validate_standard_target(series)
                pd.testing.assert_series_equal(series, before)
        self.assertEqual(detect_target_type(pd.Series([2024, 2025]), 'Date / Time', 'Year'), 'temporal')

    def test_delivery_references_without_mutation(self):
        frame = pd.DataFrame({'Order Date': ['2025-01-01', '2025-01-02'],
                              'Delivery Date': ['2025-01-04', '2025-01-06']})
        before = frame.copy(deep=True)
        context = guidance_context(frame, 'Delivery Date', infer_metadata(frame))
        self.assertEqual(context['reference_columns'], ['Order Date'])
        recommendation = fallback_guidance(context)
        self.assertEqual(recommendation['analysis'], 'event')
        self.assertIn('Delivery Date − Order Date', recommendation['why'])
        pd.testing.assert_frame_equal(frame, before)
        reverse = guidance_context(frame, 'Order Date', infer_metadata(frame))
        self.assertNotIn('event', reverse['candidates'])

    def test_failure_status(self):
        frame = pd.DataFrame({'Installation Date': ['2025-01-01'] * 2,
                              'Failure Date': ['2025-02-01', None],
                              'Operating Hours': [400, 200], 'Failed': [True, False]})
        context = guidance_context(frame, 'Failure Date', infer_metadata(frame))
        result = fallback_guidance(context)
        self.assertEqual(result['analysis'], 'survival')
        self.assertIn('has not occurred yet', result['why'])
        self.assertIn('Failed', result['why'])

    def test_forecast_requires_explicit_numerical_goal(self):
        frame = pd.DataFrame({'Date': pd.date_range('2025-01-01', periods=3), 'Sales': [1., 2., 3.]})
        metadata = infer_metadata(frame)
        context = guidance_context(frame, 'Date', metadata)
        self.assertNotIn('forecast', context['candidates'])
        with self.assertRaises(ValueError):
            validate_guidance({'analysis': 'forecast', 'interpretation': 'Forecast dates', 'why': 'Dates'}, context)
        context = guidance_context(frame, 'Date', metadata, 'Forecast a numerical value over time', 'Sales')
        self.assertEqual(fallback_guidance(context)['analysis'], 'forecast')
        context = guidance_context(frame, 'Date', metadata, 'Forecast a numerical value over time', 'Date')
        self.assertNotIn('forecast', context['candidates'])

    def test_all_training_entry_points_reject_raw_datetime(self):
        from utils import modelling
        frame = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=3), 'x': [1, 2, 3]})
        for name in ['run_random_forest_regressor', 'run_random_forest_classifier',
                     'run_random_forest_multiclass_classifier', 'run_regression_comparison',
                     'run_binary_classification_comparison', 'run_multiclass_classification_comparison']:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Temporal targets'):
                getattr(modelling, name)(frame, 'date', ['x'])


SCRIPT = '''
import pandas as pd
import streamlit as st
from utils.variable_metadata import infer_metadata, effective_data
from utils.dataset_views import render_predictive
raw = pd.DataFrame({'Order Date': ['2025-01-01', '2025-01-02'], 'Delivery Date': ['2025-01-04', '2025-01-06'], 'Sales': [10., 20.]})
metadata = infer_metadata(raw)
st.session_state['variable_metadata'] = metadata
df, types = effective_data(raw, metadata)
render_predictive(df, types, {c:m['variable_type'] for c,m in metadata.items()}, metadata, lambda *args: None)
'''


class GuidanceUITests(unittest.TestCase):
    def test_workspace_preserves_controls_without_persisting_action(self):
        from tests.test_dataset_workspace import FIXTURE
        from utils.dataset_workspace import add_dataset
        frame = pd.DataFrame({'Order Date': ['2025-01-01', '2025-01-02'],
                              'Delivery Date': ['2025-01-04', '2025-01-06'], 'Sales': [10., 20.]})
        state = {}
        dataset_id = add_dataset(state, frame, 'Delivery', 'Test')
        state['ws_datasets'][dataset_id]['workflow'] = 'Analyze'
        app = AppTest.from_string(FIXTURE)
        for key, value in state.items():
            app.session_state[key] = value
        with patch('utils.ai_interpreter.recommend_temporal_target', side_effect=RuntimeError('offline')):
            app.run(timeout=60)
            self.assertFalse(app.exception)
            app.checkbox(key=target_key('Delivery Date')).check().run(timeout=60)
            self.assertFalse(app.exception)
            app.run(timeout=60)
            self.assertFalse(app.exception)
            app.button(key='pa_temporal_choose_target').click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['pa_target_select'], 'Sales')

    def test_ai_guidance_and_only_available_actions(self):
        response = {'analysis': 'event', 'interpretation': 'This may describe delivery duration.', 'why': 'Order Date may be a reference point for Delivery Date.'}
        with patch('utils.ai_interpreter.recommend_temporal_target', return_value=response) as ai:
            app = AppTest.from_string(SCRIPT)
            app.session_state['pa_target_select'] = 'Delivery Date'
            app.run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(ai.call_count, 1)
            self.assertEqual(ai.call_args.args[0]['reference_columns'], ['Order Date'])
            self.assertTrue(any(m.value == '### Temporal target detected' for m in app.markdown))
            self.assertTrue(any(m.value == 'Time-to-Event Analysis' for m in app.markdown))
            self.assertFalse(any(c.key.startswith('pa_pred_') for c in app.checkbox))
            self.assertFalse(any(b.key == 'pa_run_btn' for b in app.button))
            self.assertEqual([b.label for b in app.button], ['Choose Another Target'])
            self.assertEqual(AVAILABLE_ANALYSES, {'target'})
            app.run()
            self.assertEqual(ai.call_count, 1)
            app.button(key='pa_temporal_choose_target').click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['pa_target_select'], 'Sales')
            self.assertTrue(app.checkbox(key=target_key('Sales')).value)
            self.assertFalse(any(m.value == '### Temporal target detected' for m in app.markdown))

    def test_ai_failure_and_invalid_ai_response_fallback(self):
        for kwargs in [{'side_effect': RuntimeError('offline')},
                       {'return_value': {'analysis': 'forecast', 'interpretation': 'Unsupported', 'why': 'Unsupported'}}]:
            with self.subTest(kwargs=kwargs), patch('utils.ai_interpreter.recommend_temporal_target', **kwargs):
                app = AppTest.from_string(SCRIPT)
                app.session_state['pa_target_select'] = 'Delivery Date'
                app.run(timeout=60)
                self.assertFalse(app.exception)
                self.assertTrue(any('AI guidance is unavailable' in c.value for c in app.caption))
                self.assertEqual(app.session_state['pa_temporal_guidance']['result']['analysis'], 'event')
