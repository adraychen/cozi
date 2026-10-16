import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.numerical_formatting import (DisplaySettings, format_statistic as fmt, format_p_value,
    format_confidence_interval, format_significant_digits, current_settings, format_results,
    format_plot_hover, _parse_config, validate_display_narrative)


class NumericalFormattingTests(unittest.TestCase):
    def setUp(self):
        # Tests use a private precision document, independent of user edits.
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name)/'precision.toml'
        path.write_text('[display]\nprecision="Standard"\n', encoding='utf-8')
        self.config_patch = patch('utils.numerical_formatting.CONFIG_PATH', path)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def test_approved_defaults(self):
        expected = {'mean':'25.36', 'median':'25.36', 'minimum':'25.36', 'maximum':'25.36', 'quantile':'25.36',
                    'sd':'25.365', 'se':'25.365', 'f':'25.365', 't':'25.365', 'z':'25.365',
                    'cohens_d':'25.365', 'cohens_f2':'25.365', 'eta_squared':'25.365',
                    'correlation':'25.365', 'r2':'25.365', 'adjusted_r2':'25.365',
                    'ss':'25.365', 'ms':'25.365', 'variance':'25.365'}
        for kind, output in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(fmt(25.36482, kind), output)
        self.assertEqual(fmt(12345.678, 'mean'), '12,345.68')
        self.assertEqual(fmt(150, 'count'), '150')
        self.assertEqual(fmt(2**60+1, 'count'), f'{2**60+1:,}')
        self.assertEqual(fmt(148., 'df'), '148')
        self.assertEqual(fmt(96.4213, 'df'), '96.42')
        self.assertEqual(fmt(24.938, 'percentage'), '24.9%')
        self.assertEqual(fmt(.24938, 'proportion'), '24.9%')
        self.assertEqual(fmt(1.2838, 'odds_ratio'), '1.28')
        self.assertEqual(fmt(.85712, 'hazard_ratio'), '0.857')
        self.assertEqual(fmt(.002348, 'coefficient'), '0.002348')
        self.assertEqual(format_significant_digits(12345.678, 4), '12,350')

    def test_special_values_and_small_nonzero(self):
        self.assertEqual(fmt(None, 'mean'), '—')
        self.assertEqual(fmt(pd.NA, 'mean'), '—')
        self.assertEqual(fmt(np.nan, 'mean'), '—')
        self.assertEqual(fmt('N/A', 'mean'), 'N/A')
        self.assertEqual(fmt(np.inf, 'f'), '—')
        self.assertEqual(fmt(np.inf, 'ci', allow_infinite=True), '∞')
        self.assertEqual(fmt(-np.inf, 'ci', allow_infinite=True), '−∞')
        self.assertEqual(fmt(-0., 'sd'), '0.000')
        self.assertNotEqual(float(fmt(.00001, 'sd')), 0)
        self.assertEqual(fmt(.00002345, 'coefficient'), '2.345e-5')
        self.assertEqual(fmt(12345678.9, 'coefficient'), '1.235e7')
        with self.assertRaises(ValueError): fmt(1.2, 'icc')

    def test_p_values_and_alpha_boundaries(self):
        for p, result in [(0.23864,'0.239'),(.02437,'0.024'),(.00456,'0.005'),(.0004,'<0.001'),(8.2e-9,'<0.001'),(0.,'<0.001')]:
            self.assertEqual(format_p_value(p), result)
        self.assertEqual(format_p_value(.0004, mode='Detailed'), '4.00e-4')
        self.assertEqual(format_p_value(.0000234, mode='Detailed'), '2.34e-5')
        self.assertEqual(format_p_value(8.2e-9, mode='Detailed'), '8.20e-9')
        self.assertEqual(format_p_value(.02437, mode='Detailed'), '0.02437')
        self.assertEqual(format_p_value(.0004, context='narrative'), 'p < 0.001')
        for p in (-.1, 1.1, np.inf):
            self.assertIn(format_p_value(p), ('—', 'Invalid p-value'))
        for alpha in (.05, .001, .0001):
            for p in (alpha-1e-10, alpha+1e-10):
                text = format_p_value(p, alpha)
                if text.startswith('<'):
                    self.assertLessEqual(float(text[1:]), alpha)
                    self.assertLess(p, alpha)
                else:
                    self.assertEqual(float(text) < alpha, p < alpha)
        self.assertEqual(format_p_value(0, alpha=.0001, mode='Detailed'), '<1.00e-4')

    def test_settings_and_document_reload(self):
        setting = DisplaySettings(5, 'Detailed')
        self.assertEqual(fmt(1.234567, 'mean', setting), '1.23457')
        self.assertEqual(fmt(12, 'count', setting), '12')
        self.assertEqual(fmt(.002348, 'coefficient', setting), '0.002348')
        self.assertEqual(format_confidence_interval(20.354,30.421), '[20.35, 30.42]')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'precision.toml'
            with patch('utils.numerical_formatting.CONFIG_PATH', path):
                path.write_text('[decimal_places]\nmean = 4\n', encoding='utf-8')
                self.assertEqual(fmt(1.23456, 'mean'), '1.2346')
                path.write_text('[decimal_places]\nmean = 5\n', encoding='utf-8')
                self.assertEqual(fmt(1.23456, 'mean'), '1.23456')
        for bad in ('[decimal_places]\nicc=3', '[decimal_places]\nmean=-1', '[display]\nprecision=7'):
            with self.assertRaises(ValueError): _parse_config(bad)

    def test_styling_and_exports_preserve_raw_numbers(self):
        raw = pd.DataFrame({'Mean':[25.36482], 'p-value':[.0004], 'ICC':[.123456789]})
        before = raw.copy(deep=True)
        styled = format_results(raw, {'Mean':'mean', 'p-value':'p'})
        self.assertIn('25.36', styled.to_html())
        self.assertIn('&lt;0.001', styled.to_html()) if '&lt;' in styled.to_html() else self.assertIn('<0.001', styled.to_html())
        pd.testing.assert_frame_equal(styled.data, before)
        pd.testing.assert_frame_equal(raw, before)
        self.assertIn('25.36482', raw.to_csv(index=False))
        self.assertIn('0.123456789', raw.to_csv(index=False))
        # A raw dataset with the same column names is never auto-formatted.
        from utils import presentation
        with patch.object(presentation.st, 'dataframe') as show:
            presentation.dataframe(raw)
        pd.testing.assert_frame_equal(show.call_args.args[0], before)

    def test_chart_hover_preserves_plotted_values(self):
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=[1,2], y=[1.234567, 8.765432], hovertemplate='%{y:.6g}'))
        values = list(fig.data[0].y)
        format_plot_hover(fig, {'y':'mean'})
        self.assertEqual(list(fig.data[0].y), values)
        self.assertEqual(fig.data[0].customdata[:,0].tolist(), ['1.23', '8.77'])

    def test_ai_cannot_reround_numeric_evidence(self):
        self.assertEqual(validate_display_narrative('F = 5.287.', 'F = 5.287'), 'F = 5.287.')
        with self.assertRaises(ValueError): validate_display_narrative('F = 5.29.', 'F = 5.287')

    def test_new_report_display_is_frozen_and_legacy_is_unchanged(self):
        from utils.report_capture import _capture, record_table
        from utils.report_export import table_groups
        raw = pd.DataFrame({'Mean':[25.36482], 'p-value':[.0004], 'ICC':[.123456789]})
        capture = {'tables':[], 'charts':[], 'content':[]}
        token = _capture.set(capture)
        try: record_table(format_results(raw, {'Mean':'mean','p-value':'p'}))
        finally: _capture.reset(token)
        table = capture['tables'][0]
        self.assertEqual(table['rows'][0][0], 25.36482)
        self.assertEqual(table['display_rows'][0], ['25.36','<0.001',.123456789])
        before = copy.deepcopy(table)
        with patch('utils.numerical_formatting.current_settings', return_value=DisplaySettings(5, 'Detailed')):
            self.assertEqual(table_groups(table)[0][1], before['display_rows'])
        self.assertEqual(table, before)
        legacy = {'columns':['Mean'], 'rows':[[25.36482]]}
        self.assertEqual(table_groups(legacy), [(['Mean'], [[25.36482]])])

    def test_unlisted_named_like_statistics_is_not_guessed(self):
        from utils.regression_reporting import format_table
        # A DFBETA matrix may have a predictor literally called Mean or F.
        matrix = pd.DataFrame({'Mean':[1.23456789], 'F':[8.7654321], 'ICC':[.123456789]})
        display = format_table(matrix)
        self.assertEqual(display.data.Mean.iloc[0], '1.235')
        self.assertEqual(display.data.F.iloc[0], '8.765')
        self.assertEqual(display.data.ICC.iloc[0], '0.1235')

    def test_document_drives_streamlit_and_preserves_saved_report(self):
        from streamlit.testing.v1 import AppTest
        import pyarrow as pa
        def convert_arrow_bytes_to_pandas(data):
            return pa.ipc.open_stream(data).read_all().to_pandas()
        script = """
import pandas as pd
import streamlit as st
from utils import presentation
from utils.report_capture import reportable, offer_result, begin_result
@reportable
def render():
    begin_result()
    raw = pd.DataFrame({'Mean':[1.23456789], 'p-value':[.0004]})
    presentation.dataframe(raw)
    presentation.dataframe(raw, result_types={'Mean':'mean', 'p-value':'p'})
    offer_result('Test', 'Precision snapshot', {}, {}, {'mean':1.23456789}, key='precision_fixture')
render()
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'precision.toml'
            path.write_text('[display]\nprecision="Standard"\n', encoding='utf-8')
            with patch('utils.numerical_formatting.CONFIG_PATH', path):
                app = AppTest.from_string(script).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertEqual(app.dataframe[0].value.Mean.iloc[0], 1.23456789)
                cells = convert_arrow_bytes_to_pandas(app.dataframe[1].proto.arrow_data.styler.display_values)
                self.assertEqual(cells.iloc[0].tolist(), ['1.23', '<0.001'])
                app.button(key='report_add_precision_fixture').click().run(timeout=30)
                before = copy.deepcopy(app.session_state['cozy_report']['blocks'])
                path.write_text('[display]\nprecision=4\np_value_mode="Detailed"\n', encoding='utf-8')
                app.run(timeout=30)
                self.assertFalse(app.exception)
                cells = convert_arrow_bytes_to_pandas(app.dataframe[1].proto.arrow_data.styler.display_values)
                self.assertEqual(cells.iloc[0].tolist(), ['1.2346', '4.00e-4'])
                self.assertEqual(app.session_state['cozy_report']['blocks'], before)
                self.assertEqual(app.dataframe[0].value.Mean.iloc[0], 1.23456789)

    def test_excel_and_csv_keep_numerical_types(self):
        raw = pd.DataFrame({'Mean':[1.23456789012345], 'p-value':[.000023456789]})
        before = raw.copy(deep=True)
        format_results(raw, {'Mean':'mean','p-value':'p'}).to_html()
        excel = io.BytesIO()
        raw.to_excel(excel, index=False)
        excel.seek(0)
        exported = pd.read_excel(excel)
        self.assertTrue(pd.api.types.is_numeric_dtype(exported.Mean))
        np.testing.assert_allclose(exported.to_numpy(), before.to_numpy(), rtol=1e-14)
        csv = pd.read_csv(io.StringIO(raw.to_csv(index=False)))
        self.assertTrue(pd.api.types.is_numeric_dtype(csv.Mean))
        np.testing.assert_allclose(csv.to_numpy(), before.to_numpy(), rtol=1e-14)

    def test_precision_change_invalidates_live_ai_only(self):
        from utils.numerical_formatting import sync_formatting_context
        state = {'ai_summary':'Old precision', 'col_ai_example':'Old precision',
                 'pa_ai__test':'Old precision', 'pa_results':{'r2':.123456789},
                 'cozy_report':{'blocks':[{'text':'Historical 1.23'}]}}
        sync_formatting_context(state)
        self.assertNotIn('ai_summary', state)
        self.assertNotIn('col_ai_example', state)
        self.assertNotIn('pa_ai__test', state)
        self.assertEqual(state['pa_results'], {'r2':.123456789})
        self.assertEqual(state['cozy_report']['blocks'][0]['text'], 'Historical 1.23')

    def test_number_column_cannot_override_p_value_semantics(self):
        from utils import presentation
        raw = pd.DataFrame({'p-value':[.0004]})
        with patch.object(presentation.st, 'dataframe') as show:
            presentation.dataframe(raw, result_types={'p-value':'p'},
                column_config={'p-value':{'type_config':{'type':'number','format':'%.4g'}}})
        self.assertIsNone(show.call_args.kwargs['column_config']['p-value']['type_config']['format'])
        self.assertEqual(show.call_args.args[0].data['p-value'].iloc[0], .0004)

    def test_logistic_narrative_uses_selected_alpha_and_raw_p(self):
        from streamlit.testing.v1 import AppTest
        script = """
import pandas as pd
import streamlit as st
from utils.logistic_ui import render_findings
confidence = st.number_input('Confidence', value=.99)
r = {'spec':{'target':'Y','predictors':[]},'event':1,'adequacy':'No major issues detected',
     'issues':pd.DataFrame(), 'lr_df':1,'lr':5.0,'lr_p':.02437,'confidence':confidence,
     'active':[], 'pseudo_r2':.1, 'brier':.1, 'elimination':False}
render_findings(r,'precision-test')
"""
        app = AppTest.from_string(script).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('does not provide strong evidence of better fit' in item.value for item in app.markdown))
        app.number_input[0].set_value(.95).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('provides evidence of better fit' in item.value for item in app.markdown))

    def test_coefficient_chart_uses_significant_digits_without_rounding(self):
        from utils.charts import make_coefficient_chart
        value = .00234812345
        fig = make_coefficient_chart(pd.Series([value], index=['Predictor']))
        self.assertEqual(fig.data[0].x[0], value)
        self.assertEqual(fig.data[0].customdata[0][0], '0.002348')
