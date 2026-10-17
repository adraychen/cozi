import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from utils.factorial_anova import prepare_factorial, run_factorial
from utils.factorial_diagnostics import diagnostic_summary
from utils.factorial_guidance import (guided_items, coverage_table, dependence_indicators,
                                     analysis_context, INDEPENDENT, REPEATED, UNSURE)
from utils.guided_diagnostics import Diagnostic, diagnostic_status


def design():
    rng = np.random.default_rng(331)
    return pd.DataFrame([(a, b, a * 3 + b + rng.normal()) for a in range(3)
                         for b in range(2) for _ in range(20)], columns=['A', 'B', 'Y'])


class StructuredDiagnosticsTests(unittest.TestCase):
    def test_status_priorities_and_unresolved_assumptions(self):
        item = Diagnostic('x', 'model', 'ok', 'Title', 'Found', 'Impact', 'Action')
        self.assertEqual(diagnostic_status([item]), 'Checks passed')
        for severity in ('review', 'unresolved'):
            self.assertEqual(diagnostic_status([replace(item, severity=severity)]), 'Review advised')
        self.assertEqual(diagnostic_status([replace(item, severity='blocking')]), 'Model issue')

    def test_actual_counts_empty_singletons_and_bounded_design(self):
        df = design()
        df = df.loc[~((df.A == 2) & (df.B == 1))]
        df = df.loc[~((df.A == 0) & (df.B == 0)) | (df.index == 0)]
        prepared = prepare_factorial(df, 'Y', ['A', 'B'])
        table, truncated = coverage_table(prepared)
        self.assertFalse(truncated)
        self.assertEqual(len(table), 6)
        self.assertEqual(table.Observations.sum(), len(df))
        self.assertEqual(int(table.Observations.eq(0).sum()), 1)
        self.assertEqual(int(table.Observations.eq(1).sum()), 1)
        observed, truncated = coverage_table(prepared, limit=2)
        self.assertTrue(truncated)
        self.assertEqual(len(observed), 5)
        result = run_factorial(prepared)
        items = guided_items(prepared, result, diagnostic_summary(prepared, result), INDEPENDENT)
        coverage = next(i for i in items if i.diagnostic_id == 'cell_coverage')
        self.assertEqual(coverage.severity, 'review')
        self.assertIn('5 observed / 6', coverage.observed_result)
        self.assertIn('do not automatically invalidate', coverage.explanation)

    def test_blocking_rank_failure_and_no_fabricated_rank(self):
        df = design().loc[lambda d: ~((d.A == 2) & (d.B == 1))]
        prepared = prepare_factorial(df, 'Y', ['A', 'B'], [('A', 'B')])
        with self.assertRaises(ValueError) as caught:
            run_factorial(prepared)
        items = guided_items(prepared, failure=str(caught.exception))
        self.assertEqual(diagnostic_status(items), 'Model issue')
        self.assertIn('rank deficient', next(i.observed_result for i in items if i.diagnostic_id == 'estimability'))

    def test_indicators_use_analyzed_source_rows_without_proving_dependence(self):
        df = design()
        df['SubjectID'] = ['same'] + [str(i) for i in range(1, len(df))]
        df.loc[0, 'Y'] = np.nan
        df.loc[1, 'SubjectID'] = 'same'
        prepared = prepare_factorial(df, 'Y', ['A', 'B'])
        self.assertTrue(dependence_indicators(df, prepared, {}).empty)
        df.loc[2, 'SubjectID'] = 'same'
        evidence = dependence_indicators(df, prepared, {})
        self.assertEqual(len(evidence), 1)
        self.assertIn('2 rows share', evidence.iloc[0]['Observed result'])
        self.assertIn('treatment labels', evidence.iloc[0]['Observed result'])

    def test_context_changes_with_data_metadata_and_model(self):
        df = design()
        args = ('Y', ['A', 'B'], [], {'A': 'Fixed', 'B': 'Fixed'}, .05, {})
        first = analysis_context(df, *args)
        self.assertEqual(first, analysis_context(df.copy(), *args))
        changed = df.copy(); changed.loc[0, 'Y'] += 1
        self.assertNotEqual(first, analysis_context(changed, *args))
        self.assertNotEqual(first, analysis_context(df, *args[:-1], {'A': {'variable_subtype': 'Identifier'}}))

    def test_repeated_unknown_and_independent_are_qualified(self):
        prepared = prepare_factorial(design(), 'Y', ['A', 'B'])
        result = run_factorial(prepared)
        for answer in (None, UNSURE, REPEATED, INDEPENDENT):
            items = guided_items(prepared, result, answer=answer)
            independence = next(i for i in items if i.diagnostic_id == 'independence')
            self.assertEqual(independence.severity, 'ok' if answer == INDEPENDENT else 'unresolved')
            self.assertIn('No data-only check proves', independence.explanation)
            if answer == REPEATED:
                self.assertIn('unsupported', independence.recommended_action)


SCRIPT = '''
import streamlit as st
from tests.test_factorial_guided_workflow import design
from utils.factorial_anova_ui import render_factorial_anova
st.session_state.setdefault('fac_interaction_mode', 'Select interactions manually')
df = design()
if st.session_state.get('break_design'):
    df = df.loc[~((df.A == 2) & (df.B == 1))]
render_factorial_anova(df, ['Y'], ['A','B'])
'''


class GuidedWorkflowUITests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string(SCRIPT).run(timeout=40)
        self.assertFalse(app.exception)
        return app

    def test_overview_classification_and_real_navigation(self):
        app = self.app()
        overview = app.tabs[0]
        self.assertEqual(overview.label, 'Model Overview')
        self.assertTrue(any('Original observations: 120' in m.value for m in overview.markdown))
        self.assertTrue(any('Type II Sum' in m.value for m in overview.caption))
        self.assertFalse(overview.warning)
        self.assertTrue(any('Review advised' in m.value for m in overview.markdown))
        self.assertTrue(any('Cell Coverage' in m.label for m in app.tabs[1].expander))
        app.button(key='fac_open_diagnostics').click().run(timeout=40)
        self.assertEqual(app.session_state['fac_results_tab'], 'Diagnostics')
        app.button(key='fac_action_cell_coverage').click().run(timeout=40)
        self.assertEqual(app.session_state['fac_diagnostic_focus'], 'coverage')
        self.assertTrue(any(e.label == 'Cell-count details' and e.proto.expanded for e in app.expander))
        self.assertTrue(any('#fac-model-configuration' in m.value for m in app.markdown))
        app.button(key='fac_action_hc3').click().run(timeout=40)
        self.assertEqual(app.session_state['fac_results_tab'], 'Model Overview')
        self.assertFalse(app.checkbox(key='fac_hc3').value)

    def test_confirm_terms_no_stale_results_and_answer_reset(self):
        app = self.app()
        app.selectbox(key='fac_study_answer').set_value(INDEPENDENT).run(timeout=40)
        app.checkbox(key='fac_plot_all').check().run(timeout=40)
        self.assertEqual(app.selectbox(key='fac_study_answer').value, INDEPENDENT)
        app.multiselect(key='fac_interactions').set_value([('A', 'B')]).run(timeout=40)
        self.assertEqual(len(app.get('plotly_chart')), 0)
        self.assertTrue(any('pending' in m.value for m in app.info))
        app.button(key='fac_apply_model').click().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertIsNone(app.selectbox(key='fac_study_answer').value)
        self.assertTrue(any('Partial η²' in t.value.columns for t in app.dataframe))
        self.assertEqual(app.multiselect(key='fac_interactions').value, [('A', 'B')])

    def test_critical_model_status_withholds_tables_and_followups(self):
        app = self.app()
        app.multiselect(key='fac_interactions').set_value([('A', 'B')]).run(timeout=40)
        app.button(key='fac_apply_model').click().run(timeout=40)
        app.session_state['break_design'] = True
        app.run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(any('Model issue' in m.value for m in app.tabs[0].error))
        self.assertFalse(any('Partial η²' in t.value.columns for t in app.dataframe))
        self.assertEqual(len(app.get('plotly_chart')), 0)
        self.assertFalse(any(b.key == 'report_add_factorial' for b in app.button))

    def test_study_answers_and_data_changes(self):
        app = self.app()
        app.selectbox(key='fac_study_answer').set_value(REPEATED).run(timeout=40)
        self.assertTrue(any('does not account' in m.value for m in app.markdown))
        self.assertTrue(any('Partial η²' in t.value.columns for t in app.dataframe))
        app.selectbox(key='fac_study_answer').set_value(UNSURE).run(timeout=40)
        self.assertTrue(any('remains unresolved' in m.value for m in app.markdown))
        app.session_state['break_design'] = True
        app.run(timeout=40)
        self.assertIsNone(app.selectbox(key='fac_study_answer').value)

    def test_report_snapshot_is_immutable_and_excludes_controls(self):
        from copy import deepcopy
        app = self.app()
        app.button(key='report_add_factorial').click().run(timeout=40)
        before = deepcopy(app.session_state['cozy_report']['blocks'][0])
        content = str(before['snapshot']['content'])
        self.assertIn('Study design remains unresolved', content)
        self.assertNotIn('Apply changes and rerun', content)
        self.assertNotIn('Review Interactions / Model Terms', content)
        app.selectbox(key='fac_study_answer').set_value(INDEPENDENT).run(timeout=40)
        self.assertFalse(app.button(key='report_add_factorial').disabled)
        app.multiselect(key='fac_interactions').set_value([('A', 'B')]).run(timeout=40)
        self.assertNotIn('report_latest', app.session_state)
        app.button(key='fac_apply_model').click().run(timeout=40)
        self.assertEqual(app.session_state['cozy_report']['blocks'][0], before)


if __name__ == '__main__':
    unittest.main()
