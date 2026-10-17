import unittest
from copy import deepcopy
from itertools import combinations, product
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import stats
from patsy import dmatrix
from utils.factorial_anova import prepare_factorial, run_factorial
from utils.factorial_interactions import (EXPLORE, MANUAL, selected_pairs, interaction_structure,
    support_table, fixed_assessment, qualified_fixed_result, interaction_contrast,
    compare_fixed_reduction, random_assessment)
from utils.anova_effects import prepare_random, fit_random


def design():
    rng = np.random.default_rng(922)
    return pd.DataFrame([(a, b, c, 4*a + b + 8*a*b + 2*b*c + rng.normal())
                         for a, b, c in product(range(2), repeat=3) for _ in range(12)],
                        columns=['A', 'B', 'C', 'Y'])


class InteractionEngineTests(unittest.TestCase):
    def test_selection_structure_and_higher_orders(self):
        factors = ['A', 'B', 'C', 'D']
        self.assertEqual(selected_pairs(factors, EXPLORE), list(combinations(factors, 2)))
        self.assertTrue(all(len(p) == 2 for p in selected_pairs(factors, EXPLORE)))
        self.assertEqual(selected_pairs(factors, MANUAL, [('C', 'B'), ('A', 'removed')]), [('B', 'C')])
        for a, b in product(['Fixed', 'Random'], repeat=2):
            self.assertEqual(interaction_structure(('A', 'B'), {'A': a, 'B': b}),
                             'Fixed interaction' if a == b == 'Fixed' else 'Random interaction')

    def test_fixed_type_ii_against_independent_projection_and_holm(self):
        df = design()
        pairs = selected_pairs(['A', 'B', 'C'], EXPLORE)
        prepared = prepare_factorial(df, 'Y', ['A', 'B', 'C'], pairs)
        result = run_factorial(prepared)
        y = df.Y.to_numpy()
        X = np.asarray(result['model'].model.exog)
        # Independently project out all other terms, not another ANOVA invocation.
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual_ss = np.sum((y-X@beta)**2)
        span = dmatrix(prepared['formula'].split('~', 1)[1], prepared['internal']).design_info.term_name_slices['C(f0):C(f1)']
        reduced = np.delete(X, np.arange(X.shape[1])[span], axis=1)
        reduced_ss = np.sum((y-reduced@np.linalg.lstsq(reduced, y, rcond=None)[0])**2)
        reference_f = (reduced_ss-residual_ss)/(residual_ss/(len(y)-X.shape[1]))
        row = result['table'].set_index('Source').loc['A × B']
        self.assertAlmostEqual(row.F, reference_f, places=8)
        assessment = fixed_assessment(prepared, result, EXPLORE)
        raw = assessment['p-value'].to_numpy()
        order = np.argsort(raw)
        expected = np.empty(len(raw))
        expected[order] = np.minimum(1, np.maximum.accumulate(raw[order]*np.arange(len(raw), 0, -1)))
        np.testing.assert_allclose(assessment['Adjusted p-value'], expected)
        qualified = qualified_fixed_result(prepared, result, assessment)
        np.testing.assert_allclose(qualified['table']['p-value'], result['table']['p-value'], equal_nan=True)
        self.assertIn('A', qualified['significant_interaction_factors'])

    def test_contrast_direction_magnitude_and_intervals(self):
        df = design()
        prepared = prepare_factorial(df, 'Y', ['A', 'B'], [('A', 'B')])
        result = run_factorial(prepared)
        contrast = interaction_contrast(prepared, result, ('A', 'B'))
        means = df.groupby(['A', 'B']).Y.mean()
        expected = means[1, 1]-means[0, 1]-means[1, 0]+means[0, 0]
        self.assertAlmostEqual(contrast['Difference of differences'], expected)
        self.assertLess(contrast['95% simultaneous CI lower'], expected)
        self.assertGreater(contrast['95% simultaneous CI upper'], expected)

    def test_reduction_identical_rows_hierarchy_and_no_mutation(self):
        prepared = prepare_factorial(design(), 'Y', ['A', 'B', 'C'], [('A', 'B'), ('B', 'C')])
        result = run_factorial(prepared)
        before = deepcopy(prepared['interactions'])
        comparison = compare_fixed_reduction(prepared, result, [('A', 'B')])
        self.assertEqual(comparison['Remaining interactions'], [('B', 'C')])
        self.assertEqual(comparison['Main effects retained'], ['A', 'B', 'C'])
        self.assertEqual(prepared['interactions'], before)
        self.assertEqual(comparison['DF removed'], 1)
        self.assertLess(comparison['p-value'], .001)

    def test_support_and_invalid_full_model_withheld(self):
        df = design().loc[lambda d: ~((d.A == 1) & (d.B == 1))]
        table = support_table(df, 'Y', dict.fromkeys(['A', 'B', 'C'], 'Fixed'), [('A', 'B')])
        self.assertEqual(table.iloc[0]['Empty pairs'], 1)
        prepared = prepare_factorial(df, 'Y', ['A', 'B', 'C'], [('A', 'B')])
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            run_factorial(prepared)
        df = pd.DataFrame([(a, b, float(a+b)) for a, b in product(range(2), repeat=2)], columns=['A', 'B', 'Y'])
        prepared = prepare_factorial(df, 'Y', ['A', 'B'], [('A', 'B')])
        with self.assertRaisesRegex(ValueError, 'Insufficient observations'):
            run_factorial(prepared)
        with self.assertRaisesRegex(ValueError, 'confounded'):
            prepare_random(df, 'Y', {'A': 'Random', 'B': 'Random'}, [('A', 'B')])

    def test_random_interactions_against_balanced_expected_mean_squares(self):
        a, b = np.arange(6)-2.5, np.arange(5)-2
        df = pd.DataFrame([(str(i), str(j), 10+4*a[i]+3*b[j]+a[i]*b[j]+e)
                           for i in range(6) for j in range(5) for e in [-1., 0., 1.]], columns=['A', 'B', 'Y'])
        prepared = prepare_random(df, 'Y', {'A': 'Random', 'B': 'Random'}, [('A', 'B')])
        result = fit_random(prepared)
        grand, means_a, means_b = df.Y.mean(), df.groupby('A').Y.mean(), df.groupby('B').Y.mean()
        cells = df.groupby(['A', 'B']).Y.mean()
        mse = df.groupby(['A', 'B']).Y.var(ddof=1).mean()
        msab = sum(3*(value-means_a[i]-means_b[j]+grand)**2 for (i, j), value in cells.items())/20
        expected = [(15*means_a.var(ddof=1)-msab)/15, (18*means_b.var(ddof=1)-msab)/18, (msab-mse)/3, mse]
        np.testing.assert_allclose(result['variance'].Variance, expected, rtol=.005)
        self.assertAlmostEqual(result['variance']['Percentage of modeled variance'].sum(), 100)
        assessment = random_assessment(prepared, result, [('A', 'B')])
        self.assertEqual(assessment.iloc[0].Assessment, 'Inconclusive')
        self.assertNotIn('p-value', assessment)
        self.assertEqual(prepared['covariance_rank'], 4)
        with self.assertRaisesRegex(ValueError, 'confounded'):
            prepare_random(df, 'Y', {'A': 'Random', 'B': 'Random'}, [('A', 'B')], {'B': 'A'})

    def test_mixed_complexity_is_bounded_before_dense_design_allocation(self):
        df = pd.DataFrame({'A': np.arange(120), 'B': np.arange(120),
                           'Group': np.tile(np.arange(6), 20), 'Y': np.arange(120, dtype=float)})
        with patch('utils.anova_effects.dmatrix') as matrix:
            with self.assertRaisesRegex(ValueError, 'design-matrix limits'):
                prepare_random(df, 'Y', {'A':'Fixed', 'B':'Fixed', 'Group':'Random'}, [('A','B')])
            matrix.assert_not_called()

    def test_mixed_fixed_and_random_interactions_against_gls(self):
        rng = np.random.default_rng(48)
        group = rng.normal(0, 3, 6)
        ac, bc = rng.normal(0, 2, (2, 6)), rng.normal(0, 2, (2, 6))
        df = pd.DataFrame([(str(a), str(b), str(c), 5*a+2*b+7*a*b+group[c]+ac[a,c]+bc[b,c]+rng.normal())
                           for a, b, c in product(range(2), range(2), range(6)) for _ in range(4)],
                          columns=['A', 'B', 'C', 'Y'])
        pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        prepared = prepare_random(df, 'Y', {'A': 'Fixed', 'B': 'Fixed', 'C': 'Random'}, pairs)
        result = fit_random(prepared)
        X = np.asarray(dmatrix('C(f0)+C(f1)+C(f0):C(f1)', prepared['data']))
        variance = result['variance'].set_index('Component').Variance
        V = np.eye(len(df))*variance['Residual (within groups)']
        for label, columns in [('C', ['C']), ('A × C', ['A', 'C']), ('B × C', ['B', 'C'])]:
            codes = pd.factorize(pd.MultiIndex.from_frame(df[columns]))[0]
            V += variance[label]*(codes[:, None] == codes[None, :])
        covariance = np.linalg.inv(X.T @ np.linalg.solve(V, X))
        beta = covariance @ X.T @ np.linalg.solve(V, df.Y)
        np.testing.assert_allclose(result['fixed'].Estimate, beta, rtol=1e-7, atol=1e-7)
        test = result['fixed_interactions']['A × B']
        self.assertAlmostEqual(test['Wald χ²'], beta[-1]**2/covariance[-1, -1], places=5)
        self.assertAlmostEqual(test['p-value'], stats.chi2.sf(test['Wald χ²'], 1))
        assessment = random_assessment(prepared, result, pairs)
        self.assertEqual(assessment.Structure.tolist(), ['Fixed interaction', 'Random interaction', 'Random interaction'])
        self.assertTrue(assessment.loc[assessment.Structure == 'Random interaction', 'p-value'].isna().all())


SCRIPT = '''
import streamlit as st
from tests.test_factorial_interactions import design
from utils.factorial_anova_ui import render_factorial_anova
df = design()
factors = ['A','B'] if st.session_state.get('remove_c') else ['A','B','C']
render_factorial_anova(df, ['Y'], factors)
'''


class InteractionUITests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string(SCRIPT).run(timeout=60)
        self.assertFalse(app.exception)
        return app

    def apply(self, app):
        buttons = [b for b in app.button if b.key == 'fac_apply_model']
        if buttons:
            buttons[0].click().run(timeout=60)
        self.assertFalse(app.exception)

    def test_default_explore_manual_roundtrip_factor_change_and_no_stale_output(self):
        app = self.app()
        self.assertEqual(app.radio(key='fac_interaction_mode').value, EXPLORE)
        self.assertFalse([m for m in app.multiselect if m.key == 'fac_interactions'])
        self.assertEqual(app.session_state['report_latest']['settings']['interactions'], [['A','B'], ['A','C'], ['B','C']])
        app.radio(key='fac_interaction_mode').set_value(MANUAL).run(timeout=60)
        self.assertNotIn('report_latest', app.session_state)
        app.multiselect(key='fac_interactions').set_value([('A','B'), ('A','C')]).run(timeout=60)
        self.apply(app)
        app.radio(key='fac_interaction_mode').set_value(EXPLORE).run(timeout=60)
        self.apply(app)
        app.run(timeout=60)
        app.radio(key='fac_interaction_mode').set_value(MANUAL).run(timeout=60)
        self.assertEqual(app.multiselect(key='fac_interactions').value, [('A','B'), ('A','C')])
        self.apply(app)
        app.session_state['remove_c'] = True
        app.run(timeout=60)
        self.assertEqual(app.multiselect(key='fac_interactions').value, [('A','B')])
        self.apply(app)
        self.assertEqual(app.session_state['report_latest']['settings']['interactions'], [['A','B']])

    def test_explicit_reduction_and_snapshot_preservation(self):
        app = self.app()
        app.button(key='report_add_factorial').click().run(timeout=60)
        snapshot = deepcopy(app.session_state['cozy_report']['blocks'])
        app.multiselect(key='fac_refinement_remove').set_value([('A','B')]).run(timeout=60)
        self.assertEqual(len(app.session_state['report_latest']['settings']['interactions']), 3)
        self.assertTrue(any('Full residual DF' in t.value.columns for t in app.dataframe))
        app.button(key='fac_action_refine_confirm').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key='fac_interaction_mode').value, MANUAL)
        self.assertNotIn(('A','B'), app.multiselect(key='fac_interactions').value)
        self.assertEqual(app.session_state['cozy_report']['blocks'], snapshot)
        content = str(snapshot)
        self.assertIn('Interaction Insights', content)
        self.assertNotIn('Confirm removal and refit', content)

    def test_ai_unavailable_keeps_deterministic_insights(self):
        app = self.app()
        with patch('utils.ai_interpreter.interpret_factorial_interactions', side_effect=RuntimeError('offline')):
            app.button(key='fac_action_ai_generate').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any('AI guidance is unavailable' in x.value for x in app.info))
        self.assertTrue(any('relationship with Y' in m.value for m in app.markdown))
        self.assertTrue(app.button(key='report_add_factorial'))

    def test_random_guided_design_fit_failure_and_descriptive_report(self):
        from streamlit.testing.v1 import AppTest
        script = '''
import streamlit as st
import numpy as np
import pandas as pd
from utils.factorial_anova_ui import render_factorial_anova
a, b = np.arange(6)-2.5, np.arange(5)-2
df = pd.DataFrame([(str(i), str(j), 10+4*a[i]+3*b[j]+a[i]*b[j]+e)
    for i in range(6) for j in range(5) for e in [-1.,0.,1.]], columns=['A','B','Y'])
st.session_state.setdefault('fac_effects_context', (None, 'Y'))
st.session_state.setdefault('fac_effects', {'A':'Random','B':'Random'})
render_factorial_anova(df, ['Y'], ['A','B'])
'''
        app = AppTest.from_string(script).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotIn('report_latest', app.session_state)
        self.assertEqual(app.tabs[1].label, 'Diagnostics')
        from utils.factorial_study_design import factor_key, SEPARATE
        for factor in ('A', 'B'):
            app.button(key='fac_action_design_confirm_' + factor_key(factor)).click().run(timeout=60)
        app.radio(key='fac_design_collection').set_value(SEPARATE).run(timeout=60)
        self.assertFalse(app.exception)
        variance = next(t.value for t in app.dataframe if 'Percentage of modeled variance' in t.value and 'Component' in t.value)
        self.assertAlmostEqual(variance['Percentage of modeled variance'].sum(), 100)
        self.assertFalse(app.tabs[0].warning)
        self.assertTrue(any('Inconclusive' in t.value.get('Assessment', []).tolist()
                            for t in app.dataframe if 'Assessment' in t.value))
        app.button(key='report_add_fac_random').click().run(timeout=60)
        snapshot = deepcopy(app.session_state['cozy_report']['blocks'])
        self.assertIn('Random-effects', str(snapshot))
        app.selectbox(key='fac_design_assignment_' + factor_key('A')).select('within:1').run(timeout=60)
        self.assertNotIn('report_latest', app.session_state)
        app.button(key='fac_action_design_confirm_' + factor_key('A')).click().run(timeout=60)
        self.assertTrue(any('changes are pending' in x.value for x in app.tabs[1].info))
        app.button(key='fac_action_random_apply').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotIn('report_latest', app.session_state)
        self.assertTrue(any('confounded' in m.value for m in app.tabs[1].markdown))
        self.assertEqual(app.session_state['cozy_report']['blocks'], snapshot)

    def test_ai_rejects_invented_numerical_results(self):
        from types import SimpleNamespace
        from langchain_core.runnables import RunnableLambda
        from utils.ai_interpreter import interpret_factorial_interactions
        evidence = {'deterministic_findings': 'A × B: variance 2.300. Inconclusive.'}
        with patch('utils.ai_interpreter._get_llm', return_value=RunnableLambda(
                lambda _: SimpleNamespace(content='A × B has variance 9.999.'))):
            with self.assertRaises(ValueError):
                interpret_factorial_interactions(evidence)
        with patch('utils.ai_interpreter._get_llm', return_value=RunnableLambda(
                lambda _: SimpleNamespace(content='A × B has estimated variance 2.300; the result is inconclusive.'))):
            self.assertIn('inconclusive', interpret_factorial_interactions(evidence))


if __name__ == '__main__':
    unittest.main()
