import unittest
import numpy as np
import pandas as pd
from scipy import stats
from utils.regression_analysis import regression


class RegressionTests(unittest.TestCase):
    def test_simple_fit_matches_independent_scipy(self):
        x = np.arange(1., 41.)
        y = 5 + 2 * x + np.sin(x)
        result = regression(pd.DataFrame({'x': x, 'y': y}), 'y', ['x'])
        reference = stats.linregress(x, y)
        row = result['coefficients'].iloc[1]
        self.assertAlmostEqual(row['Coefficient'], reference.slope)
        self.assertAlmostEqual(row['Standard error'], reference.stderr)
        self.assertAlmostEqual(row['p-value'], reference.pvalue)
        self.assertAlmostEqual(row['95% lower'], reference.slope - stats.t.ppf(.975, 38) * reference.stderr)
        np.testing.assert_allclose(result['mean_ci'].mean(axis=1), result['fitted'])

    def test_cleaning_preserves_original_row_positions(self):
        data = pd.DataFrame({'x': [1, 2, np.inf, 4, 'bad', 6], 'y': [3, 6, 8, 7, 9, 14]}, index=[4]*6)
        r = regression(data, 'y', ['x'], categorical=[])
        self.assertEqual(r['excluded'], 2)
        np.testing.assert_array_equal(r['source_rows'], [1, 2, 4, 6])

    def test_invalid_inputs(self):
        for df, target, predictors in [
            (pd.DataFrame({'x': [1, 2], 'y': [3, 5]}), 'y', ['x']),
            (pd.DataFrame({'x': [1]*4, 'y': [1, 2, 3, 4]}), 'y', ['x']),
            (pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1]*4}), 'y', ['x']),
            (pd.DataFrame({'x': [1, 2, 3, 4], 'z': [2, 4, 6, 8], 'y': [1, 3, 4, 9]}), 'y', ['x', 'z']),
        ]:
            with self.subTest(predictors=predictors), self.assertRaises(ValueError):
                regression(df, target, predictors)

    def test_perfect_fit_suppresses_inference(self):
        r = regression(pd.DataFrame({'x': np.arange(30), 'y': 2 + 3*np.arange(30)}), 'y', ['x'])
        self.assertTrue(r['perfect_fit'])
        self.assertTrue(r['coefficients']['p-value'].isna().all())
        self.assertTrue(r['diagnostics'].empty)

    def test_small_sample_omnibus_unavailable(self):
        r = regression(pd.DataFrame({'x': range(5), 'y': [1, 4, 2, 6, 3]}), 'y', ['x'])
        omni = r['diagnostics'].set_index('Diagnostic').loc['Omnibus']
        self.assertTrue(np.isnan(omni['p-value']))
        self.assertIn('Unavailable', omni['Interpretation'])

    def test_unequal_variance_flagged(self):
        rng = np.random.default_rng(982)
        x = np.linspace(1, 20, 1000)
        r = regression(pd.DataFrame({'x': x, 'y': 3*x + rng.normal(size=len(x))*x}), 'y', ['x'])
        bp = r['diagnostics'].set_index('Diagnostic').loc['Breusch–Pagan (Koenker LM)']
        self.assertLess(bp['p-value'], .05)

    def test_multiple_predictors_and_dw(self):
        rng = np.random.default_rng(42)
        x, z = rng.normal(size=(2, 100))
        y = 1 + 2*x - 3*z + rng.normal(size=100)*.01
        r = regression(pd.DataFrame({'x': x, 'z': z, 'y': y}), 'y', ['x', 'z'])
        np.testing.assert_allclose(r['coefficients']['Coefficient'], [1, 2, -3], atol=.01)
        e = r['residual']
        self.assertAlmostEqual(r['dw'], np.sum(np.diff(e)**2)/np.sum(e**2))


class RegressionUITests(unittest.TestCase):
    def test_controls_and_order_guard(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import numpy as np
import pandas as pd
from utils.relationship_ui import render_relationship_analysis
x = np.arange(1., 41.)
df = pd.DataFrame({'x': x, 'y': 2*x + np.sin(x), 'z': np.cos(x)})
render_relationship_analysis(df, ['x', 'y', 'z'], [], [])
''').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(any(m.label == 'Durbin–Watson statistic' for m in app.metric))
        app.checkbox(key='rel_reg_ordered').check().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(m.label == 'Durbin–Watson statistic' for m in app.metric))
        app.multiselect(key='rel_predictors').set_value(['y','z']).run(timeout=30)
        self.assertEqual(len(app.exception), 0)

    def test_perfect_fit_and_invalid_ui(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import pandas as pd
from utils.relationship_ui import render_relationship_analysis
df = pd.DataFrame({'x': range(20), 'y': [2*v + 1 for v in range(20)], 'constant': [1]*20})
render_relationship_analysis(df, list(df.columns), [], [])
''').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('Perfect or near-perfect' in w.value for w in app.warning))
        app.multiselect(key='rel_predictors').set_value(['constant']).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('constant' in w.value for w in app.error))


if __name__ == '__main__':
    unittest.main()
