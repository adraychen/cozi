from tests.factorial_selection_helper import interaction_selection
import unittest
from itertools import product
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from utils.factorial_anova import prepare_factorial, run_factorial, posthoc, interaction_summary


def fixture(unbalanced=False):
    rng = np.random.default_rng(241)
    rows = []
    for a, b, c in product(range(3), range(2), range(2)):
        for _ in range(5 + (a + b if unbalanced else 0)):
            rows.append((str(a), str(b), str(c), 2*a + b + c + 4*a*b + rng.normal()))
    return pd.DataFrame(rows, columns=['A', 'B', 'D', 'Y'])


class FactorialTests(unittest.TestCase):
    def test_reference_balanced_unbalanced_main_and_interactions(self):
        for unbalanced, interaction, three in product([False, True], repeat=3):
            df = fixture(unbalanced)
            factors = ['A', 'B', 'D'] if three else ['A', 'B']
            pairs = [('A', 'B')] if interaction else []
            prepared = prepare_factorial(df, 'Y', factors, pairs)
            result = run_factorial(prepared)
            formula = 'Y ~ C(A) + C(B)' + (' + C(D)' if three else '') + (' + C(A):C(B)' if interaction else '')
            ref = anova_lm(ols(formula, df).fit(), typ=2)
            for actual, expected in [('DF', 'df'), ('Sum of Squares', 'sum_sq'), ('F', 'F'), ('p-value', 'PR(>F)')]:
                np.testing.assert_allclose(result['table'][actual], ref[expected], rtol=1e-10, atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(result['table']['Partial η²'][:-1], ref.sum_sq[:-1]/(ref.sum_sq[:-1]+ref.sum_sq.iloc[-1]))
            reverse = run_factorial(prepare_factorial(df, 'Y', list(reversed(factors)), pairs))['table']
            # Main-effect inference is invariant to factor order.
            for name in factors:
                self.assertAlmostEqual(result['table'].set_index('Source').loc[name, 'F'], reverse.set_index('Source').loc[name, 'F'])

    def test_empty_cells_and_singular_models(self):
        df = fixture()
        df = df.loc[~((df.A == '0') & (df.B == '0'))]
        prepared = prepare_factorial(df, 'Y', ['A', 'B'])
        self.assertTrue(any('empty' in w for w in prepared['warnings']))
        run_factorial(prepared)  # additive model remains estimable
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            run_factorial(prepare_factorial(df, 'Y', ['A', 'B'], [('A', 'B')]))
        df = fixture()
        df['Duplicate'] = df.A
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            run_factorial(prepare_factorial(df, 'Y', ['A', 'Duplicate']))

    def test_multiple_selected_interactions_and_complexity_guard(self):
        df = fixture(True)
        prepared = prepare_factorial(df, 'Y', ['A','B','D'], [('A','B'), ('B','D')])
        result = run_factorial(prepared)
        ref = anova_lm(ols('Y ~ C(A) + C(B) + C(D) + C(A):C(B) + C(B):C(D)', df).fit(), typ=2)
        np.testing.assert_allclose(result['table']['Sum of Squares'], ref.sum_sq)
        self.assertNotIn('A × D', result['table'].Source.to_list())
        large = pd.DataFrame({'Y':np.arange(2004.), 'A':np.repeat(np.arange(1002), 2), 'B':np.tile([0,1],1002)})
        with self.assertRaisesRegex(ValueError, 'too complex'):
            run_factorial(prepare_factorial(large, 'Y', ['A','B']))
        with self.assertRaisesRegex(ValueError, 'two different'):
            prepare_factorial(df, 'Y', ['A','B'], [('A','A')])

    def test_sparse_insufficient_constant_and_single_level(self):
        df = fixture().groupby(['A', 'B']).head(1)
        prepared = prepare_factorial(df, 'Y', ['A', 'B'])
        self.assertTrue(any('only one' in w for w in prepared['warnings']))
        self.assertIsNone(run_factorial(prepared)['levene'])
        with self.assertRaisesRegex(ValueError, 'Insufficient observations'):
            run_factorial(prepare_factorial(df, 'Y', ['A', 'B'], [('A', 'B')]))
        df = fixture()
        df['Y'] = 1.
        with self.assertRaisesRegex(ValueError, 'residual error'):
            run_factorial(prepare_factorial(df, 'Y', ['A', 'B']))
        with self.assertRaisesRegex(ValueError, 'two observed levels'):
            run_factorial(prepare_factorial(df[df.A == '0'], 'Y', ['A', 'B']))

    def test_missing_labels_and_plot(self):
        df = fixture().rename(columns={'A': 'Store type', 'B': 'Promo: C(x)', 'Y': 'Mean'})
        df.loc[0, 'Mean'] = np.nan
        df.loc[1, 'Store type'] = None
        df.loc[2, 'Mean'] = np.inf
        pair = ('Store type', 'Promo: C(x)')
        prepared = prepare_factorial(df, 'Mean', list(pair), [pair])
        self.assertEqual((prepared['original'], prepared['n'], prepared['missing'], prepared['nonfinite']), (60,57,2,1))
        result = run_factorial(prepared)
        self.assertIn('Store type × Promo: C(x)', result['table'].Source.to_list())
        summary = interaction_summary(prepared, pair)
        reference = prepared['clean'].groupby(list(pair), sort=False)['Mean'].mean().to_numpy()
        np.testing.assert_allclose(summary['Mean'], reference)
        self.assertEqual(summary.N.sum(), 57)

    def test_posthoc_model_contrasts_and_interaction_guard(self):
        df = fixture(True)
        prepared = prepare_factorial(df, 'Y', ['A', 'B'])
        result = run_factorial(prepared)
        table = posthoc(prepared, result, 'A')
        ref = ols('Y ~ C(A) + C(B)', df).fit().t_test('C(A)[T.1] = 0')
        self.assertAlmostEqual(table.iloc[0]['p-value'], float(ref.pvalue))
        self.assertAlmostEqual(table.iloc[0]['Adjusted mean difference'], -float(ref.effect.item()))
        self.assertTrue((table['Holm-adjusted p-value'] >= table['p-value']).all())
        prepared = prepare_factorial(df, 'Y', ['A', 'B'], [('A', 'B')])
        result = run_factorial(prepared)
        self.assertEqual(result['significant_interaction_factors'], {'A', 'B'})
        with self.assertRaisesRegex(ValueError, 'Simple-effects'):
            posthoc(prepared, result, 'A')


class FactorialUITests(unittest.TestCase):
    def test_all_significant_terms_have_plots(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import numpy as np
from itertools import product, combinations
from utils.factorial_anova import prepare_factorial, run_factorial
from utils.factorial_anova_ui import render_effect_plots
rng = np.random.default_rng(124)
rows = [(a,b,c,5*a+6*b+4*c+10*a*b+8*b*c+rng.normal())
        for a,b,c in product([0,1],repeat=3) for _ in range(25)]
df = pd.DataFrame(rows, columns=['A','B','C','Y'])
p = prepare_factorial(df,'Y',['A','B','C'],list(combinations(['A','B','C'],2)))
r = run_factorial(p)
st.session_state['expected_plots'] = int((r['table']['Significant'] == 'Yes').sum())
st.session_state['expected_interactions'] = int(((r['table']['Significant'] == 'Yes') & (r['table']['Effect'] == 'Interaction effect')).sum())
render_effect_plots(p,r)
''').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertGreaterEqual(app.session_state['expected_interactions'],2)
        self.assertEqual(len(app.get('plotly_chart')),app.session_state['expected_plots'])
        self.assertTrue(any('not in isolation' in w.value for w in app.info))
        self.assertFalse(any(s.key == 'fac_plot' for s in app.selectbox))
        app.checkbox(key='fac_plot_all').check().run()
        self.assertEqual(len(app.get('plotly_chart')),6)
        self.assertFalse(app.exception)

    def test_no_significant_terms_and_optional_exploration(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import pandas as pd
from utils.factorial_anova import prepare_factorial,run_factorial
from utils.factorial_anova_ui import render_effect_plots
df = pd.DataFrame([(a,b,y) for a in ['a','b'] for b in ['c','d'] for y in [1.,2.,3.,4.]], columns=['A','B','Y'])
p = prepare_factorial(df,'Y',['A','B'],[('A','B')])
render_effect_plots(p,run_factorial(p))
''').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('plotly_chart')),0)
        self.assertTrue(any('No selected terms' in m.value for m in app.info))
        app.checkbox(key='fac_plot_all').check().run()
        self.assertEqual(len(app.get('plotly_chart')),3)

    def test_metadata_filter_and_numeric_categorical_codes(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import streamlit as st
from tests.test_factorial_anova import fixture
from utils.factorial_anova_ui import render_factorial_anova
df = fixture()
df['B'] = df.B.astype(int)
st.session_state['variable_metadata'] = {
    'A': {'variable_type':'Categorical', 'variable_subtype':'Ordinal'},
    'B': {'variable_type':'Categorical', 'variable_subtype':'Binary'},
    'D': {'variable_type':'Numerical', 'variable_subtype':'Discrete'},
    'Y': {'variable_type':'Numerical', 'variable_subtype':'Continuous'}}
render_factorial_anova(df, ['Y','B'], ['A','B','D'])
''').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(any(w.key == 'fac_y' for w in app.selectbox))
        self.assertFalse(any(w.key == 'fac_x' for w in app.multiselect))
        self.assertTrue(any('**Response (Y):** Y' in m.value for m in app.markdown))
        self.assertTrue(any('**Factors (X):** A, B' in m.value for m in app.markdown))
        self.assertTrue(any('Partial η²' in t.value for t in app.dataframe))

    def test_controls_results_and_stale_selections(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import streamlit as st
from tests.test_factorial_anova import fixture
from utils.comparison_ui import render_comparison
st.session_state.setdefault('fac_interaction_mode', 'Select interactions manually')
df = fixture()
render_comparison(df, {'numerical':['Y'], 'categorical':['A','B','D'], 'datetime':[]})
''').run(timeout=30)
        app.button(key='gs_cmp_all_numerical').click().run()
        app.button(key='gs_cmp_all_categorical').click().run()
        app.selectbox(key='gs_cmp_analysis').select('Factorial ANOVA').run()
        self.assertFalse(app.exception)
        interaction_selection(app).set_value([('A', 'B')]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('depends on' in w.value for w in app.info))
        self.assertTrue(any('Partial η²' in t.value for t in app.dataframe))
        interaction_selection(app).set_value([]).run()
        app.checkbox(key='fac_posthoc_show').check().run()
        self.assertTrue(any('Holm-adjusted p-value' in t.value for t in app.dataframe))
        app.checkbox(key='gs_cmp_variable_D').uncheck().run()
        self.assertTrue(any('**Factors (X):** A, B' in m.value for m in app.markdown))
        app.checkbox(key='gs_cmp_variable_B').uncheck().run()
        self.assertFalse(any('Partial η²' in t.value for t in app.dataframe))
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
