import unittest
import numpy as np
import pandas as pd
from scipy import stats
from utils.anova_analysis import prepare_anova, run_anova


def prepare(*groups):
    return prepare_anova(pd.DataFrame({'group': np.repeat(np.arange(len(groups)), list(map(len, groups))),
                                      'value': np.concatenate(groups)}), 'group', 'value')


class AnovaTests(unittest.TestCase):
    def test_classical_known_result(self):
        result = run_anova(prepare([1, 2, 3], [7, 8, 9]), 'Classical', True)
        self.assertAlmostEqual(result['F'], 54)
        self.assertAlmostEqual(result['p-value'], stats.f.sf(54, 1, 4))
        self.assertEqual(result['df denominator'], 4)

    def test_welch_matches_two_sample_welch_t(self):
        a, b = [1., 2, 2, 3, 4], [2., 8, 12, 13, 19, 24, 30]
        result = run_anova(prepare(a, b), 'Welch', True)
        reference = stats.ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(result['F'], reference.statistic**2)
        self.assertAlmostEqual(result['p-value'], reference.pvalue)

    def test_three_group_welch_matches_scipy(self):
        groups = [[1., 4, 2, 3], [2., 8, 12, 16, 18], [3., 5, 4, 6, 8, 5]]
        result = run_anova(prepare(*groups), 'Welch', True)
        reference = stats.f_oneway(*groups, equal_var=False)
        self.assertAlmostEqual(result['F'], reference.statistic)
        self.assertAlmostEqual(result['p-value'], reference.pvalue)

    def test_cleaning_and_lost_group_report(self):
        frame = pd.DataFrame({'g': ['a', 'a', 'b', 'b', 'c', None], 'y': [1, 2, 3, 4, np.inf, 9]})
        result = prepare_anova(frame, 'g', 'y')
        self.assertEqual(result['excluded'], 2)
        self.assertEqual(len(result['groups']), 2)
        self.assertTrue(any('excluded entirely' in note for note in result['warnings']))
        self.assertEqual(result['n'], 4)

    def test_invalid_data_blocked(self):
        for inputs in [([1, 2, 3],), ([1], [2, 3]), ([1, 1], [2, 2])]:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                run_anova(prepare(*inputs), 'Classical', True)
        with self.assertRaises(ValueError):
            prepare_anova(pd.DataFrame({'g': ['a', 'b'], 'v': ['1', '2']}), 'g', 'v')
        with self.assertRaises(ValueError):
            run_anova(prepare([1, 2, 3], [3, 4, 6]), 'Welch', False)

    def test_zero_variance_welch_blocked(self):
        prepared = prepare([1, 1, 1], [2, 3, 4])
        with self.assertRaises(ValueError):
            run_anova(prepared, 'Welch', True)
        self.assertTrue(np.isfinite(run_anova(prepared, 'Classical', True)['F']))

    def test_normality_limits_and_variance_warning(self):
        prepared = prepare([1, 2], [3, 4, 5])
        self.assertTrue(np.isnan(prepared['normality'].iloc[0]['p-value']))
        rng = np.random.default_rng(31)
        prepared = prepare(rng.normal(size=100), rng.normal(size=100)*20)
        self.assertLess(prepared['levene_p'], .05)
        self.assertTrue(any('unequal variances' in note for note in prepared['warnings']))
        large = prepare(np.arange(5001.), [1., 2, 3])
        self.assertIn('Not run', large['normality'].iloc[0]['Interpretation'])

    def test_unused_categories_and_empty_data(self):
        frame = pd.DataFrame({'g': pd.Categorical(['a','a','b','b'], categories=['a','b','c']), 'v':[1.,2,3,5]})
        self.assertEqual(len(prepare_anova(frame, 'g', 'v')['groups']), 2)
        empty = prepare_anova(frame.iloc[:0], 'g', 'v')
        self.assertTrue(empty['errors'])


class AnovaUITests(unittest.TestCase):
    def test_result_guard_method_and_dataset_change(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import streamlit as st
import pandas as pd
from utils.relationship_ui import render_relationship_analysis
dataset = st.selectbox('Dataset', ['one', 'two'], key='fixture_dataset')
st.session_state['file_id'] = dataset
df = pd.DataFrame({'g': ['a']*5 + ['b']*5, 'v': [1.,2,4,3,6, 3,5,8,6,10]})
render_relationship_analysis(df, ['v'], ['g'], [])
''').run(timeout=30)
        app.selectbox(key='rel_method').select('Categorical–numerical: group summaries and ANOVA').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(any('one-way ANOVA**' in m.value for m in app.markdown))
        app.checkbox(key='rel_anova_independent').check().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('Welch one-way ANOVA' in m.value for m in app.markdown))
        app.selectbox(key='rel_anova_method').select('Classical').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('Classical one-way ANOVA' in m.value for m in app.markdown))
        app.selectbox(key='fixture_dataset').select('two').run(timeout=30)
        self.assertFalse(app.checkbox(key='rel_anova_independent').value)
        self.assertFalse(any('one-way ANOVA**' in m.value for m in app.markdown))

    def test_invalid_groups_keep_summary(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import pandas as pd
from utils.anova_ui import render_anova
render_anova(pd.DataFrame({'g':['a','b','b'], 'v':[1.,2,4]}), 'g', 'v')
''').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertTrue(app.dataframe)
        self.assertFalse(app.checkbox)


if __name__ == '__main__':
    unittest.main()
