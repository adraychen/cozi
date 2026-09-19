import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from utils.ttest_analysis import prepare_ttest, run_ttest, numerical_candidates, TTestInputError
from utils.ttest_reporting import interpretation, export_result
from utils.variable_metadata import infer_metadata, apply_user_override, effective_data


class TTestNumericalTests(unittest.TestCase):
    def assert_result(self, actual, reference, ci, se):
        np.testing.assert_allclose([actual['t'], actual['p'], actual['df']],
                                   [reference.statistic, reference.pvalue, reference.df], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose([actual['low'], actual['high']], ci, rtol=1e-11, atol=1e-12)
        self.assertAlmostEqual(actual['se'], se)

    def test_one_all_alternatives_and_confidence(self):
        x = np.array([3., 4., 8., 2., 10., 7.])
        prepared = prepare_ttest(pd.DataFrame({'x': x}), 'one', 'x')
        for alt, sm_alt in [('two-sided', 'two-sided'), ('greater', 'larger'), ('less', 'smaller')]:
            for confidence in [.90, .95, .99]:
                with self.subTest(alt=alt, confidence=confidence):
                    r = run_ttest(prepared, 4., alt, confidence)
                    test = stats.ttest_1samp(x, 4., alternative=alt)
                    ci = np.array(DescrStatsW(x).tconfint_mean(alpha=1-confidence, alternative=sm_alt)) - 4.
                    self.assert_result(r, test, ci, np.std(x, ddof=1)/np.sqrt(len(x)))
                    self.assertAlmostEqual(r['estimate'], np.mean(x)-4)
                    self.assertAlmostEqual(r['effect'], (np.mean(x)-4)/np.std(x, ddof=1))
                    self.assertEqual(r['effect_name'], "Cohen's d")
        d = prepared['descriptive'].iloc[0]
        np.testing.assert_allclose([d['N'], d['Mean'], d['Standard deviation'], d['Minimum'], d['Maximum']],
                                   [len(x), np.mean(x), np.std(x, ddof=1), min(x), max(x)])

    def test_one_missing_constant_and_small(self):
        p = prepare_ttest(pd.DataFrame({'x': [1., np.nan, 3., np.inf, -np.inf]}), 'one', 'x')
        self.assertEqual(len(p['samples'][0][1]), 2)
        self.assertEqual(p['accounting'].iloc[-1]['Count'], 3)
        self.assertTrue(np.isfinite(run_ttest(p)['t']))
        for values in [[], [1.], [1., 1., 1.], [np.nan, np.inf]]:
            with self.subTest(values=values):
                p = prepare_ttest(pd.DataFrame({'x': pd.Series(values, dtype=float)}), 'one', 'x')
                with self.assertRaises(TTestInputError):
                    run_ttest(p)

    def test_independent_methods_alternatives_unequal_sizes_variances(self):
        x = np.array([3., 5., 6., 9.])
        y = np.array([2., 1., 10., 25., 18., 2., 9.])
        frame = pd.DataFrame({'x': pd.Series(x), 'y': pd.Series(y)})
        p = prepare_ttest(frame, 'independent', 'x', 'y')
        self.assertEqual([len(s) for _, s in p['samples']], [4, 7])
        v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
        pooled = (3*v1 + 6*v2)/9
        for equal in [False, True]:
            for alt, sm_alt in [('two-sided', 'two-sided'), ('greater', 'larger'), ('less', 'smaller')]:
                with self.subTest(equal=equal, alt=alt):
                    r = run_ttest(p, alternative=alt, equal_var=equal)
                    ref = stats.ttest_ind(x, y, equal_var=equal, alternative=alt)
                    cm = CompareMeans(DescrStatsW(x), DescrStatsW(y))
                    ci = cm.tconfint_diff(alpha=.05, alternative=sm_alt, usevar='pooled' if equal else 'unequal')
                    se = np.sqrt(pooled*(1/4+1/7) if equal else v1/4+v2/7)
                    self.assert_result(r, ref, ci, se)
                    self.assertAlmostEqual(r['estimate'], np.mean(x)-np.mean(y))
                    self.assertAlmostEqual(r['effect'], r['estimate']/np.sqrt(pooled))
                    expected_df = 9 if equal else (v1/4+v2/7)**2 / ((v1/4)**2/3+(v2/7)**2/6)
                    self.assertAlmostEqual(r['df'], expected_df)
        self.assertIn('Welch', run_ttest(p)['method'])

    def test_stacked_multicategory_missing_and_direction(self):
        df = pd.DataFrame({'v': [1., 3., 7., 9., 11., 50., 51., np.nan, np.inf, 2.],
                           'g': ['A','A','B','B','B','C','C','A','B',None]})
        p = prepare_ttest(df, 'independent', 'v', group='g', levels=['B','A'])
        r = run_ttest(p, alternative='greater')
        self.assertAlmostEqual(r['estimate'], 7.)
        self.assertEqual([len(s) for _, s in p['samples']], [3, 2])
        self.assertEqual(list(p['accounting']['Count']), [10, 5, 3, 2])
        reverse = run_ttest(prepare_ttest(df, 'independent', 'v', group='g', levels=['A','B']), alternative='less')
        self.assertAlmostEqual(reverse['t'], -r['t'])
        self.assertAlmostEqual(reverse['p'], r['p'])
        self.assertAlmostEqual(reverse['high'], -r['low'])
        self.assertIn('g = B - g = A', interpretation(r))

    def test_pairwise_missing_all_alternatives(self):
        frame = pd.DataFrame({'x': [10., 8., np.nan, 9., 20., 15., np.inf],
                              'y': [5., np.nan, 4., 3., 12., 8., 1.]})
        p = prepare_ttest(frame, 'paired', 'x', 'y')
        diff = np.array([5., 6., 8., 7.])
        np.testing.assert_array_equal(p['diagnostic_samples'][0][1].to_numpy(), diff)
        self.assertEqual(list(p['accounting']['Count']), [7, 4, 3])
        self.assertEqual(list(p['descriptive']['N']), [4, 4, 4])
        self.assertEqual(len(p['diagnostics']), 1)
        for alt, sm_alt in [('two-sided', 'two-sided'), ('greater', 'larger'), ('less', 'smaller')]:
            r = run_ttest(p, alternative=alt)
            ref = stats.ttest_rel([10.,9.,20.,15.], [5.,3.,12.,8.], alternative=alt)
            ci = DescrStatsW(diff).tconfint_mean(alpha=.05, alternative=sm_alt)
            self.assert_result(r, ref, ci, np.std(diff,ddof=1)/2)
            self.assertAlmostEqual(r['effect'], np.mean(diff)/np.std(diff,ddof=1))
            self.assertEqual(r['effect_name'], "Cohen's dz")

    def test_independent_columns_not_pairwise_dropped(self):
        df = pd.DataFrame({'x': [1., 4., np.nan, 8., np.inf], 'y': [np.nan, 2., 5., 10., 11.]})
        p = prepare_ttest(df, 'independent', 'x', 'y')
        np.testing.assert_array_equal(p['samples'][0][1], [1.,4.,8.])
        np.testing.assert_array_equal(p['samples'][1][1], [2.,5.,10.,11.])
        self.assertEqual(list(p['accounting']['Count']), [5,3,2,4,1])
        paired = prepare_ttest(df, 'paired', 'x', 'y')
        self.assertEqual(len(paired['diagnostic_samples'][0][1]), 2)

    def test_degenerate_independent_and_paired(self):
        for mode, x, y in [('paired',[2.,3.,4.],[1.,2.,3.]), ('paired',[1.,np.nan],[2.,4.]),
                           ('independent',[1.,1.,1.],[2.,2.,2.]), ('independent',[1.],[2.])]:
            p = prepare_ttest(pd.DataFrame({'x':x, 'y':y}), mode, 'x', 'y')
            with self.assertRaises(TTestInputError): run_ttest(p)
        # One constant independent group is mathematically estimable when the other varies.
        p = prepare_ttest(pd.DataFrame({'x':[1.,1.,1.],'y':[2.,4.,8.]}), 'independent', 'x', 'y')
        self.assertTrue(np.isfinite(run_ttest(p)['p']))

    def test_invalid_roles_and_options(self):
        df = pd.DataFrame({'x':[1.,3.], 'g':['a','b'], 'flag':[True,False]})
        for kwargs in [dict(mode='paired',first='x',second='x'),dict(mode='one',first='g'),
                       dict(mode='one',first='missing'),dict(mode='one',first='flag'),
                       dict(mode='independent',first='x',group='g',levels=['a','a']),
                       dict(mode='independent',first='x',group='g',levels=['a','z'])]:
            with self.assertRaises(TTestInputError): prepare_ttest(df, **kwargs)
        p = prepare_ttest(df, 'one', 'x')
        for kwargs in [dict(null_mean=np.inf),dict(confidence=1.),dict(confidence=0.),dict(alternative='bad'),dict(null_mean='invalid'),dict(confidence=None)]:
            with self.assertRaises(TTestInputError): run_ttest(p, **kwargs)

    def test_no_formal_normality_gate_and_no_outlier_removal(self):
        df = pd.DataFrame({'x': [1.,2.,3.,4.,5.,100.]}, index=[0,0,0,0,0,0])
        with patch('scipy.stats.shapiro', side_effect=AssertionError('Shapiro must not run')):
            p = prepare_ttest(df, 'one', 'x')
            r = run_ttest(p)
        self.assertEqual(len(p['samples'][0][1]), 6)
        self.assertIn(6, list(p['outliers']['Original row position (1-based)']))
        self.assertTrue(np.isfinite(r['t']))

    def test_semantic_types_overrides_and_identifiers(self):
        df = pd.DataFrame({'score':[1.,3.,7.,8.], 'CustomerID':[100,101,102,103], 'binary':[0,1,0,1]})
        metadata = infer_metadata(df)
        metadata = apply_user_override(metadata,'binary',df['binary'],'Categorical','Binary')
        data, types = effective_data(df,metadata)
        self.assertEqual(numerical_candidates(data,list(df),metadata), ['score'])
        metadata = apply_user_override(metadata,'CustomerID',df['CustomerID'],'Numerical','Discrete')
        self.assertIn('CustomerID',numerical_candidates(data,types['numerical'],metadata))
        metadata = apply_user_override(metadata,'score',df['score'],'Categorical','Nominal')
        self.assertNotIn('score',numerical_candidates(df,list(df),metadata))

    def test_export_and_directional_interpretation(self):
        p = prepare_ttest(pd.DataFrame({'x':[1.,3.,4.,8.]}),'one','x')
        for alt, word in [('greater','greater than zero'),('less','less than zero'),('two-sided','different from zero')]:
            r = run_ttest(p, null_mean=2, alternative=alt)
            text = interpretation(r)
            self.assertIn(word,text)
            self.assertIn('x - hypothesized mean (2)',text)
            self.assertIn('does not establish equal means',text)
            output = export_result(r).decode('utf-8-sig')
            self.assertIn('confidence',output)
            self.assertIn(alt,output)


if __name__ == '__main__': unittest.main()
