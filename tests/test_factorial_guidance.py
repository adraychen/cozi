import unittest
import json
from itertools import product

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from tests.test_factorial_anova import fixture
from utils.factorial_anova import prepare_factorial, run_factorial, posthoc
from utils.factorial_followup import (MEAN, estimated_marginal_means, emm_vectors,
                                      simple_effects, simple_effects_family, robust_type2,
                                      _estimates)
from utils.factorial_diagnostics import diagnostic_summary


class AdjustedMeansTests(unittest.TestCase):
    def test_balanced_unbalanced_emms_intervals_against_reference(self):
        for unbalanced in [False, True]:
            df = fixture(unbalanced)
            prepared = prepare_factorial(df, 'Y', ['A', 'B', 'D'], [('A', 'B')])
            result = run_factorial(prepared)
            reference = ols('Y ~ C(A) + C(B) + C(D) + C(A):C(B)', df).fit()
            grid = pd.DataFrame(product(['0','1','2'], ['0','1'], ['0','1']), columns=['A','B','D'])
            x = dmatrix('C(A) + C(B) + C(D) + C(A):C(B)', grid, return_type='dataframe')[reference.model.exog_names]
            actual = estimated_marginal_means(prepared, result, ['A'])
            for i, level in enumerate(['0','1','2']):
                vector = x.loc[grid.A == level].mean().to_numpy()
                ref = reference.t_test(vector)
                self.assertAlmostEqual(actual.iloc[i][MEAN], ref.effect.item())
                self.assertAlmostEqual(actual.iloc[i]['SE'], ref.sd.item())
                np.testing.assert_allclose(actual.iloc[i][['95% CI lower','95% CI upper']].to_numpy(dtype=float), ref.conf_int()[0])
            # In an unbalanced design, equal-weight model means are not raw means.
            if unbalanced:
                self.assertFalse(np.allclose(actual[MEAN], df.groupby('A').Y.mean()))
            labels, vectors = emm_vectors(prepared, result, ['A','B'])
            self.assertEqual(len(labels), 6)
            np.testing.assert_allclose(vectors @ result['model'].params,
                                       reference.predict(grid).groupby([grid.A,grid.B]).mean())

    def test_simple_effects_reference_and_familywise_adjustment(self):
        df = fixture(True)
        prepared = prepare_factorial(df, 'Y', ['A','B','D'], [('A','B')])
        result = run_factorial(prepared)
        for factor, partner in [('A','B'), ('B','A')]:
            family = simple_effects_family(prepared, result, factor)
            self.assertEqual(len(family), len(prepared['levels'][partner]))
            tables = [item['comparisons'] for item in family]
            raw_p = np.concatenate([table['p-value'] for table in tables])
            expected_p = multipletests(raw_p, method='holm')[1]
            np.testing.assert_allclose(np.concatenate([t['Holm-adjusted p-value'] for t in tables]), expected_p)
            for item in family:
                labels, vectors = emm_vectors(prepared, result, [factor], item['fixed'])
                test = result['model'].t_test(vectors[0]-vectors[1])
                row = item['comparisons'].iloc[0]
                self.assertAlmostEqual(row['Adjusted mean difference'], test.effect.item())
                self.assertAlmostEqual(row['SE'], test.sd.item())
                self.assertAlmostEqual(row['p-value'], test.pvalue.item())
                half = stats.t.ppf(1-.05/(2*len(raw_p)), result['model'].df_resid)*test.sd.item()
                self.assertAlmostEqual(row['95% simultaneous CI lower'], test.effect.item()-half)

    def test_nonsignificant_interaction_allows_global_pairs_two_levels(self):
        df = pd.DataFrame([(a,b,10*a+2*b+noise) for a,b in product([0,1],repeat=2)
                           for noise in [-2.,-1.,1.,2.]],columns=['A','B','Y'])
        prepared = prepare_factorial(df,'Y',['A','B'],[('A','B')])
        result = run_factorial(prepared)
        self.assertFalse(result['significant_interaction_factors'])
        table = posthoc(prepared,result,'A')
        self.assertAlmostEqual(table.iloc[0]['Adjusted mean difference'],-10.)
        self.assertEqual(table.iloc[0]['Significant'],'Yes')

    def test_sparse_estimable_additive_and_unestimable_interactions(self):
        df = fixture()
        df = df.loc[~((df.A == '0') & (df.B == '0'))]
        prepared = prepare_factorial(df,'Y',['A','B'])
        result = run_factorial(prepared)
        self.assertTrue(np.isfinite(estimated_marginal_means(prepared,result,['A'])[MEAN]).all())
        self.assertIn('extrapolate', diagnostic_summary(prepared,result)['table'].set_index('Check').loc['Cell coverage','Interpretation'])
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            run_factorial(prepare_factorial(df,'Y',['A','B'],[('A','B')]))
        singular = ols('Y ~ C(A)*C(B)',df).fit()
        grid = pd.DataFrame(product(['0','1','2'],['0','1']),columns=['A','B'])
        vector = dmatrix('C(A)*C(B)',grid,return_type='dataframe')[singular.model.exog_names].iloc[0].to_numpy()
        with self.assertRaisesRegex(ValueError,'not estimable'):
            _estimates({'model':singular}, vector)

    def test_multiple_partners_require_conditioning_and_missing_strata_withheld(self):
        rng=np.random.default_rng(224)
        df=pd.DataFrame([(a,b,c,5*a+7*a*b+9*a*c+rng.normal())
                         for a,b,c in product([0,1],repeat=3) if (a,b,c)!=(0,0,0)
                         for _ in range(20)],columns=['A','B','C','Y'])
        prepared=prepare_factorial(df,'Y',['A','B','C'],[('A','B'),('A','C')])
        result=run_factorial(prepared)
        with self.assertRaisesRegex(ValueError,'every significant'):
            simple_effects(prepared,result,'A',{'B':1})
        family=simple_effects_family(prepared,result,'A')
        self.assertEqual(len(family),4)
        self.assertEqual(sum('error' in item for item in family),1)
        self.assertIn('no observations',next(item['error'] for item in family if 'error' in item))

    def test_reference_grid_size_guard(self):
        prepared=prepare_factorial(fixture(),'Y',['A','B'])
        result=run_factorial(prepared)
        prepared['possible_cells']=100001
        with self.assertRaisesRegex(ValueError,'too large'):
            estimated_marginal_means(prepared,result,['A'])


class DiagnosticsTests(unittest.TestCase):
    def test_normality_variance_independence_and_replication_language(self):
        df=fixture()
        prepared=prepare_factorial(df,'Y',['A','B'],[('A','B')])
        result=run_factorial(prepared)
        diagnostic=diagnostic_summary(prepared,result)
        checks=diagnostic['table'].set_index('Check')
        self.assertEqual(checks.loc['Independence','Status'],'Cannot assess')
        self.assertEqual(checks.loc['Cell coverage','Status'],'OK')
        self.assertIn('does not establish',checks.loc['Residual normality','Interpretation'])
        df=df.groupby(['A','B']).head(1)
        prepared=prepare_factorial(df,'Y',['A','B'])
        result=run_factorial(prepared)
        diagnostic=diagnostic_summary(prepared,result)
        self.assertIn('could not be formally assessed',diagnostic['table'].set_index('Check').loc['Equal variance','Interpretation'])
        self.assertEqual(diagnostic['table'].set_index('Check').loc['Replication','Status'],'Caution')

    def test_heteroskedasticity_spread_and_hc3_reference(self):
        rng=np.random.default_rng(25)
        df=pd.DataFrame([(a,b,20*a+5*b+rng.normal(scale=1+5*a))
                         for a,b in product(range(4),range(2)) for _ in range(80)],columns=['A','B','Y'])
        prepared=prepare_factorial(df,'Y',['A','B'])
        result=run_factorial(prepared)
        diagnostic=diagnostic_summary(prepared,result)
        self.assertEqual(diagnostic['spread'],'increasing')
        self.assertEqual(diagnostic['table'].set_index('Check').loc['Equal variance','Status'],'Problem')
        before=result['table'].copy()
        actual=robust_type2(prepared,result)
        reference=anova_lm(ols('Y ~ C(A)+C(B)',df).fit(),typ=2,robust='hc3').iloc[:-1]
        np.testing.assert_allclose(actual['HC3 robust F'],reference.F)
        np.testing.assert_allclose(actual['HC3 p-value'],reference['PR(>F)'])
        pd.testing.assert_frame_equal(before,result['table'])

    def test_skew_transform_domain_safeguards_no_mutation(self):
        rng=np.random.default_rng(414)
        original=pd.DataFrame([(a,b,np.exp(rng.normal(scale=1.5))+a+b)
                               for a,b in product(range(3),range(2)) for _ in range(45)],columns=['A','B','Y'])
        for domain in ['positive','zero','negative']:
            df=original.copy()
            if domain=='zero': df.loc[0,'Y']=0.
            if domain=='negative': df.loc[0,'Y']=-1.
            before=df.copy(deep=True)
            prepared=prepare_factorial(df,'Y',['A','B'])
            result=run_factorial(prepared)
            diagnostic=diagnostic_summary(prepared,result)
            text=' '.join(diagnostic['recommendations'])
            if domain=='positive':
                self.assertIn('Consider a log transformation',text)
            elif domain=='zero':
                self.assertNotIn('Consider a log transformation',text)
                self.assertIn('undefined for zero',text)
                self.assertIn('square-root transformation',text)
            else:
                self.assertNotIn('Consider a log transformation',text)
                self.assertNotIn('square-root transformation is also',text)
                self.assertIn('not valid',text)
            self.assertEqual(diagnostic['table'].set_index('Check').loc['Residual normality','Status'],'Problem')
            pd.testing.assert_frame_equal(df,before)

    def test_hc3_rejects_unit_leverage(self):
        df=fixture()[['A','B','Y']]
        df=pd.concat([df.loc[~((df.A=='0') & (df.B=='0'))], df.loc[(df.A=='0') & (df.B=='0')].head(1)])
        prepared=prepare_factorial(df,'Y',['A','B'],[('A','B')])
        result=run_factorial(prepared)
        with self.assertRaisesRegex(ValueError,'leverage'):
            robust_type2(prepared,result)

    def test_decreasing_spread_curvature_and_extreme_residual_screens(self):
        rng=np.random.default_rng(633)
        df=pd.DataFrame([(a,b,20*a+5*b+rng.normal(scale=20-6*a))
                         for a,b in product(range(4),range(2)) for _ in range(100)],columns=['A','B','Y'])
        prepared=prepare_factorial(df,'Y',['A','B'])
        self.assertEqual(diagnostic_summary(prepared,run_factorial(prepared))['spread'],'decreasing')
        df=pd.DataFrame([(a,b,10*a+3*b+8*a*b+rng.normal(scale=.5))
                         for a,b in product([0,1],repeat=2) for _ in range(50)],columns=['A','B','Y'])
        prepared=prepare_factorial(df,'Y',['A','B'])
        self.assertTrue(diagnostic_summary(prepared,run_factorial(prepared))['curvature'])
        df.loc[0,'Y']+=1000
        prepared=prepare_factorial(df,'Y',['A','B'])
        self.assertGreater(diagnostic_summary(prepared,run_factorial(prepared))['extreme_residuals'],0)


class GuidedUITests(unittest.TestCase):
    def test_factorial_controls_are_dataset_scoped(self):
        from utils.dataset_workspace import add_dataset, activate, save_active
        state = {}
        first = add_dataset(state, fixture(), 'First', 'Test')
        activate(state, first)
        state.update(fac_alpha=.01, fac_interactions=[('A','B')], fac_simple_show=True)
        save_active(state)
        second = add_dataset(state, fixture(), 'Second', 'Test')
        activate(state, second)
        self.assertNotIn('fac_alpha',state)
        self.assertNotIn('fac_interactions',state)
        state['fac_alpha']=.10
        activate(state, first)
        self.assertEqual(state['fac_alpha'],.01)
        self.assertEqual(state['fac_interactions'],[('A','B')])
        self.assertTrue(state['fac_simple_show'])

    def test_complete_guided_workflow_hc3_simple_effects_swaps_and_alpha(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_string('''
import streamlit as st
from tests.test_factorial_anova import fixture
from utils.factorial_anova_ui import render_factorial_anova
st.session_state.setdefault('fac_interaction_mode', 'Select interactions manually')
render_factorial_anova(fixture(True), ['Y'], ['A','B','D'])
''').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(any('Model validity' in item.value for item in app.subheader))
        self.assertTrue(any('Adjusted mean (Estimated Marginal Mean)' in t.value.columns for t in app.dataframe))
        charts = [json.loads(chart.proto.spec) for chart in app.get('plotly_chart')]
        adjusted_charts = [chart for chart in charts if 'adjusted mean (EMM)' in chart['layout'].get('yaxis', {}).get('title', {}).get('text', '')]
        self.assertTrue(adjusted_charts)
        self.assertTrue(all(chart['layout']['xaxis']['type'] == 'category' for chart in adjusted_charts))
        self.assertTrue(all(trace.get('error_y') for chart in adjusted_charts for trace in chart['data']))
        app.multiselect(key='fac_interaction_choices').set_value([('A','B')]).run(timeout=40)
        app.checkbox(key='fac_hc3').check().run(timeout=40)
        app.checkbox(key='fac_simple_show').check().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(any('HC3 robust F' in t.value.columns for t in app.dataframe))
        self.assertTrue(any('95% simultaneous CI lower' in t.value.columns for t in app.dataframe))
        self.assertNotIn('A',app.selectbox(key='fac_posthoc').options)
        app.selectbox(key='fac_simple_factor').select('B').run(timeout=40)
        app.checkbox(key='fac_swap_3').check().run(timeout=40)
        app.number_input(key='fac_alpha').set_value(.01).run(timeout=40)
        self.assertFalse(app.exception)
        app.multiselect(key='fac_interaction_choices').set_value([]).run(timeout=40)
        self.assertTrue(any('A' in s.options for s in app.selectbox if s.key=='fac_posthoc'))
        self.assertFalse(any(s.key=='fac_simple_factor' for s in app.selectbox))


if __name__=='__main__':
    unittest.main()
