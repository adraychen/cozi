import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from utils.regression_analysis import regression, predict
from utils.regression_design import dependencies
from utils.regression_reporting import findings, p_text
from utils.regression_plots import fitted_curve


class ExtendedRegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(819)
        self.data = pd.DataFrame({'x':rng.normal(size=160)+5,'z':rng.normal(size=160),
                                  'g':np.tile(['A','B'],80),'h':np.tile(['C','C','D','D'],40)})
        self.data['y'] = 3+2*self.data.x-.5*self.data.z+rng.normal(size=160)

    def test_simple_statistics_and_prediction(self):
        d = self.data; r = regression(d,'y',['x']); ref = smf.ols('y ~ x',d).fit()
        np.testing.assert_allclose(r['model'].params,ref.params)
        np.testing.assert_allclose(r['coefficients']['Standard error'],ref.bse)
        np.testing.assert_allclose(r['coefficients']['t-statistic'],ref.tvalues)
        np.testing.assert_allclose(r['coefficients']['p-value'],ref.pvalues)
        np.testing.assert_allclose(r['coefficients'][['95% lower','95% upper']],ref.conf_int())
        for key,value in [('r2',ref.rsquared),('adjusted_r2',ref.rsquared_adj),('residual_df',ref.df_resid),('f_stat',ref.fvalue),('f_p',ref.f_pvalue),('rmse',np.sqrt(np.mean(ref.resid**2))),('mae',np.mean(abs(ref.resid))),('residual_se',np.sqrt(ref.mse_resid))]:
            self.assertAlmostEqual(r[key],value)
        new = pd.DataFrame({'x':[4.,5.,6.]})
        np.testing.assert_allclose(predict(r,new),ref.get_prediction(new).summary_frame())
        pred = predict(r,new)
        self.assertTrue(((pred.obs_ci_upper-pred.obs_ci_lower) > (pred.mean_ci_upper-pred.mean_ci_lower)).all())
        names = [trace.name for trace in fitted_curve(r).data]
        self.assertIn('95% prediction band',names); self.assertIn('95% confidence band',names)

    def test_multiple_diagnostics_vif_influence(self):
        d = self.data; r = regression(d,'y',['x','z']); ref=smf.ols('y ~ x + z',d).fit()
        self.assertAlmostEqual(r['vif'].iloc[0].VIF,variance_inflation_factor(ref.model.exog,1))
        self.assertAlmostEqual(r['vif'].iloc[0].Tolerance,1/r['vif'].iloc[0].VIF)
        self.assertAlmostEqual(r['dw'],durbin_watson(ref.resid))
        bp = r['diagnostics'].set_index('Diagnostic').loc['Breusch–Pagan (Koenker LM)']
        np.testing.assert_allclose(bp[['Statistic','p-value']].astype(float),het_breuschpagan(ref.resid,ref.model.exog)[:2])
        inf = ref.get_influence()
        np.testing.assert_allclose(r['influence']['Studentized residual'],inf.resid_studentized_internal)
        np.testing.assert_allclose(r['influence']['Leverage'],inf.hat_matrix_diag)
        np.testing.assert_allclose(r['influence']["Cook's distance"],inf.cooks_distance[0])

    def test_interaction_types_and_category_prediction(self):
        for predictors, pair, formula in [(['x','z'],('x','z'),'y ~ x*z'),(['x','g'],('x','g'),'y ~ x*C(g)'),(['g','h'],('g','h'),'y ~ C(g)*C(h)')]:
            with self.subTest(pair=pair):
                r=regression(self.data,'y',predictors,interactions=[pair]); ref=smf.ols(formula,self.data).fit()
                np.testing.assert_allclose(r['fitted'],ref.fittedvalues,atol=1e-10)
                np.testing.assert_allclose(predict(r,self.data.iloc[:8]),ref.get_prediction(self.data.iloc[:8]).summary_frame(),atol=1e-10)
                self.assertTrue(dependencies(('interaction',*pair)).issubset(r['active']))
                if r['coding']:
                    self.assertTrue(r['coefficients']['Standardized coefficient'].iloc[1:].isna().all())

    def test_polynomial_hierarchy_and_prediction(self):
        for degree in [2,3]:
            r=regression(self.data,'y',['x'],degrees={'x':degree})
            ref=smf.ols('y ~ x + I(x**2)'+(' + I(x**3)' if degree==3 else ''),self.data).fit()
            np.testing.assert_allclose(r['fitted'],ref.fittedvalues,atol=1e-9)
            np.testing.assert_allclose(predict(r,self.data.iloc[:4]),ref.get_prediction(self.data.iloc[:4]).summary_frame(),atol=1e-9)
            self.assertIn(('power','x',2),r['active'])
            self.assertTrue(r['spec']['centered'])

    def test_selection_removes_noise_and_protects_interaction(self):
        d=self.data.copy(); x=d.x-d.x.mean(); z=d.z-d.z.mean()
        design=np.column_stack([np.ones(len(d)),x,z,x*z])
        noise=np.random.default_rng(3).normal(size=len(d))
        noise-=design@np.linalg.lstsq(design,noise,rcond=None)[0]
        d.y=3+4*x*z+noise
        r=regression(d,'y',['x','z'],interactions=[('x','z')],elimination=True)
        self.assertEqual(len(r['active']),3)
        tests=r['term_tests'].set_index('Term')
        self.assertGreater(tests.loc['x','p-value'],.05)
        self.assertTrue(tests.loc['x','Hierarchy protected'])
        d.y=3+4*x+noise
        r=regression(d,'y',['x','z'],elimination=True)
        self.assertNotIn(('main','z'),r['active'])
        self.assertIn('z',r['history']['Removed term'].tolist())

    def test_selection_protects_cubic_hierarchy(self):
        d=self.data.copy(); x=d.x-d.x.mean(); design=np.column_stack([np.ones(len(d)),x,x*x,x**3])
        noise=np.random.default_rng(8).normal(size=len(d)); noise-=design@np.linalg.lstsq(design,noise,rcond=None)[0]
        d.y=5+2*x**3+noise
        r=regression(d,'y',['x'],degrees={'x':3},elimination=True)
        self.assertEqual(r['active'],[('main','x'),('power','x',2),('power','x',3)])

    def test_grouped_categorical_removal(self):
        d=self.data.copy(); ref=smf.ols('y ~ x + C(g)',d).fit()
        r=regression(d,'y',['x','g'])
        test=r['term_tests'].set_index('Term').loc['g','p-value']
        self.assertAlmostEqual(test,float(ref.f_test('C(g)[T.B] = 0').pvalue))

    def test_vif_and_artificial_influence(self):
        d=self.data.copy(); d.z=d.x+np.random.default_rng(4).normal(size=len(d))*.001
        r=regression(d,'y',['x','z']); self.assertTrue((r['vif'].VIF>10).all())
        d=self.data.copy(); d.loc[0,['x','y']]=[100,900]
        r=regression(d,'y',['x','z']); self.assertTrue(r['influence'].iloc[0].Flagged)
        self.assertEqual(r['n'],len(d))

    def test_invalid_specs_prediction_and_missing(self):
        for kwargs in [dict(degrees={'g':2}),dict(interactions=[('x','bad')]),dict(degrees={'x':4})]:
            with self.assertRaises(ValueError): regression(self.data,'y',['x','g'],**kwargs)
        r=regression(self.data,'y',['x','g'])
        with self.assertRaises(ValueError): predict(r,pd.DataFrame({'x':[5], 'g':['Unknown']}))
        d=self.data.copy(); d.loc[1,'x']=np.nan; d.loc[3,'g']=None
        r=regression(d,'y',['x','g'],interactions=[('x','g')])
        self.assertEqual(r['excluded'],2); self.assertEqual(r['original_n'],160)
        self.assertNotIn(2,r['source_rows']); self.assertNotIn(4,r['source_rows'])

    def test_findings_format_and_no_claimed_visual_pattern(self):
        r=regression(self.data,'y',['x'])
        text=' '.join(findings(r))
        self.assertIn('associated',text.replace('association','associated'))
        self.assertIn('Inspect residual',text)
        self.assertEqual(p_text(.0000001),'< 0.001')
        self.assertEqual(p_text(np.nan),'N/A')


class ExtendedRegressionUITests(unittest.TestCase):
    def test_controls_prediction_and_cache(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_string('''
import pandas as pd
import numpy as np
from utils.regression_ui import render_regression
x=np.arange(1.,61.)
df=pd.DataFrame({'y':2*x+.1*x*x+np.sin(x),'x':x,'g':['A','B']*30})
render_regression(df,['y','x'],['g'])
''').run(timeout=40)
        self.assertFalse(app.exception)
        from utils.regression_controls import widget_key
        app.checkbox(key=widget_key('predictor','g')).check().run(timeout=40)
        app.checkbox(key=widget_key('interaction',('x','g'))).check().run(timeout=40)
        app.checkbox(key=widget_key('power',('x',3))).check().run(timeout=40)
        self.assertFalse(app.exception)
        app.selectbox(key='reg_elimination').select('Backward Elimination — Hierarchy Preserved').run(timeout=40)
        self.assertFalse(app.exception)
        from utils import regression_ui
        with patch.object(regression_ui,'regression',side_effect=AssertionError('Unexpected refit')):
            next(b for b in app.button if b.label=='Predict response').click().run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label.startswith('Predicted') for m in app.metric))
        self.assertTrue(any('prediction interval for an individual' in m.value for m in app.markdown))
        self.assertTrue(any(t.label=='Regression Findings' for t in app.tabs))


if __name__=='__main__': unittest.main()
