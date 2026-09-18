import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from utils.logistic_analysis import logistic_regression,predict_logistic,separation_status
from utils.regression_design import dependencies

class LogisticTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(401)
        self.d=pd.DataFrame({'x':rng.normal(size=360),'z':rng.normal(size=360),'g':np.tile(['A','B','C'],120),'b':np.tile(['No','Yes'],180)})
        self.d['y']=rng.binomial(1,expit(-.3+.7*self.d.x-.4*self.d.z+.5*(self.d.g=='B')))

    def test_numeric_fit_matches_logit(self):
        r=logistic_regression(self.d,'y',['x','z'],1,continuous=[])
        ref=sm.Logit(self.d.y,sm.add_constant(self.d[['x','z']])).fit(disp=False)
        np.testing.assert_allclose(r['model'].params,ref.params,atol=1e-7)
        np.testing.assert_allclose(r['model'].bse,ref.bse,atol=1e-6)
        np.testing.assert_allclose(r['coefficients']['Odds Ratio'],np.exp(ref.params),atol=1e-7)
        np.testing.assert_allclose(r['coefficients']['OR CI Lower'],np.exp(ref.conf_int().iloc[:,0]),atol=1e-6)
        self.assertAlmostEqual(r['lr'],ref.llr,places=6)
        self.assertAlmostEqual(r['lr_p'],ref.llr_pvalue,places=6)
        self.assertAlmostEqual(r['bic'],ref.bic,places=6)
        self.assertEqual(r['events'],int(self.d.y.sum()))

    def test_event_direction_and_probability_intervals(self):
        a=logistic_regression(self.d,'y',['x','z'],1,confidence=.9,continuous=[])
        b=logistic_regression(self.d,'y',['x','z'],0,confidence=.99,continuous=[])
        np.testing.assert_allclose(a['model'].params,-b['model'].params,atol=1e-7)
        pred=predict_logistic(a,pd.DataFrame({'x':[0,1],'z':[1,0]}))
        ref=a['model'].get_prediction(np.array([[1,0,1],[1,1,0]])).summary_frame(alpha=.1)
        np.testing.assert_allclose(pred,ref)
        reverse=predict_logistic(b,pd.DataFrame({'x':[0,1],'z':[1,0]}))
        np.testing.assert_allclose(pred['mean'],1-reverse['mean'])
        self.assertTrue(((reverse.mean_ci_upper-reverse.mean_ci_lower)>(pred.mean_ci_upper-pred.mean_ci_lower)).all())
        with self.assertRaises(ValueError): predict_logistic(a,{'x':[np.nan],'z':[0]})

    def test_categories_references_interactions_polynomials(self):
        r=logistic_regression(self.d,'y',['x','g','b'],1,categorical=['g','b'],references={'g':'B','b':'Yes'},interactions=[('x','g'),('g','b')],degrees={'x':3},continuous=[])
        ref=smf.glm('y ~ x*C(g, Treatment(reference="B")) + C(g, Treatment(reference="B"))*C(b, Treatment(reference="Yes")) + I(x**2) + I(x**3)',self.d,family=sm.families.Binomial()).fit()
        np.testing.assert_allclose(r['model'].fittedvalues,ref.fittedvalues,atol=1e-7)
        self.assertEqual(r['spec']['levels']['g'][0],'B')
        pred=predict_logistic(r,self.d.iloc[:4])
        np.testing.assert_allclose(pred['mean'],ref.get_prediction(self.d.iloc[:4]).predicted_mean,atol=1e-7)
        self.assertTrue(any("vs 'B'" in name for name in r['names']))
        with self.assertRaises(ValueError): predict_logistic(r,pd.DataFrame({'x':[0],'g':['new'],'b':['No']}))

    def test_ordinal_and_missing(self):
        d=self.d.copy();d['rating']=pd.Categorical(np.tile(['High','Low','Medium'],120),categories=['Low','Medium','High'],ordered=True)
        d.loc[3,'x']=np.nan;d.loc[5,'y']=np.nan;d.loc[6,'x']=np.inf
        r=logistic_regression(d,'y',['x','rating'],1,categorical=['rating'],continuous=[])
        self.assertEqual(r['excluded'],3)
        self.assertEqual(r['spec']['levels']['rating'][0],'Low')
        self.assertEqual(len(r['indices'][('main','rating')]),2)
        self.assertNotIn(4,r['influence']['Original data row'].tolist())

    def test_separation_complete_and_quasi(self):
        for d,expected in [(pd.DataFrame({'x':[-3,-2,-1,1,2,3],'y':[0,0,0,1,1,1]}),'Complete separation'),(pd.DataFrame({'x':[0,0,1,2,3,4],'y':[0,1,1,1,1,1]}),'Quasi-complete separation')]:
            with self.assertRaisesRegex(ValueError,expected): logistic_regression(d,'y',['x'],1)

    def test_invalid_and_nonconvergence(self):
        with self.assertRaisesRegex(ValueError,'did not converge'): logistic_regression(self.d,'y',['x','z'],1,maxiter=1)
        d=self.d.copy();d['z']=d.x
        with self.assertRaisesRegex(ValueError,'singular'): logistic_regression(d,'y',['x','z'],1)
        d['y']=2
        with self.assertRaises(ValueError): logistic_regression(d,'y',['x'],2)
        d=self.d.copy();d['z']=1
        with self.assertRaises(ValueError): logistic_regression(d,'y',['z'],1)
        d=self.d.copy();d.loc[d.y==1,'x']=np.nan
        with self.assertRaises(ValueError): logistic_regression(d,'y',['x'],1)
        with self.assertRaises(ValueError): logistic_regression(self.d,'y',['x'],1,interactions=[('x','missing')])

    def test_hierarchy_aic_and_fixed_sample(self):
        r=logistic_regression(self.d,'y',['x','z'],1,interactions=[('x','z')],degrees={'x':3},elimination=True,continuous=[])
        for block in r['active']: self.assertTrue(dependencies(block).issubset(set(r['active'])))
        self.assertEqual(r['n'],len(self.d))
        if not r['history'].empty: self.assertTrue((r['history']['Refitted AIC']<r['history']['Previous AIC']).all())

    def test_nonlinearity_and_influence(self):
        rng=np.random.default_rng(71);x=rng.uniform(-2,2,700)
        d=pd.DataFrame({'x':x,'y':rng.binomial(1,expit(-1+1.4*x*x))})
        r=logistic_regression(d,'y',['x'],1)
        self.assertEqual(r['linearity'].iloc[0]['Assessment'],'Possible nonlinearity')
        self.assertTrue(np.isfinite(r['influence']['Deviance Residual']).all())
        self.assertTrue(np.isfinite(r['influence']["Cook's Distance"]).all())
        reference=r['model'].get_influence()
        np.testing.assert_allclose(r['influence'].Leverage,reference.hat_matrix_diag)
        np.testing.assert_allclose(r['influence']["Cook's Distance"],reference.cooks_distance[0])

    def test_rare_levels_imbalance_and_vif(self):
        d=self.d.copy();d['rare']='usual';d.loc[:3,'rare']='rare';d.loc[:3,'y']=[0,1,0,1]
        r=logistic_regression(d,'y',['x','rare'],1,categorical=['rare'],continuous=[])
        self.assertTrue(r['sparse'].Flag.str.contains('Rare').any())
        d=self.d.copy();d['z']=d.x+np.random.default_rng(5).normal(0,.03,len(d))
        r=logistic_regression(d,'y',['x','z'],1,continuous=[])
        self.assertTrue((r['vif'].VIF>10).all())
        rng=np.random.default_rng(2);d['y']=rng.binomial(1,.035,len(d))
        r=logistic_regression(d,'y',['x'],1,continuous=[])
        self.assertIn('Limited outcome information',r['issues'].Issue.tolist())

    def test_unsupported_inputs_and_influential_row(self):
        d=self.d.copy();d['text']=['this is a long unprocessed narrative sentence '*6]*len(d)
        with self.assertRaisesRegex(ValueError,'free-text'): logistic_regression(d,'y',['text'],1,categorical=['text'])
        d['complex']=d.x.astype(complex)+1j
        with self.assertRaisesRegex(ValueError,'complex'): logistic_regression(d,'y',['complex'],1)
        d=self.d.copy();d.loc[0,'x']=10.;d.loc[0,'y']=0
        r=logistic_regression(d,'y',['x','z'],1,continuous=[])
        self.assertTrue(bool(r['influence'].iloc[0]['Flag']))

    def test_probability_plots(self):
        from utils.regression_effect_plots import adjusted_curves,effect_figure
        r=logistic_regression(self.d,'y',['x','g'],1,categorical=['g'],interactions=[('x','g')],degrees={'x':2},continuous=[])
        curves,settings=adjusted_curves(r,'x','g',prediction_function=predict_logistic)
        self.assertEqual(len(curves),3)
        for curve in curves:
            self.assertTrue(curve['prediction']['mean'].between(0,1).all())
        self.assertGreater(len(effect_figure(r,'x',curves,'Probability').data),0)

if __name__=='__main__': unittest.main()
