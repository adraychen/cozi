import unittest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from utils.regression_analysis import regression
from utils.regression_effect_plots import adjusted_curves, effect_figure, retained_interactions, retained_polynomials

class EffectPlots(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(199)
        self.df=pd.DataFrame(dict(x=rng.uniform(2,10,240),z=rng.normal(5,1,240),g=['A','B','C']*80,h=['E','E','W','W']*60))
        d=self.df
        d['y']=2+d.x+d.x*d.z+.3*d.x**2+3*(d.g=='B')+2*d.x*(d.g=='C')+rng.normal(size=240)

    def compare(self,r,x,moderator,formula):
        curves,settings=adjusted_curves(r,x,moderator)
        reference=smf.ols(formula,self.df).fit()
        for c in curves:
            np.testing.assert_allclose(c['prediction'],reference.get_prediction(c['input']).summary_frame(),rtol=1e-8,atol=1e-8)
            for v in set(r['spec']['predictors'])-{x,moderator}:
                expected=r['spec']['levels'][v][0] if v in r['spec']['categorical'] else r['clean'][v].mean()
                self.assertTrue((c['input'][v]==expected).all())
        return curves,settings

    def test_numeric_interaction(self):
        r=regression(self.df,'y',['x','z','g'],categorical=['g'],interactions=[('x','z')])
        curves,_=self.compare(r,'x','z','y ~ x*z + C(g)')
        np.testing.assert_allclose([c['input'].z.iloc[0] for c in curves],self.df.z.quantile([.25,.5,.75]))
        fig=effect_figure(r,'x',curves,'test')
        self.assertEqual(sum(t.fill=='tonexty' for t in fig.data),3)

    def test_category_interaction_and_polynomial(self):
        r=regression(self.df,'y',['x','z','g'],categorical=['g'],interactions=[('x','g')],degrees={'x':3})
        curves,_=self.compare(r,'x','g','y ~ x*C(g) + I(x**2) + I(x**3) + z')
        self.assertEqual(retained_polynomials(r),['x'])
        for c in curves:
            observed=self.df.loc[self.df.g==c['input'].g.iloc[0],'x']
            self.assertAlmostEqual(c['input'].x.min(),observed.min())
            self.assertAlmostEqual(c['input'].x.max(),observed.max())

    def test_two_categories(self):
        r=regression(self.df,'y',['g','h','x'],categorical=['g','h'],interactions=[('g','h')])
        curves,_=self.compare(r,'g','h','y ~ C(g)*C(h) + x')
        fig=effect_figure(r,'g',curves,'test')
        self.assertEqual(fig.layout.xaxis.type,'category')
        self.assertTrue(all(t.error_y.array is not None for t in fig.data))

    def test_multiple_predictor_polynomial(self):
        r=regression(self.df,'y',['x','z','g'],categorical=['g'],degrees={'x':2})
        curves,settings=self.compare(r,'x',None,'y ~ x + I(x**2) + z + C(g)')
        self.assertEqual(len(curves),1)
        self.assertEqual(set(settings.Predictor),{'z','g'})

    def test_eliminated_terms_hidden(self):
        d=self.df.copy();x=d.x-d.x.mean();z=d.z-d.z.mean()
        design=np.column_stack([np.ones(len(d)),x,z,x*x,x*z])
        noise=np.random.default_rng(83).normal(size=len(d))
        noise-=design@np.linalg.lstsq(design,noise,rcond=None)[0]
        d['y']=3+2*x+noise
        r=regression(d,'y',['x','z'],interactions=[('x','z')],degrees={'x':2},elimination=True)
        self.assertEqual(retained_interactions(r),[])
        self.assertEqual(retained_polynomials(r),[])

    def test_perfect_fit_and_invalid_selection(self):
        d=self.df.copy();d['y']=2*d.x*d.z
        r=regression(d,'y',['x','z'],interactions=[('x','z')])
        self.assertTrue(r['perfect_fit'])
        curves,_=adjusted_curves(r,'x','z')
        self.assertEqual(len(effect_figure(r,'x',curves,'test').data),3)
        r=regression(self.df,'y',['x','g'],categorical=['g'],interactions=[('x','g')])
        with self.assertRaises(ValueError):
            adjusted_curves(r,'x','g',['unknown'])

class EffectPlotUI(unittest.TestCase):
    def test_retained_controls_and_cached_display(self):
        from streamlit.testing.v1 import AppTest
        from unittest.mock import patch
        from utils import regression_ui
        app=AppTest.from_string("""
import pandas as pd
import numpy as np
from utils.regression_ui import render_regression
x=np.linspace(2,10,120)
df=pd.DataFrame({'y':2+x+x*x+.2*x**3+np.sin(x),'x':x,'g':['A','B']*60})
render_regression(df,['y','x'],['g'])
""").run(timeout=40)
        self.assertFalse(app.exception)
        self.assertFalse(any(s.label=='Retained interaction to plot' for s in app.selectbox))
        app.multiselect(key='rel_predictors').set_value(['x','g']).run(timeout=40)
        app.multiselect(key='reg_interactions').set_value([('x','g')]).run(timeout=40)
        app.selectbox(key='reg_degree_x').select(3).run(timeout=40)
        self.assertFalse(app.exception)
        labels={s.label for s in app.selectbox}
        self.assertTrue({'Retained interaction to plot','Predictor with retained polynomial terms','Interacting predictor for separate curves'} <= labels)
        with patch.object(regression_ui,'regression',side_effect=AssertionError('Unexpected refit')):
            app.run(timeout=40)
        self.assertFalse(app.exception)

if __name__=='__main__':
    unittest.main()
