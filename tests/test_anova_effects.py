import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from utils.anova_effects import model_family, sync_effects, prepare_random, fit_random


def balanced():
    # Orthogonal balanced fixture: within MS=2.5; between MS=200.
    return pd.DataFrame([(str(g), 10+g*2+e) for g in range(10) for e in [-2,-1,0,1,2]], columns=['Group','Y'])


class EffectsTests(unittest.TestCase):
    def test_defaults_and_routes(self):
        self.assertEqual(sync_effects({}, ['A','B']), {'A':'Fixed','B':'Fixed'})
        self.assertEqual(sync_effects({'A':'Random','B':'Fixed'}, ['A','C']), {'A':'Random','C':'Fixed'})
        self.assertIn('Welch', model_family({'A':'Fixed'}))
        self.assertIn('Fixed-effects', model_family({'A':'Fixed','B':'Fixed'}))
        self.assertIn('Mixed-effects', model_family({'A':'Fixed','B':'Random'}))
        self.assertIn('Random-effects', model_family({'A':'Random','B':'Random'}))

    def test_balanced_independent_anova_reference(self):
        df=balanced(); p=prepare_random(df,'Y',{'Group':'Random'}); r=fit_random(p)
        means=df.groupby('Group').Y.mean()
        msb=5*means.var(ddof=1)
        mse=df.groupby('Group').Y.var(ddof=1).mean()
        np.testing.assert_allclose(r['variance'].Variance, [(msb-mse)/5,mse],rtol=.002)
        self.assertAlmostEqual(r['icc'], ((msb-mse)/5)/((msb-mse)/5+mse),places=4)
        self.assertFalse(any('p-value' in c for c in r['variance']))

    def test_nested_and_crossed(self):
        rng=np.random.default_rng(18)
        rows=[]
        a=rng.normal(0,3,6); b=rng.normal(0,2,5)
        for i in range(6):
            for j in range(5):
                for k in range(4):
                    rows.append((str(i),str(j),a[i]+b[j]+rng.normal()))
        df=pd.DataFrame(rows,columns=['A','B','Y'])
        crossed=prepare_random(df,'Y',{'A':'Random','B':'Random'})
        nested=prepare_random(df,'Y',{'A':'Random','B':'Random'},parents={'B':'A'})
        self.assertEqual(crossed['groups'].Groups.tolist(),[6,5])
        self.assertEqual(nested['groups'].Groups.tolist(),[6,30])
        self.assertTrue(np.isfinite(fit_random(crossed)['variance'].Variance).all())
        self.assertTrue(np.isfinite(fit_random(nested)['variance'].Variance).all())
        mixed=fit_random(prepare_random(df.iloc[:-3],'Y',{'A':'Fixed','B':'Random'}))
        self.assertIn('SE',mixed['fixed'])
        from scipy.stats import norm
        np.testing.assert_allclose(mixed['fixed']['p-value (asymptotic)'],2*norm.sf(abs(mixed['fixed']['Estimate']/mixed['fixed']['SE'])))
        self.assertEqual(mixed['method'],'REML')

    def test_crossed_and_nested_independent_mean_square_references(self):
        # Balanced orthogonal constructions: REML interior solutions equal EMS estimates.
        a=np.arange(6)-2.5; b=np.arange(5)-2
        rows=[(str(i),str(j),float(10+4*a[i]+3*b[j]+a[i]*b[j]+e))
              for i in range(6) for j in range(5) for e in [-1.,0.,1.]]
        df=pd.DataFrame(rows,columns=['A','B','Y'])
        grand=df.Y.mean(); cell=df.groupby(['A','B']).Y.mean()
        ma=df.groupby('A').Y.mean(); mb=df.groupby('B').Y.mean()
        mse=df.groupby(['A','B']).Y.var(ddof=1).mean()
        msa=15*ma.var(ddof=1); msb=18*mb.var(ddof=1)
        ssab=sum(3*(v-ma[i]-mb[j]+grand)**2 for (i,j),v in cell.items())
        msab=ssab/20
        expected=[(msa-msab)/15,(msb-msab)/18,(msab-mse)/3,mse]
        actual=fit_random(prepare_random(df,'Y',{'A':'Random','B':'Random'},[('A','B')]))
        np.testing.assert_allclose(actual['variance'].Variance,expected,rtol=.005)
        # In the explicitly nested interpretation, B labels are local to A.
        msnested=sum(3*(v-ma[i])**2 for (i,j),v in cell.items())/(6*4)
        expected=[(msa-msnested)/15,(msnested-mse)/3,mse]
        actual=fit_random(prepare_random(df,'Y',{'A':'Random','B':'Random'},parents={'B':'A'}))
        np.testing.assert_allclose(actual['variance'].Variance,expected,rtol=.005)

    def test_confounds_replication_levels(self):
        df=balanced()
        with self.assertRaisesRegex(ValueError,'two observed'):
            prepare_random(df.assign(Group='one'),'Y',{'Group':'Random'})
        with self.assertRaisesRegex(ValueError,'confounded'):
            prepare_random(df.assign(Group=np.arange(len(df))),'Y',{'Group':'Random'})
        df['Copy']=df.Group
        with self.assertRaisesRegex(ValueError,'confounded'):
            prepare_random(df,'Y',{'Group':'Random','Copy':'Random'})
        with self.assertRaisesRegex(ValueError,'cycle'):
            prepare_random(df,'Y',{'Group':'Random','Copy':'Random'},parents={'Group':'Copy','Copy':'Group'})

    def test_nonfinite_and_small_group_warning(self):
        df=balanced().iloc[:15].astype({'Y':float});df.loc[0,'Y']=np.inf
        p=prepare_random(df,'Y',{'Group':'Random'})
        self.assertEqual(p['excluded'],1)
        r=fit_random(p)
        self.assertFalse(r['inference'])
        self.assertNotIn('p-value (asymptotic)',r['fixed'])

    def test_unbalanced_independent_profile_reml_reference(self):
        from scipy.optimize import minimize_scalar
        df=balanced().drop(index=[0,1,7,13,22])
        codes=pd.factorize(df.Group)[0]; z=np.eye(df.Group.nunique())[codes]
        y=df.Y.to_numpy(); x=np.ones(len(y)); n=len(y)
        def profile(logratio, return_scale=False):
            v=np.eye(n)+np.exp(logratio)*(z@z.T)
            vi_y=np.linalg.solve(v,y); vi_x=np.linalg.solve(v,x)
            beta=x@vi_y/(x@vi_x); residual=y-beta
            scale=residual@np.linalg.solve(v,residual)/(n-1)
            if return_scale: return scale
            return .5*(np.linalg.slogdet(v)[1]+np.log(x@vi_x)+(n-1)*np.log(scale))
        ref=minimize_scalar(profile,bounds=(-15,15),method='bounded')
        scale=profile(ref.x,True)
        r=fit_random(prepare_random(df,'Y',{'Group':'Random'}))
        np.testing.assert_allclose(r['variance'].Variance,[np.exp(ref.x)*scale,scale],rtol=.002)

    def test_boundary_and_redundant_nested_interaction(self):
        df=balanced(); df['Y']=np.tile([-2.,-1.,0.,1.,2.],10)
        with self.assertRaisesRegex(ValueError,'Singular|did not converge'):
            fit_random(prepare_random(df,'Y',{'Group':'Random'}))
        df['Parent']=np.repeat(['a','b'],25)
        with self.assertRaisesRegex(ValueError,'confounded'):
            prepare_random(df,'Y',{'Group':'Random','Parent':'Random'},[('Group','Parent')],{'Group':'Parent'})

    def test_nonconvergence(self):
        p=prepare_random(balanced(),'Y',{'Group':'Random'})
        with patch('utils.anova_effects.MixedLM.fit') as fit:
            fit.return_value.converged=False
            with self.assertRaisesRegex(ValueError,'did not converge'):
                fit_random(p)

class EffectsUITests(unittest.TestCase):
    def test_switching_report_and_dataset(self):
        from streamlit.testing.v1 import AppTest
        script = """
import pandas as pd
import streamlit as st
from utils.anova_ui import render_anova
st.session_state['file_id'] = st.selectbox('Dataset', ['one','two'])
df=pd.DataFrame([(str(g),float(10+g*2+e)) for g in range(10) for e in [-2,-1,0,1,2]],columns=['Group','Y'])
render_anova(df,'Group','Y')
"""
        app=AppTest.from_string(script).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value,'Fixed')
        app.radio[0].set_value('Random').run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('One-factor random' in m.value for m in app.markdown))
        self.assertFalse(any('Variance' in t.value for t in app.dataframe))
        next(w for w in app.selectbox if 'dependence remains' in w.label).select('None expected: group intercepts and selected interactions describe the dependence').run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('Variance' in t.value for t in app.dataframe))
        self.assertEqual([t.label for t in app.tabs], ['Overview', 'Group Comparisons', 'Diagnostics', 'ANOVA Findings'])
        self.assertEqual(len(app.tabs[0].get('plotly_chart')), 1)
        self.assertEqual(len(app.tabs[1].get('plotly_chart')), 2)
        self.assertEqual(len(app.tabs[2].get('plotly_chart')), 2)
        self.assertFalse(any('Adjusted p-value' in t.value for t in app.dataframe))
        app.button(key='report_add_rel_anova_random').click().run(timeout=45)
        import copy
        before=copy.deepcopy(app.session_state['cozy_report']['blocks'])
        app.radio[0].set_value('Fixed').run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual(before,app.session_state['cozy_report']['blocks'])
        app.selectbox[0].select('two').run(timeout=45)
        self.assertEqual(app.radio[0].value,'Fixed')

    def test_factorial_controls_and_removed_factor(self):
        from streamlit.testing.v1 import AppTest
        script = """
import pandas as pd
from utils.factorial_anova_ui import render_factorial_anova
df=pd.DataFrame([(str(a),str(b),float(a*3+b+e)) for a in range(6) for b in range(5) for e in [-1,0,1]],columns=['A','B','Y'])
render_factorial_anova(df,['Y'],['A','B'])
"""
        app=AppTest.from_string(script).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual([w.value for w in app.radio if w.key != 'fac_interaction_mode'],['Fixed','Fixed'])
        self.assertTrue(any(m.value == '**Factor Effect Type**' for m in app.markdown))
        summary = next(t.value for t in app.dataframe if 'Model detail' in t.value.columns
                       and 'Analysis method' in t.value['Model detail'].tolist())
        self.assertEqual(summary.loc[summary['Model detail'] == 'Analysis method', 'Value'].iloc[0],
                         'Fixed-effects Factorial ANOVA')
        app.radio[1].set_value('Random').run(timeout=45)
        self.assertFalse(app.exception)
        self.assertFalse(any('Automatically selected model:' in m.value for m in app.markdown))
        self.assertFalse(any('Sum of Squares' in t.value for t in app.dataframe))
        app.radio[0].set_value('Random').run(timeout=45)
        self.assertFalse(any('Automatically selected model:' in m.value for m in app.markdown))
        self.assertFalse(any(w.key == 'fac_x' for w in app.multiselect))
        self.assertTrue(any('**Factors (X):** A, B' in m.value for m in app.markdown))

if __name__=='__main__': unittest.main()
