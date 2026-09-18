import unittest
import numpy as np
import pandas as pd
from utils.variable_metadata import *
from utils.regression_analysis import regression

class MetadataTests(unittest.TestCase):
    def test_defaults(self):
        cases=[([0,1,2,3,4],'Numerical','Discrete'),([1.2,3.5,8.9],'Numerical','Continuous'),([0,1],'Numerical','Discrete'),(['Yes','No'],'Categorical','Binary'),(['Red','Blue','Green'],'Categorical','Nominal'),(['High','Low','Medium'],'Categorical','Ordinal'),([True,False],'Categorical','Binary')]
        for values,t,sub in cases:
            m=infer_metadata(pd.DataFrame({'x':values}))['x']
            self.assertEqual((m['variable_type'],m['variable_subtype']),(t,sub))
        self.assertEqual(infer_metadata(pd.DataFrame({'x':['High','Low','Medium']}))['x']['order'],['Low','Medium','High'])

    def test_validation(self):
        for values,t,sub in [(['A','B','C'],'Categorical','Binary'),(['A',None],'Categorical','Binary'),(['a','b'],'Numerical','Continuous'),([1.2,2.5],'Numerical','Discrete'),([0,1],'Numerical','Binary'),([0,1],'Categorical','Continuous')]:
            with self.assertRaises(ValueError): validate_type_subtype(pd.Series(values),t,sub)
        validate_type_subtype(pd.Series([1.,2.,3.]),'Numerical','Discrete')
        validate_type_subtype(pd.Series(['1','2',None]),'Numerical','Continuous')
        validate_type_subtype(pd.Series(['A','B',None]),'Categorical','Binary')
        with self.assertRaises(ValueError): validate_type_subtype(pd.Series(['A','B']),'Categorical','Ordinal',['A','A'])

    def test_temporal_and_numeric_conversion(self):
        d=pd.DataFrame({'when':pd.date_range('2020-01-01',periods=3),'number':['1','2',None]})
        m=infer_metadata(d)
        self.assertEqual(m['when']['variable_type'],'Date / Time')
        m=apply_user_override(m,'number',d.number,'Numerical','Discrete')
        data,groups=effective_data(d,m)
        self.assertTrue(pd.api.types.is_numeric_dtype(data.number))
        self.assertEqual(data.number.isna().sum(),1)
        self.assertEqual(d.number.iloc[0],'1')
        self.assertIn('when',groups['datetime'])

    def test_persistence_replacement_and_pruning(self):
        d=pd.DataFrame({'x':[0,1],'z':[2,3]});state={}
        m=ensure_metadata(state,d,'one')
        state['variable_metadata']=apply_user_override(m,'x',d.x,'Categorical','Binary')
        self.assertEqual(ensure_metadata(state,d,'one')['x']['source'],'User override')
        self.assertNotIn('z',ensure_metadata(state,d[['x']],'one'))
        self.assertEqual(ensure_metadata(state,d,'two')['x']['variable_type'],'Numerical')

    def test_effective_regression_and_targets(self):
        rng=np.random.default_rng(1)
        d=pd.DataFrame({'x':[0,1]*30,'y':rng.normal(size=60)})
        m=infer_metadata(d);effective,groups=effective_data(d,m)
        self.assertIn('x',groups['numerical'])
        self.assertFalse(regression(effective,'y',['x'],categorical=groups['categorical'])['spec']['categorical'])
        m=apply_user_override(m,'x',d.x,'Categorical','Binary');effective,groups=effective_data(d,m)
        self.assertNotIn('x',groups['numerical'])
        self.assertIn('x',regression(effective,'y',['x'],categorical=groups['categorical'])['spec']['categorical'])
        for values in [[0,1],[True,False],['Yes','No'],['A','B']]:
            self.assertEqual(metadata_target_type(pd.Series(values),'Categorical','Binary'),'binary')
        self.assertEqual(metadata_target_type(d.x,'Numerical','Discrete'),'continuous_numerical')

    def test_ordinal_frequency(self):
        from utils.exploration import frequency_table
        d=pd.DataFrame({'x':['High','Low','Medium','High']});m=infer_metadata(d)
        data,_=effective_data(d,m)
        self.assertEqual(frequency_table(data.x).Value.tolist(),['Low','Medium','High'])
        self.assertEqual(d.x.tolist(),['High','Low','Medium','High'])

    def test_logistic_labels_and_categorical_predictors(self):
        from utils.modelling import detect_target_type, _build_preprocessor
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, roc_auc_score
        for labels in [[0,1],[True,False],['Yes','No'],['Pass','Fail']]:
            d=pd.DataFrame({'target':labels*20,'flag':[0,1]*20,'measurement':np.arange(40)})
            m=infer_metadata(d)
            for col in ['target','flag']:
                m=apply_user_override(m,col,d[col],'Categorical','Binary')
            data,groups=effective_data(d,m)
            self.assertEqual(detect_target_type(data.target,'Categorical','Binary'),'binary')
            self.assertIsInstance(data.flag.dtype,pd.CategoricalDtype)
            y,original=encode_binary_target(data.target)
            pipeline=Pipeline([('preprocessor',_build_preprocessor(['measurement'],['flag'],True)),('model',LogisticRegression())])
            pipeline.fit(data[['flag','measurement']],y)
            self.assertEqual(f1_score(y,pipeline.predict(data[['flag','measurement']])),1)
            self.assertEqual(roc_auc_score(y,pipeline.predict_proba(data[['flag','measurement']])[:,1]),1)
            self.assertEqual(set(original),set(labels))

class MetadataUI(unittest.TestCase):
    def test_editor_and_session(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_string('''
import pandas as pd
import streamlit as st
from utils.variable_metadata_ui import render_column_overview
new=st.checkbox('Replace dataset',key='replace')
df=pd.DataFrame({'flag':[0,1,0], 'text':['A','B','C']})
st.subheader('Column Overview')
render_column_overview(df,'two' if new else 'one')
''').run(timeout=30)
        self.assertFalse(app.exception)
        from utils.variable_overview_display import overview_html
        markup=overview_html(pd.DataFrame({'flag':[0,1,0], 'text':['A','B','C']}),app.session_state['variable_metadata'])
        self.assertIn('<th scope="col">Variable Type</th><th scope="col">Variable Subtype</th>',markup)
        self.assertNotIn('Classification Source',markup)
        self.assertNotIn('Category Order',markup)
        typ=next(s for s in app.selectbox if s.label=='Variable Type')
        typ.select('Categorical').run()
        sub=next(s for s in app.selectbox if s.label=='Variable Subtype')
        self.assertEqual(sub.options,['Binary','Nominal','Ordinal'])
        self.assertEqual(sub.value,'Binary')
        app.button(key='vm_apply').click().run()
        self.assertFalse(app.exception)
        app.run()
        self.assertEqual(app.session_state['variable_metadata']['flag']['variable_subtype'],'Binary')
        app.selectbox(key='vm_column').select('text').run()
        next(s for s in app.selectbox if s.label=='Variable Subtype').select('Binary').run()
        app.button(key='vm_apply').click().run()
        self.assertTrue(any('exactly two' in e.value for e in app.error))
        self.assertEqual(app.session_state['variable_metadata']['text']['variable_subtype'],'Nominal')
        app.checkbox(key='replace').check().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['variable_metadata']['flag']['variable_type'],'Numerical')

if __name__=='__main__': unittest.main()
