import unittest
import numpy as np
import pandas as pd
from utils.regression_analysis import regression


class MixedRegressionTests(unittest.TestCase):
    def test_numeric_and_categorical_coefficients(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=120)
        group = np.array(['A', 'B', 'C']*40)
        y = 3 + 2*x + (group == 'B')*5 - (group == 'C')*4
        result = regression(pd.DataFrame({'x':x, 'group':group, 'y':y}), 'y', ['x','group'])
        np.testing.assert_allclose(result['coefficients']['Coefficient'], [3,2,5,-4], atol=1e-10)
        self.assertEqual(result['k'], 4)
        self.assertEqual(len(result['coding']), 2)

    def test_categorical_only_with_missing(self):
        data = pd.DataFrame({'g':['A','B','A','B',None,'B'], 'y':[1.,8,3,10,999,12]})
        result = regression(data,'y',['g'])
        np.testing.assert_allclose(result['coefficients']['Coefficient'], [2,8])
        self.assertEqual(result['excluded'], 1)

    def test_numeric_category_override(self):
        data = pd.DataFrame({'g':[1,2,3]*5, 'y':np.arange(15.)})
        result = regression(data,'y',['g'], categorical=['g'])
        self.assertEqual(len(result['coding']), 2)

    def test_invalid_encoding(self):
        for data in [pd.DataFrame({'g':['A']*5, 'y':range(5)}),
                     pd.DataFrame({'g':['A','B','C'], 'y':[1,2,3]})]:
            with self.assertRaises(ValueError):
                regression(data,'y',['g'])

    def test_mixed_ui(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import numpy as np
import pandas as pd
from utils.relationship_ui import render_relationship_analysis
x = np.arange(30.)
df = pd.DataFrame({'y': 2*x + np.sin(x), 'x':x, 'g':['A','B','C']*10})
render_relationship_analysis(df, ['y','x'], ['g'], [])
''').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='rel_method').value, 'Linear regression')
        self.assertEqual(app.selectbox(key='rel_target_False').label, 'Target / dependent variable (Y)')
        app.multiselect(key='rel_predictors').set_value(['x','g']).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('0/1 indicators' in item.value for item in app.caption))
        app.multiselect(key='rel_predictors').set_value(['g']).run(timeout=30)
        self.assertFalse(app.exception)
        app.multiselect(key='rel_predictors').set_value([]).run(timeout=30)
        self.assertTrue(any('at least one independent' in item.value for item in app.info))


if __name__ == '__main__':
    unittest.main()
