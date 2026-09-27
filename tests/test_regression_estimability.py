import unittest
import numpy as np
import pandas as pd
from utils.regression_analysis import regression
from utils.regression_design import specification, design, label, dependencies
from utils.regression_estimability import RegressionDesignError, explain_singularity


class EstimabilityTests(unittest.TestCase):
    def test_empty_interaction_cells_named_and_suggestion_repairs_rank(self):
        rng = np.random.default_rng(12)
        df = pd.DataFrame({'a':np.repeat(['A','A','B'], 15),
                           'b':np.repeat(['X','Y','X'], 15), 'y':rng.normal(size=45)})
        with self.assertRaises(RegressionDesignError) as caught:
            regression(df, 'y', ['a','b'], interactions=[('a','b')], elimination=True)
        error = caught.exception
        self.assertEqual([r['Term'] for r in error.details], ['a × b'])
        self.assertIn('only zeros', error.details[0]['Issue'])
        self.assertEqual((error.n, error.coefficients, error.rank), (45,4,3))
        regression(df, 'y', ['a','b'])

    def test_overlap_and_hierarchy(self):
        x = np.arange(30.)
        df = pd.DataFrame({'x':x, 'duplicate':2*x, 'y':x + np.sin(x)})
        spec, clean, _ = specification(df, 'y', ['x','duplicate'], interactions=[('x','duplicate')], degrees={'x':3,'duplicate':3})
        matrix, names, indices = design(clean, spec)
        error = explain_singularity(matrix, names, indices, spec['blocks'])
        removed = {r['Term'] for r in error.details}
        self.assertIn('duplicate', removed)
        self.assertIn('x × duplicate', removed)
        self.assertIn('x (centered)', error.details[0]['Issue'])
        retained = [b for b in spec['blocks'] if label(b) not in removed]
        repaired, _, _ = design(clean, spec, retained)
        self.assertEqual(np.linalg.matrix_rank(repaired), repaired.shape[1])
        for block in retained:
            self.assertTrue(dependencies(block).issubset(retained))

    def test_binary_polynomial_and_scaling_only(self):
        df = pd.DataFrame({'x':[0.,1.]*20, 'y':np.arange(40.)})
        spec, clean, _ = specification(df, 'y', ['x'], degrees={'x':3})
        matrix, names, indices = design(clean, spec)
        error = explain_singularity(matrix, names, indices, spec['blocks'])
        self.assertEqual([r['Term'] for r in error.details], ['x²','x³'])
        self.assertIn('constant', error.details[0]['Issue'])
        rng = np.random.default_rng(14)
        df = pd.DataFrame({'large':rng.normal(size=40)*1e18, 'small':rng.normal(size=40), 'y':rng.normal(size=40)})
        with self.assertRaises(RegressionDesignError) as caught:
            regression(df, 'y', ['large','small'])
        self.assertIn('no exact redundancy', str(caught.exception))
        self.assertTrue(all('Rescale' in row['Suggested action'] for row in caught.exception.details))

    def test_ui_shows_term_guidance_without_changing_selections(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import numpy as np
from utils.regression_ui import render_regression
st.session_state['rel_interaction_terms'] = [('a','b')]
df = pd.DataFrame({'a':np.repeat(['A','A','B'],15), 'b':np.repeat(['X','Y','X'],15), 'y':np.arange(45.)})
render_regression(df, ['y'], ['a','b'], selection=('y',['a','b']))
''').run(timeout=40)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertTrue(any('Suggested action' in d.value.columns for d in app.dataframe))
        self.assertTrue(any('Backward elimination cannot start' in i.value for i in app.info))
        self.assertEqual(app.session_state['rel_interaction_terms'], [('a','b')])


if __name__ == '__main__':
    unittest.main()
