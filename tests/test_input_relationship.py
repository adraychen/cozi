import unittest
import numpy as np
import pandas as pd
from utils.manual_input import parse_csv
from utils.relationship_stats import regression, group_anova


class InputRelationshipTests(unittest.TestCase):
    def test_csv(self):
        d = parse_csv('Name,Value\n"East, West",12\nOther,13')
        self.assertEqual(d.iloc[0,0], 'East, West')
        self.assertEqual(d.Value.sum(),25)
        self.assertEqual(parse_csv('ID\n001').iloc[0,0],'001')

    def test_limits_and_validation(self):
        self.assertEqual(len(parse_csv('A\n'+'1\n'*500)),500)
        for text in ['A\n'+'1\n'*501, 'A,A\n1,2', 'A,B\n1', 'A,B\n,', ',B\n1,2', 'A\n"bad']:
            with self.assertRaises(ValueError): parse_csv(text)
        headers=','.join('C'+str(i) for i in range(26))
        with self.assertRaises(ValueError): parse_csv(headers+'\n'+','.join(['1']*26))

    def test_regression(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=100); z=rng.normal(size=100)
        result=regression(pd.DataFrame({'y':2+3*x-4*z,'x':x,'z':z}), 'y',['x','z'])
        np.testing.assert_allclose(result['coefficients'].Coefficient,[2,3,-4],atol=1e-10)
        self.assertAlmostEqual(result['r2'],1)
        with self.assertRaises(ValueError): regression(pd.DataFrame({'y':x,'x':x,'z':x}), 'y',['x','z'])

    def test_anova(self):
        table,f,p,n=group_anova(pd.DataFrame({'group':['A']*3+['B']*3,'value':[1,2,3,7,8,9]}),'group','value')
        self.assertAlmostEqual(f,54)
        self.assertLess(p,.01)
        self.assertEqual(n,0)


if __name__=='__main__': unittest.main()
