import unittest
import numpy as np
from utils.distribution_fit import fit_distributions, cdf_distance
from scipy import stats


class DistributionTests(unittest.TestCase):
    def test_bedrooms_reject_count_families(self):
        r = fit_distributions(np.tile([2,3,3,4,3,2,5,4,3], 60))
        self.assertEqual(r['type'], 'Discrete')
        self.assertEqual(r['quality'], 'Poor')
        self.assertIn('No suitable', r['conclusion'])

    def test_lognormal(self):
        x = np.random.default_rng(15).lognormal(1, .8, 600)
        r = fit_distributions(x)
        self.assertEqual(r['best']['Distribution'], 'Lognormal')
        self.assertEqual(r['quality'], 'Not rejected')

    def test_discrete_distance(self):
        self.assertEqual(cdf_distance(np.array([0,0,1,1]), stats.bernoulli(.5), True), 0)

    def test_degenerate_and_support(self):
        self.assertIsNone(fit_distributions([1]*40)['best'])
        self.assertIsNone(fit_distributions([-2,-1]*30)['best'])
        self.assertIsNone(fit_distributions([1,2,3])['best'])
        with self.assertRaises(ValueError):
            fit_distributions([1,2,3]*20, trials=2)

    def test_binomial_only_when_known(self):
        x = np.random.default_rng(10).binomial(10,.4,500)
        r = fit_distributions(x, trials=10)
        self.assertEqual(r['best']['Distribution'], 'Binomial')
        self.assertEqual(r['quality'], 'Not rejected')


if __name__ == '__main__':
    unittest.main()
