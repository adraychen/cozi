"""Exploratory distribution comparison with refitted parametric bootstrap."""
import numpy as np
from scipy import stats


def fit_model(name, x, trials=None):
    if name == 'Normal':
        params = stats.norm.fit(x)
        return stats.norm(*params), 2
    if name == 'Lognormal':
        params = stats.lognorm.fit(x, floc=0)
        return stats.lognorm(*params), 2
    if name == 'Exponential':
        return stats.expon(loc=0, scale=float(x.mean())), 1
    if name == 'Uniform':
        return stats.uniform(loc=float(x.min()), scale=float(np.ptp(x))), 2
    if name == 'Poisson':
        return stats.poisson(mu=float(x.mean())), 1
    if name == 'Geometric (failures before success)':
        return stats.geom(p=1 / (float(x.mean()) + 1), loc=-1), 1
    if name == 'Binomial':
        return stats.binom(n=trials, p=float(x.mean()) / trials), 1
    raise ValueError('Unknown distribution')


def cdf_distance(x, model, discrete):
    values, counts = np.unique(x, return_counts=True)
    right = np.cumsum(counts) / len(x)
    left = right - counts / len(x)
    before = model.cdf(values - 1) if discrete else model.cdf(values)
    return float(max(np.max(np.abs(right - model.cdf(values))), np.max(np.abs(left - before))))


def fit_distributions(values, kind='Auto', trials=None, simulations=199):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    original_n = len(x)
    discrete = bool(len(x) and np.all(x == np.floor(x))) if kind == 'Auto' else kind == 'Discrete'
    result = {'type': 'Discrete' if discrete else 'Continuous', 'n': original_n, 'used': original_n,
              'candidates': [], 'best': None, 'quality': 'Not assessed', 'conclusion': ''}
    if len(x) < 20 or np.unique(x).size < 2:
        result['conclusion'] = 'Insufficient data: need at least 20 finite observations and two distinct values.'
        return result
    if discrete and (np.any(x < 0) or np.any(x != np.floor(x))):
        result['conclusion'] = 'No suitable theoretical distribution identified among the supported nonnegative integer families.'
        return result
    if trials is not None and (trials < 1 or trials != int(trials) or np.any(x > trials)):
        raise ValueError('Binomial trials must be a positive integer at least as large as every observed count.')
    rng = np.random.default_rng(42)
    if len(x) > 3000:
        x = rng.choice(x, 3000, replace=False)
    result['used'] = len(x)
    if discrete:
        names = ['Poisson', 'Geometric (failures before success)']
        if trials is not None:
            names.append('Binomial')
    else:
        names = ['Normal', 'Uniform']
        if np.all(x > 0):
            names += ['Lognormal', 'Exponential']
    # Common bins per comparison, with explicit tails. SSE compares probability masses.
    edges = np.histogram_bin_edges(x, bins=min(30, max(5, int(np.sqrt(len(x))))))
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    observed = np.histogram(x, bins=edges)[0] / len(x)
    for name in names:
        try:
            model, k = fit_model(name, x, trials)
            log_likelihood = np.sum(model.logpmf(x) if discrete else model.logpdf(x))
            if not np.isfinite(log_likelihood):
                continue
            distance = cdf_distance(x, model, discrete)
            # Histogram bins are [left,right), with the final infinite tail included.
            cdf_edges = model.cdf(np.ceil(edges) - 1) if discrete else model.cdf(edges)
            sse = float(np.sum((observed - np.diff(cdf_edges)) ** 2))
            exceed = 0
            successful = 0
            for _ in range(simulations):
                sample = model.rvs(size=len(x), random_state=rng)
                try:
                    fitted, _ = fit_model(name, sample, trials)
                    d = cdf_distance(sample, fitted, discrete)
                    if np.isfinite(d):
                        successful += 1
                        exceed += d >= distance
                except (ValueError, FloatingPointError):
                    pass
            p = (exceed + 1) / (successful + 1) if successful == simulations else None
            quality = 'Not assessed' if p is None else ('Poor' if p < 0.05 else 'Not rejected')
            result['candidates'].append({'Distribution': name, 'SSE': sse, 'AIC': float(2*k - 2*log_likelihood),
                                          'KS statistic': distance, 'Bootstrap p-value': p, 'Fit quality': quality})
        except (ValueError, FloatingPointError):
            continue
    result['candidates'].sort(key=lambda item: item['AIC'])
    if not result['candidates']:
        result['conclusion'] = 'No suitable theoretical distribution identified among the tested families.'
        return result
    result['best'] = result['candidates'][0]
    result['quality'] = result['best']['Fit quality']
    if all(c['Fit quality'] == 'Poor' for c in result['candidates']):
        result['conclusion'] = 'No suitable theoretical distribution identified among the tested families.'
    elif result['quality'] == 'Poor':
        result['conclusion'] = 'The lowest-AIC candidate fits poorly. Review the other candidates; no distribution is established.'
    else:
        result['conclusion'] = 'The candidate is not rejected by this exploratory check; this does not establish a theoretical generating process.' if result['quality'] == 'Not rejected' else 'Goodness of fit could not be assessed reliably.'
    return result
