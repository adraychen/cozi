"""Equal-weight reference-grid means and contrasts for fitted factorial models.

All intervals use the fitted classical OLS covariance and residual t distribution.
No model terms, response values, or fitting choices are changed here.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import stats
from statsmodels.stats.multitest import multipletests


MEAN = 'Adjusted mean (Estimated Marginal Mean)'


def significant_interactions(prepared, result):
    return [members for term, (_, kind, members) in prepared['terms'].items()
            if kind == 'Interaction effect' and result['term_pvalues'][term] < result['alpha']]


def _reference_grid(prepared, result):
    if '_emm_grid' in result:
        return result['_emm_grid']
    model = result['model']
    size = prepared['possible_cells']
    if size > 100_000 or size * len(model.params) > 2_000_000:
        raise ValueError('The equal-weight reference grid is too large for interactive follow-up. Reduce unnecessary factors or levels to request adjusted means; the fitted ANOVA is unchanged.')
    factors = prepared['factors']
    grid = pd.DataFrame(product(*(range(len(prepared['levels'][f])) for f in factors)),
                        columns=[prepared['aliases'][f] for f in factors])
    for f in factors:
        alias = prepared['aliases'][f]
        grid[alias] = pd.Categorical(grid[alias], categories=range(len(prepared['levels'][f])))
    design = dmatrix(prepared['formula'].split('~', 1)[1], grid, return_type='dataframe')
    matrix = design[model.model.exog_names].to_numpy()
    result['_emm_grid'] = grid, matrix
    return grid, matrix


def emm_vectors(prepared, result, target, fixed=None):
    """Return labels and design-row averages; nuisance levels have equal weights."""
    target, fixed = list(target), dict(fixed or {})
    if (not target or len(set(target)) != len(target) or set(target) & set(fixed)
            or any(f not in prepared['factors'] for f in target + list(fixed))):
        raise ValueError('Choose distinct model factors for adjusted means and conditioning levels.')
    grid, matrix = _reference_grid(prepared, result)
    mask = np.ones(len(grid), dtype=bool)
    for factor, value in fixed.items():
        if value not in prepared['levels'][factor]:
            raise ValueError(f'Unknown conditioning level for {factor}.')
        code = prepared['levels'][factor].index(value)
        mask &= np.asarray(grid[prepared['aliases'][factor]] == code)
    labels, vectors = [], []
    for codes in product(*(range(len(prepared['levels'][f])) for f in target)):
        selected = mask.copy()
        for f, code in zip(target, codes):
            selected &= np.asarray(grid[prepared['aliases'][f]] == code)
        if not selected.any():
            raise ValueError('This adjusted mean has no reference-grid support.')
        labels.append(tuple(prepared['levels'][f][code] for f, code in zip(target, codes)))
        vectors.append(matrix[selected].mean(axis=0))
    return labels, np.asarray(vectors)


def _estimates(result, vectors):
    model = result['model']
    vectors = np.atleast_2d(vectors)
    # Explicit row-space check also protects callers supplying a singular fit.
    if '_emm_projection' not in result:
        result['_emm_projection'] = model.model.pinv_wexog @ model.model.exog
    projection = result['_emm_projection']
    if not np.allclose(vectors @ projection, vectors, atol=1e-8, rtol=1e-8):
        raise ValueError('Requested adjusted means or contrasts are not estimable from this model. Review empty cells and confounded factors.')
    means = vectors @ np.asarray(model.params)
    variances = np.einsum('ij,jk,ik->i', vectors, np.asarray(model.cov_params()), vectors)
    if (model.df_resid <= 0 or not np.isfinite(means).all()
            or not np.isfinite(variances).all() or (variances <= 0).any()):
        raise ValueError('Uncertainty for the requested adjusted means or contrasts cannot be estimated reliably.')
    return means, np.sqrt(variances)


def estimated_marginal_means(prepared, result, target, fixed=None):
    labels, vectors = emm_vectors(prepared, result, target, fixed)
    means, se = _estimates(result, vectors)
    half = stats.t.ppf(.975, result['model'].df_resid) * se
    # Generic level-column names avoid collisions with arbitrary dataset names.
    table = pd.DataFrame(labels, columns=['Level' if len(target) == 1 else f'Factor {i+1} level'
                                         for i in range(len(target))])
    table[MEAN], table['SE'] = means, se
    table['95% CI lower'], table['95% CI upper'] = means-half, means+half
    return table


def pairwise_emmeans(prepared, result, factor, fixed=None):
    if factor not in prepared['factors'] or not 2 <= len(prepared['levels'][factor]) <= 30:
        raise ValueError('Pairwise comparisons require a factor with 2–30 observed levels.')
    labels, vectors = emm_vectors(prepared, result, [factor], fixed)
    pairs = list(combinations(range(len(vectors)), 2))
    contrasts = np.asarray([vectors[a]-vectors[b] for a, b in pairs])
    difference, se = _estimates(result, contrasts)
    df = result['model'].df_resid
    pvalues = 2 * stats.t.sf(np.abs(difference / se), df)
    adjusted = multipletests(pvalues, method='holm')[1]
    # Conservative simultaneous intervals for the same family of all level pairs.
    half = stats.t.ppf(1 - .05/(2*len(pairs)), df) * se
    return pd.DataFrame({
        'Comparison': [f'{labels[a][0]} − {labels[b][0]}' for a, b in pairs],
        'Adjusted mean difference': difference, 'SE': se,
        '95% simultaneous CI lower': difference-half,
        '95% simultaneous CI upper': difference+half,
        'p-value': pvalues, 'Holm-adjusted p-value': adjusted,
        'Significant': np.where(adjusted < result['alpha'], 'Yes', 'No')})


def simple_effects(prepared, result, factor, fixed):
    """Conditional means/pairs; all significant partners must be held fixed."""
    partners = {f for pair in significant_interactions(prepared, result)
                if factor in pair for f in pair if f != factor}
    if not partners or not partners.issubset(fixed):
        raise ValueError('Simple-effects comparisons must condition on every significant interaction partner of this factor.')
    # Do not extrapolate a simple comparison into an unobserved target/stratum.
    for value in prepared['levels'][factor]:
        mask = prepared['clean'][factor].eq(value)
        for f, level in fixed.items():
            if f not in prepared['factors'] or f == factor:
                raise ValueError('Choose other model factors as conditioning variables.')
            mask &= prepared['clean'][f].eq(level)
        if not mask.any():
            raise ValueError(f'Simple effect withheld: {factor} = {value} has no observations in this stratum. Sparse cells do not support this comparison; collect missing combinations or review the selected model.')
    return (estimated_marginal_means(prepared, result, [factor], fixed),
            pairwise_emmeans(prepared, result, factor, fixed))


def simple_effects_family(prepared, result, factor):
    """Both directions are supported via factor; adjust across all its strata."""
    partners = [f for f in prepared['factors'] if f != factor and any(
        factor in pair and f in pair for pair in significant_interactions(prepared, result))]
    if not partners:
        raise ValueError('Select a factor in a significant two-way interaction.')
    count = int(np.prod([len(prepared['levels'][f]) for f in partners]))
    levels = len(prepared['levels'][factor])
    if count > 60 or count * levels*(levels-1)//2 > 2000:
        raise ValueError('Too many conditional comparisons for an interpretable display. Review unnecessary factors or interactions before requesting simple effects.')
    output = []
    for values in product(*(prepared['levels'][f] for f in partners)):
        fixed = dict(zip(partners, values))
        try:
            means, comparisons = simple_effects(prepared, result, factor, fixed)
            output.append(dict(fixed=fixed, means=means, comparisons=comparisons))
        except ValueError as exc:
            output.append(dict(fixed=fixed, error=str(exc)))
    tables = [o['comparisons'] for o in output if 'comparisons' in o]
    if tables:
        adjusted = multipletests(np.concatenate([t['p-value'] for t in tables]), method='holm')[1]
        critical = stats.t.ppf(1-.05/(2*len(adjusted)), result['model'].df_resid)
        start = 0
        for table in tables:
            table['Holm-adjusted p-value'] = adjusted[start:start+len(table)]
            table['Significant'] = np.where(table['Holm-adjusted p-value'] < result['alpha'], 'Yes', 'No')
            table['95% simultaneous CI lower'] = table['Adjusted mean difference'] - critical*table['SE']
            table['95% simultaneous CI upper'] = table['Adjusted mean difference'] + critical*table['SE']
            start += len(table)
    return output


def robust_type2(prepared, result):
    """Explicit HC3 sensitivity analysis, never replacing classical Type II SS."""
    import warnings
    from statsmodels.stats.anova import anova_lm
    model = result['model']
    if np.any(model.get_influence().hat_matrix_diag >= 1 - 1e-10):
        raise ValueError('HC3 inference is unavailable because an observation has leverage near one. Additional replication or a simpler model is needed.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        raw = anova_lm(model, typ=2, robust='hc3').iloc[:-1]
    if caught or not np.isfinite(raw[['F', 'PR(>F)', 'df']].to_numpy()).all():
        raise ValueError('HC3 covariance cannot support all selected Type II tests reliably.')
    return pd.DataFrame({'Source': [prepared['terms'][t][0] for t in raw.index],
                         'DF': raw['df'].to_numpy(), 'Residual DF': model.df_resid,
                         'HC3 robust F': raw['F'].to_numpy(),
                         'HC3 p-value': raw['PR(>F)'].to_numpy()})
