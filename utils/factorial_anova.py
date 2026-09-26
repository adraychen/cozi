"""Estimable, hierarchical factorial OLS models with Type II inference."""
from itertools import combinations
from math import prod
import warnings

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


def prepare_factorial(df, outcome, factors, interactions=()):
    factors = list(factors)
    if (not df.columns.is_unique or len(factors) < 2 or len(set(factors)) != len(factors)
            or outcome in factors or any(c not in df for c in [outcome] + factors)):
        raise ValueError('Choose one numerical response and at least two distinct categorical factors with unique column names.')
    y = df[outcome]
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y) or pd.api.types.is_complex_dtype(y):
        raise ValueError('The response must be a numerical variable.')
    pairs = []
    for pair in interactions:
        if len(pair) != 2 or pair[0] == pair[1] or any(c not in factors for c in pair):
            raise ValueError('Interactions must contain two different selected factors.')
        pair = tuple(sorted(pair, key=factors.index))
        if pair not in pairs:
            pairs.append(pair)
    data = df[[outcome] + factors].copy()
    missing = data.isna().any(axis=1)
    finite = np.isfinite(y.to_numpy(dtype=float, na_value=np.nan))
    clean = data.loc[~missing & finite].reset_index(drop=True)
    errors, notes = [], []
    # Safe internal aliases prevent arbitrary column names becoming formula code.
    internal = pd.DataFrame({'response': clean[outcome].astype(float)})
    aliases, levels = {}, {}
    for i, factor in enumerate(factors):
        alias = f'f{i}'
        aliases[factor] = alias
        codes, values = pd.factorize(clean[factor], sort=False)
        levels[factor] = list(values)
        internal[alias] = pd.Categorical(codes, categories=range(len(values)))
        if len(values) < 2:
            errors.append(f'{factor} needs at least two observed levels after complete-case filtering.')
    cells = clean.groupby(factors, observed=True, sort=False).size()
    possible = prod(len(levels[f]) for f in factors)
    empty = possible - len(cells)
    sparse = int((cells < 2).sum())
    if empty:
        notes.append(f'{empty} of {possible} factor-level combinations are empty. Missing cells can confound effects; only a full-rank selected model will be reported.')
    if sparse:
        notes.append(f'{sparse} observed factor-level combinations have only one observation. Cell variance cannot be assessed there; interpret inference cautiously.')
    terms = {f'C({aliases[f]})': (str(f), 'Main effect', (f,)) for f in factors}
    terms.update({':'.join(f'C({aliases[f]})' for f in pair):
                  (' × '.join(map(str, pair)), 'Interaction effect', pair) for pair in pairs})
    parameters = 1 + sum(len(levels[f])-1 for f in factors)
    parameters += sum((len(levels[a])-1)*(len(levels[b])-1) for a, b in pairs)
    if len(clean) <= parameters:
        errors.append(f'Insufficient observations: {len(clean)} usable rows for {parameters} model parameters. Positive residual degrees of freedom are required; add observations or simplify the model.')
    # Bound dense design construction before Patsy allocates the matrix.
    if parameters > 1000 or len(clean) * parameters > 10_000_000:
        errors.append('The selected model is too complex for interactive analysis. Reduce factors, levels, or interactions.')
    if 0 < len(clean) - parameters < max(5, parameters):
        notes.append('Few residual degrees of freedom relative to model complexity. Estimates and diagnostics may be unstable; consider more observations or a simpler model.')
    return dict(clean=clean, internal=internal, aliases=aliases, levels=levels,
                factors=factors, outcome=outcome, interactions=pairs, terms=terms,
                formula='response ~ ' + ' + '.join(terms), original=len(df), n=len(clean),
                missing=int(missing.sum()), nonfinite=int((~missing & ~finite).sum()),
                excluded=len(df)-len(clean), cells=cells, errors=errors, warnings=notes)


def run_factorial(prepared, alpha=.05):
    if prepared['errors']:
        raise ValueError(' '.join(prepared['errors']))
    if not 0 < alpha < 1:
        raise ValueError('Significance level must be between zero and one.')
    specification = ols(prepared['formula'], data=prepared['internal'])
    design = specification.exog
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('The model is rank deficient: some effects cannot be estimated separately. Check empty cells or confounded factors, remove redundant factors/interactions, or collect missing combinations.')
    model = specification.fit()
    near_perfect = np.linalg.norm(model.resid) <= 100 * np.finfo(float).eps * max(np.linalg.norm(model.model.endog), np.finfo(float).tiny)
    if model.df_resid <= 0 or not np.isfinite(model.mse_resid) or near_perfect:
        raise ValueError('The model cannot estimate residual error reliably. Add replication or simplify the model; a perfectly fitted response cannot support an F test.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        raw = anova_lm(model, typ=2)
    values = raw.iloc[:-1][['df', 'sum_sq', 'F', 'PR(>F)']].to_numpy()
    if caught or not np.isfinite(values).all() or (values[:, :2] < 0).any():
        raise ValueError('Some Type II effects could not be estimated reliably. Review sparse cells, redundant factors, and model complexity.')
    rows = []
    significant_factors = set()
    for term, row in raw.iterrows():
        residual = term == 'Residual'
        name, kind, members = ('Residual / Error', 'Residual', ()) if residual else prepared['terms'][term]
        significant = not residual and row['PR(>F)'] < alpha
        if significant and kind == 'Interaction effect':
            significant_factors.update(members)
        rows.append({'Source': name, 'Effect': kind, 'DF': row['df'],
                     'Sum of Squares': row['sum_sq'], 'Mean Square': row['sum_sq']/row['df'],
                     'F': row['F'], 'p-value': row['PR(>F)'],
                     'Partial η²': np.nan if residual else row['sum_sq']/(row['sum_sq'] + model.ssr),
                     'Significant': '—' if residual else ('Yes' if significant else 'No')})
    groups = [g[prepared['outcome']].to_numpy(dtype=float) for _, g in
              prepared['clean'].groupby(prepared['factors'], observed=True, sort=False)]
    levene = None
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            test = stats.levene(*groups, center='median')
        if np.isfinite(test.pvalue):
            levene = (float(test.statistic), float(test.pvalue))
    return dict(model=model, table=pd.DataFrame(rows), alpha=alpha,
                significant_interaction_factors=significant_factors, levene=levene)


def interaction_summary(prepared, pair):
    """Observed complete-case means; other factors are pooled, not adjusted."""
    data = prepared['clean'][list(pair) + [prepared['outcome']]].copy()
    data.columns = ['First factor', 'Second factor', 'Response']
    return data.groupby(['First factor', 'Second factor'], observed=True, sort=False)['Response'].agg(
        Mean='mean', N='size').reset_index()


def posthoc(prepared, result, factor):
    """Model-adjusted contrasts, averaging over observed other-factor rows.

    Only factors absent from selected interactions are supported: their contrasts
    are constant across the other factors and need no extrapolated reference grid.
    """
    if factor not in prepared['factors'] or len(prepared['levels'][factor]) <= 2:
        raise ValueError('Choose a factor with more than two observed levels.')
    if any(factor in pair for pair in prepared['interactions']):
        raise ValueError('This factor participates in an interaction. Simple-effects comparisons within relevant factor levels are not implemented; marginal comparisons are withheld.')
    if len(prepared['levels'][factor]) > 30:
        raise ValueError('Too many levels for an interpretable pairwise table (maximum 30).')
    model = result['model']
    vectors = []
    for code in range(len(prepared['levels'][factor])):
        frame = prepared['internal'].copy()
        frame[prepared['aliases'][factor]] = pd.Categorical(
            [code] * len(frame), categories=range(len(prepared['levels'][factor])))
        matrix = dmatrix(prepared['formula'].split('~', 1)[1], frame, return_type='dataframe')
        vectors.append(matrix[model.model.exog_names].to_numpy().mean(axis=0))
    rows = []
    for a, b in combinations(range(len(vectors)), 2):
        test = model.t_test(vectors[a]-vectors[b])
        rows.append({'Comparison': f"{prepared['levels'][factor][a]} − {prepared['levels'][factor][b]}",
                     'Adjusted mean difference': float(np.asarray(test.effect).item()),
                     'SE': float(np.asarray(test.sd).item()), 'p-value': float(test.pvalue)})
    table = pd.DataFrame(rows)
    table['Holm-adjusted p-value'] = multipletests(table['p-value'], method='holm')[1]
    table['Significant'] = table['Holm-adjusted p-value'] < result['alpha']
    return table
