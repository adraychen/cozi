"""Model-specific interaction assessment; never selects or removes fitted terms."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from utils.factorial_anova import prepare_factorial, run_factorial

EXPLORE = 'Explore all interactions'
MANUAL = 'Select interactions manually'


def selected_pairs(factors, mode, manual=()):
    candidates = list(combinations(factors, 2))
    if mode not in (EXPLORE, MANUAL):
        raise ValueError('Choose an interaction-selection mode.')
    return candidates if mode == EXPLORE else [p for p in candidates if p in manual or p[::-1] in manual]


def interaction_structure(pair, effects):
    return 'Fixed interaction' if all(effects[f] == 'Fixed' for f in pair) else 'Random interaction'


def support_table(df, outcome, effects, interactions):
    """Observed complete-case support, without allocating the Cartesian product."""
    raw = df[[outcome, *effects]].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for pair in interactions:
        counts = raw.groupby(list(pair), observed=True).size()
        possible = int(raw[pair[0]].nunique()) * int(raw[pair[1]].nunique())
        rows.append({'Interaction': ' × '.join(map(str, pair)),
                     'Structure': interaction_structure(pair, effects),
                     'Possible pairs': possible, 'Observed pairs': len(counts),
                     'Empty pairs': possible - len(counts),
                     'Single-observation pairs': int(counts.eq(1).sum()),
                     'Minimum N': int(counts.min()) if len(counts) else 0,
                     'Support': 'Review coverage' if len(counts) < possible or counts.eq(1).any() else 'Replicated pairs'})
    return pd.DataFrame(rows)


def fixed_assessment(prepared, result, mode):
    rows = result['table'].loc[lambda t: t.Effect == 'Interaction effect'].copy()
    if rows.empty:
        return rows
    # One family: every interaction in this jointly fitted model, in both modes.
    rows['Adjusted p-value'] = multipletests(rows['p-value'], method='holm')[1]
    rows['Assessment'] = np.where(rows['Adjusted p-value'] < result['alpha'],
                                  'Evidence of interaction', 'Limited evidence')
    support = support_table(prepared['clean'], prepared['outcome'],
                            dict.fromkeys(prepared['factors'], 'Fixed'), prepared['interactions'])
    rows = rows.merge(support, left_on='Source', right_on='Interaction', validate='one_to_one')
    rows['Estimability'] = 'Full-rank jointly fitted model'
    rows['Selection mode'] = mode
    return rows


def qualified_fixed_result(prepared, result, assessment):
    """Use Holm decisions for interaction narratives; preserve raw ANOVA values."""
    if assessment.empty:
        return result
    result = {**result, 'table': result['table'].copy(),
              'term_pvalues': dict(result['term_pvalues']), 'significant_interaction_factors': set()}
    adjusted = assessment.set_index('Source')['Adjusted p-value']
    for term, (label, kind, members) in prepared['terms'].items():
        if kind != 'Interaction effect':
            continue
        pvalue = float(adjusted[label])
        result['term_pvalues'][term] = pvalue
        mask = (result['table'].Source == label) & (result['table'].Effect == kind)
        result['table'].loc[mask, 'Adjusted interaction p-value'] = pvalue
        result['table'].loc[mask, 'Significant'] = 'Yes' if pvalue < result['alpha'] else 'No'
        if pvalue < result['alpha']:
            result['significant_interaction_factors'].update(members)
    return result


def interaction_contrast(prepared, result, pair):
    """First/last observed levels: one prespecified difference of differences.

    This describes a direction and response-unit magnitude, not the entire
    omnibus interaction. Other factors are averaged with equal weights.
    """
    from utils.factorial_followup import emm_vectors, _estimates
    from scipy.stats import t
    labels, vectors = emm_vectors(prepared, result, pair)
    a0, a1 = prepared['levels'][pair[0]][0], prepared['levels'][pair[0]][-1]
    b0, b1 = prepared['levels'][pair[1]][0], prepared['levels'][pair[1]][-1]
    lookup = dict(zip(labels, vectors))
    vector = lookup[(a1, b1)] - lookup[(a0, b1)] - lookup[(a1, b0)] + lookup[(a0, b0)]
    means, se = _estimates(result, [vector])
    # One contrast per selected interaction, with simultaneous 95% intervals.
    half = t.ppf(1 - .05/(2*len(prepared['interactions'])), result['model'].df_resid) * se[0]
    return {'Interaction': ' × '.join(map(str, pair)),
            'Contrast': f'({pair[0]}: {a1} − {a0}) at {pair[1]}={b1} minus at {pair[1]}={b0}',
            'Difference of differences': float(means[0]),
            '95% simultaneous CI lower': float(means[0]-half),
            '95% simultaneous CI upper': float(means[0]+half)}


def random_assessment(prepared, result, interactions, alpha=.05):
    """Variance magnitudes are descriptive; no boundary p-values are invented."""
    rows = []
    for pair in interactions:
        label = ' × '.join(map(str, pair))
        row = {'Interaction': label, 'Structure': interaction_structure(pair, prepared['effects']),
               'Assessment': 'Inconclusive', 'Uncertainty': 'Not available',
               'Estimability': 'Validated residual-subspace covariance rank'}
        if row['Structure'] == 'Random interaction':
            component = result['variance'].set_index('Component').loc[label]
            row.update({'Variance': component['Variance'],
                        'Percentage of modeled variance': component['Percentage of modeled variance'],
                        'Interpretation': 'Descriptive estimate; practical importance and uncertainty are not established.'})
        else:
            test = result.get('fixed_interactions', {}).get(label)
            if test and result['inference']:
                row.update(test)
                row['Uncertainty'] = 'Large-sample joint Wald test; no small-sample correction'
            else:
                row['Interpretation'] = 'Fixed interaction inference withheld by model diagnostics.'
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty and 'p-value' in table:
        mask = table['p-value'].notna()
        table.loc[mask, 'Adjusted p-value'] = multipletests(table.loc[mask, 'p-value'], method='holm')[1]
        table.loc[mask, 'Assessment'] = np.where(table.loc[mask, 'Adjusted p-value'] < alpha,
                                                'Evidence of interaction', 'Limited evidence')
    return table


def joint_fixed_interactions(model, fit, prepared, inference):
    """Omnibus Wald chi-square for fixed interactions in a validated MixedLM."""
    if not inference:
        return {}
    tests = {}
    covariance = np.asarray(fit.cov_params())[:model.k_fe, :model.k_fe]
    for term, bounds in prepared['fixed_term_slices'].items():
        if ':' not in term:
            continue
        names = prepared['fixed_columns'][slice(*bounds)]
        if any(name not in model.exog_names for name in names):
            continue  # Never test an unmatched coefficient layout.
        indices = [model.exog_names.index(name) for name in names]
        members = [f for f, alias in prepared['names'].items() if f'C({alias})' in term.split(':')]
        beta = np.asarray(fit.fe_params)[indices]
        cov = covariance[np.ix_(indices, indices)]
        if len(members) != 2 or not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 0:
            continue
        statistic = float(beta @ np.linalg.solve(cov, beta))
        tests[' × '.join(map(str, members))] = {'Wald χ²': statistic, 'DF': len(beta),
                                               'p-value': float(chi2.sf(statistic, len(beta)))}
    return tests


def compare_fixed_reduction(prepared, result, removed):
    """Classical nested OLS F test, identical observations, retained main effects."""
    if not removed or any(tuple(p) not in prepared['interactions'] for p in removed):
        raise ValueError('Choose currently fitted interactions to remove.')
    remaining = [p for p in prepared['interactions'] if p not in removed]
    reduced = prepare_factorial(prepared['clean'], prepared['outcome'], prepared['factors'], remaining)
    reduced_result = run_factorial(reduced, result['alpha'])
    full = result['model']
    if full.nobs != reduced_result['model'].nobs:
        raise ValueError('Model comparisons require identical observations.')
    statistic, pvalue, degrees = full.compare_f_test(reduced_result['model'])
    if degrees <= 0 or not np.isfinite([statistic, pvalue, degrees]).all():
        raise ValueError('The requested nested-model comparison is unavailable.')
    return {'F': float(statistic), 'p-value': float(pvalue), 'DF removed': float(degrees),
            'Full residual DF': float(full.df_resid), 'Reduced residual DF': float(reduced_result['model'].df_resid),
            'Remaining interactions': remaining, 'Main effects retained': list(prepared['factors'])}
