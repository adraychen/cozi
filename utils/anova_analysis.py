"""Welch-only one-way inference and descriptive variation; no method switching."""
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway, effectsize_oneway

PAIRWISE_GROUP_LIMIT = 30
EFFECT_SIZE = 'Welch-compatible Cohen f² (variance-weighted)'
VARIATION_NOTE = ("The variation table shows how total variation is divided between groups and within groups. "
                  "Welch's ANOVA uses variance-adjusted calculations, so its F-statistic is not calculated "
                  "directly from the mean squares shown here.")


def validate_alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('The significance level must be between zero and one.')


def prepare_anova(df, category, outcome):
    if category == outcome or not df.columns.is_unique or category not in df or outcome not in df:
        raise ValueError('Choose distinct grouping and outcome columns with unique names.')
    data = df[[category, outcome]].reset_index(drop=True).copy()
    if (not pd.api.types.is_numeric_dtype(data[outcome]) or pd.api.types.is_bool_dtype(data[outcome])
            or pd.api.types.is_complex_dtype(data[outcome])):
        raise ValueError('ANOVA requires a quantitative numeric outcome, not text, boolean or complex values.')
    data[outcome] = data[outcome].astype(float).replace([np.inf, -np.inf], np.nan)
    data[category] = data[category].mask(data[category].isin([np.inf, -np.inf]))
    clean = data.dropna()
    pairs = [(label, group[outcome].to_numpy(dtype=float))
             for label, group in clean.groupby(category, observed=True, sort=False)]
    groups = [values for _, values in pairs]
    errors, notes, rows, distribution = [], [], [], []
    if len(groups) < 2:
        errors.append('At least two groups with usable observations are required.')
    dropped_groups = data[category].nunique(dropna=True) - len(groups)
    if dropped_groups:
        notes.append(f'{dropped_groups} groups have no usable outcomes and were excluded entirely.')
    for label, g in pairs:
        with warnings.catch_warnings(), np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = float(np.mean(g))
            variance = float(np.var(g, ddof=1)) if len(g) > 1 else np.nan
            q1, q3 = np.percentile(g, [25, 75])
            outliers = int(np.sum((g < q1 - 1.5*(q3-q1)) | (g > q3 + 1.5*(q3-q1))))
            skew = float(stats.skew(g, bias=False)) if len(g) >= 3 and variance > 0 else np.nan
            kurtosis = float(stats.kurtosis(g, bias=False)) if len(g) >= 4 and variance > 0 else np.nan
            half = (float(stats.t.ppf(.975, len(g)-1) * np.sqrt(variance/len(g)))
                    if len(g) > 1 and np.isfinite(variance) and variance > 0 else np.nan)
        if len(g) < 2:
            errors.append(f'Group {label!s} needs at least two usable observations; it was not silently removed.')
        elif not np.isfinite(mean) or not np.isfinite(variance):
            errors.append(f'Group {label!s} has non-finite statistics. Rescale the outcome and retry.')
        elif variance <= 0:
            errors.append(f'Group {label!s} has zero variance. Welch ANOVA and Games–Howell require positive variance in every group.')
        elif not np.isfinite(half):
            errors.append(f'Group {label!s} has non-finite uncertainty. Rescale the outcome and retry.')
        severe = (np.isfinite(skew) and abs(skew) >= 2) or (np.isfinite(kurtosis) and abs(kurtosis) >= 7)
        status = ('Severe skew/tail screening flag; review plots' if severe else
                  'Insufficient distribution information' if not np.isfinite(skew) else
                  'No severe skew/tail flag; normality is not established')
        rows.append({'Group': label, 'Count': len(g), 'Mean': mean, 'Median': float(np.median(g)),
                     'SD': np.sqrt(variance), 'Variance': variance, 'Minimum': min(g), 'Maximum': max(g),
                     'IQR outlier flags': outliers, '95% CI Lower': mean-half, '95% CI Upper': mean+half})
        distribution.append({'Group': label, 'Skewness': skew, 'Excess kurtosis': kurtosis,
                             'IQR outlier flags': outliers, 'Assessment': status})
    summary = pd.DataFrame(rows, columns=['Group', 'Count', 'Mean', 'Median', 'SD', 'Variance',
                                        'Minimum', 'Maximum', 'IQR outlier flags', '95% CI Lower', '95% CI Upper'])
    size_min = min(map(len, groups)) if groups else 0
    size_max = max(map(len, groups)) if groups else 0
    imbalance = size_max / size_min if size_min else np.nan
    variance_ratio = np.nan
    if groups:
        variances = summary.Variance.to_numpy()
        if np.all(np.isfinite(variances)) and np.all(variances > 0):
            with np.errstate(over='ignore', divide='ignore'):
                variance_ratio = float(max(variances)/min(variances))
        if size_min < 20:
            notes.append('Some groups have fewer than 20 observations. This screening caution is not a universal sample-size rule; small groups weaken diagnostics and power.')
        if imbalance > 4:
            notes.append('Group sizes differ by more than 4:1. Review imbalance alongside group variances; this is a screening heuristic.')
        if summary['IQR outlier flags'].sum():
            notes.append('Potential outliers were flagged using within-group 1.5 × IQR fences. None were removed automatically.')
        if any('Severe' in item['Assessment'] for item in distribution):
            notes.append('Severe skew or heavy tails were flagged descriptively. Small-sample Welch and Games–Howell inference may be unreliable; review the plots. No method switch is automatic.')
    lv, lp = np.nan, np.nan
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            lv, lp = map(float, stats.levene(*groups, center='median'))
        if np.isfinite(lp) and lp < .05:
            notes.append('Median-centered Levene screening suggests unequal variances (p < 0.05). Welch remains the active method.')
        elif not np.isfinite(lp):
            notes.append('The variance diagnostic is undefined; equal variances have not been established.')
    return {'clean': clean, 'groups': groups, 'labels': [label for label, _ in pairs],
            'category': category, 'outcome': outcome, 'summary': summary,
            'distribution': pd.DataFrame(distribution), 'errors': errors, 'warnings': notes,
            'excluded': len(df)-len(clean), 'n': len(clean), 'original_n': len(df),
            'missing_response': int(data[outcome].isna().sum()), 'missing_factor': int(data[category].isna().sum()),
            'min_group_size': size_min, 'max_group_size': size_max, 'size_ratio': imbalance,
            'variance_ratio': variance_ratio, 'levene_stat': lv, 'levene_p': lp}


def variation_decomposition(prepared):
    """Conventional SS, checked against total centered SS; never Welch's F."""
    groups = prepared['groups']
    n, k = prepared['n'], len(groups)
    if not n:
        raise ValueError('No usable observations for variation decomposition.')
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.concatenate(groups)
        grand = np.mean(values)
        between = float(sum(len(g)*(np.mean(g)-grand)**2 for g in groups))
        within = float(sum(np.sum((g-np.mean(g))**2) for g in groups))
        total = float(np.sum((values-grand)**2))
    if not np.all(np.isfinite([between, within, total])):
        raise ValueError('Variation decomposition exceeds the numerical range. Rescale the outcome.')
    if not np.isclose(between+within, total, rtol=1e-9, atol=np.finfo(float).eps*max(1., total)*100):
        raise ValueError('The sum-of-squares identity could not be verified numerically. Center or rescale the outcome.')
    return pd.DataFrame([
        ['Between Groups', k-1, between, between/(k-1) if k>1 else np.nan],
        ['Within Groups (Residual)', n-k, within, within/(n-k) if n>k else np.nan],
        ['Total', n-1, total, np.nan],
    ], columns=['Source', 'DF', 'Sum of Squares (SS)', 'Mean Square (MS)'])


def run_anova(prepared, method='Welch', independent=None, alpha=.05, independence=None, design=None):
    """Only Welch; ambiguous repeated/clustered design is resolved explicitly."""
    validate_alpha(alpha)
    if method != 'Welch':
        raise ValueError('One-way ANOVA supports Welch only. Classical ANOVA is not available.')
    if prepared['errors']:
        raise ValueError(' '.join(prepared['errors']))
    if independent is False:
        raise ValueError('The observations are not independent; use repeated-measures or mixed-effects analysis.')
    if independence and independence['requires_clarification'] and design != 'independent':
        raise ValueError('Resolve the detected repeated or clustered structure before interpreting ordinary Welch ANOVA. A repeated-measures or mixed-effects analysis may be appropriate.')
    groups = prepared['groups']
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = anova_oneway(groups, use_var='unequal', welch_correction=True)
        f2 = float(effectsize_oneway(prepared['summary'].Mean.to_numpy(),
                                    prepared['summary'].Variance.to_numpy(),
                                    prepared['summary'].Count.to_numpy(), use_var='unequal'))
    f, p, dfn, dfd = map(float, [result.statistic, result.pvalue, result.df_num, result.df_denom])
    if not np.all(np.isfinite([f, p, dfn, dfd, f2])) or dfn <= 0 or dfd <= 0 or f2 < 0:
        raise ValueError('Welch ANOVA is numerically undefined. Review group variation and rescale the outcome.')
    return {'method': 'Welch', 'F': f, 'p-value': p, 'df between': dfn,
            'df denominator': dfd, EFFECT_SIZE: f2, 'alpha': alpha}


def games_howell(prepared, alpha=.05):
    """All pairs with variance-specific SE/df and studentized-range p and CI.

    No omnibus significance gate. The range distribution includes all k groups.
    """
    validate_alpha(alpha)
    lower, upper = f'{100*(1-alpha):g}% CI Lower', f'{100*(1-alpha):g}% CI Upper'
    columns = ['Comparison', 'Mean Difference', 'Adjusted p-value', lower, upper, 'Significant',
               'Group A', 'Group B', 'Standard error', 'Welch–Satterthwaite DF']
    result = {'table': pd.DataFrame(columns=columns), 'alpha': alpha, 'confidence': 1-alpha,
              'lower': lower, 'upper': upper, 'reason': None}
    k = len(prepared['groups'])
    if prepared['errors']:
        result['reason'] = 'Games–Howell unavailable: ' + ' '.join(prepared['errors'])
        return result
    if k < 3:
        result['reason'] = 'Games–Howell pairwise comparisons require at least three valid groups in this workflow.'
        return result
    if k > PAIRWISE_GROUP_LIMIT:
        result['reason'] = f'Games–Howell is limited to {PAIRWISE_GROUP_LIMIT} groups ({PAIRWISE_GROUP_LIMIT*(PAIRWISE_GROUP_LIMIT-1)//2} comparisons) for numerical workload. No groups or pairs were silently dropped.'
        return result
    summary = prepared['summary']
    counts, means, variances = (summary[c].to_numpy(dtype=float) for c in ['Count', 'Mean', 'Variance'])
    pairs = list(combinations(range(k), 2))
    a, b = np.array(pairs).T
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        va, vb = variances[a]/counts[a], variances[b]/counts[b]
        se = np.sqrt(va+vb)
        df = (va+vb)**2/(va**2/(counts[a]-1)+vb**2/(counts[b]-1))
        difference = means[a]-means[b]
        q = np.sqrt(2)*abs(difference)/se
    if not np.all(np.isfinite(np.r_[se, df, difference, q])) or np.any(se <= 0) or np.any(df <= 0):
        result['reason'] = 'Games–Howell pair statistics are not finite or estimable. Rescale the outcome; no intervals or p-values were fabricated.'
        return result
    p = np.asarray(stats.studentized_range.sf(q, k, df))
    critical = np.asarray(stats.studentized_range.ppf(1-alpha, k, df))
    half = critical*se/np.sqrt(2)
    lo, hi = difference-half, difference+half
    if not np.all(np.isfinite(np.r_[p, lo, hi])):
        result['reason'] = 'Studentized-range calculations failed; no Games–Howell intervals or p-values are available.'
        return result
    labels = [str(label) for label in prepared['labels']]
    if len(set(labels)) < len(labels):
        labels = [repr(label) for label in prepared['labels']]
    result['table'] = pd.DataFrame({
        'Comparison': [f'{labels[i]} − {labels[j]}' for i,j in pairs],
        'Mean Difference': difference, 'Adjusted p-value': p, lower: lo, upper: hi,
        'Significant': p < alpha, 'Group A': [labels[i] for i in a], 'Group B': [labels[i] for i in b],
        'Standard error': se, 'Welch–Satterthwaite DF': df,
    })
    return result
