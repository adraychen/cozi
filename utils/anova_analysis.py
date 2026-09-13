"""Validation and one-way independent-group mean comparisons."""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway


def prepare_anova(df, category, outcome):
    if category == outcome or not df.columns.is_unique:
        raise ValueError('Choose distinct grouping and outcome columns with unique names.')
    data = df[[category, outcome]].copy()
    if not pd.api.types.is_numeric_dtype(data[outcome]) or pd.api.types.is_bool_dtype(data[outcome]):
        raise ValueError('ANOVA requires a numeric outcome, not text or boolean values.')
    data[outcome] = data[outcome].replace([np.inf, -np.inf], np.nan)
    clean = data.dropna()
    pairs = [(label, group[outcome].to_numpy(dtype=float))
             for label, group in clean.groupby(category, observed=True, sort=False)]
    groups = [values for _, values in pairs]
    errors, notes, rows, normality = [], [], [], []
    if len(groups) < 2:
        errors.append('At least two groups with usable observations are required.')
    if any(len(g) < 2 for g in groups):
        errors.append('Each group needs at least two usable observations. Singleton groups are not silently removed.')
    dropped_groups = data[category].nunique(dropna=True) - len(groups)
    if dropped_groups:
        notes.append(f'{dropped_groups} groups have no usable outcomes and were excluded entirely.')
    for label, g in pairs:
        variance = float(np.var(g, ddof=1)) if len(g) > 1 else np.nan
        q1, q3 = np.percentile(g, [25, 75])
        outliers = int(np.sum((g < q1 - 1.5*(q3-q1)) | (g > q3 + 1.5*(q3-q1))))
        rows.append({'Group': label, 'Count': len(g), 'Mean': np.mean(g),
                     'Median': np.median(g), 'SD': np.sqrt(variance),
                     'Variance': variance, 'Minimum': min(g), 'Maximum': max(g),
                     'IQR outlier flags': outliers})
        w, p = np.nan, np.nan
        if len(g) < 3:
            status = 'Unavailable: fewer than 3 observations'
        elif np.ptp(g) == 0:
            status = 'Unavailable: constant group'
        elif len(g) > 5000:
            status = 'Not run: above 5,000 observations; use the Q–Q plot'
        else:
            w, p = map(float, stats.shapiro(g))
            status = 'Review warning' if p < .05 else 'No strong evidence of non-normality'
        normality.append({'Group': label, 'Shapiro–Wilk W': w, 'p-value': p, 'Interpretation': status})
    summary = pd.DataFrame(rows)
    if groups:
        if any(len(g) < 20 for g in groups):
            notes.append('Some groups have fewer than 20 observations. This is a caution threshold, not a universal sample-size requirement; small groups weaken diagnostics and power.')
        if max(map(len, groups)) / min(map(len, groups)) > 4:
            notes.append('Group sizes differ by more than 4:1. Review imbalance alongside group variances; this is a screening heuristic.')
        if any(np.ptp(g) == 0 for g in groups):
            notes.append('At least one group is constant. Welch ANOVA is unavailable for zero group variance; classical ANOVA also needs positive pooled within-group variation.')
        if summary['IQR outlier flags'].sum():
            notes.append('Potential within-group outliers were flagged using 1.5 × IQR fences. Investigate them; none were removed automatically.')
        if any(item['Interpretation'] == 'Review warning' for item in normality):
            notes.append('Some groups show evidence of non-normality. Review their Q–Q plots, particularly for small groups; neither ANOVA variant fixes severe skew or outliers.')
    lv, lp = np.nan, np.nan
    if not errors:
        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            lv, lp = map(float, stats.levene(*groups, center='median'))
        if np.isfinite(lp) and lp < .05:
            notes.append('Median-centered Levene test suggests unequal variances (p < 0.05). Prefer Welch ANOVA over classical ANOVA.')
        elif not np.isfinite(lp):
            notes.append('The variance diagnostic is undefined for these groups; equal variances have not been established.')
    return {'clean': clean, 'groups': groups, 'labels': [label for label, _ in pairs],
            'summary': summary, 'normality': pd.DataFrame(normality), 'errors': errors,
            'warnings': notes, 'excluded': len(df)-len(clean), 'n': len(clean),
            'levene_stat': lv, 'levene_p': lp}


def run_anova(prepared, method='Welch', independent=False):
    if method not in ('Welch', 'Classical'):
        raise ValueError('Choose Welch or Classical ANOVA.')
    if prepared['errors']:
        raise ValueError(' '.join(prepared['errors']))
    if not independent:
        raise ValueError('Confirm independent observations before running ANOVA. Repeated or clustered measurements require a suitable alternative model.')
    groups = prepared['groups']
    variances = np.array([np.var(g, ddof=1) for g in groups])
    if not np.all(np.isfinite(variances)):
        raise ValueError('Group variances are not numerically finite. Rescale the outcome and retry.')
    if method == 'Welch' and np.any(variances <= 0):
        raise ValueError('Welch ANOVA requires positive variance in every group. Review the constant groups; do not switch methods merely to obtain a result.')
    if np.dot(variances, [len(g)-1 for g in groups]) <= 0:
        raise ValueError('ANOVA cannot estimate error: pooled within-group variance is zero.')
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = anova_oneway(groups, use_var='unequal' if method == 'Welch' else 'equal', welch_correction=True)
    f, p = float(result.statistic), float(result.pvalue)
    if not np.isfinite(f) or not np.isfinite(p):
        raise ValueError('ANOVA is numerically undefined for these data. Review group variation and outcome scale.')
    return {'method': method, 'F': f, 'p-value': p,
            'df between': float(result.df_num), 'df denominator': float(result.df_denom)}
