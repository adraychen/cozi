"""Validated classical t-tests. No formal normality tests or automatic filtering."""
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from utils.exploration import detect_outliers


class TTestInputError(ValueError):
    """A validation message suitable for display to the user."""


def numerical_candidates(df, columns, metadata=None):
    """Respect semantic classifications; never infer quantitative roles from integer storage.

    The app has no Identifier subtype. Conservatively screen identifier-like names
    unless the user explicitly overrides them to Numerical in Column Overview.
    """
    metadata = metadata or {}
    result = []
    for col in columns:
        if col not in df or not df.columns.is_unique:
            continue
        meta = metadata.get(col, {})
        if meta.get('variable_type', 'Numerical') != 'Numerical':
            continue
        if meta.get('variable_subtype', 'Continuous') not in ('Continuous', 'Discrete'):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_complex_dtype(df[col]):
            continue
        identifier = bool(re.search(r'(^|[_\s-])(id|identifier)($|[_\s-])', str(col), re.I) or str(col).endswith('ID'))
        if identifier and meta.get('source', 'Automatic') == 'Automatic':
            continue
        result.append(col)
    return result


def _numeric(df, column):
    if column not in df.columns:
        raise TTestInputError('Choose a numerical variable for each required role.')
    s = df[column]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_complex_dtype(s):
        raise TTestInputError('Responses must be quantitative numerical variables. Review Column Overview.')
    return s.astype(float).replace([np.inf, -np.inf], np.nan)


def describe(label, s):
    n = len(s)
    sd = float(s.std(ddof=1)) if n > 1 else np.nan
    return {'Sample': str(label), 'N': n, 'Mean': float(s.mean()), 'Standard deviation': sd,
            'Standard error': sd / np.sqrt(n) if n else np.nan,
            'Minimum': float(s.min()), 'Maximum': float(s.max())}


def prepare_ttest(df, mode, first, second=None, group=None, levels=None):
    """Clean by design; separate independent columns never lose unmatched values.

    Internal indexes are zero-based row positions, retained for outlier reporting.
    Stacked counts distinguish missing/nonfinite rows from unselected categories.
    """
    if mode not in ('one', 'independent', 'paired') or not df.columns.is_unique:
        raise TTestInputError('Choose a supported t-test and use unique column names.')
    frame = df.reset_index(drop=True)
    x = _numeric(frame, first)
    samples, accounting = [], []
    if mode == 'one':
        samples = [(str(first), x.dropna())]
        accounting = [('Original rows', len(df)), ('Used observations', x.notna().sum()),
                      ('Excluded missing/nonfinite rows', x.isna().sum())]
    elif mode == 'independent' and group is not None:
        if group not in frame or group == first:
            raise TTestInputError('Choose a distinct categorical grouping variable.')
        if levels is None or len(levels) != 2 or levels[0] == levels[1]:
            raise TTestInputError('Choose two different groups to compare.')
        g = frame[group]
        if any(level not in list(g.dropna().unique()) for level in levels):
            raise TTestInputError('Choose two groups present in the active dataset.')
        valid = x.notna() & g.notna()
        samples = [(f'{group} = {level}', x[valid & g.eq(level)]) for level in levels]
        accounting = [('Original rows', len(df)), ('Used observations', sum(len(s) for _, s in samples)),
                      ('Excluded missing/nonfinite response or group', (~valid).sum()),
                      ('Other groups not selected (valid rows)', (valid & ~g.isin(levels)).sum())]
    else:
        if first == second:
            raise TTestInputError('Choose two different numerical variables.')
        y = _numeric(frame, second)
        if mode == 'paired':
            valid = x.notna() & y.notna()
            x, y = x[valid], y[valid]
            samples = [(str(first), x), (str(second), y)]
            accounting = [('Original rows', len(df)), ('Complete pairs used', valid.sum()),
                          ('Excluded incomplete/nonfinite pairs', (~valid).sum())]
        else:
            samples = [(str(first), x.dropna()), (str(second), y.dropna())]
            accounting = [('Original rows', len(df)), ('Sample 1 used', x.notna().sum()),
                          ('Sample 1 missing/nonfinite', x.isna().sum()), ('Sample 2 used', y.notna().sum()),
                          ('Sample 2 missing/nonfinite', y.isna().sum())]
    diagnostic_samples = samples
    descriptive_samples = samples[:]
    if mode == 'paired':
        with np.errstate(over='ignore', invalid='ignore'):
            differences = samples[0][1] - samples[1][1]
        if not np.isfinite(differences.to_numpy()).all():
            raise TTestInputError('Paired differences exceed the numerical range. Rescale the measurements before retrying.')
        diagnostic_samples = [('Paired differences (1 - 2)', differences)]
        descriptive_samples += diagnostic_samples
    errors, notes, diagnostics, outliers = [], [], [], []
    for label, s in diagnostic_samples:
        if len(s) < 2:
            errors.append(f'{label}: at least two usable observations are required.')
        elif not np.isfinite(s.std(ddof=1)):
            errors.append(f'{label}: variation exceeds the numerical range. Rescale the measurements.')
        elif s.std(ddof=1) == 0:
            if mode in ('one', 'paired'):
                errors.append(f'{label}: variation must be positive; a t-test cannot estimate uncertainty from constant values.')
            else:
                notes.append(f'{label} is constant. Interpret the variance assumptions with care.')
        if len(s) < 20:
            notes.append(f'{label}: the sample is small (N = {len(s)}). Inspect skewness and extreme values; this is a caution, not a cutoff for validity.')
        info, flagged = detect_outliers(s)
        skew = float(s.skew()) if len(s) >= 3 and s.nunique() > 1 else np.nan
        diagnostics.append({'Sample': label, 'N': len(s), 'Skewness': skew, 'IQR outlier flags': info['count']})
        outliers += [{'Sample': label, 'Original row position (1-based)': int(i) + 1, 'Value': float(v)} for i, v in flagged.items()]
        if info['count']:
            notes.append(f'{label}: {info["count"]} potential outliers flagged using 1.5 x IQR fences. All remain in the analysis.')
        if np.isfinite(skew) and abs(skew) > 1:
            notes.append(f'{label}: substantial skewness is a reason to review the plots, especially with a small sample; it does not automatically invalidate the test.')
    return {'mode': mode, 'samples': samples, 'diagnostic_samples': diagnostic_samples,
            'descriptive': pd.DataFrame([describe(label, s) for label, s in descriptive_samples]),
            'accounting': pd.DataFrame(accounting, columns=['Data detail', 'Count']),
            'diagnostics': pd.DataFrame(diagnostics), 'outliers': pd.DataFrame(outliers),
            'errors': errors, 'warnings': notes}


def run_ttest(prepared, null_mean=0., alternative='two-sided', confidence=.95, equal_var=False):
    try:
        null_mean, confidence = float(null_mean), float(confidence)
    except (TypeError, ValueError):
        raise TTestInputError('Use finite numerical values for the hypothesized mean and confidence level.') from None
    if alternative not in ('two-sided', 'greater', 'less'):
        raise TTestInputError('Choose a two-sided, greater, or less alternative.')
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise TTestInputError('Confidence level must be between 0 and 100 percent.')
    if not np.isfinite(null_mean):
        raise TTestInputError('The hypothesized mean must be a finite numerical value.')
    if prepared['errors']:
        raise TTestInputError(' '.join(prepared['errors']))
    mode = prepared['mode']
    x = prepared['samples'][0][1].to_numpy()
    with warnings.catch_warnings(), np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        if mode in ('one', 'paired'):
            values = x if mode == 'one' else prepared['diagnostic_samples'][0][1].to_numpy()
            sd = float(np.std(values, ddof=1))
            mean = float(np.mean(values))
            estimate = mean - null_mean if mode == 'one' else mean
            se = sd / np.sqrt(len(values))
            if not np.isfinite(se) or se <= 0:
                raise TTestInputError('The test needs positive, finite variation. Review or rescale the measurements.')
            if mode == 'one':
                test = stats.ttest_1samp(values, null_mean, alternative=alternative)
                method = 'One-Sample t-Test'
            else:
                test = stats.ttest_rel(x, prepared['samples'][1][1].to_numpy(), alternative=alternative)
                method = 'Paired Samples t-Test'
            effect = estimate / sd
            effect_name = "Cohen's d" if mode == 'one' else "Cohen's dz"
            effect_formula = 'Mean difference / sample SD' if mode == 'one' else 'Mean paired difference / SD of paired differences'
        else:
            y = prepared['samples'][1][1].to_numpy()
            n1, n2 = len(x), len(y)
            v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
            pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            se = np.sqrt(pooled * (1 / n1 + 1 / n2) if equal_var else v1 / n1 + v2 / n2)
            if not np.isfinite(se) or se <= 0 or not np.isfinite(pooled) or pooled <= 0:
                raise TTestInputError('The samples need positive, finite combined variation. Review or rescale the measurements.')
            estimate = float(np.mean(x) - np.mean(y))
            test = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
            method = 'Pooled Two-Sample t-Test' if equal_var else "Welch's Two-Sample t-Test"
            # Cohen's d uses the sample-size-weighted pooled SD for BOTH methods.
            # For Welch this is a descriptive standardizer, not an equal-variance
            # assumption: SE, df, p and CI still use Welch-Satterthwaite inference.
            effect = estimate / np.sqrt(pooled)
            effect_name = "Cohen's d (pooled SD standardizer)"
            effect_formula = '(Mean 1 - Mean 2) / sqrt(((n1-1)*SD1^2 + (n2-1)*SD2^2) / (n1+n2-2))'
        interval = test.confidence_interval(confidence_level=confidence)
        low, high = float(interval.low), float(interval.high)
        if mode == 'one':
            low, high = low - null_mean, high - null_mean
        if not np.isfinite([estimate, se, test.statistic, test.df, test.pvalue, effect]).all():
            raise TTestInputError('The test is numerically undefined for these measurements. Review their scale and variation.')
        if (alternative != 'less' and not np.isfinite(low)) or (alternative != 'greater' and not np.isfinite(high)):
            raise TTestInputError('The confidence bounds exceed the numerical range. Rescale the measurements.')
    direction = (f'{prepared["samples"][0][0]} - hypothesized mean ({null_mean:g})' if mode == 'one'
                 else f'{prepared["samples"][0][0]} - {prepared["samples"][1][0]}')
    return {'method': method, 'difference': direction, 'estimate': float(estimate), 'se': float(se),
            't': float(test.statistic), 'df': float(test.df), 'p': float(test.pvalue),
            'low': low, 'high': high, 'confidence': confidence, 'alternative': alternative,
            'effect_name': effect_name, 'effect': float(effect), 'effect_formula': effect_formula,
            'null_mean': float(null_mean) if mode == 'one' else 0., 'mode': mode}
