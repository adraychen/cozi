"""Conservative diagnostic summaries and optional next steps for factorial OLS."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import jarque_bera


def diagnostic_summary(prepared, result):
    model, alpha = result['model'], result['alpha']
    residuals, fitted = np.asarray(model.resid), np.asarray(model.fittedvalues)
    jb = tuple(map(float, jarque_bera(residuals))) if len(residuals) >= 3 else None
    flags, patterns = [], []
    rows = []

    def add(check, status, explanation):
        rows.append({'Check': check, 'Status': status, 'Interpretation': explanation})
        if status in ('Caution', 'Problem'):
            flags.append(explanation)

    if jb is None or not np.isfinite(jb).all():
        add('Residual normality', 'Cannot assess', 'Too little usable residual information for the normality diagnostic.')
    elif jb[1] < alpha:
        substantial = abs(jb[2]) > 1 or jb[3] > 5
        add('Residual normality', 'Problem' if substantial else 'Caution',
            'Residuals show substantial departure from normality; inspect curvature and tails in the Q–Q plot.' if substantial
            else 'The residual normality diagnostic flags a departure; inspect the Q–Q plot and sample size.')
    else:
        add('Residual normality', 'Caution' if len(residuals) < 20 else 'OK',
            'No departure detected by Jarque–Bera; this does not establish normality. Small samples limit sensitivity and the asymptotic test can be unreliable.')

    # Exploratory screens, deliberately not presented as formal hypothesis tests.
    spread = None
    curvature = False
    if len(residuals) >= 24 and np.ptp(fitted) > 1e-10:
        bins = pd.qcut(fitted, q=4, duplicates='drop')
        binned = pd.DataFrame({'fit': fitted, 'resid': residuals, 'bin': bins}).groupby('bin', observed=True).agg(
            N=('resid', 'size'), Fitted=('fit', 'mean'), Spread=('resid', 'std'), Mean=('resid', 'mean'))
        binned = binned.loc[binned.N >= 6]
        if len(binned) >= 2 and binned.Spread.iloc[0] > 0:
            ratio = binned.Spread.iloc[-1] / binned.Spread.iloc[0]
            correlation = stats.spearmanr(fitted, np.abs(residuals)).statistic
            if ratio >= 1.75 and correlation > .25:
                spread = 'increasing'
            elif ratio <= 1/1.75 and correlation < -.25:
                spread = 'decreasing'
        if len(binned) >= 3:
            scale = np.std(residuals, ddof=1)
            curvature = bool(scale > 0 and np.max(np.abs(binned.Mean)) > .5*scale)
    if spread:
        patterns.append(f'Possible {spread} residual spread across fitted values; review the residual-vs-fitted plot for a funnel shape.')
    if curvature:
        patterns.append('Binned residual means show a possible systematic pattern or curvature. Review model specification and the residual-vs-fitted plot.')
    with np.errstate(divide='ignore', invalid='ignore'):
        standardized = np.asarray(model.get_influence().resid_studentized_internal)
    extreme = int(np.sum(np.isfinite(standardized) & (np.abs(standardized) > 3)))
    if extreme:
        patterns.append(f'{extreme} observations have internally studentized residuals beyond ±3. Investigate their context and data quality; do not automatically delete them.')
    if not np.isfinite(standardized).all():
        patterns.append('Some standardized residuals cannot be assessed because of high leverage or limited replication.')
    if not patterns:
        patterns.append('No obvious pattern was flagged by the limited automated screens. Inspect the plots; this is not confirmation of a correctly specified model.')

    if result['levene'] is None:
        add('Equal variance', 'Caution' if spread else 'Cannot assess',
            'Equal variances could not be formally assessed using the selected factor-cell structure.'
            + (f' The exploratory screen suggests {spread} residual spread.' if spread else ''))
    elif result['levene'][1] < alpha:
        add('Equal variance', 'Problem', 'Median-centered Levene flags unequal cell variances; classical F tests may be unreliable, especially with unequal cell sizes.')
    elif spread:
        add('Equal variance', 'Caution', f'An exploratory screen suggests {spread} residual spread despite a non-significant Levene test; inspect the plot.')
    else:
        add('Equal variance', 'OK', 'Levene did not detect unequal cell variances; a non-significant test does not establish equal variance.')
    add('Independence', 'Cannot assess', 'Independence depends on the data-collection design. Repeated or clustered observations need an appropriate repeated-measures or mixed model.')
    add('Cell coverage', 'Caution' if prepared['empty_cells'] else 'OK',
        f"{prepared['empty_cells']} of {prepared['possible_cells']} possible factor-level combinations are empty."
        + (' Some estimates rely on the selected model to extrapolate across missing combinations.' if prepared['empty_cells'] else ' All combinations are observed.'))
    add('Replication', 'Caution' if prepared['singleton_cells'] else 'OK',
        f"{prepared['singleton_cells']} observed combinations contain only one observation."
        + (' Cell-level variance cannot be estimated for these combinations.' if prepared['singleton_cells'] else ' All observed combinations have replication.'))
    low_df = model.df_resid < max(5, len(model.params))
    add('Model estimability / rank', 'Caution' if low_df else 'OK',
        f'Full-rank selected model; {len(model.params)} parameters and {model.df_resid:g} residual degrees of freedom.'
        + (' Limited residual degrees of freedom can make inference unstable.' if low_df else ''))
    if extreme or curvature:
        flags.append('Explore the flagged residual patterns or extreme observations.')
    overall = ('Problem' if any(r['Status'] == 'Problem' for r in rows) else 'Caution') if flags else 'Cannot assess'
    reliability = ('Interpret F tests cautiously. ' + ' '.join(flags)) if flags else (
        'No major issues were flagged by these diagnostics. Overall reliability still depends on independence, study design, and visual checks; it cannot be established from these data alone.')
    add('Overall model reliability', overall, reliability)

    response = prepared['clean'][prepared['outcome']].to_numpy(dtype=float)
    response_skew = float(stats.skew(response, bias=False)) if len(response) > 2 else 0.
    transform_flag = response_skew > 1 or (jb is not None and jb[1] < alpha) or spread == 'increasing'
    recommendations = []
    if transform_flag:
        if np.all(response > 0):
            recommendations.append('Consider a log transformation of the response and rerun the analysis. Compare the diagnostics before and after transformation. All analyzed responses are positive; check any additional rows before transforming them. Effects would then be interpreted on the log scale.')
        elif np.all(response >= 0):
            recommendations.append('Log(Y) is undefined for zero responses. Do not apply it directly or add an arbitrary offset without a substantive reason.')
        else:
            recommendations.append('The response contains negative values: log(Y) and square-root(Y) are not valid for all analyzed observations. Review model specification or a suitable alternative response model.')
        if np.all(response >= 0) and response_skew > 1:
            recommendations.append('A square-root transformation is also mathematically valid for this nonnegative, right-skewed response, particularly for count-like outcomes. Compare diagnostics and interpretation before choosing it.')
    if flags:
        recommendations.append('Consider the optional HC3 robust-inference comparison for unequal variances. It does not fix non-independence, missing design support, outliers, or an incorrect mean model. Welch ANOVA is a one-way alternative, not a general factorial remedy.')
    recommendations.append('Confirm the sampling or experimental design and investigate flagged observations. Observational associations do not establish causation.')
    return dict(table=pd.DataFrame(rows), jb=jb, spread=spread, curvature=curvature,
                extreme_residuals=extreme, patterns=patterns, recommendations=recommendations,
                overall=overall, reliability=reliability, response_skew=response_skew)
