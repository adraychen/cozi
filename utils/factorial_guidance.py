"""Deterministic factorial guidance. No fitting, term selection or data mutation."""
import hashlib
import re
from itertools import product
import numpy as np
import pandas as pd
from utils.guided_diagnostics import Diagnostic
from utils.numerical_formatting import format_statistic as fmt, format_p_value

INDEPENDENT = 'No, observations were collected independently.'
REPEATED = 'Yes, some observations belong to the same unit or group.'
UNSURE = "I'm not sure."


def analysis_context(df, outcome, factors, interactions, effects, alpha, metadata):
    """Include all source columns: changed identifiers/design invalidate answers too."""
    try:
        source = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except TypeError:
        source = df.to_json(date_format='iso', default_handler=str).encode()
    digest = hashlib.sha256(source)
    digest.update(repr((list(df.columns), list(map(str, df.dtypes)), outcome, factors,
                        interactions, effects, alpha, metadata)).encode())
    return digest.hexdigest()


def dependence_indicators(df, prepared, metadata):
    # Keep source row positions, including after incomplete-case exclusions.
    selected = df[[prepared['outcome']] + prepared['factors']]
    mask = selected.notna().all(axis=1) & np.isfinite(selected[prepared['outcome']].to_numpy(dtype=float))
    frame = df.loc[mask]
    rows = []
    for column in frame:
        if column == prepared['outcome']:
            continue
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', str(column)).lower()
        tokens = set(re.split(r'[^a-z0-9]+', name))
        meta = metadata.get(column, {})
        temporal = meta.get('variable_subtype') in ('Date', 'DateTime', 'Year-Month') or pd.api.types.is_datetime64_any_dtype(frame[column])
        candidate = tokens & {'id', 'identifier', 'subject', 'participant', 'patient', 'unit', 'product',
                              'operator', 'machine', 'device', 'batch', 'cluster', 'site', 'household'}
        candidate = candidate or meta.get('variable_subtype') in ('Identifier', 'ID')
        if temporal or tokens & {'time', 'date', 'timestamp'}:
            rows.append({'Column': str(column), 'Indicator': 'Possible temporal ordering',
                         'Observed result': f'{frame[column].notna().sum()} analyzed rows have values. Ordering alone does not prove dependence.'})
        if candidate:
            values = frame[column].dropna()
            repeats = int(values.duplicated(keep=False).sum())
            if repeats:
                rows.append({'Column': str(column), 'Indicator': 'Repeated unit or group labels',
                             'Observed result': f'{values.nunique()} labels; {repeats} rows share labels. Confirm what a label represents; treatment labels can also recur among independent units.'})
    return pd.DataFrame(rows, columns=['Column', 'Indicator', 'Observed result'])


def coverage_table(prepared, factors=None, limit=5000):
    """Bound Cartesian expansion; never allocate an enormous unobserved design."""
    factors = list(factors or prepared['factors'])
    possible = int(np.prod([len(prepared['levels'][f]) for f in factors], dtype=object))
    counts = prepared['clean'].groupby(factors, observed=True, sort=False).size()
    if possible <= limit:
        index = pd.MultiIndex.from_tuples(list(product(*(prepared['levels'][f] for f in factors))), names=factors)
        counts = counts.reindex(index, fill_value=0)
    table = counts.rename('Observations').reset_index()
    table['Coverage'] = np.select([table.Observations.eq(0), table.Observations.eq(1), table.Observations.eq(2)],
                                  ['Empty', 'Single observation', 'Two observations'], default='Replicated')
    return table, possible > limit


def guided_items(prepared, result=None, summary=None, answer=None, evidence=None, failure=None):
    context = {k: prepared[k] for k in ('outcome', 'factors', 'interactions', 'n')}
    items = []

    def add(key, category, severity, title, observed, explanation, action, target=None):
        items.append(Diagnostic(key, category, severity, title, observed, explanation, action,
                                'review' if target else None, target, context))

    cells = prepared['cells']
    low = int(cells.eq(2).sum())
    add('cell_coverage', 'design', 'review' if prepared['empty_cells'] or prepared['singleton_cells'] else 'ok',
        'Cell Coverage',
        f"{len(cells)} observed / {prepared['possible_cells']} possible combinations; {prepared['empty_cells']} empty; "
        f"{prepared['singleton_cells']} with one observation; {low} with two. Observed counts range from "
        f"{int(cells.min()) if len(cells) else 0} to {int(cells.max()) if len(cells) else 0}.",
        'Empty cells can limit separability or require extrapolation, but do not automatically invalidate an additive model. '
        'Single observations cannot estimate within-cell variance. Two observations provide little replication; this is descriptive, not a universal minimum sample rule.',
        ('Inspect affected combinations and classification. Seek additional observations where feasible. Review interactions using domain knowledge; never drop an important term solely to clear a diagnostic.'
         if prepared['empty_cells'] or prepared['singleton_cells'] else
         'All combinations are observed with replication. Inspect counts to understand design balance; no model change is indicated by this check.'), 'coverage')
    model = result['model'] if result else None
    parameters = len(model.params) if model is not None else prepared['parameters']
    residual_df = model.df_resid if model is not None else prepared['n'] - parameters
    low_df = residual_df < max(5, parameters)
    add('complexity', 'design', 'review' if low_df else 'ok', 'Model Complexity',
        f"{prepared['n']} usable observations; {parameters} model parameters; {residual_df:g} residual degrees of freedom" +
        (f'; fitted design rank {parameters} / {parameters}.' if model is not None else '; rank not confirmed by a valid fit.'),
        'Few residual degrees of freedom can reduce precision and diagnostic sensitivity. The relative-DF flag is an exploratory screen, not a minimum observations-per-parameter requirement.',
        'Review the scientific need for each selected term and available design support. Confirm any model change and rerun.', 'terms')
    if failure:
        add('estimability', 'model', 'blocking', 'Model Estimability', str(failure),
            'The requested inference is unavailable. No ANOVA p-values or fitted follow-up conclusions are reported.',
            'Inspect empty combinations, confounded factors and selected terms. Collect missing support or explicitly revise the scientifically justified model; rerun before interpreting results.', 'terms')
    else:
        add('estimability', 'model', 'ok', 'Model Estimability', 'The fitted design is full rank and the engine validated Type II inference.',
            'Selected effects can be estimated separately under this model. Individual follow-up contrasts still require their own estimability checks.',
            'Review cell support before interpreting extrapolated means or comparisons.', 'coverage')
    observed = ('User-provided assumption: observations were collected independently.' if answer == INDEPENDENT else
                'User reports repeated or clustered observations.' if answer == REPEATED else
                'Study design remains unresolved; independence has not been established.')
    if evidence is not None and len(evidence):
        observed += f' {len(evidence)} structural indicators need context; labels and ordering are not proof.'
    add('independence', 'assumption', 'ok' if answer == INDEPENDENT else 'unresolved', 'Independence of Observations', observed,
        'Rows from the same experimental unit may share variation, making independent-observation F tests misleading. No data-only check proves independence.',
        ('Review the existing fixed/random effect controls to specify appropriate grouping. The available Gaussian mixed model supports group intercepts and independent interaction components; correlated slopes, serial correlation and unequal residual variances are unsupported.'
         if answer == REPEATED else 'Confirm the sampling unit with the study owner. If unsure, conclusions remain conditional on independent observations.'), 'study')
    if summary:
        rows = summary['table'].set_index('Check')
        for key, check, title, target, action in [
            ('residuals', 'Residual normality', 'Residual Diagnostics', 'residuals', 'Inspect the Q–Q and residual plots, investigate unusual observations in context, and review response-scale suggestions below. Do not delete observations automatically.'),
            ('variance', 'Equal variance', 'Variance Assessment', 'variance', 'Inspect residual spread and the cell structure. If appropriate, request the existing HC3 sensitivity comparison in Model Overview; it does not repair dependence or missing cells.')]:
            row = rows.loc[check]
            observed = row.Interpretation
            if key == 'residuals':
                if summary['jb']:
                    observed += f" Jarque–Bera statistic {summary['jb'][0]:.4g}; {format_p_value(summary['jb'][1], result['alpha'], context='narrative')}."
                observed += f" {summary['extreme_residuals']} internally studentized residuals beyond ±3. " + ' '.join(summary['patterns'])
            elif result['levene'] is not None:
                observed += f" Levene statistic {fmt(result['levene'][0], 'f')}; {format_p_value(result['levene'][1], result['alpha'], context='narrative')}."
            review = row.Status != 'OK' or (key == 'residuals' and (summary['extreme_residuals'] or summary['curvature']))
            add(key, 'residual', 'review' if review else 'ok', title, observed,
                'Distribution departures, changing spread or mean patterns can affect classical uncertainty estimates. Automated screens are exploratory; a non-significant test does not prove an assumption.', action, target)
    return items
