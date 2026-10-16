from utils.numerical_formatting import format_statistic as fmt, format_p_value, format_row_results
"""Tables and direction-aware interpretation for classical t-tests."""
import numpy as np
import pandas as pd
from utils.regression_reporting import p_text


def number(value):
    if np.isnan(value):
        return 'N/A'
    if np.isinf(value):
        return '+infinity' if value > 0 else '-infinity'
    return f'{value:.5g}'


def result_table(r):
    rows = [('Mean difference', r['estimate']), ('Standard error of difference', r['se']),
            ('t statistic', r['t']), ('Degrees of freedom', r['df']), ('p-value', r['p']),
            (f'{100*r["confidence"]:g}% CI lower bound', r['low']),
            (f'{100*r["confidence"]:g}% CI upper bound', r['high']),
            (r['effect_name'], r['effect'])]
    schema = {'Standard error of difference':'se', 't statistic':'t', 'Degrees of freedom':'df',
              'p-value':'p', f'{100*r["confidence"]:g}% CI lower bound':'ci',
              f'{100*r["confidence"]:g}% CI upper bound':'ci'}
    if r['mode'] == 'one':
        rows = [('Hypothesized mean', r['null_mean']), ('Observed sample mean', r['estimate']+r['null_mean'])]+rows
        schema.update({'Hypothesized mean':'mean', 'Observed sample mean':'mean'})
    if r['effect_name'] != "Cohen's dz":
        schema[r['effect_name']] = 'cohens_d'
    frame = pd.DataFrame(rows, columns=['Statistic', 'Value'])
    return format_row_results(frame, 'Statistic', 'Value', schema, alpha=1-r['confidence'], fallback=number)


def interpretation(r):
    relation = {'two-sided': 'different from zero', 'greater': 'greater than zero', 'less': 'less than zero'}[r['alternative']]
    alpha = 1 - r['confidence']
    evidence = ('providing evidence that the population mean difference is ' + relation if r['p'] < alpha else
                'so the data do not provide sufficient evidence that the population mean difference is ' + relation)
    p = format_p_value(r['p'], alpha, context='narrative')
    return (f'Difference = {r["difference"]}. The test evaluates whether this population mean difference is {relation}. '
            f'The estimated difference is {number(r["estimate"])} units '
            f'({100*r["confidence"]:g}% CI: {fmt(r['low'], 'ci', allow_infinite=True)} to {fmt(r['high'], 'ci', allow_infinite=True)}). '
            f'{r["method"]} gives t = {fmt(r['t'], 't')}, df = {fmt(r['df'], 'df')}, {p}, '
            f'{evidence} at alpha = {alpha:.3g}. '
            f'{r["effect_name"]} = {number(r['effect']) if r['effect_name'] == "Cohen's dz" else fmt(r['effect'], 'cohens_d')}. '
            'Insufficient evidence does not establish equal means. Practical importance depends on the measurement scale and context.')


def export_result(r):
    """Export raw precision with method/direction, including infinite one-sided bounds."""
    return pd.DataFrame([r]).to_csv(index=False).encode('utf-8-sig')
