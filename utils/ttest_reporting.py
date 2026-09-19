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
    rows = [('Mean difference', number(r['estimate'])), ('Standard error of difference', number(r['se'])),
            ('t statistic', number(r['t'])), ('Degrees of freedom', number(r['df'])),
            ('p-value', p_text(r['p'])),
            (f'{100*r["confidence"]:g}% CI lower bound', number(r['low'])),
            (f'{100*r["confidence"]:g}% CI upper bound', number(r['high'])),
            (r['effect_name'], number(r['effect']))]
    if r['mode'] == 'one':
        rows = [('Hypothesized mean', number(r['null_mean'])),
                ('Observed sample mean', number(r['estimate'] + r['null_mean']))] + rows
    return pd.DataFrame(rows, columns=['Statistic', 'Value'])


def interpretation(r):
    relation = {'two-sided': 'different from zero', 'greater': 'greater than zero', 'less': 'less than zero'}[r['alternative']]
    alpha = 1 - r['confidence']
    evidence = ('providing evidence that the population mean difference is ' + relation if r['p'] < alpha else
                'so the data do not provide sufficient evidence that the population mean difference is ' + relation)
    p = ('p < 0.001' if r['p'] < .001 else f'p = {r["p"]:.4g}')
    return (f'Difference = {r["difference"]}. The test evaluates whether this population mean difference is {relation}. '
            f'The estimated difference is {number(r["estimate"])} units '
            f'({100*r["confidence"]:g}% CI: {number(r["low"])} to {number(r["high"])}). '
            f'{r["method"]} gives t = {number(r["t"])}, df = {number(r["df"])}, {p}, '
            f'{evidence} at alpha = {alpha:.3g}. '
            f'{r["effect_name"]} = {number(r["effect"])}. '
            'Insufficient evidence does not establish equal means. Practical importance depends on the measurement scale and context.')


def export_result(r):
    """Export raw precision with method/direction, including infinite one-sided bounds."""
    return pd.DataFrame([r]).to_csv(index=False).encode('utf-8-sig')
