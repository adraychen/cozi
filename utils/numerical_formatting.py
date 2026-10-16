"""Semantic display formatting. Never use formatted strings for calculations."""
from dataclasses import dataclass
from functools import partial, lru_cache
from pathlib import Path
import tomllib
import math
import numbers
import re


@dataclass(frozen=True)
class DisplaySettings:
    decimals: int | None = None
    p_mode: str = 'Standard'
    fixed: tuple = ()
    significant: tuple = ()
    p_decimals: int = 5
    p_digits: int = 3

    def __post_init__(self):
        if self.decimals not in (None, 2, 3, 4, 5):
            raise ValueError('Precision must be Standard or 2–5 decimal places.')
        if type(self.p_decimals) is not int or not 3 <= self.p_decimals <= 10 or type(self.p_digits) is not int or not 3 <= self.p_digits <= 10:
            raise ValueError('Detailed p-value precision must be 3–10 digits/decimals.')
        if self.p_mode not in ('Standard', 'Detailed'):
            raise ValueError('Choose Standard or Detailed p-values.')


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'NUMERICAL_PRECISION.toml'


@lru_cache(maxsize=8)
def _parse_config(content):
    config = tomllib.loads(content)
    if set(config) - {'display', 'decimal_places', 'significant_digits'}:
        raise ValueError('Unknown section in NUMERICAL_PRECISION.toml.')
    if any(not isinstance(value, dict) for value in config.values()):
        raise ValueError('Precision settings must use TOML sections.')
    display = config.get('display', {})
    if set(display) - {'precision', 'p_value_mode', 'p_value_detailed_decimals', 'p_value_scientific_digits'}:
        raise ValueError('Unknown display setting in NUMERICAL_PRECISION.toml.')
    precision = display.get('precision', 'Standard')
    precision = None if precision == 'Standard' else precision
    fixed = config.get('decimal_places', {})
    significant = config.get('significant_digits', {})
    if set(fixed) - set(FIXED) or set(significant) - set(SIGNIFICANT):
        raise ValueError('Only approved result types can be configured.')
    if any(type(n) is not int or not 0 <= n <= 10 for n in fixed.values()):
        raise ValueError('Decimal places must be integers from 0 to 10.')
    if any(type(n) is not int or not 1 <= n <= 10 for n in significant.values()):
        raise ValueError('Significant digits must be integers from 1 to 10.')
    return DisplaySettings(precision, display.get('p_value_mode', 'Standard'), tuple(fixed.items()), tuple(significant.items()), display.get('p_value_detailed_decimals', 5), display.get('p_value_scientific_digits', 3))


def current_settings():
    # Read on rendering, not import: document edits take effect on the next rerun.
    # Fail visibly rather than silently applying a partially invalid document.
    return _parse_config(CONFIG_PATH.read_text(encoding='utf-8'))


FIXED = {'mean': 2, 'median': 2, 'minimum': 2, 'maximum': 2, 'quantile': 2,
         'sd': 3, 'se': 3, 'ci': 2, 'f': 3, 't': 3, 'z': 3,
         'cohens_d': 3, 'cohens_f2': 3, 'eta_squared': 3,
         'correlation': 3, 'r2': 3, 'adjusted_r2': 3, 'ss': 3, 'ms': 3, 'variance': 3, 'df': 2, 'percentage': 1}
SIGNIFICANT = {'coefficient': 4, 'odds_ratio': 3, 'hazard_ratio': 3}
RESULT_TYPES = set(FIXED) | set(SIGNIFICANT) | {'count', 'df', 'p', 'percentage', 'proportion'}


def _value(value, allow_infinite=False):
    if value is None:
        return None, '—'
    if isinstance(value, str):
        return None, value
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None, '—'
    if math.isnan(number):
        return None, '—'
    if math.isinf(number):
        return None, ('∞' if number > 0 else '−∞') if allow_infinite else '—'
    return number, None


def _science(value, digits):
    mantissa, exponent = f'{value:.{digits - 1}e}'.split('e')
    return f'{mantissa}e{int(exponent)}'


def _fixed(value, decimals):
    # A real tiny effect must not disappear into rounded zero. Ordinary negative
    # zero (including tiny negative values that round to zero) has no minus sign.
    if value and (abs(value) < 1e-4 or abs(value) >= 1e6):
        return _science(value, max(3, decimals + 1))
    rendered = f'{value:,.{decimals}f}'
    if value and float(rendered.replace(',', '')) == 0:
        if abs(value) < 0.5 * 10 ** (-decimals):
            return _science(value, max(3, decimals + 1))
    return rendered.lstrip('-') if value == 0 else rendered


def format_significant_digits(value, digits=4, *, allow_infinite=False):
    number, special = _value(value, allow_infinite)
    if special is not None:
        return special
    if digits < 1:
        raise ValueError('At least one significant digit is required.')
    if number == 0:
        return '0'
    if abs(number) < 1e-4 or abs(number) >= 1e6:
        return _science(number, digits)
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(number))))
    rounded = round(number, digits - 1 - math.floor(math.log10(abs(number))))
    return f'{rounded:,.{decimals}f}'


def format_p_value(value, alpha=.05, mode=None, *, context='cell', settings=None):
    settings = settings or current_settings()
    mode = mode or settings.p_mode
    if mode not in ('Standard', 'Detailed'):
        raise ValueError('Choose Standard or Detailed p-values.')
    p, special = _value(value)
    if special is not None:
        return '—' if special == '—' else 'Invalid p-value'
    if not 0 <= p <= 1:
        return 'Invalid p-value'
    if alpha is not None and (not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('Significance level must be between zero and one.')
    # Zero may be numerical underflow: do not invent an exact small p-value.
    if p == 0:
        rendered = '<0.001' if alpha is None or alpha >= .001 else f'<{_science(alpha, 3)}'
    elif mode == 'Standard' and p < .001 and (alpha is None or alpha >= .001):
        rendered = '<0.001'
    else:
        if p < .001:
            rendered = _science(p, settings.p_digits)
        elif mode == 'Detailed':
            rendered = f'{p:.{settings.p_decimals}f}'
        else:
            rendered = f'{p:.3f}'
        # Rounded values must not imply equality or the opposite decision at α.
        if alpha is not None and p != alpha:
            digits = 3
            while (float(rendered) == alpha or (float(rendered) < alpha) != (p < alpha)) and digits <= 17:
                rendered = format(p, f'.{digits}g')
                digits += 1
    if context == 'narrative':
        return 'p < ' + rendered[1:] if rendered.startswith('<') else ('p = ' + rendered if rendered not in ('—', 'Invalid p-value') else rendered)
    return rendered


def format_statistic(value, result_type, settings=None, *, alpha=.05, allow_infinite=False):
    if result_type not in RESULT_TYPES:
        raise ValueError(f'Unapproved numerical result type: {result_type}')
    settings = settings or current_settings()
    if result_type in ('count', 'df') and isinstance(value, numbers.Integral):
        return f'{int(value):,}'
    if result_type == 'p':
        return format_p_value(value, alpha, settings=settings)
    number, special = _value(value, allow_infinite)
    if special is not None:
        return special
    if result_type == 'count':
        return f'{int(number):,}' if number.is_integer() else 'Invalid count'
    if result_type == 'df':
        return f'{int(number):,}' if number.is_integer() else _fixed(number, settings.decimals if settings.decimals is not None else dict(settings.fixed).get('df', 2))
    if result_type in SIGNIFICANT:
        return format_significant_digits(number, dict(settings.significant).get(result_type, SIGNIFICANT[result_type]), allow_infinite=allow_infinite)
    if result_type in ('percentage', 'proportion'):
        return _fixed(number * (100 if result_type == 'proportion' else 1), settings.decimals if settings.decimals is not None else dict(settings.fixed).get('percentage', 1)) + '%'
    return _fixed(number, settings.decimals if settings.decimals is not None else dict(settings.fixed).get(result_type, FIXED[result_type]))


def format_percentage(value, settings=None):
    return format_statistic(value, 'percentage', settings)


def format_confidence_interval(lower, upper, settings=None):
    settings = settings or current_settings()
    return '[' + format_statistic(lower, 'ci', settings, allow_infinite=True) + ', ' + format_statistic(upper, 'ci', settings, allow_infinite=True) + ']'


def format_results(data, result_types, *, alpha=.05, settings=None):
    """Opt-in schemas at result sites; never infer semantics from raw columns."""
    settings = settings or current_settings()
    frame = data.data if hasattr(data, 'data') and hasattr(data.data, 'columns') else data
    styled = data if frame is not data else frame.style
    result_types = {**getattr(styled, '_cozy_result_types', {}), **result_types}
    functions = {column: partial(format_statistic, result_type=kind, settings=settings, alpha=alpha, allow_infinite=kind == 'ci')
                 for column, kind in result_types.items() if column in frame.columns}
    styled = styled.format(functions, subset=list(functions)) if functions else styled
    styled._cozy_result_types = result_types
    styled._cozy_alpha = alpha
    styled._cozy_settings = settings or current_settings()
    return styled


def display_rows(frame, result_types, *, alpha=.05, settings=None):
    settings = settings or current_settings()
    return [[format_statistic(value, result_types[column], settings, alpha=alpha, allow_infinite=result_types[column] == 'ci')
             if column in result_types else value
             for column, value in zip(frame.columns, row)]
            for row in frame.itertuples(index=False, name=None)]


def plot_number_format(kind, settings=None):
    """D3 format for chart labels; axes themselves keep scale-aware defaults."""
    settings = settings or current_settings()
    if kind == 'count':
        return ',d'
    if kind in SIGNIFICANT:
        return f'.{dict(settings.significant).get(kind, SIGNIFICANT[kind])}g'
    return f',.{settings.decimals if settings.decimals is not None else dict(settings.fixed).get(kind, FIXED.get(kind, 2))}f'


# Schemas describe known statistical outputs, never arbitrary input datasets.
DESCRIPTIVE = {
    'N': 'count', 'Count': 'count', 'Mean': 'mean', 'Median': 'median',
    'SD': 'sd', 'Std Dev': 'sd', 'Standard deviation': 'sd', 'Standard error': 'se', 'SE': 'se',
    'Variance': 'variance', 'Minimum': 'minimum', 'Maximum': 'maximum',
    'Min': 'minimum', 'Max': 'maximum', '25%': 'quantile', '50%': 'quantile', '75%': 'quantile',
    '95% CI Lower': 'ci', '95% CI Upper': 'ci', '95% CI lower': 'ci', '95% CI upper': 'ci',
    'IQR outlier flags': 'count', 'Adjusted mean (EMM)': 'mean', 'Adjusted mean (Estimated Marginal Mean)': 'mean',
}
INFERENCE = {'F': 'f', 't': 't', 'z': 'z', 'DF': 'df', 'df': 'df',
             'df numerator': 'df', 'df denominator': 'df', 'df between': 'df',
             'p-value': 'p', 'Adjusted p-value': 'p', 'Holm-adjusted p-value': 'p', 'HC3 p-value': 'p', 'HC3 F': 'f', '95% simultaneous CI lower': 'ci', '95% simultaneous CI upper': 'ci', 
             "Effect Size (Cohen's f²)": 'cohens_f2', 'Sum of Squares': 'ss',
             'Sum of Squares (SS)': 'ss', 'Mean Square': 'ms', 'Mean Square (MS)': 'ms',
             'Variation (%)': 'percentage', 'SE': 'se'}
REGRESSION = {'Coefficient': 'coefficient', 'Estimate': 'coefficient', 'Std. Error': 'se',
              'Std Error': 'se', 'Standard Error': 'se', 'SE': 'se',
              't': 't', 't-statistic': 't', 'Standard error': 'se', 'z': 'z', 'F': 'f', 'df': 'df', 'DF': 'df',
              'p-value': 'p', 'p-value (asymptotic)': 'p', 'Wald z (asymptotic)': 'z', 'Odds Ratio': 'odds_ratio'}


def approved_schema(frame, candidates):
    return {column: kind for column, kind in candidates.items() if column in frame.columns}


def format_row_results(frame, label_column, value_column, result_types, *, alpha=.05, settings=None, fallback=None):
    """Schemas for authored Statistic/Value tables with mixed numerical meanings."""
    settings = settings or current_settings()
    styled = frame if hasattr(frame, 'data') and hasattr(frame.data, 'columns') else frame.style
    frame = styled.data
    rendered = [list(row) for row in getattr(styled, '_cozy_display_rows', list(frame.itertuples(index=False, name=None)))]
    value_index = frame.columns.get_loc(value_column)
    if fallback is not None:
        styled.format(fallback, subset=[value_column])
        for row in rendered:
            row[value_index] = fallback(row[value_index])
    for i, (index, row) in enumerate(frame.iterrows()):
        kind = result_types.get(row[label_column])
        if kind:
            fn = partial(format_statistic, result_type=kind, settings=settings, alpha=alpha, allow_infinite=kind == 'ci')
            styled.format(fn, subset=([index], [value_column]))
            rendered[i][value_index] = fn(row[value_column])
    styled._cozy_display_rows = rendered
    return styled


def format_plot_hover(figure, bindings, *, alpha=.05):
    """Append display-only hover fields while keeping original plotted arrays."""
    import numpy as np
    settings = current_settings()
    for trace in figure.data:
        template = trace.hovertemplate
        if not template or not isinstance(template, str):
            continue
        length = len(trace.x) if trace.x is not None else len(trace.y) if trace.y is not None else 0
        if not length:
            continue
        custom = np.asarray(trace.customdata, dtype=object) if trace.customdata is not None else np.empty((length, 0), dtype=object)
        if custom.ndim == 1:
            custom = custom.reshape(-1, 1)
        if custom.shape[0] != length:
            continue
        for field, kind in bindings.items():
            pattern = r'%\{' + re.escape(field) + r'(?::[^}]*)?\}'
            if not re.search(pattern, template):
                continue
            if field.startswith('customdata['):
                index = int(field[len('customdata['):-1])
                if index >= custom.shape[1]:
                    continue
                values = custom[:, index]
            else:
                values = getattr(trace, field, None)
                if values is None or len(values) != length:
                    continue
            rendered = [format_statistic(value, kind, settings, alpha=alpha, allow_infinite=kind == 'ci') for value in values]
            index = custom.shape[1]
            custom = np.column_stack((custom, rendered))
            template = re.sub(pattern, '%{customdata[' + str(index) + ']}', template)
        trace.customdata = custom
        trace.hovertemplate = template
    return figure


def validate_display_narrative(text, evidence):
    """Reject newly rounded/invented numeric literals rather than rewrite prose.

    Callers retain their deterministic fallback. This validates formatting, not
    the statistical truth of every AI-generated sentence.
    """
    pattern = r'(?<![A-Za-z])[-+−]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?%?'
    allowed = set(re.findall(pattern, str(evidence)))
    actual = set(re.findall(pattern, str(text)))
    if not actual.issubset(allowed):
        raise ValueError('AI interpretation changed numerical display precision; deterministic findings remain available.')
    return text


def sync_formatting_context(state):
    """Invalidate live AI prose after document edits; never touch report snapshots."""
    key = 'view_numerical_context'
    context = repr(current_settings())
    if state.get(key) != context:
        for name in list(state):
            if name in ('ai_summary', 'rel_anova_ai') or isinstance(name, str) and name.startswith(('col_ai_', 'pa_ai__')):
                state.pop(name, None)
        state[key] = context
