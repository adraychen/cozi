"""Deterministic, model-aware descriptions and display formatting."""
import numpy as np
from utils.regression_design import label


def p_text(p):
    return 'N/A' if not np.isfinite(p) else '< 0.001' if p < .001 else f'{p:.3f}'


def format_table(frame):
    display = frame.copy()
    for c in display.columns:
        if c == 'p-value':
            display[c] = display[c].map(p_text)
        elif display[c].dtype.kind in 'fc':
            display[c] = display[c].map(lambda v:'N/A' if np.isnan(v) else f'{v:.4g}')
    return display


def equation(result):
    values = result['model'].params
    return result['spec']['target'] + ' = ' + f'{values[0]:.5g}' + ''.join(
        f' {b:+.5g} × {name}' for name,b in zip(result['names'][1:],values[1:]))


def findings(result, ordered=False, visual_concerns=()):
    spec = result['spec']; y = spec['target']; lines = []
    lines.append(f'The model explains {100*result["r2"]:.1f}% of the observed variation in {y} (Adjusted R² = {result["adjusted_r2"]:.3f}).')
    p = result['f_p']
    if np.isfinite(p):
        lines.append(f'The overall regression {"shows evidence of an association" if p < .05 else "does not show strong evidence of an association"} (F = {result["f_stat"]:.4g}, p {"" if p < .001 else "= "}{p_text(p)}).')
    else:
        lines.append('An overall regression F-test is unavailable for an intercept-only or perfect-fit model.')
    involved = {v for b in result['active'] if b[0]!='main' for v in (b[1:3] if b[0]=='interaction' else [b[1]])}
    for row in result['term_tests'].to_dict('records'):
        b = row['block']; p = row['p-value']
        if not np.isfinite(p):
            continue
        if b[0] == 'interaction' and p < .05:
            lines.append(f'{label(b)}: the relationship between {b[1]} and {y} depends on the value or level of {b[2]} (joint p {"" if p < .001 else "= "}{p_text(p)}).')
        elif b[0] == 'power' and p < .05:
            lines.append(f'{label(b)} provides evidence of curvature in the relationship with {y}; interpret the fitted curve over the observed range, not this coefficient as a constant slope.')
        elif b[0] == 'main' and b[1] not in involved:
            c = b[1]
            if c in spec['categorical']:
                lines.append(f'{c}: {"evidence" if p < .05 else "no strong evidence"} of adjusted differences from reference {spec["levels"][c][0]!r} (joint p {"" if p < .001 else "= "}{p_text(p)}).')
            else:
                i = result['indices'][b][0]; coef = result['coefficients'].iloc[i]
                lines.append(f'{c} has an estimated {"positive" if coef["Coefficient"] >= 0 else "negative"} association with {y}: {coef["Coefficient"]:.4g} units per one-unit increase, holding other predictors constant (95% CI {coef["95% lower"]:.4g} to {coef["95% upper"]:.4g}; p {"" if p < .001 else "= "}{p_text(p)}; {"statistically significant" if p < .05 else "not statistically significant"} at α = 0.05).')
        elif b[0] == 'main':
            lines.append(f'{b[1]} participates in a polynomial or interaction: its coefficient is conditional at centered zero/reference levels and is not an overall constant slope.')
    if not result['vif'].empty:
        count = int((result['vif']['VIF'] >= 5).sum())
        lines.append(f'{count} encoded model terms have VIF ≥ 5. These thresholds are guidance; generated terms can remain correlated after centering.')
    for item in result['diagnostics'].to_dict('records'):
        if item['Diagnostic'].startswith('Breusch'):
            p = item['p-value']
            if np.isfinite(p):
                lines.append('The variance diagnostic suggests nonconstant residual variance.' if p < .05 else 'No strong evidence of heteroscedasticity was detected; constant variance is not proven.')
        elif item['Diagnostic'] == 'Jarque–Bera' and item['p-value'] < .05:
            lines.append('The residual normality diagnostic suggests departures. Inspect the Q–Q plot for their nature and magnitude.')
    for concern in visual_concerns:
        lines.append(f'User-reviewed visual diagnostic: {concern}. Consider model-form and inference limitations.')
    if not visual_concerns:
        lines.append('Inspect residual and Q–Q plots for systematic patterns and large departures; numerical tests do not establish their absence.')
    if ordered and np.isfinite(result['dw']):
        lines.append(f'Durbin–Watson = {result["dw"]:.3f}; values toward 0 or 4 can suggest positive or negative first-order autocorrelation. No fixed pass/fail cutoff is applied.')
    else:
        lines.append('Autocorrelation is not interpreted without meaningful observation order; independence remains a study-design question.')
    lines.append(f'{int(result["influence"]["Flagged"].sum())} observations are flagged for review by residual, leverage or influence heuristics. None were deleted.')
    lines.append('Confidence intervals quantify uncertainty in a mean response; prediction intervals describe uncertainty for an individual future observation, conditional on the model. They are not guarantees.')
    if result['elimination']:
        lines.append('Backward elimination makes the reported inference exploratory and conditional on selection; selection uncertainty is not included.')
    lines.append('These associations do not imply causation.')
    return lines


def selected_terms_table(result):
    """Describe selected blocks using the fitted training centers, including removed terms."""
    import pandas as pd
    spec=result['spec']
    rows=[]
    for block in spec['blocks']:
        variables=list(block[1:]) if block[0]=='interaction' else [block[1]]
        numeric=[c for c in variables if c in spec['numeric']]
        if not numeric:
            means=transformation='Not applicable'
        else:
            means='; '.join(f'{c}: {spec["means"][c]:.6g}' for c in numeric) if spec['centered'] else 'Not applied'
            factors=[]
            for c in variables:
                if c in spec['categorical']:
                    factors.append(f'indicators({c})')
                elif spec['centered']:
                    factors.append(f'({c} − {spec["means"][c]:.6g})')
                else:
                    factors.append(str(c))
            transformation=' × '.join(factors)
            if block[0]=='power':
                transformation+={2:'²',3:'³'}[block[2]]
        rows.append({'Term':label(block),'Type':{'main':'Main effect','power':'Polynomial','interaction':'Interaction'}[block[0]],
                     'Final status':'Retained' if block in result['active'] else 'Removed',
                     'Centering - Mean Subtracted':means,'Centering - Transformation':transformation})
    return pd.DataFrame(rows,columns=['Term','Type','Final status','Centering - Mean Subtracted','Centering - Transformation'])
