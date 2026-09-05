"""Escaped HTML contingency report with stacked count/expected/residual cells."""
from html import escape


def contingency_table_html(result, row_name, column_name):
    observed = result['observed']
    label = lambda value: escape(str(value), quote=True)
    total_label = 'All (total)' if 'All' in observed.index or 'All' in observed.columns else 'All'
    parts = ['''<style>
    .ct-report {overflow-x:auto; padding:8px 0 16px;}
    .ct-report h4 {color:#0077b6; margin:0 0 18px;}
    .ct-report table {border-collapse:collapse; width:auto; font-variant-numeric:tabular-nums;}
    .ct-report th,.ct-report td {border:0; padding:8px 18px; text-align:right; vertical-align:top; white-space:nowrap;}
    .ct-report thead {border-bottom:2px solid currentColor;}
    .ct-report th:first-child {text-align:left;}
    .ct-report tbody td,.ct-report tbody th {padding-top:22px; padding-bottom:22px;}
    .ct-report .ct-cell div {line-height:1.9;}
    .ct-report .ct-total {line-height:1.9;}
    .ct-report tfoot {border-top:1px solid currentColor;}
    .ct-report .ct-legend {margin:18px 0 0 18px; font-style:italic; line-height:1.6;}
    </style><div class="ct-report">''',
    f'<h4>Rows: {label(row_name)} &nbsp; Columns: {label(column_name)}</h4>',
    '<table aria-label="Contingency table: count, expected count, standardized residual"><thead><tr><th scope="col"></th>']
    parts.extend(f'<th scope="col">{label(c)}</th>' for c in observed.columns)
    parts.append(f'<th scope="col">{total_label}</th></tr></thead><tbody>')
    for r in observed.index:
        parts.append(f'<tr><th scope="row" class="ct-total">{label(r)}</th>')
        for c in observed.columns:
            parts.append(f'<td class="ct-cell"><div>{observed.loc[r,c]:,.0f}</div>'
                         f'<div>{result["expected"].loc[r,c]:,.2f}</div>'
                         f'<div>{result["residual"].loc[r,c]:.4f}</div></td>')
        parts.append(f'<td class="ct-total">{observed.loc[r].sum():,.0f}</td></tr>')
    parts.append(f'</tbody><tfoot><tr><th scope="row">{total_label}</th>')
    parts.extend(f'<td>{observed[c].sum():,.0f}</td>' for c in observed.columns)
    parts.append(f'<td>{result["n"]:,}</td></tr></tfoot></table>'
                 '<div class="ct-legend">Cell Contents<br>&emsp;Count<br>&emsp;Expected count<br>'
                 '&emsp;Standardized residual</div></div>')
    return ''.join(parts)
