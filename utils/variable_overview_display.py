"""Escaped, accessible overview table with inline classification/value details."""
from html import escape
import pandas as pd
import streamlit as st
from utils.loader import get_dataframe_info
from utils.temporal import SUBTYPES, parse_temporal, temporal_value, infer_temporal_subtype


def value_details(series, metadata):
    values=list(series.dropna().unique())
    subtype=metadata['variable_subtype']
    if not values:
        return 'No non-missing values', []
    if subtype=='Continuous' or (subtype=='Discrete' and len(values)>10):
        numeric=pd.to_numeric(series.dropna(),errors='raise')
        return 'Range', [('Minimum',numeric.min()),('Maximum',numeric.max())]
    if subtype in SUBTYPES or subtype == 'Date / Time':
        subtype = infer_temporal_subtype(series) if subtype == 'Date / Time' else subtype
        parsed = parse_temporal(series, subtype or 'DateTime').dropna()
        return f'{subtype} range', [('Minimum',temporal_value(parsed.min(),subtype)),('Maximum',temporal_value(parsed.max(),subtype))] if len(parsed) else []
    if subtype=='Ordinal':
        return 'All values, lowest to highest', [('Value',v) for v in metadata['order'] if v in values]
    if subtype=='Nominal':
        return 'First 5 unique values in observed order', [('Value',v) for v in values[:5]]
    return 'All unique values', [('Value',v) for v in values]


def overview_html(raw, metadata):
    def text(value):
        return escape(str(value),quote=True)
    def classified(value, source):
        value = 'Temporal' if value == 'Date / Time' else value
        manual=source=='User override'
        description='Manual Override' if manual else 'Automatic classification'
        badge='M' if manual else 'A'
        return text(value)+f' <span class="vm-badge" tabindex="0" title="{description}" aria-label="{description}">{badge}<span class="vm-tooltip" role="tooltip">{description}</span></span>'
    headers=['Column Name','Data Type','Variable Type','Variable Subtype','Unique Values','Total Values','Missing Values','Missing %']
    rows=[]
    for index, row in get_dataframe_info(raw).iterrows():
        column=row['Column Name'];m=metadata[column]
        heading,values=value_details(raw[column],m)
        items=''.join('<li>'+ (text(label)+': ' if label!='Value' else '')+text(value)+'</li>' for label,value in values)
        detail=f'<details class="vm-details"><summary aria-label="Value details for {text(column)}" title="Show value details">ⓘ</summary><div class="vm-values"><strong>{text(heading)}</strong><ul>{items}</ul></div></details>'
        missing=float(row['Missing %'])
        style=' class="vm-missing-high"' if missing>=20 else ' class="vm-missing-medium"' if missing>=5 else ''
        cells=[text(column),text(row['Data Type']),classified(m['variable_type'],m['source']),classified(m['variable_subtype'],m['source']),f'<span>{raw[column].nunique(dropna=True)}</span> {detail}',text(row['Total Values']),text(row['Missing Values'])]
        rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+f'<td{style}>{missing:.2f}</td></tr>')
    return '''<style>
.vm-overview {width:fit-content; max-width:100%; overflow:auto; max-height:540px; border:1px solid #8885; border-radius:8px;}
.vm-overview table {width:max-content; max-width:1200px; border-collapse:collapse; font-size:14px; text-align:left;}
.vm-overview th,.vm-overview td {padding:10px 12px; border-bottom:1px solid #8883; vertical-align:top;}
.vm-overview th {white-space:nowrap; font-weight:600;}
.vm-badge {display:inline-block; position:relative; border:1px solid #8888; border-radius:5px; padding:0 4px; font-size:11px; font-weight:600; cursor:help;}
.vm-tooltip {display:none; position:absolute; top:100%; left:0; z-index:10; background:#263445; color:white; padding:5px 8px; border-radius:4px; white-space:nowrap; font-size:12px; font-weight:400;}
.vm-badge:hover .vm-tooltip,.vm-badge:focus .vm-tooltip {display:block;}
.vm-details {display:inline-block; vertical-align:top;}
.vm-details summary {display:inline; cursor:pointer; color:#3978b7; font-size:16px;}
.vm-details summary:focus-visible {outline:2px solid #3978b7; outline-offset:3px;}
.vm-values {min-width:180px; max-width:360px; max-height:220px; overflow:auto; padding:8px; border:1px solid #8885; border-radius:5px; margin-top:6px;}
.vm-values ul {padding-left:20px; margin:4px 0;}
.vm-missing-high {background:#f8d7da; color:#842029;}
.vm-missing-medium {background:#fff3cd; color:#664d03;}
</style><div class="vm-overview"><table aria-label="Column Overview"><thead><tr>'''+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


@st.fragment
def _render_value_details(raw, metadata):
    """Refresh the selected column's details independently of workspace navigation."""
    from utils import presentation

    column = st.selectbox('Column for value details', list(raw), key='vm_value_details_column')
    heading, values = value_details(raw[column], metadata[column])
    # Read-only dataframes ignore widget keys; key the output container so the
    # browser replaces the previous column's heading and table together.
    with st.container(key=f"vm_value_details_result_{st.session_state.get('file_id', '')}_{column}"):
        st.write(heading)
        if values:
            presentation.dataframe(
                pd.DataFrame([(str(label), str(value)) for label, value in values],
                             columns=['Detail', 'Value']), hide_index=True,
            )


def render_overview(raw, metadata):
    """Native preview-style table; retain classification and value explanations."""
    from utils import presentation

    overview = get_dataframe_info(raw).copy()
    overview['Unique Values'] = [raw[column].nunique(dropna=True) for column in overview['Column Name']]
    for field, title in [('variable_type', 'Variable Type'), ('variable_subtype', 'Variable Subtype')]:
        overview[title] = [
            ('Temporal' if metadata[column][field] == 'Date / Time' else metadata[column][field])
            + (' (M)' if metadata[column]['source'] == 'User override' else ' (A)')
            for column in overview['Column Name']
        ]
    overview = overview[['Column Name', 'Data Type', 'Variable Type', 'Variable Subtype',
                         'Unique Values', 'Total Values', 'Missing Values', 'Missing %']]
    def missing_color(value):
        return ('background-color: #f8d7da; color: #842029' if value >= 20 else
                'background-color: #fff3cd; color: #664d03' if value >= 5 else '')
    styled = overview.style.format({'Missing %': '{:.2f}'}).map(missing_color, subset=['Missing %'])
    presentation.dataframe(styled, hide_index=True, column_config={
        'Variable Type': st.column_config.TextColumn(help='A = Automatic classification; M = Manual Override.'),
        'Variable Subtype': st.column_config.TextColumn(help='A = Automatic classification; M = Manual Override.'),
        'Unique Values': st.column_config.NumberColumn(help='Open Unique value details below for values or ranges.'),
    })
    with st.expander('Unique value details'):
        _render_value_details(raw, metadata)
