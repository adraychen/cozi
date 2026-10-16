"""Descriptive display for clock times and elapsed durations (no invented dates)."""
import pandas as pd
import streamlit as st
from utils import presentation
import plotly.express as px
from utils.charts import make_histogram_boxplot
from utils.temporal import temporal_value


def render_noncalendar_temporal(series, subtype, key_prefix, *, interactive_summary=False):
    clean = series.dropna()
    if subtype in ('Year', 'Month', 'Year-Month', 'Quarter', 'Year-Quarter'):
        st.caption('Year: a calendar year, without a month or day.' if subtype == 'Year' else
                   'Month: month of the year, without an associated year. Values from different years are pooled.' if subtype == 'Month' else
                   'Quarter: quarter of the year, without an associated year. Repeated quarters are pooled.' if subtype == 'Quarter' else
                   'Year-Quarter: a specific calendar quarter; no particular day is implied.' if subtype == 'Year-Quarter' else
                   'Year-Month: a specific calendar month. The first day is only an internal period anchor, not an observed day.')
        counts = clean.value_counts().sort_index()
        table = pd.DataFrame({subtype: [temporal_value(v, subtype) for v in counts.index], 'Count': counts.to_numpy()})
        st.write(f'Used values: {len(clean)} · Missing / invalid values: {series.isna().sum()} · Unique values: {clean.nunique()}')
        presentation.dataframe(table, result_types={'Count': 'count'}, hide_index=True, width='stretch')
        if len(clean):
            presentation.plotly_chart(px.bar(table, x=subtype, y='Count'), key=key_prefix+'_distribution', use_container_width=True)
        if subtype in ('Year', 'Month', 'Quarter'):
            st.caption('These components are not full dates. Dated monthly trends and period comparisons require a Year-Month, Date, or DateTime variable.')
        return
    st.caption(f'{subtype}: ' + ('time of day; no calendar date is attached.' if subtype == 'Time' else 'elapsed time, not a calendar timestamp.'))
    rows = [('Used values', str(len(clean))), ('Missing / invalid values', str(series.isna().sum())),
            ('Unique values', str(clean.nunique()))]
    if len(clean):
        rows += [('Minimum', temporal_value(clean.min(),subtype)), ('Maximum',temporal_value(clean.max(),subtype))]
        if subtype == 'Duration':
            rows += [('Mean duration',str(clean.mean())), ('Median duration',str(clean.median()))]
        seconds = clean.dt.total_seconds()
        fig = make_histogram_boxplot(seconds)
        fig.update_xaxes(title_text='Seconds since midnight' if subtype == 'Time' else 'Elapsed seconds', row=2, col=1)
        presentation.plotly_chart(fig, use_container_width=True, key=key_prefix+'_distribution')
    summary = pd.DataFrame(rows, columns=['Statistic', 'Value'])
    if interactive_summary:
        presentation.dataframe(summary, hide_index=True, use_container_width=True)
    else:
        presentation.table(summary)
    st.caption('Calendar-based trends and period comparisons require Date or DateTime.')
