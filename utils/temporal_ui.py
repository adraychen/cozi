"""Descriptive display for clock times and elapsed durations (no invented dates)."""
import pandas as pd
import streamlit as st
from utils.charts import make_histogram_boxplot
from utils.temporal import temporal_value


def render_noncalendar_temporal(series, subtype, key_prefix):
    clean = series.dropna()
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
        st.plotly_chart(fig, use_container_width=True, key=key_prefix+'_distribution')
    st.table(pd.DataFrame(rows,columns=['Statistic','Value']))
    st.caption('Calendar-based trends and period comparisons require Date or DateTime.')
