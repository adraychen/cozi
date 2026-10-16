from utils.numerical_formatting import format_plot_hover
"""Guided controls and captured outputs for the existing period comparison."""
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st
from utils import presentation
from utils.report_capture import begin_result, offer_result
from utils.period_comparison import interpret_time, frequency_rules, build_timeline, calculate, interpret_result


def comparison_charts(result, metric, frequency):
    periods = result.sort_values('_period')['Period'].drop_duplicates().tolist()
    actual = go.Figure()
    changes = go.Figure()
    for index, (segment, rows) in enumerate(result.groupby('Segment', sort=False)):
        color = qualitative.Plotly[index % len(qualitative.Plotly)]
        rows = rows.sort_values('_period')
        values = rows['Value'].where(rows['Value'].notna(), None).astype(object).replace({np.nan: None}).tolist()
        actual.add_trace(go.Scatter(x=rows['Period'].tolist(), y=values, name=segment,
                                   mode='lines+markers', connectgaps=False, line={'color': color},
                                   customdata=rows[['Missing period', 'Missing value', 'Inserted zero']].to_numpy(),
                                   hovertemplate='%{x}<br>Value: %{y:,.2f}<br>Inserted zero: %{customdata[2]}<extra>%{fullData.name}</extra>'))
        percentages = rows['Change %'].astype(object).replace({np.nan: None}).tolist()
        changes.add_trace(go.Bar(x=rows['Period'].tolist(), y=percentages, name=segment,
                                marker_color=color, customdata=rows[['Previous period', 'Change', 'Comparison']].astype(object).fillna('N/A').to_numpy(),
                                hovertemplate='%{x}<br>Change: %{y:+.2f}%<br>Baseline: %{customdata[0]}<br>Absolute change: %{customdata[1]}<br>%{customdata[2]}<extra>%{fullData.name}</extra>'))
        unavailable = rows[rows['Change %'].isna()]
        if len(unavailable):
            changes.add_trace(go.Scatter(x=unavailable['Period'].tolist(), y=[0.] * len(unavailable),
                                        mode='text', text=['N/A'] * len(unavailable), textposition='top center',
                                        textfont={'color': color}, name=segment, showlegend=False,
                                        customdata=unavailable['Comparison'].tolist(),
                                        hovertemplate='%{x}<br>Change %: N/A<br>%{customdata}<extra>%{fullData.name}</extra>'))
    for figure, title, y_title in ((actual, 'Actual values over time', metric), (changes, 'Period-over-period percentage change', 'Change (%)')):
        figure.update_layout(title=title, xaxis_title=f'{frequency} period', yaxis_title=y_title,
                             xaxis={'type': 'category', 'categoryorder': 'array', 'categoryarray': periods},
                             legend_title_text='Segment', margin={'l': 30, 'r': 15, 't': 55, 'b': 35})
    changes.update_layout(barmode='group')
    changes.add_hline(y=0, line_color='#666', line_width=1)
    return actual, changes


def render_period_comparison(df, measure, dimensions, date, statistic, key_prefix):
    metadata = st.session_state.get('variable_metadata', {})
    raw = st.session_state.get('ws_datasets', {}).get(st.session_state.get('ws_active'), {}).get('frame')
    source = raw[date] if raw is not None and date in raw and raw.index.equals(df.index) else None
    try:
        axis = interpret_time(df[date], metadata.get(date, {}).get('variable_subtype'), source)
        from utils.temporal_validation_ui import guide_temporal_selection
        context = {'tab': 'Comparison', 'numerical': [measure] if measure is not None else [],
                   'categorical': dimensions, 'analysis': 'Period-over-Period Comparison'}
        allowed, assessment, objective = guide_temporal_selection(df, date, metadata, context,
            'temporal_guidance_' + key_prefix + '_' + str(date), allow_seasonal=True)
        if not allowed:
            return
        rules = frequency_rules(axis, statistic)
        frequencies = [name for name, (status, _) in rules.items() if status != 'Unavailable']
        if not frequencies:
            presentation.info(next(iter(rules.values()))[1])
            return
        frequency_key = key_prefix + '_frequency'
        context = (st.session_state.get('file_id'), date, axis.kind, measure, tuple(dimensions), statistic)
        if st.session_state.get(key_prefix + '_frequency_context') != context:
            st.session_state[key_prefix + '_frequency_context'] = context
            # A clarification or gap policy belongs to this metric and temporal selection.
            for suffix in ('_missing_policy', '_month_reporting'):
                st.session_state.pop(key_prefix + suffix, None)
        if st.session_state.get(frequency_key) not in frequencies:
            st.session_state.pop(frequency_key, None)
        default = 'Monthly' if 'Monthly' in frequencies else frequencies[0]
        frequency = st.selectbox('Period frequency', frequencies, index=frequencies.index(default), key=frequency_key)
        with st.expander('Frequency compatibility'):
            presentation.dataframe(pd.DataFrame([{'Frequency': name, 'Status': status, 'Explanation': reason}
                                                for name, (status, reason) in rules.items()]), hide_index=True)
        presentation.caption('Choose the statistic for your metric: Sum for additive amounts, Mean for averages, or Period-end value for dated balances. Names alone do not determine the statistic.')
        clarification = None
        if axis.kind == 'Month' and frequency == 'Quarterly':
            presentation.warning(rules[frequency][1])
            clarification = st.selectbox('What do the month labels represent?', [None, 'Monthly observations', 'Quarterly reports labelled by month'],
                                         format_func=lambda value: 'Choose the reporting meaning…' if value is None else value,
                                         key=key_prefix + '_month_reporting')
            if clarification is None:
                presentation.info('Choose the reporting meaning to calculate a defensible quarterly result.')
                return
        planned = build_timeline(df, measure, dimensions, axis, frequency, statistic, clarification,
                                 objective, metadata, assessment.year_column)
        gap_count = len(planned['gaps'])
        zero = False
        policy_key = key_prefix + '_missing_policy'
        gap_digest = hashlib.sha256(pd.util.hash_pandas_object(planned['gaps'], index=False).values.tobytes()).hexdigest()
        gap_context = (context, frequency, gap_digest)
        if st.session_state.get(key_prefix + '_gap_context') != gap_context:
            st.session_state[key_prefix + '_gap_context'] = gap_context
            st.session_state.pop(policy_key, None)
        if gap_count:
            if not dimensions:
                first, last = planned['timeline'].iloc[[0, -1]]['Period']
                gap_message = f'{gap_count:,} {frequency.lower()} periods are missing between {first} and {last}.'
            else:
                gap_message = f'{gap_count:,} {frequency.lower()} segment-periods are missing within the observed reporting range of each segment.'
            presentation.warning(gap_message)
            with st.expander('Missing periods'):
                presentation.dataframe(planned['gaps'].head(100), hide_index=True)
                if gap_count > 100:
                    presentation.caption('Showing the first 100 gaps. All periods remain in the results table.')
            policy = st.radio('How should missing periods be handled?',
                              ['Keep missing periods empty (recommended)', 'Treat missing periods as zero'],
                              key=policy_key)
            zero = policy == 'Treat missing periods as zero'
            presentation.caption('This applies only to periods with no rows. Existing blank metric values remain blank; actual observed zeros are retained.')
        else:
            policy = 'Keep missing periods empty (no missing periods)'
            st.session_state.pop(policy_key, None)
        mode = 'Previous period'
        lag = 1
        if axis.kind not in ('Month', 'Quarter') and frequency in ('Monthly', 'Quarterly', 'Yearly'):
            comparison_key = key_prefix + '_comparison_' + frequency
            mode = st.selectbox('Compare with', ['Previous period', 'Same period last year'], key=comparison_key)
            if mode == 'Same period last year':
                lag = {'Monthly': 12, 'Quarterly': 4, 'Yearly': 1}[frequency]
        result = calculate(planned['timeline'], planned['keys'], zero, lag)
        warnings = list(planned['warnings'])
        seasonal = axis.kind == 'Month' and objective != 'Chronological reporting periods'
        if assessment.years and not assessment.missing_year_rows:
            warnings = [warning for warning in warnings if not warning.startswith('No unambiguous year is available')]
        if seasonal:
            warnings.append('Month-of-year values are pooled intentionally. Changes compare seasonal period aggregates, not chronological reporting periods across years.')
            result.loc[result['Comparison'].eq('Consecutive periods'), 'Comparison'] = 'Adjacent pooled month-of-year periods'
        if zero:
            warnings.append(f'Inserted zeros into {gap_count:,} absent segment-periods for this analysis only. Percentage changes from zero are N/A.')
        if (result['Value'] < 0).any():
            warnings.append('Percentage change divides by the absolute previous value. With negative metrics, signs describe numerical movement, not a conventional growth rate.')
        if axis.kind not in ('Month', 'Quarter'):
            current = pd.Timestamp.now(tz='America/Toronto').tz_localize(None).to_period({'Daily': 'D', 'Weekly': 'W-SUN', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}[frequency])
            if result['_period'].max() >= current.ordinal:
                warnings.append('The data includes the current or a future period. The current period may be incomplete.')
        interpretation = interpret_result(result, frequency)
        if seasonal:
            interpretation = 'Seasonal month profile: these are changes between pooled month-of-year aggregates, not dated month-over-month changes.\n\n' + interpretation
        validation_state = 'Warning' if warnings else 'Supported'
        segments = result['Segment'].unique().tolist()
        chart_key = key_prefix + '_chart_Period-over-Period Comparison'
        if chart_key in st.session_state:
            st.session_state[chart_key] = [s for s in st.session_state[chart_key] if s in segments]
        else:
            st.session_state[chart_key] = segments[:10]
        chosen = st.multiselect('Segments to chart', segments, key=chart_key)
        chart = result[result['Segment'].isin(chosen)]
        chart_limit = 2000
        if len(chart) > chart_limit:
            # A contiguous final window preserves gaps; no resampling or bridging.
            bounds = chart['_period'].drop_duplicates().sort_values().tail(max(1, chart_limit // max(1, len(chosen))))
            chart = chart[chart['_period'].isin(bounds)]
            warnings.append('Charts show a final contiguous window of at most 2,000 segment-period points. The table, interpretation, CSV, and saved result retain all periods.')
        begin_result()
        presentation.subheader('Period-over-Period Comparison')
        presentation.caption(f'Analysis validation: {validation_state}. Temporal precision, aggregation, and observed coverage were checked.')
        for warning in warnings:
            presentation.warning(warning)
        presentation.caption(f'{frequency} · {statistic} · {mode} · {policy}. Baselines use calendar periods, never silently skip gaps. First-period and zero-baseline percentages are N/A.')
        if len(chart):
            actual, changes = comparison_charts(chart, measure or 'Row count', frequency)
            if seasonal:
                actual.update_layout(title='Seasonal period values (years pooled)')
                changes.update_layout(title='Percentage change between pooled seasonal periods')
            kind = {'Mean': 'mean', 'Median': 'median', 'Minimum': 'minimum', 'Maximum': 'maximum', 'Standard deviation': 'sd', 'Non-missing count': 'count', 'Row count': 'count'}.get(statistic)
            if kind:
                format_plot_hover(actual, {'y': kind})
            presentation.plotly_chart(actual, use_container_width=True, key=key_prefix + '_chart')
            format_plot_hover(changes, {'y': 'percentage'})
            presentation.plotly_chart(changes, use_container_width=True, key=key_prefix + '_change_chart')
        columns = ['Period', 'Value', 'Previous', 'Change', 'Change %', 'Segment', 'Previous period', 'Comparison', 'Missing period', 'Missing value', 'Inserted zero']
        display = result[columns]
        value_kind = {'Mean': 'mean', 'Median': 'median', 'Minimum': 'minimum', 'Maximum': 'maximum', 'Standard deviation': 'sd', 'Non-missing count': 'count', 'Row count': 'count'}.get(statistic)
        schema = {'Change %': 'percentage'}
        if value_kind:
            schema.update({'Value': value_kind, 'Previous': value_kind})
        presentation.dataframe(display.style.format({'Value': '{:,.2f}', 'Previous': '{:,.2f}', 'Change': '{:+,.2f}', 'Change %': '{:+.2f}%'}, na_rep='N/A'), result_types=schema, hide_index=True)
        st.download_button('Download statistics (CSV)', display.to_csv(index=False, na_rep='N/A').encode('utf-8-sig'),
                           file_name='comparison.csv', mime='text/csv', key=key_prefix + '_download')
        presentation.markdown('##### Interpretation')
        presentation.write(interpretation)
        presentation.caption('Descriptive comparisons do not establish causes or forecast future values. Chart segment selection does not change calculations or table results.')
        offer_result('Period-over-Period Comparison', 'Period-over-Period Comparison',
                     {'temporal_column': date, 'measure': measure, 'dimensions': dimensions},
                     {'temporal_subtype': axis.kind, 'frequency': frequency, 'statistic': statistic,
                      'comparison': mode, 'lag': lag, 'missing_period_handling': policy,
                      'month_reporting': clarification, 'chart_segments': chosen,
                      'validation_state': validation_state,
                      'objective': objective, 'year_sources': assessment.year_columns, 'observed_years': assessment.years,
                      'percentage_denominator': 'Absolute previous value', 'chart_point_limit': chart_limit},
                     {'Summary rows': len(result), 'Missing periods': gap_count, 'Inserted zeros': int(result['Inserted zero'].sum())},
                     diagnostics={'invalid_temporal_rows': planned['invalid'], 'blank_metric_periods': planned['blank_periods']},
                     warnings=warnings, interpretation=interpretation, key=key_prefix)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        presentation.info(f'Unable to compare these periods: {exc}')
