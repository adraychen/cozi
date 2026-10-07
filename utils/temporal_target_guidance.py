"""Metadata-only AI explanation above deterministic temporal-target rules."""
import hashlib
import json
import re
import pandas as pd
import streamlit as st
from utils.temporal import parse_temporal


ANALYSES = {
    'event': ('Time-to-Event Analysis', 'Reliability Analysis'),
    'survival': ('Survival Analysis', 'Reliability Analysis'),
    'forecast': ('Time Series Forecasting', 'Time Series'),
    'target': ('Choose a numerical or categorical target', 'Predictive Analytics'),
}
# These specialist analyses have no implemented page in the current application.
AVAILABLE_ANALYSES = {'target'}


def guidance_context(df, target, metadata, goal='Not sure', measurement=None):
    item = metadata[target]
    context = {'target': str(target), 'variable_type': item['variable_type'],
               'variable_subtype': item['variable_subtype'], 'goal': goal,
               'columns': [{'name': str(c), 'type': m['variable_type'],
                            'subtype': m['variable_subtype']} for c, m in metadata.items()],
               'reference_columns': [], 'event_status_columns': [],
               'candidates': ['target'], 'measurement': None}
    # Calendar differences are only meaningful for compatible calendar subtypes.
    if item['variable_subtype'] in ('Date', 'DateTime'):
        end = parse_temporal(df[target], item['variable_subtype'])
        for column, meta in metadata.items():
            if column == target or meta['variable_type'] != 'Date / Time' or meta['variable_subtype'] not in ('Date', 'DateTime'):
                continue
            start = parse_temporal(df[column], meta['variable_subtype'])
            try:
                pairs = start.notna() & end.notna()
                if pairs.any() and ((end[pairs] - start[pairs]).dt.total_seconds() >= 0).all():
                    context['reference_columns'].append(str(column))
            except (TypeError, ValueError):
                continue
        if context['reference_columns']:
            context['candidates'].insert(0, 'event')
    for column, meta in metadata.items():
        if (re.search(r'failed|failure|event|status|delivered', str(column), re.I)
                and meta['variable_type'] in ('Numerical', 'Categorical')
                and df[column].nunique(dropna=True) == 2):
            context['event_status_columns'].append(str(column))
    durations = [str(c) for c, m in metadata.items()
                 if re.search(r'hours|lifetime|duration', str(c), re.I)
                 and m['variable_type'] == 'Numerical'
                 and pd.to_numeric(df[c], errors='coerce').dropna().ge(0).all()
                 and df[c].notna().any()]
    context['duration_columns'] = durations
    if context['event_status_columns'] and (context['reference_columns'] or durations):
        context['candidates'].insert(0, 'survival')
    if (goal == 'Forecast a numerical value over time' and measurement in metadata
            and metadata[measurement]['variable_type'] == 'Numerical'
            and item['variable_subtype'] in ('Date', 'DateTime', 'Year-Month', 'Year')
            and df[[target, measurement]].dropna().shape[0] >= 3):
        context['measurement'] = str(measurement)
        context['candidates'].insert(0, 'forecast')
    return context


def fallback_guidance(context):
    choice = context['candidates'][0]
    if choice == 'survival':
        why = ('These columns may describe lifetime or failure events. ' +
               ', '.join(context['event_status_columns']) +
               ' may indicate whether the event occurred, so observations where it has not occurred yet can remain part of the analysis. Confirm what each status value means and the last observation date.')
    elif choice == 'event':
        reference = context['reference_columns'][0]
        why = (f'{reference} may provide a reference point for {context["target"]}. '
               f'An elapsed duration could be defined as {context["target"]} − {reference}. '
               'Confirm that these dates refer to the same event process. No duration column has been created.')
    elif choice == 'forecast':
        why = f'Your stated goal is to forecast {context["measurement"]} over future periods. Use the temporal column as the time axis, with a numerical measurement as the outcome.'
    else:
        why = 'This date/time target cannot be used directly with standard Regression or Classification. Choose a numerical or categorical target, or clarify your goal.'
    return {'analysis': choice, 'interpretation': 'A date/time target can describe when an event occurs rather than a normal numerical value or category.', 'why': why}


def validate_guidance(result, context):
    if (not isinstance(result, dict) or result.get('analysis') not in context['candidates']
            or any(not isinstance(result.get(key), str) or not result[key].strip()
                   or len(result[key]) > 1500 for key in ('interpretation', 'why'))):
        raise ValueError('Invalid temporal guidance response.')
    return {key: result[key] for key in ('analysis', 'interpretation', 'why')}


def choose_another_target(options):
    from utils.predictive_controls import choose_target
    if options:
        choose_target(options[0], list(st.session_state['variable_metadata']))


def render_temporal_guidance(df, target, metadata):
    st.markdown('### Temporal target detected')
    st.write(f'You selected: {target}')
    st.write('A date/time target usually means you want to understand or predict when an event occurs rather than a normal numerical value or category.')
    goal = st.selectbox('What do you want to understand?',
                        ['Not sure', 'Time until an event occurs', 'Lifetime or failure time',
                         'Forecast a numerical value over time'], key='pa_temporal_goal')
    measurement = None
    if goal == 'Forecast a numerical value over time':
        numerical = [c for c, m in metadata.items() if m['variable_type'] == 'Numerical']
        measurement = st.selectbox('Numerical value to forecast', numerical, index=None,
                                   key='pa_temporal_measurement')
    context = guidance_context(df, target, metadata, goal, measurement)
    # Context contains column metadata and validated candidates, never raw records.
    digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    cache = st.session_state.get('pa_temporal_guidance', {})
    if cache.get('key') != digest:
        result = fallback_guidance(context)
        failed = False
        try:
            from utils.ai_interpreter import recommend_temporal_target
            with st.spinner('Preparing guidance…'):
                result = validate_guidance(recommend_temporal_target(context), context)
        except Exception:
            failed = True
        cache = {'key': digest, 'result': result, 'failed': failed}
        st.session_state['pa_temporal_guidance'] = cache
    result = cache['result']
    if cache['failed']:
        st.caption('AI guidance is unavailable. Showing guidance based on the detected column types and your goal.')
    st.markdown('### What this may mean')
    st.text(result['interpretation'])
    st.markdown('### Recommended analysis')
    name, location = ANALYSES[result['analysis']]
    st.write(name)
    st.write(f'Location: {location}')
    st.markdown('### Why')
    st.text(result['why'])
    if result['analysis'] not in AVAILABLE_ANALYSES:
        st.caption(f'{name} is not available in this app yet. This is guidance for choosing an analysis, not a runnable workflow.')
    st.markdown('### Other possible goal')
    st.write('Select a numerical or categorical target for standard Predictive Analytics. If you want to predict time until an event occurs, Time-to-Event Analysis under Reliability Analysis may be appropriate. If you want to forecast a numerical measurement over future periods, use Time Series Forecasting. These specialist analyses are not available in this app yet.')
    options = [c for c, m in metadata.items() if m['variable_type'] in ('Numerical', 'Categorical')]
    st.button('Choose Another Target', key='pa_temporal_choose_target',
              on_click=choose_another_target, args=(options,), disabled=not options)
