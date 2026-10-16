"""Content-aware result sizing shared by the application's renderers.

Native content widths stop at the parent container, including on mobile. Tables
keep their native scrolling, sorting, formatting, and download behavior.
"""
import math
import base64
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _narrative(name, *args, **kwargs):
    from utils.report_capture import record_text
    if not kwargs.get('unsafe_allow_html'):
        for value in args:
            record_text(value, heading=name in ('header', 'subheader'))
    return getattr(st, name)(*args, **kwargs)


def markdown(*args, **kwargs):
    return _narrative('markdown', *args, **kwargs)


def write(*args, **kwargs):
    return _narrative('write', *args, **kwargs)


def caption(*args, **kwargs):
    return _narrative('caption', *args, **kwargs)


def subheader(*args, **kwargs):
    return _narrative('subheader', *args, **kwargs)


def info(*args, **kwargs):
    return _narrative('info', *args, **kwargs)


def warning(*args, **kwargs):
    return _narrative('warning', *args, **kwargs)


def success(*args, **kwargs):
    return _narrative('success', *args, **kwargs)


def error(*args, **kwargs):
    return _narrative('error', *args, **kwargs)


def metric(target, label, value, *args, **kwargs):
    from utils.report_capture import record_text
    record_text(f'{label}: {value}')
    return target.metric(label, value, *args, **kwargs)


def _table_options(data, options, static=False):
    """Let the browser fit displayed content and scroll at the container boundary."""
    options = dict(options)
    options.pop('use_container_width', None)
    options['width'] = 'content'
    if options.get('column_config'):
        options['column_config'] = {
            key: {**value, 'width': None} if isinstance(value, dict) else value
            for key, value in options['column_config'].items()
        }
    frame = data.data if isinstance(getattr(data, 'data', None), pd.DataFrame) else data
    if static and options.get('hide_index') is None and isinstance(getattr(frame, 'index', None), pd.RangeIndex):
        options['hide_index'] = True
    return options


def dataframe(data=None, **kwargs):
    from utils.report_capture import record_table
    from utils.numerical_formatting import format_results
    result_types = kwargs.pop('result_types', None)
    alpha = kwargs.pop('alpha', .05)
    if result_types:
        data = format_results(data, result_types, alpha=alpha)
    record_table(data)
    options = _table_options(data, kwargs)
    if getattr(data, '_cozy_result_types', None) and options.get('column_config'):
        options['column_config'] = {
            key: {**config, 'type_config': {**config['type_config'], 'format': None}}
            if key in data._cozy_result_types and isinstance(config, dict) and isinstance(config.get('type_config'), dict)
            else config for key, config in options['column_config'].items()
        }
    frame = data.data if isinstance(getattr(data, 'data', None), pd.DataFrame) else data
    if isinstance(options.get('height'), int) and hasattr(frame, '__len__'):
        options['height'] = min(options['height'], max(73, 38 + 35*len(frame)))
    return st.dataframe(data, **options)


def table(data=None, **kwargs):
    return dataframe(data, **_table_options(data, kwargs, static=True))


def _sizing_values(values):
    """Decode Plotly's JSON typed arrays for sizing without changing chart data."""
    if isinstance(values, Mapping):
        if 'bdata' not in values or 'dtype' not in values:
            return None
        array = np.frombuffer(base64.b64decode(values['bdata']), dtype=values['dtype'])
        if values.get('shape'):
            shape = tuple(int(n.strip()) for n in str(values['shape']).split(','))
            array = array.reshape(shape)
        return array
    return values


def _labels(values):
    if values is None:
        return []
    return list(dict.fromkeys(str(v) for v in values[:256]))


def chart_size(figure):
    """Size by displayed structure, never by the number of scatter observations."""
    width, height = 640, 360
    layout = figure.layout
    legend_names = set()
    for trace in figure.data:
        if trace.showlegend is not False and trace.name:
            legend_names.add(str(trace.name))
        horizontal = getattr(trace, 'orientation', None) == 'h'
        axis_name = (getattr(trace, 'yaxis', None) if horizontal else getattr(trace, 'xaxis', None)) or ('y' if horizontal else 'x')
        axis = getattr(layout, axis_name[0]+'axis'+axis_name[1:], None)
        values = _sizing_values(getattr(trace, 'y' if horizontal else 'x', None))
        labels = _labels(values)
        categorical = (axis is not None and axis.type == 'category') or trace.type == 'bar'
        if values is not None and len(values):
            categorical |= isinstance(values[0], str)
        if categorical and labels:
            count = len(labels)
            longest = max(map(len, labels))
            if horizontal:
                width = max(width, 540 + min(300, longest*7))
                height = max(height, 130 + 27*count)
            else:
                width = max(width, 220 + count*max(48, min(110, longest*7)))
                if longest > 15:
                    height = max(height, 420)
        if trace.type in ('heatmap', 'surface'):
            z = _sizing_values(getattr(trace, 'z', None))
            shape = np.shape(z)
            if len(shape) == 2:
                width = max(width, 200 + 42*shape[1])
                height = max(height, 140 + 32*shape[0])
        if trace.type in ('pie', 'sunburst', 'treemap'):
            labels = _sizing_values(getattr(trace, 'labels', None))
            count = len(labels) if labels is not None else 0
            width = max(width, 560 + min(400, count*20))
            height = max(height, 420)
    width = max(width, 620 + 35*max(0, len(legend_names)-1))
    if legend_names and layout.showlegend is not False:
        width = max(width, 500 + min(400, 6*max(map(len, legend_names))))
        if layout.legend.orientation != 'h' and len(legend_names) > 10:
            height = max(height, 100 + 22*len(legend_names))
    # Separate subplot domains need space; overlaid axes do not count twice.
    axes = layout.to_plotly_json()
    xdomains = {tuple(v.get('domain', [0,1])) for k,v in axes.items() if k.startswith('xaxis') and isinstance(v, dict)}
    ydomains = {tuple(v.get('domain', [0,1])) for k,v in axes.items() if k.startswith('yaxis') and isinstance(v, dict)}
    width = max(width, 480*len(xdomains))
    height = max(height, 230*len(ydomains))
    if layout.height:
        height = max(height, min(900, layout.height))
    return min(1200, int(width)), min(900, int(height))


def _compact_category_marks(figure):
    """Keep sparse bars and boxes from expanding across the entire plot."""
    if any(trace.type == 'bar' for trace in figure.data):
        if figure.layout.bargap is None:
            figure.update_layout(bargap=.4, bargroupgap=.15)
    if any(trace.type == 'box' for trace in figure.data):
        if figure.layout.boxgap is None:
            figure.update_layout(boxgap=.55, boxgroupgap=.15)
    categories = {}
    for trace in figure.data:
        if trace.type not in ('bar', 'box'):
            continue
        horizontal = getattr(trace, 'orientation', None) == 'h'
        direction = 'y' if horizontal else 'x'
        reference = getattr(trace, direction+'axis', None) or direction
        axis_name = direction+'axis'+reference[1:]
        axis = getattr(figure.layout, axis_name)
        values = _sizing_values(getattr(trace, direction, None))
        if values is None and trace.type == 'box':
            values = [trace.name or '']
        if values is None or not len(values):
            continue
        if axis.type not in (None, 'category'):
            continue
        if axis.type != 'category' and not isinstance(values[0], str):
            continue
        categories.setdefault(axis_name, set()).update(_labels(values))
    for axis_name, labels in categories.items():
        axis = getattr(figure.layout, axis_name)
        count = len(labels)
        if count < 5 and axis.range is None:
            # Center the real categories in at least five category slots.
            center = (count-1)/2
            axis.update(type='category', range=[center-2.5, center+2.5])


def plotly_chart(figure_or_data, **kwargs):
    figure = go.Figure(figure_or_data)
    _compact_category_marks(figure)
    width, height = chart_size(figure)
    panel_rows = kwargs.pop('panel_rows', None)
    if panel_rows is not None:
        height = max(height, 420*panel_rows)
    kwargs.pop('use_container_width', None)
    if kwargs.get('width') in (None, 'stretch', 'content'):
        kwargs['width'] = width
    # Apply height to the figure so Plotly lays out axes rather than cropping it.
    figure.update_layout(height=height, autosize=True, width=None)
    from utils.report_capture import record_chart
    record_chart(figure)
    return st.plotly_chart(figure, **kwargs)


def pyplot(fig=None, **kwargs):
    from utils.report_capture import record_pyplot
    record_pyplot(fig)
    kwargs.pop('use_container_width', None)
    if kwargs.get('width') in (None, 'stretch', 'content'):
        native = fig.get_size_inches()[0]*100 if fig is not None else 720
        kwargs['width'] = int(min(1200, max(560, native)))
    return st.pyplot(fig, **kwargs)
