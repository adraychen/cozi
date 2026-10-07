"""Content-aware result sizing shared by the application's renderers.

Native content widths stop at the parent container, including on mobile. Tables
keep their native scrolling, sorting, formatting, and download behavior.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
    record_table(data)
    options = _table_options(data, kwargs)
    frame = data.data if isinstance(getattr(data, 'data', None), pd.DataFrame) else data
    if isinstance(options.get('height'), int) and hasattr(frame, '__len__'):
        options['height'] = min(options['height'], max(73, 38 + 35*len(frame)))
    return st.dataframe(data, **options)


def table(data=None, **kwargs):
    return dataframe(data, **_table_options(data, kwargs, static=True))


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
        values = getattr(trace, 'y' if horizontal else 'x', None)
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
            z = getattr(trace, 'z', None)
            shape = np.shape(z)
            if len(shape) == 2:
                width = max(width, 200 + 42*shape[1])
                height = max(height, 140 + 32*shape[0])
        if trace.type in ('pie', 'sunburst', 'treemap'):
            labels = getattr(trace, 'labels', None)
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


def plotly_chart(figure_or_data, **kwargs):
    figure = go.Figure(figure_or_data)
    width, height = chart_size(figure)
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
