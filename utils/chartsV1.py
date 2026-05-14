import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHART_COLOR = "#4C72B0"
GRID_COLOR = "#f0f0f0"

# Columns with fewer than this many unique values are treated as discrete.
# Documented in README.md.
DISCRETE_THRESHOLD = 15


def make_histogram_boxplot(series: pd.Series) -> go.Figure:
    """
    Stacked histogram (top) + boxplot (bottom) for continuous numerical
    or date/time columns. Shared x-axis.
    """
    clean = series.dropna()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Histogram(
            x=clean,
            marker_color=CHART_COLOR,
            showlegend=False,
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(
            x=clean,
            marker_color=CHART_COLOR,
            showlegend=False,
            boxmean=True,
            hovertemplate="%{x}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return fig


def make_bar_chart(series: pd.Series, max_bars: int = 10) -> tuple:
    """
    Vertical bar chart of value counts, sorted descending.
    Returns (figure, was_truncated, total_unique).
    """
    counts = series.value_counts()
    total_unique = len(counts)
    truncated = total_unique > max_bars
    counts = counts.head(max_bars)

    fig = go.Figure(
        go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            marker_color=CHART_COLOR,
            showlegend=False,
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title="Count",
    )
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return fig, truncated, total_unique
