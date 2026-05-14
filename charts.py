import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

CHART_COLOR = "#4C72B0"
GRID_COLOR  = "#e8e8e8"
FONT_COLOR  = "#333333"
AXIS_COLOR  = "#444444"

# Columns with fewer than this many unique values are treated as discrete.
# Documented in README.md.
DISCRETE_THRESHOLD = 15

# Colour palette for multi-series line charts
LINE_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _base_layout(height: int) -> dict:
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=FONT_COLOR, size=12),
    )


def _style_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(color=AXIS_COLOR, size=11),
        title_font=dict(color=AXIS_COLOR, size=12),
        linecolor="#cccccc",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(color=AXIS_COLOR, size=11),
        title_font=dict(color=AXIS_COLOR, size=12),
        linecolor="#cccccc",
    )
    return fig


def make_histogram_boxplot(series: pd.Series, height: int = 300) -> go.Figure:
    """Stacked histogram (top) + boxplot (bottom). Shared x-axis."""
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
    fig.update_layout(**_base_layout(height))
    return _style_axes(fig)


def make_bar_chart(series: pd.Series, max_bars: int = 10, height: int = 300) -> tuple:
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
    layout = _base_layout(height)
    layout["yaxis_title"] = "Count"
    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=False,
        tickangle=-30,
        tickfont=dict(color=AXIS_COLOR, size=11),
        linecolor="#cccccc",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(color=AXIS_COLOR, size=11),
        linecolor="#cccccc",
    )
    return fig, truncated, total_unique


def make_violin(series: pd.Series, height: int = 300) -> go.Figure:
    """Violin plot with embedded boxplot and mean line."""
    clean = series.dropna()
    fig = go.Figure(
        go.Violin(
            y=clean,
            box_visible=True,
            meanline_visible=True,
            fillcolor=CHART_COLOR,
            opacity=0.7,
            line_color=CHART_COLOR,
            showlegend=False,
            hovertemplate="%{y}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(height))
    return _style_axes(fig)


def make_time_series(series: pd.Series, freq: str = "M", height: int = 360) -> go.Figure:
    """Line chart of record counts aggregated over time."""
    # Normalise resample aliases to Period-compatible codes
    freq_map = {"ME": "M", "YE": "Y", "QE": "Q"}
    period_freq = freq_map.get(freq, freq)

    clean = pd.to_datetime(series.dropna(), errors="coerce").dropna()
    counts = clean.dt.to_period(period_freq).value_counts().sort_index()
    x = [str(p) for p in counts.index]

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=counts.values,
            mode="lines+markers",
            line=dict(color=CHART_COLOR, width=2),
            marker=dict(color=CHART_COLOR, size=5),
            showlegend=False,
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    layout["yaxis_title"] = "Count"
    fig.update_layout(**layout)
    return _style_axes(fig)


def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    height: int = 400,
) -> go.Figure:
    """
    Multi-series line chart.
    x_col: column name used as x-axis (datetime or numerical).
    y_cols: one or more column names plotted as separate lines.
    """
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        mask = df[x_col].notna() & df[col].notna()
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, x_col],
                y=df.loc[mask, col],
                mode="lines+markers",
                name=col,
                line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2),
                marker=dict(size=4),
                hovertemplate=f"{col}: %{{y}}<extra></extra>",
            )
        )
    layout = _base_layout(height)
    layout["xaxis_title"] = x_col
    layout["legend"] = dict(
        font=dict(color=FONT_COLOR, size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1,
    )
    fig.update_layout(**layout)
    return _style_axes(fig)


def make_heatmap(corr_df: pd.DataFrame, height: int = 460) -> go.Figure:
    """
    Correlation heatmap with values annotated in each cell.
    Uses a red–white–blue diverging colour scale (red = positive, blue = negative).
    """
    z      = corr_df.values.round(2)
    labels = corr_df.columns.tolist()
    text   = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#222222"),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=AXIS_COLOR, size=11),
                title=dict(text="r", font=dict(color=AXIS_COLOR, size=12)),
            ),
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    fig.update_layout(**layout)
    fig.update_xaxes(
        tickfont=dict(color=AXIS_COLOR, size=11),
        tickangle=-30,
        linecolor="#cccccc",
        showgrid=False,
    )
    fig.update_yaxes(
        tickfont=dict(color=AXIS_COLOR, size=11),
        linecolor="#cccccc",
        showgrid=False,
    )
    return fig


def make_scatter(
    x: pd.Series,
    y: pd.Series,
    x_name: str,
    y_name: str,
    height: int = 400,
) -> go.Figure:
    """Scatter / dot plot for two numerical columns."""
    mask = x.notna() & y.notna()
    fig = go.Figure(
        go.Scatter(
            x=x[mask],
            y=y[mask],
            mode="markers",
            marker=dict(color=CHART_COLOR, size=6, opacity=0.55),
            showlegend=False,
            hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    layout["xaxis_title"] = x_name
    layout["yaxis_title"] = y_name
    fig.update_layout(**layout)
    return _style_axes(fig)


def make_density_contour(
    x: pd.Series,
    y: pd.Series,
    x_name: str,
    y_name: str,
    height: int = 420,
) -> go.Figure:
    """
    2D density contour plot with faint scatter points underneath.
    Shows where data clusters rather than individual points.
    """
    mask = x.notna() & y.notna()
    fig = go.Figure()

    # Faint scatter underneath for context
    fig.add_trace(
        go.Scatter(
            x=x[mask], y=y[mask],
            mode="markers",
            marker=dict(color=CHART_COLOR, size=4, opacity=0.22),
            showlegend=False,
            hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<extra></extra>",
        )
    )
    # Density contour on top
    fig.add_trace(
        go.Histogram2dContour(
            x=x[mask], y=y[mask],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=AXIS_COLOR, size=11),
                title=dict(text="Density", font=dict(color=AXIS_COLOR, size=12)),
            ),
            hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<br>Density: %{{z}}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    layout["xaxis_title"] = x_name
    layout["yaxis_title"] = y_name
    fig.update_layout(**layout)
    return _style_axes(fig)


def make_occurrence_heatmap(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    max_cats: int = 10,
    height: int = 460,
) -> go.Figure:
    """
    Pivot-table heatmap showing how often each combination of
    two categorical columns occurs. Blues colour scale; count in cells.
    """
    top_a = df[col_a].value_counts().head(max_cats).index.tolist()
    top_b = df[col_b].value_counts().head(max_cats).index.tolist()

    filtered = df[df[col_a].isin(top_a) & df[col_b].isin(top_b)]
    pivot = (
        filtered.groupby([col_a, col_b])
        .size()
        .unstack(col_b, fill_value=0)
        .reindex(index=top_a, columns=top_b, fill_value=0)
    )

    z    = pivot.values
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#222222"),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=AXIS_COLOR, size=11),
                title=dict(text="Count", font=dict(color=AXIS_COLOR, size=12)),
            ),
            hovertemplate=f"{col_a}: %{{y}}<br>{col_b}: %{{x}}<br>Count: %{{z}}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    layout["xaxis_title"] = col_b
    layout["yaxis_title"] = col_a
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-30, showgrid=False, tickfont=dict(color=AXIS_COLOR, size=11), linecolor="#cccccc")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=AXIS_COLOR, size=11), linecolor="#cccccc")
    return fig


def make_grouped_bar(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    max_cats: int = 10,
    height: int = 420,
) -> go.Figure:
    """
    Grouped bar chart. For each value of col_a (x-axis),
    draws side-by-side bars coloured by col_b values.
    """
    top_a = df[col_a].value_counts().head(max_cats).index.tolist()
    top_b = df[col_b].value_counts().head(max_cats).index.tolist()

    filtered = df[df[col_a].isin(top_a) & df[col_b].isin(top_b)]

    fig = go.Figure()
    for i, val_b in enumerate(top_b):
        subset = filtered[filtered[col_b] == val_b]
        counts = subset[col_a].value_counts().reindex(top_a, fill_value=0)
        fig.add_trace(
            go.Bar(
                name=str(val_b),
                x=[str(v) for v in top_a],
                y=counts.values,
                marker_color=LINE_COLORS[i % len(LINE_COLORS)],
                hovertemplate=f"{col_b}={val_b} | {col_a}: %{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )

    layout = _base_layout(height)
    layout["barmode"]     = "group"
    layout["xaxis_title"] = col_a
    layout["yaxis_title"] = "Count"
    layout["legend"]      = dict(
        title=dict(text=col_b, font=dict(color=FONT_COLOR, size=12)),
        font=dict(color=FONT_COLOR, size=11),
        bgcolor="rgba(255,255,255,0.0)",
        bordercolor="#cccccc",
        borderwidth=1,
    )
    fig.update_layout(**layout)
    return _style_axes(fig)
