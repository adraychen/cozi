from utils.numerical_formatting import format_statistic as fmt, format_plot_hover, current_settings
import numpy as np
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
    counts = series.value_counts(sort=not (isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered))
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
    settings = current_settings()
    z      = corr_df.values
    labels = corr_df.columns.tolist()
    text   = [[fmt(v, 'correlation', settings) for v in row] for row in z]

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
            hovertemplate="%{y} × %{x}: %{text}<extra></extra>",
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


def make_boxplot_by_category(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    max_cats: int = 15,
    height: int = 420,
) -> tuple:
    """
    Boxplot of a numerical column grouped by a categorical column.
    One box per category, sorted by median descending.
    Returns (figure, was_truncated, total_cats).
    """
    total_cats = df[cat_col].nunique()
    top_cats   = df[cat_col].value_counts().head(max_cats).index.tolist()
    truncated  = total_cats > max_cats

    # Sort categories by median for easier reading
    medians = (
        df[df[cat_col].isin(top_cats)]
        .groupby(cat_col)[num_col]
        .median()
        .sort_values(ascending=False)
    )
    ordered_cats = medians.index.tolist()

    filtered = df[df[cat_col].isin(ordered_cats)].copy()
    filtered[cat_col] = filtered[cat_col].astype(str)

    fig = go.Figure(
        go.Box(
            x=filtered[cat_col],
            y=filtered[num_col],
            marker_color=CHART_COLOR,
            showlegend=False,
            boxmean=True,
            hovertemplate=f"{cat_col}: %{{x}}<br>{num_col}: %{{y}}<extra></extra>",
        )
    )

    layout = _base_layout(height)
    layout["xaxis_title"] = cat_col
    layout["yaxis_title"] = num_col
    layout["xaxis"]       = {"categoryorder": "array", "categoryarray": ordered_cats}
    fig.update_layout(**layout)
    return _style_axes(fig), truncated, total_cats


def make_feature_importance_chart(feat_imp: "pd.Series", height: int = 450) -> "go.Figure":
    """Horizontal bar chart of top 20 feature importances."""
    top    = feat_imp.head(20)
    labels = [n.split("__", 1)[-1] for n in top.index]   # strip num__ / cat__ prefix

    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=labels,
            orientation="h",
            marker_color=CHART_COLOR,
            showlegend=False,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    layout["xaxis_title"] = "Importance Score"
    layout["yaxis"]       = {"autorange": "reversed",
                              "tickfont": {"color": AXIS_COLOR, "size": 11}}
    fig.update_layout(**layout)
    return _style_axes(fig)


def make_learning_curve_chart(
    train_sizes:  "np.ndarray",
    train_scores: "np.ndarray",
    val_scores:   "np.ndarray",
    height: int = 380,
    score_label: str = "Score",
) -> "go.Figure":
    """
    Line chart showing training vs validation R² as training size grows.
    Shaded bands show ± 1 standard deviation across CV folds.
    """
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    fig = go.Figure()

    # Training band + line
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill="toself", fillcolor="rgba(76,114,176,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode="lines+markers", name="Training Score",
        line=dict(color=LINE_COLORS[0], width=2),
        marker=dict(size=6),
        hovertemplate="Size: %{x:,}<br>R²: %{y:.3f}<extra></extra>",
    ))

    # Validation band + line
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill="toself", fillcolor="rgba(221,132,82,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode="lines+markers", name="Validation Score",
        line=dict(color=LINE_COLORS[1], width=2),
        marker=dict(size=6),
        hovertemplate="Size: %{x:,}<br>R²: %{y:.3f}<extra></extra>",
    ))

    layout = _base_layout(height)
    layout["xaxis_title"] = "Training Set Size"
    layout["yaxis_title"] = score_label
    layout["legend"]      = dict(
        font=dict(color=FONT_COLOR, size=11),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(**layout)
    if score_label == 'R² Score':
        format_plot_hover(fig, {'x':'count', 'y':'r2'})
    return _style_axes(fig)


def make_pivot_heatmap(
    df: pd.DataFrame,
    cat_a: str,
    cat_b: str,
    num_col: str,
    max_cats: int = 10,
    height: int = 460,
) -> go.Figure:
    """
    Heatmap showing the mean of a numerical column for each combination
    of two categorical columns. Color gradient = numerical value.
    """
    top_a = df[cat_a].value_counts().head(max_cats).index.tolist()
    top_b = df[cat_b].value_counts().head(max_cats).index.tolist()

    filtered = df[df[cat_a].isin(top_a) & df[cat_b].isin(top_b)]
    pivot = (
        filtered.groupby([cat_a, cat_b])[num_col]
        .mean()
        .unstack(cat_b, fill_value=np.nan)
        .reindex(index=top_a, columns=top_b)
    )

    settings = current_settings()
    z    = pivot.values
    text = [[fmt(v, 'mean', settings) if not np.isnan(v) else '—' for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#222222"),
            colorscale="RdYlBu_r",
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=AXIS_COLOR, size=11),
                title=dict(text=f"Avg {num_col}", font=dict(color=AXIS_COLOR, size=12)),
            ),
            hovertemplate=f"{cat_a}: %{{y}}<br>{cat_b}: %{{x}}<br>Avg {num_col}: %{{text}}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    layout["xaxis_title"] = cat_b
    layout["yaxis_title"] = cat_a
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-30, showgrid=False, tickfont=dict(color=AXIS_COLOR, size=11), linecolor="#cccccc")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=AXIS_COLOR, size=11), linecolor="#cccccc")
    return fig


def make_confusion_matrix(
    cm: "np.ndarray",
    labels: list,
    height: int = 360,
) -> go.Figure:
    """Plotly heatmap confusion matrix with annotated counts."""
    x_labels = [f"Predicted: {l}" for l in labels]
    y_labels = [f"Actual: {l}"    for l in labels]
    text     = [[str(v) for v in row] for row in cm]

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#222222"),
            colorscale="Blues",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(height))
    fig.update_xaxes(tickfont=dict(color=AXIS_COLOR, size=11), showgrid=False, linecolor="#cccccc")
    fig.update_yaxes(tickfont=dict(color=AXIS_COLOR, size=11), showgrid=False, linecolor="#cccccc")
    return fig


def make_roc_curve(
    fpr: "np.ndarray",
    tpr: "np.ndarray",
    auc_score: float,
    height: int = 360,
) -> go.Figure:
    """ROC curve with diagonal reference line and AUC in legend."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC  (AUC = {auc_score:.3f})",
        line=dict(color=CHART_COLOR, width=2),
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#aaaaaa", width=1, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))
    layout = _base_layout(height)
    layout["xaxis_title"] = "False Positive Rate"
    layout["yaxis_title"] = "True Positive Rate"
    layout["legend"]      = dict(font=dict(color=FONT_COLOR, size=11), bgcolor="rgba(0,0,0,0)")
    fig.update_layout(**layout)
    return _style_axes(fig)


def make_coefficient_chart(coef_series: "pd.Series", height: int = 450) -> "go.Figure":
    """
    Horizontal bar chart of Lasso coefficients (top 20 by absolute value).
    Positive coefficients in blue, negative in orange.
    """
    top    = coef_series.head(20)
    labels = [n.split("__", 1)[-1] for n in top.index]
    colors = [CHART_COLOR if v >= 0 else LINE_COLORS[1] for v in top.values]

    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=labels,
            orientation="h",
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    layout = _base_layout(height)
    layout["xaxis_title"] = "Coefficient"
    layout["yaxis"]       = {"autorange": "reversed",
                              "tickfont": {"color": AXIS_COLOR, "size": 11}}
    # Reference line at zero
    layout["shapes"] = [dict(
        type="line", x0=0, x1=0, y0=-0.5, y1=len(top) - 0.5,
        line=dict(color="#888888", width=1, dash="dot"),
    )]
    fig.update_layout(**layout)
    return _style_axes(format_plot_hover(fig, {'x':'coefficient'}))
