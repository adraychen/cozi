from utils.numerical_formatting import format_plot_hover
"""Plots consume the same group summaries and Games–Howell intervals as tables."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def group_mean_plot(prepared):
    summary = prepared['summary'].copy()
    summary['Group'] = summary['Group'].map(str)
    valid = summary[['Mean', '95% CI Lower', '95% CI Upper']].apply(np.isfinite).all(axis=1)
    summary = summary[valid]
    if summary.empty:
        return None
    fig = go.Figure(go.Scatter(
        x=summary.Group.tolist(), y=summary.Mean.tolist(), mode='markers',
        error_y=dict(type='data', symmetric=False,
                     array=(summary['95% CI Upper']-summary.Mean).tolist(),
                     arrayminus=(summary.Mean-summary['95% CI Lower']).tolist()),
        customdata=summary[['Count', '95% CI Lower', '95% CI Upper']].to_numpy(),
        hovertemplate='%{x}<br>Mean: %{y:.4g}<br>N: %{customdata[0]}<br>95% CI: [%{customdata[1]:.4g}, %{customdata[2]:.4g}]<extra></extra>',
        name='Group mean',
    ))
    fig.update_layout(xaxis_title=prepared['category'], yaxis_title=f'{prepared["outcome"]} — mean')
    fig.update_xaxes(type='category')
    return format_plot_hover(fig, {'y': 'mean', 'customdata[0]': 'count', 'customdata[1]': 'ci', 'customdata[2]': 'ci'})


def pairwise_plot(pairwise, table=None):
    """View filtering does not change the comparison family or calculations."""
    table = pairwise['table'] if table is None else table
    if table.empty:
        return None
    fig = go.Figure()
    for significant, name, color, symbol in [(True, 'Significant', '#2878b5', 'diamond'),
                                             (False, 'Not significant', '#777777', 'circle')]:
        rows = table[table.Significant.eq(significant)]
        if rows.empty:
            continue
        difference = rows['Mean Difference']
        low, high = rows[pairwise['lower']], rows[pairwise['upper']]
        fig.add_trace(go.Scatter(
            x=difference.tolist(), y=rows.Comparison.tolist(), mode='markers', name=name,
            marker=dict(color=color, symbol=symbol, size=9),
            error_x=dict(type='data', symmetric=False, array=(high-difference).tolist(),
                         arrayminus=(difference-low).tolist(), color=color),
            customdata=rows[[pairwise['lower'], pairwise['upper'], 'Adjusted p-value']].to_numpy(),
            hovertemplate=('%{y}<br>Mean A − mean B: %{x:.4g}<br>CI: [%{customdata[0]:.4g}, '
                           '%{customdata[1]:.4g}]<br>Adjusted p: %{customdata[2]:.4g}<extra>%{fullData.name}</extra>'),
        ))
    fig.add_vline(x=0, line_dash='dash', line_color='#555555')
    fig.update_layout(xaxis_title='Mean(Group A) − Mean(Group B)', yaxis_title='Comparison',
                      height=max(360, 150+30*len(table)),
                      legend=dict(orientation='h', y=1.12))
    fig.update_yaxes(type='category', categoryorder='array', categoryarray=table.Comparison.tolist(), autorange='reversed')
    return format_plot_hover(fig, {'customdata[0]': 'ci', 'customdata[1]': 'ci', 'customdata[2]': 'p'}, alpha=pairwise['alpha'])


def residual_plot(prepared):
    clean = prepared['clean'].copy()
    outcome, category = prepared['outcome'], prepared['category']
    means = clean.groupby(category, observed=True)[outcome].transform('mean')
    # Explicit names avoid collisions with user-selected column names.
    plot = dict(Fitted=means, Residual=clean[outcome]-means, Group=clean[category].map(str))
    fig = px.scatter(plot, x='Fitted', y='Residual', color='Group',
                     labels={'Fitted': 'Fitted group mean', 'Residual': 'Within-group residual'})
    fig.add_hline(y=0, line_dash='dash')
    return fig
