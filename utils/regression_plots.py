"""Plots based on the same fitted model and transformation pipeline."""
import numpy as np
import pandas as pd
import plotly.express as px
from utils.regression_analysis import predict


def actual_fitted(result):
    fig = px.scatter(x=result['actual'], y=result['fitted'], labels={'x':'Actual response', 'y':'Fitted response'})
    lo = min(result['actual'].min(), result['fitted'].min()); hi = max(result['actual'].max(), result['fitted'].max())
    fig.add_scatter(x=[lo,hi], y=[lo,hi], mode='lines', name='Perfect agreement (45°)')
    return fig


def fitted_curve(result):
    spec = result['spec']; c = spec['predictors'][0]
    grid = pd.DataFrame({c:np.linspace(result['clean'][c].min(), result['clean'][c].max(), 150)})
    prediction = predict(result, grid)
    fig = px.scatter(result['clean'], x=c, y=spec['target'])
    if not result['perfect_fit']:
        for low, high, name, color in [('obs_ci_lower','obs_ci_upper','95% prediction band','rgba(255,165,0,0.18)'),
                                        ('mean_ci_lower','mean_ci_upper','95% confidence band','rgba(30,120,220,0.25)')]:
            fig.add_scatter(x=grid[c],y=prediction[low],mode='lines',line=dict(width=0),showlegend=False)
            fig.add_scatter(x=grid[c],y=prediction[high],mode='lines',line=dict(width=0),fill='tonexty',fillcolor=color,name=name)
    fig.add_scatter(x=grid[c], y=prediction['mean'], mode='lines', name='Fitted relationship')
    return fig


def coefficient_plot(result):
    table = result['coefficients'].iloc[1:].copy()
    table['upper error'] = table['95% upper']-table['Coefficient']
    table['lower error'] = table['Coefficient']-table['95% lower']
    fig = px.scatter(table,x='Coefficient',y='Term',error_x='upper error',error_x_minus='lower error')
    fig.add_vline(x=0,line_dash='dash')
    return fig


def partial_plot(result, c):
    blocks = [b for b in result['active'] if b[1]==c and b[0] in ('main','power')]
    positions = [i for b in blocks for i in result['indices'][b]]
    component = result['design'][:,positions] @ result['model'].params[positions]
    frame = pd.DataFrame({c:result['clean'][c].to_numpy(), 'Partial residual':result['residual']+component, 'Component':component})
    fig = px.scatter(frame,x=c,y='Partial residual')
    ordered = frame.sort_values(c)
    fig.add_scatter(x=ordered[c],y=ordered['Component'],mode='lines',name='Fitted component')
    return fig
