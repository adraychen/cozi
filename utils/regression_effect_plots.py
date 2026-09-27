"""Adjusted interaction and polynomial displays from the retained OLS model."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils import presentation
from utils.regression_analysis import predict
from utils.regression_design import label
from utils.regression_plots import fitted_curve


def retained_interactions(result):
    return [b for b in result['active'] if b[0] == 'interaction']


def retained_polynomials(result):
    return list(dict.fromkeys(b[1] for b in result['active'] if b[0] == 'power'))


def adjusted_curves(result, xvar, moderator=None, selected_levels=None, prediction_function=predict):
    """Generate original-unit prediction grids without refitting or re-centering."""
    spec, clean = result['spec'], result['clean']
    if xvar not in spec['predictors'] or moderator == xvar or (moderator is not None and moderator not in spec['predictors']):
        raise ValueError('Choose distinct model predictors for the adjusted plot.')
    defaults = {c:spec['levels'][c][0] if c in spec['categorical'] else float(clean[c].mean())
                for c in spec['predictors']}
    settings = [{'Predictor':str(c), 'Held at':repr(value) if c in spec['categorical'] else f'{value:.6g}',
                 'Setting':'Reference category' if c in spec['categorical'] else 'Complete-case mean'}
                for c,value in defaults.items() if c not in (xvar, moderator)]
    if moderator is None:
        slices = [('Adjusted mean response', None)]
    elif moderator in spec['categorical']:
        levels = spec['levels'][moderator] if selected_levels is None else selected_levels
        if not levels or any(v not in spec['levels'][moderator] for v in levels):
            raise ValueError('Select at least one known moderator category.')
        slices = [(f'{moderator} = {value!r}', value) for value in levels]
    else:
        slices = [(f'{moderator}: P{q} = {value:.4g}', float(value))
                  for q,value in zip((25,50,75), np.percentile(clean[moderator], [25,50,75]))]
    curves = []
    for name, value in slices:
        if xvar in spec['categorical']:
            xs = spec['levels'][xvar]
        else:
            # Stay inside the observed X range for this category, where applicable.
            subset = clean.loc[clean[moderator] == value, xvar] if moderator in spec['categorical'] else clean[xvar]
            xs = np.linspace(float(subset.min()), float(subset.max()), 120)
        grid = pd.DataFrame({c:[v]*len(xs) for c,v in defaults.items()})
        grid[xvar] = xs
        if moderator is not None:
            grid[moderator] = value
        curves.append({'label':name, 'input':grid, 'prediction':prediction_function(result, grid)})
    return curves, pd.DataFrame(settings, columns=['Predictor','Held at','Setting'])


def effect_figure(result, xvar, curves, title, confidence=.95):
    categorical_x = xvar in result['spec']['categorical']
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i,curve in enumerate(curves):
        color = colors[i % len(colors)]
        raw_x = curve['input'][xvar]
        x = [repr(v) for v in raw_x] if categorical_x else raw_x
        values = curve['prediction']
        group = f'slice_{i}'
        error = None
        if not result['perfect_fit']:
            if categorical_x:
                error = dict(type='data', symmetric=False,
                             array=values['mean_ci_upper']-values['mean'],
                             arrayminus=values['mean']-values['mean_ci_lower'])
            else:
                rgb = tuple(int(color[j:j+2],16) for j in (1,3,5))
                fig.add_trace(go.Scatter(x=x,y=values['mean_ci_lower'],mode='lines',line=dict(width=0),
                                        legendgroup=group,showlegend=False,hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=x,y=values['mean_ci_upper'],mode='lines',line=dict(width=0),
                                        fill='tonexty',fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)',
                                        legendgroup=group,showlegend=False,hoverinfo='skip'))
        custom = np.column_stack([values['mean_ci_lower'], values['mean_ci_upper']])
        fig.add_trace(go.Scatter(x=x,y=values['mean'],mode='lines+markers' if categorical_x else 'lines',
                                name=curve['label'],legendgroup=group,line=dict(color=color),error_y=error,
                                customdata=custom,hovertemplate='%{x}<br>Adjusted mean: %{y:.4g}<br>CI: %{customdata[0]:.4g} to %{customdata[1]:.4g}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=title,xaxis_title=str(xvar),yaxis_title=f"Adjusted predicted {result['spec']['target']}",
                      legend_title_text='Model prediction at',legend=dict(groupclick='togglegroup'))
    if categorical_x:
        fig.update_xaxes(type='category',categoryorder='array',categoryarray=[repr(v) for v in result['spec']['levels'][xvar]])
    return fig


def show_effect(result, xvar, moderator, title, widget_key):
    levels = None
    if moderator in result['spec']['categorical'] and len(result['spec']['levels'][moderator]) > 8:
        all_levels = result['spec']['levels'][moderator]
        levels = st.multiselect(f'Categories of {moderator} to display',all_levels,default=all_levels[:8],key=widget_key+'_levels')
        st.caption('Showing the selected categories; choose additional levels above to include them.')
        if not levels:
            st.info('Select at least one category for this plot.')
            return
    curves, settings = adjusted_curves(result,xvar,moderator,levels)
    presentation.plotly_chart(effect_figure(result,xvar,curves,title),use_container_width=True,key=widget_key+'_chart')
    st.caption('Adjusted model predictions in original units, evaluated using the final retained model. Shading shows pointwise 95% confidence bands for the mean response; categorical-axis plots use 95% confidence bars. These are not prediction intervals or formal tests of interaction.')
    if result['perfect_fit']:
        st.info('Confidence intervals are unavailable for a perfect or near-perfect fit.')
    if settings.empty:
        st.caption('No other predictors are held fixed.')
    else:
        st.caption('Other predictors are held at the following values:')
        presentation.dataframe(settings,hide_index=True,use_container_width=True)
    if moderator is not None and moderator not in result['spec']['categorical']:
        st.caption('Numeric moderator lines use the 25th, 50th and 75th percentiles of the complete cases; labels report original-unit values. Tied percentiles may produce overlapping lines.')
    st.caption('Numeric axes stay within observed ranges (within each moderator category when applicable). Multivariable combinations can still be sparsely supported. Lines connecting categories are visual guides, not continuous trends. Intervals remain conditional on the fitted model, including any variable selection.')


def render_effect_plots(result, key):
    interactions = retained_interactions(result)
    polynomials = retained_polynomials(result)
    if interactions:
        st.markdown('**Interaction plots**')
        interaction = st.selectbox('Retained interaction to plot',interactions,format_func=label,key='reg_effect_interaction_'+key[:12])
        a,b = interaction[1:]
        numeric = [c for c in (a,b) if c not in result['spec']['categorical']]
        if len(numeric) == 1:
            xvar = numeric[0]
        else:
            xvar = st.selectbox('Interaction horizontal axis',[a,b],key='reg_effect_axis_'+key[:12]+'_'+str(interaction))
        moderator = b if xvar==a else a
        show_effect(result,xvar,moderator,'Interaction: '+label(interaction),'reg_int_'+key[:12]+'_'+str(interaction))
    if polynomials:
        st.markdown('**Polynomial plots**')
        xvar = st.selectbox('Predictor with retained polynomial terms',polynomials,key='reg_effect_poly_'+key[:12])
        moderators = list(dict.fromkeys(b[2] if b[1]==xvar else b[1] for b in interactions if xvar in b[1:]))
        moderator = None
        if moderators:
            moderator = st.selectbox('Interacting predictor for separate curves',moderators,key='reg_effect_poly_mod_'+key[:12]+'_'+str(xvar))
            st.caption('This predictor participates in retained interactions, so curves are shown by the selected interacting predictor. Other interacting predictors are held at the settings below.')
        degrees = [b[2] for b in result['active'] if b[0]=='power' and b[1]==xvar]
        st.caption('Retained powers: '+', '.join(str(xvar)+{2:'²',3:'³'}[degree] for degree in degrees)+'. The curve uses all retained terms together.')
        if len(result['spec']['predictors']) == 1:
            fig = fitted_curve(result)
            fig.update_layout(title=f'Polynomial relationship: {xvar}')
            presentation.plotly_chart(fig,use_container_width=True,key='reg_poly_single_'+key[:12])
            st.caption('Observed points and fitted polynomial in original units. Blue shading shows the 95% mean-response confidence interval; orange shading shows the 95% prediction interval. No other predictors are held fixed.')
            if result['perfect_fit']:
                st.info('Intervals are unavailable for a perfect or near-perfect fit.')
        else:
            show_effect(result,xvar,moderator,f'Polynomial relationship: {xvar}','reg_poly_'+key[:12]+'_'+str(xvar)+'_'+str(moderator))
