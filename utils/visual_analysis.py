"""Column-driven, standalone visual exploration with explicit chart roles."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st
from utils import presentation
from utils.variable_selection import variable_checkboxes


def available_charts(numeric, categorical, dates, ordinal=()):
    if len(numeric) == 1 and len(categorical) == 1 and not dates:
        charts = ['Box plot', 'Histogram', 'Dot plot', 'Bar chart', 'Pie chart', 'Pareto chart']
        if categorical[0] in ordinal:
            charts.append('Line chart')
        return charts
    if not dates:
        combination = (len(numeric), len(categorical))
        if combination == (2, 1):
            return ['Histogram', 'Dot plot', 'Box plot', 'Scatter plot']
        if combination == (1, 2):
            return ['Grouped bar chart', 'Stacked-bar chart', 'Grouped box plot', 'Heatmap']
        if combination == (2, 0):
            return ['Histogram', 'Dot plot', 'Box plot', 'Scatter plot', 'Correlation matrix', 'Pair plot', 'Heatmap', 'Line chart', 'Area chart']
        if combination == (0, 2):
            return ['Grouped bar chart', 'Stacked-bar chart', 'Heatmap']
    charts = []
    if numeric:
        charts += ['Histogram', 'Dot plot', 'Density plot', 'Box plot', 'Q–Q plot']
    if categorical:
        charts += ['Bar chart', 'Pie chart', 'Pareto chart']
    if len(categorical) >= 2:
        charts += ['Stacked-bar chart']
    if len(numeric) == 1 and len(dates) == 1 and not categorical:
        charts.append('Line chart')
    if (dates and numeric) or len(numeric) >= 2:
        charts.append('Area chart')
    if len(numeric) >= 2:
        charts += ['Scatter plot', 'Correlation matrix', 'Pair plot']
    if len(categorical) >= 2 or len(numeric) >= 2:
        charts += ['Heatmap']
    if len(numeric) >= 3:
        charts += ['Bubble plot']
    return charts


def histogram_bins(values, bins):
    edges = np.histogram_bin_edges(np.asarray(values), bins=bins)
    # Plotly's rightmost bin edge is exclusive. Pad the range so extrema count.
    padding = (float(edges[-1])-float(edges[0]))*1e-6
    start, end = float(edges[0])-padding, float(edges[-1])+padding
    return {'start':start, 'end':end, 'size':(end-start)/bins}


def panel_bin_slider(figure, values, trace_index, panel_index, bins=30):
    suffix = str(panel_index+1) if panel_index else ''
    xdomain = figure.layout['xaxis'+suffix].domain
    ydomain = figure.layout['yaxis'+suffix].domain
    return dict(active=bins-5, x=xdomain[0]+.15*(xdomain[1]-xdomain[0]),
        y=ydomain[1]+.015, xanchor='left', yanchor='bottom',
        len=.7*(xdomain[1]-xdomain[0]), pad={'t':0,'b':0},
        currentvalue={'prefix':'Bins: ', 'font':{'size':12}}, font={'size':10},
        ticklen=0, minorticklen=0, transition={'duration':0},
        steps=[dict(label=str(n), method='restyle',
                    args=[{'nbinsx':n, 'xbins':histogram_bins(values, n)}, [trace_index]])
               for n in range(5,101)])


def dot_points(frame, value, category=None):
    columns = [value]+([category] if category is not None else [])
    clean = frame[columns].dropna()
    sampled = len(clean) > 2000
    if sampled:
        clean = clean.sample(2000, random_state=42)
    names = {value: 'Value'}
    if category is not None:
        names[category] = 'Category'
    dots = clean.rename(columns=names).copy()
    # Stack across categories so repeated values remain individually visible.
    dots['Stack'] = dots.groupby('Value').cumcount()+1
    return dots, sampled


def choose(label, options, key):
    roles = {'univariate','box_group','box_subgroup','bar_category','stack','stack_measure',
             'bar_measure','line_x','scatter_x','scatter_y','bubble_size','scatter_color',
             'heat_row','heat_col','heat_measure'}
    if key in roles:
        required = None not in options
        columns = [c for c in options if c is not None]
        selected = variable_checkboxes(label, columns, 'va_'+key, single=True,
                                       default=columns[:1] if required else [])
        if not selected and required:
            raise ValueError('Select a variable for ' + label + '.')
        return selected[0] if selected else None
    if st.session_state.get('va_' + key) not in options:
        st.session_state.pop('va_' + key, None)
    return st.selectbox(label, options, key='va_' + key)


def render_visual_analysis(df, col_types, metadata=None):
    st.markdown('#### Visual Analysis')
    st.caption('Explore patterns and trends in your data through charts.')
    selected = {}
    for label, kind in [('Numerical', 'numerical'), ('Categorical', 'categorical'), ('Temporal', 'datetime')]:
        selected[kind] = variable_checkboxes(label, col_types[kind], 'va_select_' + kind, hide_empty=True)
    nums, cats, dates = selected['numerical'], selected['categorical'], selected['datetime']
    if metadata is None:
        from utils.variable_metadata import infer_metadata
        metadata = infer_metadata(df)
    ordinal = [c for c in cats if metadata.get(c, {}).get('variable_subtype') == 'Ordinal'
               and len(metadata.get(c, {}).get('order', [])) > 1]
    choices = available_charts(nums, cats, dates, ordinal)
    if not choices:
        st.info('Select a numeric or categorical column to begin. Time-series charts also need a numeric measure.')
        return
    if st.session_state.get('va_selected_graph') not in choices:
        st.session_state['va_selected_graph'] = choices[0]
    st.markdown('**Choose a graph**')
    def select_graph(name):
        st.session_state['va_selected_graph'] = name
    for start in range(0, len(choices), 5):
        buttons = st.columns(5)
        for button, name in zip(buttons, choices[start:start + 5]):
            button.button(name, key='va_graph_button_' + name,
                          type='primary' if st.session_state['va_selected_graph'] == name else 'secondary',
                          use_container_width=True, on_click=select_graph, args=(name,))
    chart = st.session_state['va_selected_graph']
    # Rename selected columns internally so source names cannot collide with derived chart fields.
    columns = list(dict.fromkeys(nums + cats + dates))
    key = {c: f'_c{i}' for i,c in enumerate(columns)}
    data = df[columns].copy().rename(columns=key)
    labels = {v:str(k) for k,v in key.items()}
    for c in nums:
        data[key[c]] = pd.to_numeric(data[key[c]], errors='coerce').replace([np.inf,-np.inf],np.nan)
    for c in dates:
        data[key[c]] = pd.to_datetime(data[key[c]],errors='coerce')
    for c in cats:
        data[key[c]] = data[key[c]].astype('string')
        if isinstance(df[c].dtype,pd.CategoricalDtype) and df[c].cat.ordered:
            data[key[c]]=pd.Categorical(data[key[c]],categories=[str(v) for v in df[c].cat.categories],ordered=True)
    try:
        fig = None
        panel_rows = None
        if len(nums) > 1 and chart in ['Histogram', 'Dot plot', 'Box plot']:
            group = choose('Group by', cats if cats else [None], 'box_group') if chart == 'Box plot' else None
            panel_rows = (len(nums)+1)//2
            top_margin = 145 if chart == 'Histogram' else 70
            panel_height = 420*panel_rows
            row_spacing = 170/(panel_height-top_margin-80) if panel_rows > 1 else 0
            fig = make_subplots(rows=panel_rows, cols=2, subplot_titles=[str(c) for c in nums],
                                horizontal_spacing=.12, vertical_spacing=row_spacing)
            panel_color = px.colors.qualitative.Plotly[0]
            bin_sliders = []
            for index, col in enumerate(nums):
                bins = 30
                row, column = index//2+1, index%2+1
                values = data[key[col]].dropna()
                if chart == 'Box plot' and group is not None:
                    box_data = data.dropna(subset=[key[col],key[group]])
                    values = box_data[key[col]]
                if values.empty:
                    st.info(str(col)+': No finite numerical values to plot for this variable.')
                    fig.add_annotation(text='No finite numerical values', x=.5, y=.5,
                                       xref='x'+(str(index+1) if index else '')+' domain',
                                       yref='y'+(str(index+1) if index else '')+' domain', showarrow=False)
                elif chart == 'Histogram':
                    trace_index = len(fig.data)
                    fig.add_trace(go.Histogram(x=values, nbinsx=bins,
                        xbins=histogram_bins(values.to_numpy(), bins),
                        name=str(col), marker_color=panel_color, showlegend=False), row=row, col=column)
                    bin_sliders.append(panel_bin_slider(fig, values.to_numpy(), trace_index, index, bins))
                elif chart == 'Box plot':
                    fig.add_trace(go.Box(y=values, x=box_data[key[group]] if group is not None else None,
                        name=str(col), marker_color=panel_color, line_color=panel_color,
                        showlegend=False), row=row, col=column)
                else:
                    dots, sampled = dot_points(data, key[col])
                    if sampled:
                        st.caption(str(col)+': Dot plot uses a reproducible sample of 2,000 observations.')
                    fig.add_trace(go.Scatter(x=dots['Value'], y=dots['Stack'], mode='markers',
                                            name=str(col), marker_color=panel_color, showlegend=False), row=row, col=column)
                x_title = (str(group) if group is not None else None) if chart == 'Box plot' else str(col)
                y_title = str(col) if chart == 'Box plot' else 'Count' if chart == 'Histogram' else 'Stacked observations'
                fig.update_xaxes(title_text=x_title, title_standoff=15, automargin=True,
                                 visible=chart != 'Box plot' or group is not None, row=row, col=column)
                fig.update_yaxes(title_text=y_title, row=row, col=column)
            fig.update_layout(bargap=.05, margin={'t':top_margin, 'b':80, 'l':70, 'r':40}, sliders=bin_sliders)
            if chart == 'Histogram':
                for annotation in fig.layout.annotations[:len(nums)]:
                    annotation.update(yshift=95)
            if len(nums) % 2:
                fig.update_xaxes(visible=False, row=panel_rows, col=2)
                fig.update_yaxes(visible=False, row=panel_rows, col=2)
        elif chart in ['Histogram','Dot plot','Density plot','Box plot','Grouped box plot','Q–Q plot']:
            col = choose('Numerical variable', nums, 'univariate')
            x = data[key[col]].dropna()
            if x.empty:
                raise ValueError('No finite numerical values to plot.')
            if chart == 'Histogram':
                bins = st.slider('Number of bins',5,100,30,key='va_bins',width=250)
                if len(cats) == 1:
                    group = cats[0]
                    clean = data.dropna(subset=[key[col],key[group]])
                    if clean.empty:
                        raise ValueError('No complete numerical values and categories to plot.')
                    fig = px.histogram(clean,x=key[col],color=key[group],nbins=bins,
                                       barmode='overlay',opacity=.6,labels=labels)
                    fig.update_traces(bingroup='category_comparison',
                        xbins=histogram_bins(clean[key[col]].to_numpy(), bins))
                    st.caption('Each color represents a category. Groups share the same bins. Bar heights show counts, so larger groups can have taller bars. Missing categories and nonfinite values are excluded.')
                else:
                    fig = px.histogram(data,x=key[col],nbins=bins,labels=labels)
            elif chart in ['Box plot', 'Grouped box plot']:
                group = choose('Group by', cats if cats else [None], 'box_group')
                color_group = choose('Subgroup', [c for c in cats if c != group], 'box_subgroup') if chart == 'Grouped box plot' else None
                fig = px.box(data,x=key[group] if group else None,y=key[col],color=key[color_group] if color_group else None,labels=labels)
                if color_group:
                    fig.update_layout(boxmode='group')
            elif chart == 'Dot plot':
                group = cats[0] if len(cats) == 1 else None
                dots, sampled = dot_points(data, key[col], key[group] if group is not None else None)
                if dots.empty:
                    raise ValueError('No complete numerical values and categories to plot.')
                if sampled:
                    st.caption('Dot plot uses a reproducible sample of 2,000 observations.')
                fig=px.scatter(dots,x='Value',y='Stack', color='Category' if group is not None else None,
                    labels={'Value':str(col),'Stack':'Stacked observations', 'Category':str(group)})
                if group is not None:
                    st.caption('Each dot represents one observation, colored by category. Repeated values are stacked without overlap. Missing categories and nonfinite values are excluded.')
            elif chart == 'Density plot':
                if x.nunique()<2:
                    raise ValueError('Density estimation requires variation in the selected column.')
                if len(x)>10000:
                    x=x.sample(10000,random_state=42)
                    st.caption('Density estimated from a reproducible sample of 10,000 observations.')
                smooth=st.slider('Bandwidth multiplier',0.2,3.0,1.0,0.1,key='va_bandwidth')
                kde=stats.gaussian_kde(x)
                kde.set_bandwidth(kde.factor*smooth)
                grid=np.linspace(x.min(),x.max(),300)
                fig=px.line(x=grid,y=kde(grid),labels={'x':str(col),'y':'Density'})
                st.caption('Gaussian kernel density estimate; a smoothed description, not a theoretical distribution fit.')
            else:
                if len(x)<3 or x.nunique()<2:
                    raise ValueError('Q–Q plots require at least three values and variation.')
                (theory,ordered),(slope,intercept,r)=stats.probplot(x,dist='norm')
                fig=go.Figure(go.Scatter(x=theory,y=ordered,mode='markers',name='Observed'))
                fig.add_scatter(x=theory,y=slope*np.asarray(theory)+intercept,mode='lines',name='Reference line')
                fig.update_layout(xaxis_title='Theoretical normal quantiles',yaxis_title=str(col))
                st.caption('Normal Q–Q plot. Departure from the line suggests departure from normality; this is not a significance test.')
        elif chart in ['Bar chart','Grouped bar chart','Stacked-bar chart','Pie chart','Pareto chart']:
            category=choose('Category',cats,'bar_category')
            stack=choose('Stack by',[c for c in cats if c!=category],'stack') if chart in ['Stacked-bar chart','Grouped bar chart'] else None
            if chart=='Stacked-bar chart':
                additive=choose('Additive measure',['Sum','Count'] if nums else ['Count'],'stack_additive')
                if additive=='Sum':
                    measure=choose('Numerical measure to sum',nums,'stack_measure')
                else:
                    measure=None
                agg='Sum'
                st.caption('Count intentionally uses category frequencies rather than the selected numeric measure. Count counts rows within each category combination, including rows with missing numeric values. Rows with missing grouping categories are excluded.' if additive=='Count' else 'Sum adds the selected numeric values within each category combination; missing numeric values are excluded.')
                if not nums:
                    st.caption('Select a numeric column to enable Sum.')
            else:
                measure=choose('Measure',[None]+nums if chart=='Pie chart' else nums if len(nums)==1 and not dates else [None]+nums,'bar_measure')
                agg=choose('Aggregation',['Sum','Mean'],'bar_agg') if measure and chart not in ['Pareto chart','Pie chart'] else 'Sum'
            if chart=='Pareto chart':
                st.caption('Pareto uses row counts or sums, ordered from largest to smallest. The cumulative line uses the full total.')
            keys=[key[category]]+([key[stack]] if stack else [])
            clean=data.dropna(subset=keys+([key[measure]] if measure else []))
            grouped=clean.groupby(keys,observed=True)
            values=(grouped[key[measure]].sum(min_count=1) if agg=='Sum' else grouped[key[measure]].mean()) if measure else grouped.size()
            table=values.rename('Value').reset_index()
            if table.empty:
                raise ValueError('No complete values for this chart.')
            if chart=='Pie chart':
                if (table.Value<0).any() or table.Value.sum()<=0:
                    raise ValueError('Pie charts require nonnegative category totals and a positive grand total.')
                fig=px.pie(table,names=key[category],values='Value',labels=labels)
                st.caption('Slices show row counts when no measure is selected, otherwise sums. Missing categories and missing selected measures are excluded. Negative totals cannot be shown as shares.')
            elif chart=='Pareto chart':
                if (table.Value<0).any() or table.Value.sum()<=0:
                    raise ValueError('Pareto requires nonnegative category totals and a positive grand total.')
                table=table.sort_values('Value',ascending=False)
                table['Cumulative %']=table.Value.cumsum()/table.Value.sum()*100
                fig=make_subplots(specs=[[{'secondary_y':True}]])
                fig.add_trace(go.Bar(x=table[key[category]],y=table.Value,name='Value'),secondary_y=False)
                fig.add_trace(go.Scatter(x=table[key[category]],y=table['Cumulative %'],name='Cumulative %',mode='lines+markers'),secondary_y=True)
                fig.update_yaxes(range=[0,100],title_text='Cumulative %',secondary_y=True)
            else:
                fig=px.bar(table,x=key[category],y='Value',color=key[stack] if stack else None,barmode='group' if chart=='Grouped bar chart' else 'stack',labels=labels)
            presentation.dataframe(table.rename(columns=labels),hide_index=True,use_container_width=True)
        elif chart in ['Line chart','Area chart']:
            axis_options = (dates or ordinal or nums) if chart == 'Line chart' else dates+nums
            if chart == 'Line chart' and len(axis_options) == 1:
                axis = axis_options[0]
                st.caption('X variable: '+str(axis))
            else:
                axis=choose('X axis',axis_options,'line_x')
            if chart == 'Line chart' and axis in nums:
                confirm_key = 'va_line_order_confirm_'+str(axis)
                if not st.checkbox('I confirm that '+str(axis)+' has a meaningful progression and connecting successive values makes sense.', key=confirm_key):
                    st.info('Confirm meaningful order for the numerical X variable to draw a line chart. Otherwise, use a scatter plot.')
                    return
            elif chart == 'Line chart':
                st.caption('X order follows the temporal values.' if axis in dates else 'X order follows the defined ordinal categories.')
            elif not dates:
                st.caption('Use line/area charts only when the X variable has a meaningful order; otherwise use a scatter plot.')
            if chart == 'Line chart':
                ys = [c for c in nums if c != axis]
                st.caption('Y variable: '+str(ys[0]))
            else:
                ys=variable_checkboxes('Measures',[c for c in nums if c!=axis],'va_line_y',default=[c for c in nums if c!=axis][:1])
            if not ys:
                raise ValueError('Select at least one measure.')
            agg=choose('Aggregate repeated X values',['Mean','Sum'],'line_agg')
            clean=data[[key[axis]]+[key[c] for c in ys]].dropna(subset=[key[axis]])
            if axis in ordinal:
                clean[key[axis]] = pd.Categorical(clean[key[axis]], categories=[str(v) for v in metadata[axis]['order']], ordered=True)
            grouped=clean.groupby(key[axis], observed=True)[[key[c] for c in ys]]
            table=(grouped.mean() if agg=='Mean' else grouped.sum(min_count=1)).reset_index().sort_values(key[axis])
            fig=(px.line if chart=='Line chart' else px.area)(table,x=key[axis],y=[key[c] for c in ys],labels=labels)
            st.caption('X values are sorted; repeated X values are aggregated. Area measures are stacked. Missing measures are not replaced by zero.')
        elif chart in ['Scatter plot','Bubble plot']:
            x=choose('X axis',nums,'scatter_x')
            y=choose('Y axis',[c for c in nums if c!=x],'scatter_y')
            size=choose('Bubble size',[c for c in nums if c not in [x,y]],'bubble_size') if chart=='Bubble plot' else None
            color=choose('Color by',cats if len(nums)==2 and len(cats)==1 and not dates else [None]+cats,'scatter_color')
            needed=[key[x],key[y]]+([key[size]] if size else [])+([key[color]] if color else [])
            clean=data.dropna(subset=needed)
            if size and ((clean[key[size]]<0).any() or not (clean[key[size]]>0).any()):
                raise ValueError('Bubble sizes must be nonnegative, with at least one positive value.')
            if len(clean)>5000:
                clean=clean.sample(5000,random_state=42)
                st.caption('Scatter/bubble plot uses a reproducible sample of 5,000 complete rows.')
            fig=px.scatter(clean,x=key[x],y=key[y],size=key[size] if size else None,color=key[color] if color else None,labels=labels)
        elif chart=='Pair plot':
            axes=variable_checkboxes('Pair plot variables',nums,'va_pair_axes',default=nums[:4])
            if not 2<=len(axes)<=6:
                raise ValueError('Choose two to six numerical variables for the pair plot.')
            clean=data[[key[c] for c in axes]].dropna()
            if len(clean)>2000:
                clean=clean.sample(2000,random_state=42)
                st.caption('Pair plot uses a reproducible sample of 2,000 complete rows.')
            fig=px.scatter_matrix(clean,dimensions=[key[c] for c in axes],labels=labels)
            fig.update_traces(diagonal_visible=False)
        elif chart=='Correlation matrix':
            method=choose('Correlation method',['pearson','spearman'],'correlation')
            matrix=data[[key[c] for c in nums]].corr(method=method).rename(index=labels,columns=labels)
            fig=px.imshow(matrix,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',text_auto='.2f',aspect='auto')
            st.caption('Pairwise complete observations are used; sample sizes can differ by pair. Constant columns yield missing correlations.')
        else:
            modes=(['Categorical cross-tab'] if len(cats)>=2 else [])+(['Numerical values'] if len(nums)>=2 else [])
            mode=choose('Heatmap type',modes,'heat_mode')
            if mode=='Categorical cross-tab':
                row=choose('Rows',cats,'heat_row'); col=choose('Columns',[c for c in cats if c!=row],'heat_col')
                measure=choose('Cell measure',nums if len(nums)==1 and len(cats)==2 and not dates else [None]+nums,'heat_measure')
                clean=data.dropna(subset=[key[row],key[col]]+([key[measure]] if measure else []))
                if clean[key[row]].nunique()*clean[key[col]].nunique()>10000:
                    raise ValueError('Choose categories producing at most 10,000 heatmap cells.')
                matrix=clean.pivot_table(index=key[row],columns=key[col],values=key[measure],aggfunc='mean',observed=True) if measure else pd.crosstab(clean[key[row]],clean[key[col]])
                matrix.index.name=str(row); matrix.columns.name=str(col)
                st.caption('Cells show mean measure values.' if measure else 'Cells show row counts.')
            else:
                matrix=data[[key[c] for c in nums]].rename(columns=labels)
                if len(matrix)>500:
                    matrix=matrix.head(500)
                    st.caption('Numerical heatmap shows the first 500 rows in original order.')
                if st.checkbox('Standardize each column',value=True,key='va_heat_scale'):
                    matrix=(matrix-matrix.mean())/matrix.std().replace(0,np.nan)
                st.caption('Rows are records; columns are numerical variables. Missing values remain blank.')
            if matrix.empty:
                raise ValueError('No values available for the heatmap.')
            fig=px.imshow(matrix,aspect='auto',color_continuous_scale='Blues')
        if fig is not None:
            if chart != 'Pareto chart':
                for col in cats:
                    if isinstance(df[col].dtype,pd.CategoricalDtype) and df[col].cat.ordered:
                        order=[str(v) for v in df[col].cat.categories]
                        for axis in fig.select_xaxes():
                            if axis.title.text in (str(col),key[col]):
                                axis.update(categoryorder='array',categoryarray=order)
                        for axis in fig.select_yaxes():
                            if axis.title.text in (str(col),key[col]):
                                axis.update(categoryorder='array',categoryarray=order)
            presentation.plotly_chart(fig,use_container_width=True,key='va_result_chart', panel_rows=panel_rows)
        st.caption('Blank and invalid values are excluded from the chart calculations where required. Sampling or display limits are stated above.')
    except (ValueError,TypeError,np.linalg.LinAlgError) as exc:
        st.info(str(exc))
