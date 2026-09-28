"""Column Overview with a type-dependent classification editor."""
import pandas as pd
import streamlit as st
from utils import presentation
from utils.variable_overview_display import render_overview
from utils.variable_metadata import ALLOWED, ensure_metadata, infer_variable_subtype, category_order, apply_user_override, effective_data


def render_column_overview(raw, dataset_id):
    metadata=ensure_metadata(st.session_state,raw,dataset_id)
    st.caption('Variable Type defines whether a variable is treated analytically as numerical, categorical, or temporal. Variable Subtype provides its statistical classification. Edit a column below; changes apply throughout the app.')
    render_overview(raw, metadata)
    st.caption('A = Automatic classification  M = Manual Override')
    with st.expander('Edit variable classification',expanded=False):
        col=st.selectbox('Column to edit',list(raw),key='vm_column')
        m=metadata[col]
        # Keys distinguish saved metadata from drafts; apply is atomic after validation.
        prefix='vm_'+str(list(raw).index(col))+'_'+str(st.session_state.get('vm_revision',0))
        t=st.selectbox('Variable Type',list(ALLOWED),index=list(ALLOWED).index(m['variable_type']),key=prefix+'_type',format_func=lambda value: 'Temporal' if value=='Date / Time' else value)
        default=m['variable_subtype'] if t==m['variable_type'] else infer_variable_subtype(raw[col],t)
        subtype=st.selectbox('Variable Subtype',ALLOWED[t],index=ALLOWED[t].index(default),key=prefix+'_'+t+'_subtype',format_func=lambda value: 'Temporal' if value=='Date / Time' else value)
        if t == 'Date / Time':
            st.caption('Date: calendar date; Time: time of day; DateTime: date and time; Duration: elapsed time with explicit units. Year: four-digit year (1000–9999). Month: January/Jan or 1–12, without a year. Year-Month: YYYY-MM or YYYY/MM; internally anchored to the first day for period analysis. Numeric year/month columns require an explicit Temporal override. Full dates are not truncated by these overrides. HH:MM[:SS] is inferred as Time; choose Duration for elapsed clock notation. Zoned timestamps are normalized to UTC.')
        order=[]
        if subtype=='Ordinal':
            values,recognized=category_order(raw[col])
            proposed=m['order'] if m['order'] else values
            st.caption('Set lowest to highest. Enter category numbers in the desired order; every category must appear once. Unrecognized orders start in observed order, not alphabetical order.')
            presentation.table(pd.DataFrame({'Category number':range(1,len(proposed)+1),'Observed value':[repr(v) for v in proposed]}))
            order_text=st.text_input('Category order (lowest to highest)',value=', '.join(str(i) for i in range(1,len(proposed)+1)),key=prefix+'_order')
            try:
                indices=[int(x.strip()) for x in order_text.split(',')]
                if sorted(indices)!=list(range(1,len(proposed)+1)): raise ValueError()
                order=[proposed[i-1] for i in indices]
            except ValueError:
                order=[]
        if pd.api.types.is_numeric_dtype(raw[col]) and raw[col].nunique()==2:
            st.info('Two unique numeric values detected. This may represent either a discrete numerical variable or a binary categorical variable.')
        has_changes = (t != m['variable_type'] or subtype != m['variable_subtype']
                       or (subtype == 'Ordinal' and order != m.get('order', [])))
        if st.button('Apply classification',key='vm_apply',disabled=not has_changes):
            try:
                updated=apply_user_override(metadata,col,raw[col],t,subtype,order)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state['variable_metadata']=updated
                # Discard results and selections whose meaning depends on classification.
                for key in list(st.session_state):
                    if key in ('ai_summary','pa_results','pa_cache_key','pa_train_error','regression_cached_fit') or key.startswith(('col_ai_','heatmap_ai_','pa_pred_','mc_','gs_','rel_','dist_','va_')):
                        st.session_state.pop(key,None)
                st.session_state['vm_revision']=st.session_state.get('vm_revision',0)+1
                st.rerun()
    data, groups = effective_data(raw,metadata)
    groups.setdefault('temporal', list(dict.fromkeys(
        groups.get('datetime', []) + groups.get('time', []) + groups.get('duration', [])
    )))
    for c in groups['temporal']:
        invalid = int((raw[c].notna() & data[c].isna()).sum())
        if invalid:
            st.warning(f'{c}: {invalid} non-missing values could not be parsed as their Temporal subtype and are missing in analyses. Original values are preserved.')
    return data, groups
