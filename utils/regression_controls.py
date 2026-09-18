"""Checkbox specification controls with single-target and polynomial hierarchy."""
import hashlib
from itertools import combinations
import streamlit as st


def widget_key(group, value):
    return 'rel_check_'+group+'_'+hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def choose_target(column, options):
    # Clicking another target switches selection; unchecking the active target keeps it.
    st.session_state['rel_target_False']=column
    for c in options:
        st.session_state[widget_key('target',c)]=c==column


def change_selection(state_key, option, key, powers=False):
    selected=list(st.session_state.get(state_key,[]))
    if st.session_state[key]:
        if option not in selected: selected.append(option)
        if powers and option[1]==3 and (option[0],2) not in selected:
            selected.append((option[0],2))
    elif option in selected:
        selected.remove(option)
    st.session_state[state_key]=selected


def set_powers(options, selected):
    st.session_state['rel_polynomial_terms']=list(options) if selected else []


def checkbox_group(title, options, state_key, group, default=(), formatter=str, powers=False):
    st.markdown('**'+title+'**')
    chosen=[v for v in st.session_state.get(state_key,list(default)) if v in options]
    if powers:
        for c,d in list(chosen):
            if d==3 and (c,2) not in chosen: chosen.append((c,2))
    st.session_state[state_key]=chosen
    if not options:
        st.caption('No available terms for the selected predictors.')
        return []
    for start in range(0,len(options),3):
        columns=st.columns(3)
        for container,option in zip(columns,options[start:start+3]):
            key=widget_key(group,option)
            st.session_state[key]=option in chosen
            required=powers and option[1]==2 and (option[0],3) in chosen
            container.checkbox(formatter(option),key=key,disabled=required,
                               help='Required while the cubic term is selected.' if required else None,
                               on_change=change_selection,args=(state_key,option,key,powers))
    return [v for v in options if v in chosen]


def render_checkbox_specification(responses, numerical, categorical):
    st.markdown('**Target / dependent variable (Y)**')
    st.caption('Choose one target. Checking another variable switches the target.')
    target=st.session_state.get('rel_target_False')
    if target not in responses: target=responses[0]
    st.session_state['rel_target_False']=target
    for start in range(0,len(responses),3):
        columns=st.columns(3)
        for container,c in zip(columns,responses[start:start+3]):
            key=widget_key('target',c)
            st.session_state[key]=c==target
            container.checkbox(str(c),key=key,on_change=choose_target,args=(c,responses))
    options=[c for c in dict.fromkeys(numerical+categorical) if c!=target]
    predictors=checkbox_group('Predictors / independent variables (X)',options,'rel_predictors','predictor',default=options[:1])
    # Prune unavailable terms even when no predictors remain and the caller returns early.
    pairs=list(combinations(predictors,2))
    st.session_state['rel_interaction_terms']=[p for p in st.session_state.get('rel_interaction_terms',[]) if p in pairs]
    nums=[c for c in predictors if c not in categorical]
    st.session_state['rel_polynomial_terms']=[p for p in st.session_state.get('rel_polynomial_terms',[]) if p[0] in nums]
    return target,predictors


def render_term_checkboxes(predictors, nums):
    pairs=list(combinations(predictors,2))
    interactions=checkbox_group('Interaction terms',pairs,'rel_interaction_terms','interaction',formatter=lambda p:' × '.join(map(str,p)))
    terms=[(c,d) for c in nums for d in (2,3)]
    st.markdown('**Polynomial terms**')
    a,b=st.columns(2)
    a.button('Select all polynomial terms',key='rel_poly_all',on_click=set_powers,args=(terms,True),disabled=not terms)
    b.button('Clear polynomial terms',key='rel_poly_clear',on_click=set_powers,args=(terms,False),disabled=not terms)
    selected=checkbox_group('Select powers',terms,'rel_polynomial_terms','power',formatter=lambda p:str(p[0])+('²' if p[1]==2 else '³'),powers=True)
    st.caption('Cubic terms include the corresponding square. Selected predictors supply all main effects; interactions retain both main effects.')
    degrees={c:max([1]+[d for variable,d in selected if variable==c]) for c in nums}
    return interactions,degrees
