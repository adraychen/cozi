"""Understated enterprise information, enquiry form, and in-app navigation."""
from html import escape

import streamlit as st

from utils.branding import render_brand
from utils.enterprise_enquiries import DEPLOYMENTS, USER_COUNTS, get_enquiry_store, submit_enquiry


FORM_FIELDS = ('name', 'email', 'organization', 'deployment', 'job_title', 'number_of_users', 'requirements')


def preserve_page_state():
    """Detach hidden page widgets from cleanup, preserving the analytics workspace."""
    from utils.dataset_workspace import dataset_state_key
    for key in list(st.session_state):
        if (dataset_state_key(key) or key in ('ws_dataset_tabs', 'ws_workflow')
                or key.startswith('ws_workflow_') or key in {f'enterprise_form_{f}' for f in FORM_FIELDS}):
            st.session_state[key] = st.session_state[key]


def navigate(page):
    st.query_params['page'] = page


def enterprise_requested():
    return st.query_params.get('page') == 'enterprise'


def render_footer(show_enterprise=True):
    st.divider()
    if show_enterprise:
        render_brand(enterprise=True, show_subtitle=False)
        st.caption("Deploy Cozy securely in your organization's cloud or private infrastructure, with customized analytical capabilities designed around your organization's needs.")
        st.button('Learn more about Cozy Enterprise →', key='enterprise_learn_more', type='tertiary',
                  on_click=navigate, args=('enterprise',))
        st.html('<div style="margin-top:1rem; border-top:1px solid #8883;"></div>')
    st.caption('© 2026 Cozy. All rights reserved.')


def render_enquiry_form():
    if st.session_state.pop('enterprise_clear_form', False):
        for field in FORM_FIELDS:
            st.session_state[f'enterprise_form_{field}'] = None if field in ('deployment', 'number_of_users') else ''
    st.header('Interested in Cozy Enterprise?')
    st.write("Tell us a little about your organization and we'll contact you to discuss your requirements.")
    store = get_enquiry_store()
    st.caption('Fields marked * are required.')
    with st.form('enterprise_enquiry', clear_on_submit=False):
        name = st.text_input('Name *', key='enterprise_form_name', max_chars=200)
        email = st.text_input('Work Email *', key='enterprise_form_email', max_chars=254,
                              placeholder='name@company.com')
        organization = st.text_input('Company / Organization *', key='enterprise_form_organization', max_chars=300)
        deployment = st.selectbox('Deployment Interest *', DEPLOYMENTS, index=None,
                                  placeholder='Select a deployment option', key='enterprise_form_deployment')
        job_title = st.text_input('Job Title', key='enterprise_form_job_title', max_chars=200)
        users = st.selectbox('Number of Users', USER_COUNTS, index=None,
                             placeholder='Optional', key='enterprise_form_number_of_users')
        requirements = st.text_area('What would you like Cozy Enterprise to do for your organization?',
                                    key='enterprise_form_requirements', max_chars=5000, height=150,
                                    help='Tell us about your data, analytical workflows, reporting requirements, or specialized analytical features you need.')
        submitted = st.form_submit_button('Submit Enquiry', type='primary')
    if submitted:
        result = submit_enquiry(dict(name=name, email=email, organization=organization,
                                     deployment=deployment, job_title=job_title,
                                     number_of_users=users, requirements=requirements), st.session_state, store)
        st.session_state['enterprise_last_result'] = result
        if result.status in ('submitted', 'duplicate'):
            st.session_state['enterprise_clear_form'] = True
            st.rerun()
    result = st.session_state.get('enterprise_last_result')
    if result:
        render = st.success if result.status in ('submitted', 'duplicate') else st.warning
        for message in result.messages:
            render(message)


def render_enterprise_page():
    st.button('← Back to Cozy Community', key='enterprise_back', type='tertiary',
              on_click=navigate, args=('community',))
    render_brand(enterprise=True)
    st.html('''<style>
    .enterprise-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr)); gap:1rem; margin:0.5rem 0 2rem;}
    .enterprise-card {border:1px solid #8884; border-radius:8px; padding:1.25rem; min-width:0; overflow-wrap:anywhere;}
    .enterprise-card h3 {font-size:1.05rem; line-height:1.45; margin:0 0 0.65rem; padding:0;}
    .enterprise-card p {margin:0; line-height:1.65;}
    </style>''')
    st.write("Cozy Enterprise brings Cozy's guided analytics platform into your organization's own environment, with customized analytical capabilities designed around your data, workflows, and analytical needs.")
    st.header('Designed for your organization')
    st.subheader('Customized Analytics')
    st.markdown('**Cozy Enterprise adapts to the way your organization works.** '
                'We customize the platform around your data, processes, reporting requirements, and business priorities—'
                'creating an analytical environment that supports how your teams operate and make decisions.')
    st.markdown('Customization may include:\n\n'
                '- **Dashboards** tailored to your objectives, workflows, and performance indicators\n'
                '- **Data processing** designed around your data sources, structures, rules, and preparation requirements\n'
                '- **Reports** customized for different audiences, decisions, and organizational standards')
    st.markdown('Cozy Enterprise can also include specialized analytical modules. Examples include:\n\n'
                '- Quality\n- Reliability Analysis\n- Supply Chain\n- Property Management')
    st.write('These are examples rather than a fixed catalogue. Additional modules can be developed to support '
             'the analytical needs of different industries, functions, and organizations.')
    st.markdown('**The result is an enterprise analytics platform configured around what your organization '
                'needs to understand, monitor, and improve.**')
    st.header('Deployment options')
    st.write('Deploy on Azure, AWS, Google Cloud, or on-premises while maintaining greater control over your data and infrastructure.')
    st.subheader('Enterprise Support')
    st.write('Implementation, customization, deployment, and ongoing technical support.')
    st.divider()
    with st.container(width=760):
        render_enquiry_form()
    render_footer(show_enterprise=False)
