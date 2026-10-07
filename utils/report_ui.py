"""Guided vertical composition; the editor and exports share included content."""
from copy import deepcopy
import hashlib
import io
import pickle
import pandas as pd
import streamlit as st
from utils import presentation
from utils.report_state import (BLOCK_TYPES, COMPONENTS, ensure_report, new_block,
    add_snapshot, move_block, remove_block, block_parts, report_evidence, set_draft, restore_text)


def _report():
    return st.session_state['cozy_report']


def _block(block_id):
    return next(b for b in _report()['blocks'] if b['id'] == block_id)


def _edit_metadata(field, key):
    _report()['metadata'][field] = st.session_state[key]


def _edit_block(block_id, field, key):
    _block(block_id)[field] = st.session_state[key]


def _change_text(block_id, action):
    block = _block(block_id)
    if action == 'restore':
        restore_text(block)
    else:
        block['previous_text'].append(block['text'])
        block['text'] = ''
    st.session_state['report_text_'+block_id] = block['text']


def _generate(block_id):
    from utils.report_ai import generate_report_text
    block = _block(block_id)
    evidence = (report_evidence(_report()) if block['kind'] in ('Executive Summary', 'AI Interpretation')
                else [{'title': block['title'], 'analysis_type': (block.get('snapshot') or {}).get('analysis_type'),
                       'content': [(kind, value) for kind, value in block_parts(block) if kind != 'Chart']}])
    try:
        draft = generate_report_text(evidence, 'Executive summary' if block['kind'] == 'Executive Summary' else 'Analysis interpretation')
        set_draft(block, draft)
        st.session_state['report_text_'+block_id] = block['text']
        block.pop('ai_error', None)
    except Exception as exc:
        block['ai_error'] = ('Select fewer result components before generating this draft.'
                             if isinstance(exc, ValueError) and 'fewer' in str(exc) else
                             'AI could not generate a draft. Your existing text is unchanged; you can edit it manually.')


def _table_edit(block_id, key):
    block = _block(block_id)
    block['csv'] = st.session_state[key]
    try:
        from utils.report_state import structured
        frame = pd.read_csv(io.StringIO(block['csv'])) if block['csv'].strip() else pd.DataFrame()
        block['snapshot'] = {'tables': [structured(frame) | {'title': block['title']}], 'charts': []}
        block.pop('table_error', None)
    except (ValueError, pd.errors.ParserError):
        block['table_error'] = 'Enter a valid CSV table before exporting.'


def _add_section(entry, registry, dataset_id):
    kind = st.session_state['report_section_type']
    if kind == 'Analysis Result':
        if st.session_state.get('report_latest'):
            add_snapshot(_report(), st.session_state['report_latest'])
        return
    block = new_block(kind)
    if kind in ('Dataset Overview', 'Data Quality Summary', 'Data Preparation'):
        from utils.variable_metadata import infer_metadata
        from utils.report_dataset import dataset_snapshot
        metadata = st.session_state.get('variable_metadata') if st.session_state.get('variable_dataset_id') == dataset_id else infer_metadata(entry['frame'])
        block['snapshot'] = dataset_snapshot(kind, entry, metadata, registry, dataset_id, st.session_state.get('tf_draft'))
    if kind == 'Chart':
        charts = _chart_choices()
        selection = st.session_state.get('report_chart_source')
        if charts and selection is not None and selection < len(charts):
            block['snapshot'] = {'charts': [deepcopy(charts[selection])], 'tables': []}
    _report()['blocks'].append(block)


def _chart_choices():
    charts = []
    for block in _report()['blocks']:
        charts.extend((block.get('snapshot') or {}).get('charts', []))
    charts.extend(st.session_state.get('report_latest', {}).get('charts', []))
    return charts


def render_part(kind, value, chart_key=None):
    from utils.report_export import text_value
    if kind == 'Table':
        st.caption(value.get('title', 'Table'))
        presentation.dataframe(pd.DataFrame(value['rows'], columns=value['columns']), hide_index=True)
    elif kind == 'Chart':
        st.caption(value['title'])
        if 'png' in value:
            st.image(value['png'])
        else:
            import plotly.io as pio
            presentation.plotly_chart(pio.from_json(value['plotly']), key=chart_key)
    elif kind == 'Text':
        st.write(value)
    else:
        st.markdown('**'+kind+'**')
        st.text(text_value(value))


def render_report(entry, registry, dataset_id):
    report = ensure_report(st.session_state, entry)
    metadata = report['metadata']
    st.text_input('Report title', value=metadata['title'], key='report_meta_title',
                  on_change=_edit_metadata, args=('title', 'report_meta_title'))
    st.caption(f"Dataset: {metadata['dataset']} · Created: {metadata['created'][:10]} · {metadata['observations']} observations · {metadata['variables']} variables")
    with st.expander('Report details'):
        for field, label in [('author', 'Author (optional)'), ('organization', 'Organization (optional)'), ('period', 'Reporting period (optional)')]:
            key = 'report_meta_'+field
            st.text_input(label, value=metadata[field], key=key, on_change=_edit_metadata, args=(field, key))
    st.caption('Reports are kept in this session. Export a copy to keep your report.')
    st.selectbox('Section type', BLOCK_TYPES, index=BLOCK_TYPES.index('Text'), key='report_section_type')
    selected = st.session_state['report_section_type']
    if selected == 'Chart':
        choices = _chart_choices()
        st.selectbox('Captured chart', range(len(choices)), format_func=lambda i: choices[i]['title'], index=None, key='report_chart_source')
        if not choices:
            st.caption('Add an analysis with charts to the report, or complete a supported analysis first.')
    if selected == 'Analysis Result' and not st.session_state.get('report_latest'):
        st.caption('Complete a supported analysis in Analyze and select Add to Report.')
    st.button('+ Add Section', key='report_add_section', on_click=_add_section,
              args=(entry, registry, dataset_id), disabled=selected == 'Analysis Result' and not st.session_state.get('report_latest'))
    if not report['blocks']:
        st.info('Add a section here, or select Add to Report beside a completed analysis in Analyze.')
    for index, block in enumerate(report['blocks']):
        block_id = block['id']
        st.divider()
        st.subheader(block['title'])
        snapshot = block.get('snapshot') or {}
        if snapshot.get('dataset'):
            st.caption(f"{snapshot.get('analysis_type', block['kind'])} · Dataset: {snapshot['dataset']['name']} · Captured: {snapshot.get('timestamp','')}")
        with st.expander('Section options', key='report_options_'+block_id):
            title_key = 'report_title_'+block_id
            st.text_input('Section heading', value=block['title'], key=title_key, on_change=_edit_block, args=(block_id, 'title', title_key))
            if snapshot:
                key = 'report_parts_'+block_id
                st.multiselect('Include in report', COMPONENTS, default=block['included'], key=key,
                               on_change=_edit_block, args=(block_id, 'included', key))
            with st.container(horizontal=True):
                st.button('Move up', key='report_up_'+block_id, disabled=index == 0,
                          on_click=move_block, args=(report, block_id, -1))
                st.button('Move down', key='report_down_'+block_id, disabled=index == len(report['blocks'])-1,
                          on_click=move_block, args=(report, block_id, 1))
                st.button('Remove section', key='report_remove_'+block_id,
                          on_click=remove_block, args=(report, block_id))
        if block['kind'] == 'Page Break':
            st.caption('Page break in exported reports')
            continue
        if block['kind'] == 'Table':
            key = 'report_csv_'+block_id
            st.text_area('Table data (CSV)', value=block.get('csv', ''), key=key, on_change=_table_edit, args=(block_id, key))
            if block.get('table_error'):
                st.error(block['table_error'])
        for part_index, (kind, value) in enumerate(block_parts(block)):
            if kind != 'Text':
                render_part(kind, value, 'report_chart_'+block_id+'_'+str(part_index))
        if block['kind'] != 'Heading' and 'Interpretation' in block['included']:
            key = 'report_text_'+block_id
            if key not in st.session_state:
                st.session_state[key] = block['text']
            st.text_area('Interpretation' if block['kind'] == 'Analysis Result' else 'Section text',
                         key=key, height=150, on_change=_edit_block, args=(block_id, 'text', key))
        if block['kind'] in ('Analysis Result', 'Executive Summary', 'AI Interpretation'):
            st.caption('AI drafts are editable. Review them before sharing your report.')
            with st.container(horizontal=True):
                label = ('Generate Executive Summary' if block['kind'] == 'Executive Summary' else 'Generate Interpretation')
                st.button('Regenerate' if block['generated'] else label, key='report_ai_'+block_id, on_click=_generate, args=(block_id,))
                st.button('Undo AI / Restore Previous', key='report_undo_'+block_id,
                          disabled=not block['previous_text'], on_click=_change_text, args=(block_id, 'restore'))
                st.button('Clear', key='report_clear_'+block_id, on_click=_change_text, args=(block_id, 'clear'))
            if block.get('ai_error'):
                st.warning(block['ai_error'])
    st.divider()
    signature = hashlib.sha256(pickle.dumps(report)).hexdigest()
    if st.button('Prepare downloads', key='report_export', disabled=not report['blocks']):
        try:
            from utils.report_export import export_docx, export_pdf
            with st.spinner('Preparing your report…'):
                st.session_state['report_export_files'] = (signature, export_docx(report), export_pdf(report))
        except Exception as exc:
            st.session_state.pop('report_export_files', None)
            st.error(f'Could not export this report: {exc}')
    saved = st.session_state.get('report_export_files')
    if saved and saved[0] == signature:
        st.download_button('Download DOCX', saved[1], 'cozy_report.docx',
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document', key='report_docx_download')
        st.download_button('Download PDF', saved[2], 'cozy_report.pdf', 'application/pdf', key='report_pdf_download')
