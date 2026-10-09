"""Small dataset-scoped handoffs using the existing workspace navigation."""
from copy import deepcopy


def open_component_transform(state, month, year, context):
    dataset_id = state.get('file_id')
    state['tf_component_request'] = {'dataset_id': dataset_id, 'month': month, 'year': year}
    state['temporal_return_context'] = {'source_id': dataset_id, 'analysis': deepcopy(context),
                                        'month': month, 'year': year}
    state.pop('tf_result', None)
    state['ws_workflow'] = 'Transform'


def prepare_component_request(state, dataset_id, frame):
    request = state.get('tf_component_request')
    if not request or request['dataset_id'] != dataset_id:
        return
    state['tf_operation'] = 'Add column'
    state['tf_add_method'] = 'Combine Date Components'
    state['tf_add_combination'] = 'Year + Month'
    state['tf_add_year'] = request['year'] if request['year'] in frame else None
    state['tf_add_month'] = request['month'] if request['month'] in frame else None
    base = 'Month-Year'
    name, number = base, 2
    while name in frame:
        name = f'{base} ({number})'
        number += 1
    state['tf_add_name'] = name
    state['tf_add_display_format'] = 'YYYY-MM'
    state.pop('tf_component_request', None)


def attach_analysis_return(state, new_id, metadata):
    context = state.get('temporal_return_context')
    if not context or context['source_id'] != state.get('file_id'):
        return
    copied = deepcopy(context)
    copied['target_id'] = new_id
    candidates = [column for column, item in metadata.items()
                  if item.get('origin') == 'Combine Date Components' and item['variable_subtype'] == 'Year-Month'
                  and item.get('components', {}).get('month') == context['month']]
    copied['new_column'] = candidates[-1] if candidates else None
    state['ws_datasets'][new_id]['state']['temporal_return_context'] = copied


def return_to_analysis(state):
    context = state.get('temporal_return_context', {})
    analysis = context.get('analysis', {})
    state['ws_workflow'] = 'Analyze'
    state['view_analyze_tabs'] = analysis.get('tab', 'Comparison')
    available = state['df'].columns
    measures = [c for c in analysis.get('numerical', []) if c in available]
    dimensions = [c for c in analysis.get('categorical', []) if c in available]
    tab = analysis.get('tab', 'Comparison')
    if tab == 'Comparison':
        for column in available:
            state['gs_cmp_variable_' + str(column)] = False
    elif tab == 'Group Statistics':
        state['gs_measure_selected'] = measures[:1]
        state['gs_dimensions_selected'] = dimensions
        state['gs_temporal_selected'] = []
        if analysis.get('analysis'):
            state['gs_analysis'] = analysis['analysis']
    elif tab == 'Relationship':
        for column in available:
            state['mc_' + str(column)] = column in measures + dimensions
    # Temporal selection remains explicit, so returning does not run a comparison.
    state['temporal_recommendation'] = context.get('new_column')


def continue_temporal_analysis(state):
    context = state.get('temporal_return_context', {})
    analysis = context.get('analysis', {})
    column = context.get('new_column')
    if column not in state['df']:
        return
    available = state['df'].columns
    measures = [c for c in analysis.get('numerical', []) if c in available]
    dimensions = [c for c in analysis.get('categorical', []) if c in available]
    tab = analysis.get('tab', 'Comparison')
    if tab == 'Comparison':
        for name in available:
            state['gs_cmp_variable_' + str(name)] = name in measures + dimensions + [column]
    elif tab == 'Group Statistics':
        state['gs_temporal_selected'] = [column]
    elif tab == 'Relationship':
        for name in available:
            state['mc_' + str(name)] = name in measures + dimensions + [column]
    state.pop('temporal_recommendation', None)


def select_existing_temporal(state, column, context):
    tab = context.get('tab', 'Comparison')
    metadata = state.get('variable_metadata', {})
    if tab == 'Comparison':
        for name, item in metadata.items():
            if item.get('variable_type') == 'Date / Time' or item.get('variable_subtype') in ('Month', 'Year-Month', 'Date', 'DateTime', 'Quarter', 'Year-Quarter', 'Year'):
                state['gs_cmp_variable_' + str(name)] = name == column
    elif tab == 'Group Statistics':
        state['gs_temporal_selected'] = [column]
    elif tab == 'Relationship':
        for name, item in metadata.items():
            if item.get('variable_type') == 'Date / Time' or item.get('variable_subtype') in ('Month', 'Year-Month', 'Date', 'DateTime', 'Quarter', 'Year-Quarter', 'Year'):
                state['mc_' + str(name)] = name == column
