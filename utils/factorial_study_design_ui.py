"""Plain-language study choices and automatic design checks."""
import numpy as np
import streamlit as st
from utils import presentation
from utils.anova_effects import prepare_random
from utils.factorial_study_design import (UNSURE, SEPARATE, SEQUENCE, SHARED,
    study_context, factor_key, detect_relationships, assignment_parent, assignment_language,
    collection_language, time_evidence, check_time_column, collection_requirement,
    clean_design, design_diagnostic, sync_study_state)


def _reset_confirmation(key):
    st.session_state.pop(key, None)
    st.session_state.pop('report_latest', None)
    st.session_state.pop('fac_interaction_ai', None)


def _select_assignment(key, choice, checkbox_key):
    st.session_state[key] = choice if st.session_state[checkbox_key] else UNSURE
    st.session_state.pop('report_latest', None)
    st.session_state.pop('fac_interaction_ai', None)


def _explain(item):
    presentation.write('**What Cozy found:** ' + item.observed_result)
    presentation.write('**Why it matters:** ' + item.explanation)
    presentation.write('**Recommended action:** ' + item.recommended_action)
    st.markdown('**Take action**')


def render_study_design(df, outcome, effects, interactions):
    state = st.session_state
    context = study_context(df, outcome, effects, interactions, state.get('variable_metadata', {}), state.get('file_id'))
    sync_study_state(state, context)
    # Preserve conditionally hidden controls during normal reruns.
    for key in list(state):
        if key.startswith('fac_design_'):
            state[key] = state[key]
    evidence = detect_relationships(df, outcome, effects)
    items, assignments, parents = [], [], {}
    confirmed_all = True
    for relation in evidence['relationships']:
        factor = relation['factor']
        suffix = factor_key(factor)
        key = 'fac_design_assignment_' + suffix
        question, options, labels = assignment_language(factor, effects)
        if key not in state:
            state[key] = relation['suggestion'] or UNSURE
        confirmed = state[key] != UNSURE
        details = []
        for comparison in relation['comparisons']:
            other = comparison['other']
            pattern = {'fully crossed labels': f'Every {factor} label appears under every {other} level.',
                       'labels confined to one group': f'Each {factor} label appears under only one {other} level.',
                       'partially crossed labels': f'Some {factor} labels appear across {other} groups, but not every combination is observed.',
                       'ambiguous': 'No single assignment pattern can be determined.'}[comparison['pattern']]
            details.append(f"{relation['levels']} {factor} labels and {comparison['levels']} {other} levels: "
                           f"{comparison['observed_pairs']} of {comparison['possible_pairs']} pairs observed; "
                           f"{comparison['labels_across_groups']} {factor} labels appear across multiple {other} levels. "
                           f"Observed counts per pair range from {comparison['minimum_n']} to {comparison['maximum_n']}; "
                           f"{comparison['empty_pairs']} empty and {comparison['singleton_pairs']} single-observation pairs. "
                           + pattern)
        item = design_diagnostic('assignment_' + suffix, 'ok' if confirmed else 'unresolved', f'{factor} Assignment',
            ' '.join(details),
            f'The model needs to know whether identical {factor} labels identify the same people or units. Reused labels may identify different people in different groups; missing combinations alone do not establish ownership.',
            'Select the assignment using your knowledge of the study. If unsure, ask the person who collected the measurements; the model structure remains unresolved.')
        presentation.markdown('**' + str(factor) + ' assignment**')
        _explain(item)
        suggestion = relation['suggestion']
        if not suggestion:
            st.caption('The observed combinations do not support a single clear suggestion. Missing combinations do not tell Cozy whether the same people or units work across groups or belong to separate groups.')
        st.markdown('**' + question + '**')
        for option in options:
            checkbox_key = key + '_option_' + factor_key(option)
            state[checkbox_key] = state[key] == option
            st.checkbox(labels[option], key=checkbox_key, on_change=_select_assignment,
                        args=(key, option, checkbox_key))
        choice = state[key]
        confirmed = choice != UNSURE
        if confirmed:
            parent = assignment_parent(choice, factor, effects)
            if parent is not None:
                parents[factor] = parent
        else:
            confirmed_all = False
        # The selected assignment configures the next step immediately.
        from dataclasses import replace
        items.append(replace(item, severity='ok' if confirmed else 'unresolved'))
        assignments.append({'factor': factor, 'answer': choice, 'label': labels[choice],
                            'confirmed': confirmed, 'suggestion': suggestion, 'observed': relation})
    response = {'context': context, 'evidence': evidence, 'assignments': assignments,
                'collection': None, 'time_column': None, 'shared_column': None}
    decision = {'fit': False, 'inference': False, 'severity': 'unresolved'}
    effective = dict(effects)
    explicitly_applied = False
    if not confirmed_all:
        st.info('Select a known assignment above to continue to measurement collection. If unsure, clarify what the labels identify with the study owner.')
        presentation.write('Study design status: **Pending check**')
        return {'parents': parents, 'effects': effective, 'decision': decision, 'items': items,
                'answers': response, 'explicitly_applied': explicitly_applied}

    labels, descriptions = collection_language(effects)
    presentation.markdown('**Measurement collection**')
    replication = (str(evidence['minimum_n']) if evidence['minimum_n'] == evidence['maximum_n'] else
                   f"{evidence['minimum_n']} to {evidence['maximum_n']}")
    found = (f"{evidence['n']} usable observations across {evidence['observed_cells']} observed combinations. "
             f"There are {replication} measurements per combination. "
             'These counts do not establish sequence, separate sessions, or shared sources.')
    item = design_diagnostic('measurement_collection', 'unresolved', 'Measurement Collection', found,
        'Measurements collected close together or from a shared source may have additional relationships beyond the selected factors.',
        'Confirm how measurements were collected. Additional controls appear only when needed.')
    _explain(item)
    choice = st.radio('How were the measurements collected?', [SEPARATE, SEQUENCE, SHARED, UNSURE],
                      index=None, key='fac_design_collection', format_func=labels.get)
    response['collection'] = choice
    if choice:
        presentation.write(descriptions[choice])
    sequence_connection = None
    shared_confirmed = False
    if choice == SEQUENCE:
        raw = clean_design(df, outcome, list(effects))
        candidates = time_evidence(raw, [outcome, *effects], state.get('variable_metadata'))
        response['observed_time_columns'] = candidates
        usable = [r['Column'] for r in candidates if r['Usable']]
        st.caption('Possible time or sequence columns: ' + (', '.join(map(str, usable)) or 'none detected') + '. CSV row order is never treated as measurement order.')
        time_column = st.selectbox('Which column records measurement time or sequence, if available?',
            [None] + [c for c in df if c not in effects and c != outcome],
            format_func=lambda c: 'No known time or sequence column' if c is None else str(c), key='fac_design_time_column')
        response['time_column'] = time_column
        if time_column is not None:
            check = check_time_column(raw[time_column])
            response['time_column_check'] = check
            st.caption(f"{time_column}: " + ('contains usable order/time values' if check['Usable'] else 'does not contain usable varying numeric or date/time values') +
                       f"; {check['Missing or invalid']} missing/invalid values. A column does not establish which measurements share a sequence.")
        sequence_connection = st.radio('Could measurements taken close together share changing conditions beyond the selected factors?',
            ['yes', 'no', UNSURE], index=None, key='fac_design_sequence_connection',
            format_func={'yes': 'Yes', 'no': 'No additional connection is expected', UNSURE: "I'm not sure"}.get)
        response['sequence_connection'] = sequence_connection
        if sequence_connection == 'yes':
            item = design_diagnostic('measurement_collection', 'blocking', 'Measurement Collection',
                'You report an additional connection associated with measurement order.' +
                (' No usable time/sequence column was identified.' if time_column is None or not check_time_column(raw[time_column])['Usable'] else f' You identified {time_column} as an order/time column.'),
                'The current model does not account for correlation between nearby measurements; adding a time column as a factor does not resolve it.',
                'Record the measurement order/time and the person or unit followed over time. Use a mixed model with an appropriate time-correlation structure or a suitable repeated-measures analysis; those covariance models are not available here. No model is changed automatically.')
        elif sequence_connection == 'no':
            item = design_diagnostic('measurement_collection', 'ok', 'Measurement Collection',
                'You report sequential collection but expect no additional connection beyond the selected factors.',
                'This is your study-design assumption, not proof of independent residuals.',
                'Review residual diagnostics and the collection process if evidence of additional dependence emerges.')
    elif choice == SHARED:
        column = st.selectbox('Which column identifies the shared source?', [None] + [c for c in df if c != outcome],
                              format_func=lambda c: 'Choose a column, if known' if c is None else str(c),
                              key='fac_design_shared_column', on_change=_reset_confirmation,
                              args=('fac_design_shared_identity',))
        response['shared_column'] = column
        if column is not None:
            identity = st.checkbox(f'Identical {column} labels identify the same shared source across the selected factors, and no other connection is expected.',
                                    key='fac_design_shared_identity')
            response['shared_identity_confirmed'] = identity
            if effects.get(column) == 'Fixed':
                item = design_diagnostic('measurement_collection', 'blocking', 'Measurement Collection',
                    f'You identified {column} as a shared source, but it is a Fixed factor in the current model.',
                    'Fixed mean differences do not account for shared residual variation within that source.',
                    'Review whether this factor should represent sampled groups using Factor Effect Type, or specify the necessary grouping variable. Do not change its scientific meaning just to fit a model.')
            elif identity and column in parents:
                item = design_diagnostic('measurement_collection', 'blocking', 'Measurement Collection',
                    f'Your {column} assignment uses labels within {parents[column]}, but the shared-source answer says the labels identify the same source across groups.',
                    'Those identity statements imply different groupings and cannot both configure the current model.',
                    'Clarify whether identical labels identify the same real source, then update the assignment or shared-source answer above. No grouping is changed automatically.')
            elif identity:
                proposed = {**effects, column: 'Random'}
                try:
                    # Existing rank/replication/ownership checks assess eligibility;
                    # this does not fit or silently add a component.
                    prepare_random(df, outcome, proposed, interactions, parents)
                    if column in effects:
                        shared_confirmed = True
                        presentation.write(f'The selected model already accounts for the confirmed {column} shared source.')
                    else:
                        approval = (context, column, tuple(parents.items()))
                        if st.button(f'Add {column} as a shared-source grouping and refit', key='fac_action_design_add_group'):
                            state['fac_design_added_group'] = approval
                            explicitly_applied = True
                        shared_confirmed = state.get('fac_design_added_group') == approval
                        if shared_confirmed:
                            effective = proposed
                            presentation.write(f'{column} is included as an additional random grouping. The selected factor effect types and interactions are retained.')
                        else:
                            st.caption('This grouping passes the structural checks. Confirm the model update above to include it; no new interactions are added.')
                except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                    item = design_diagnostic('measurement_collection', 'blocking', 'Measurement Collection',
                        f'The requested {column} grouping cannot be represented: {exc}',
                        'A shared source cannot be ignored or added as an unidentifiable component.',
                        'Review group labels and replication. If labels are local to different groups, include the grouping in Comparison and confirm its assignment there. Collect additional information or use a suitable model when this structure remains unsupported.')
            elif not identity:
                st.caption('Confirm what shared-source labels identify. If labels refer to different sources in different groups, do not confirm this statement; review the grouping structure instead.')
    decision = collection_requirement(choice, sequence_connection, shared_confirmed)
    if item.severity == 'blocking':
        decision = {'fit': False, 'inference': False, 'severity': 'blocking'}
    if choice == SEPARATE:
        item = design_diagnostic('measurement_collection', 'ok', 'Measurement Collection',
            'You confirm separate measurements with no additional relationship expected beyond the modeled factors and combinations.',
            'The selected model can account for variation associated with ' + ', '.join(map(str, effects)) +
            ' and the selected random combinations. This is appropriate if no additional dependence remains between measurements; your confirmation is not proof of independence.',
            'Continue reviewing model and residual diagnostics.')
    elif choice == SHARED and shared_confirmed:
        item = design_diagnostic('measurement_collection', 'ok', 'Measurement Collection',
            f"You confirmed {response['shared_column']} as a modeled shared source with globally consistent labels and no other connection expected.",
            'This translates your collection-process knowledge into the existing grouping model; residual independence is still an assumption.',
            'Review identifiability, convergence and residual diagnostics after fitting.')
    elif decision['fit'] and not decision['inference']:
        item = design_diagnostic('measurement_collection', 'unresolved', 'Measurement Collection',
            'The measurement collection process or additional relationship is unconfirmed.',
            'The model summary and diagnostics wait for this clarification so they do not present unsupported model conclusions.',
            'Ask the study owner about measurement order and shared sources, then update your answer. The dataset is not declared invalid.')
    items.append(item)
    status = 'Checks passed' if decision['fit'] and decision['inference'] else ('Model issue' if decision['severity'] == 'blocking' else 'Pending check')
    presentation.write('Study design status: **' + status + '**')
    if choice:
        presentation.write('**What your answer means:** ' + item.observed_result + ' ' + item.explanation)
        presentation.write('**Recommended action:** ' + item.recommended_action)
        if item.severity == 'blocking':
            st.markdown('[Review selected factors and model terms](#fac-model-configuration)')
    response['decision'] = decision
    response['effective_effects'] = effective
    response['parents'] = parents
    with st.expander('Advanced model details'):
        st.caption('Confirmed assignments map to crossed or explicitly nested Gaussian variance components. Correlated random slopes, serial correlation and general repeated-measures covariance are unsupported. Full rank, component identifiability, convergence and singularity checks remain active.')
    return {'parents': parents, 'effects': effective, 'decision': decision, 'items': items,
            'answers': response, 'explicitly_applied': explicitly_applied}
