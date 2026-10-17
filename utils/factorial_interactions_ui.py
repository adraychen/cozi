"""Shared interaction selection, grounded insights and explicit refinement."""
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
from utils import presentation
from utils.factorial_interactions import (EXPLORE, MANUAL, selected_pairs, interaction_structure,
                                         compare_fixed_reduction, interaction_contrast)
from utils.numerical_formatting import (INFERENCE, DESCRIPTIVE, format_statistic as fmt,
                                       format_p_value, current_settings)

INTERACTION_TYPES = {**INFERENCE, 'Wald χ²': 'f', 'Variance': 'variance',
                     'Percentage of modeled variance': 'percentage', 'Adjusted p-value': 'p',
                     'Adjusted interaction p-value': 'p', 'Difference of differences': 'coefficient',
                     '95% simultaneous CI lower': 'ci', '95% simultaneous CI upper': 'ci'}


ALL_INTERACTIONS = 'All interactions'
NO_INTERACTION = 'No interaction'


def _interaction_choices_changed():
    state = st.session_state
    values = state['fac_interaction_choices']
    previous = state.get('fac_interaction_previous', [])
    added = [v for v in values if v not in previous]
    if added and added[-1] in (ALL_INTERACTIONS, NO_INTERACTION):
        values = [added[-1]]
    elif added:
        values = [v for v in values if isinstance(v, tuple)]
    if not values:
        values = [NO_INTERACTION]
    state['fac_interaction_choices'] = values
    state['fac_interaction_previous'] = values


def interaction_controls(factors, effects):
    presentation.write('**Interaction Effects**')
    presentation.caption('Interactions help identify whether the effect of one factor depends on another factor.')
    with st.expander('Which interaction effects should I select?'):
        st.write('Explore all interactions is recommended for exploratory analysis. Select specific interactions based on your study objectives or domain knowledge.')
        st.write('Choose No interaction for a model containing only the selected main effects.')
    state = st.session_state
    candidates = selected_pairs(factors, EXPLORE)
    options = [ALL_INTERACTIONS, NO_INTERACTION, *candidates]
    if 'fac_interaction_choices' not in state:
        manual = selected_pairs(factors, MANUAL, state.get('fac_interactions', []))
        state['fac_interaction_choices'] = ([ALL_INTERACTIONS] if state.get('fac_interaction_mode', EXPLORE) == EXPLORE
                                            else manual or [NO_INTERACTION])
    valid = [v for v in state['fac_interaction_choices'] if v in options]
    state['fac_interaction_choices'] = valid or [NO_INTERACTION]
    state['fac_interaction_previous'] = state['fac_interaction_choices']
    choices = st.multiselect('Interaction effects (optional)', options, key='fac_interaction_choices',
        format_func=lambda v: v if isinstance(v, str) else ' × '.join(map(str, v)),
        on_change=_interaction_choices_changed)
    mode = EXPLORE if ALL_INTERACTIONS in choices else MANUAL
    pairs = candidates if mode == EXPLORE else [v for v in choices if isinstance(v, tuple)]
    state['fac_interaction_mode'] = mode
    state['fac_interactions'] = pairs if mode == MANUAL else state.get('fac_interactions', [])
    return pairs, mode


def insights(prepared, result, table, mode, *, random=False, diagnostics=None, ai=False):
    presentation.subheader('Interaction Insights')
    presentation.markdown('**What We Found**')
    if table.empty:
        presentation.write('No interactions were selected; the fitted model retains all main effects.')
        return ''
    presentation.dataframe(table, result_types=INTERACTION_TYPES, alpha=result.get('alpha', .05), hide_index=True)
    presentation.caption(('Exploratory analysis. ' if mode == EXPLORE else 'User-selected interaction family. ') +
        'All terms were fitted together. Holm adjustment covers the fixed-interaction tests in this model, preserving raw p-values. It does not cover later exploration of other models or outcomes. Random variance components receive no p-value adjustment because no variance-component tests are reported.')
    presentation.markdown('**What It Means**')
    statements = []
    for _, row in table.iterrows():
        label = row['Interaction']
        if row['Structure'] == 'Random interaction':
            statements.append(f"{label}: the fitted variance associated with these combinations is {fmt(row['Variance'], 'variance')} "
                              f"({fmt(row['Percentage of modeled variance'], 'percentage')}% of modeled variance in {prepared['outcome']}, conditional on the fixed predictors). "
                              'This is a descriptive estimate; a nonzero value alone does not establish importance. Component confidence intervals and calibrated boundary tests are unavailable, so the assessment is inconclusive.')
        elif row['Assessment'] == 'Evidence of interaction':
            statements.append(f"{label}: the fixed-interaction test supports differences in the relationship with {prepared['outcome']} across factor levels after Holm adjustment. "
                              'Practical importance depends on the response-unit differences, uncertainty and study objective.')
        else:
            statements.append(f"{label}: the available fixed-interaction inference does not establish a clear interaction in {prepared['outcome']}. "
                              'Limited evidence or withheld inference does not prove a constant effect.')
    for statement in statements:
        presentation.write(statement)
    contrasts = []
    if not random:
        for pair in prepared['interactions']:
            try:
                contrasts.append(interaction_contrast(prepared, result, pair))
            except (ValueError, np.linalg.LinAlgError):
                continue
        if contrasts:
            presentation.dataframe(pd.DataFrame(contrasts), result_types=INTERACTION_TYPES, hide_index=True)
            presentation.caption('Direction and magnitude: last minus first observed level of the first factor, compared between last and first levels of the second. Other factors have equal weights. These illustrative contrasts do not summarize every level of an omnibus interaction. Bonferroni simultaneous 95% intervals cover one contrast per selected interaction; partial η² is conditional on the model and is not a percentage of explained variation.')
    presentation.markdown('**What to Do Next**')
    next_steps = ('Review the variance-component chart and combination distributions. Check group identity, replication and residuals in Diagnostics. Compare fixed coefficients only when inference is available. '
                  'Collect additional replicated combinations when uncertainty or sparse coverage limits conclusions.' if random else
                  'Examine Effect plots and adjusted means, then use Post-hoc comparisons for supported simple effects with adjusted comparisons. Interpret main effects alongside interactions. Review Diagnostics before drawing conclusions.')
    presentation.write(next_steps)
    presentation.caption('Associations do not establish causation. No terms are removed automatically. All conclusions remain conditional on model assumptions and the confirmed study design.')
    deterministic = '\n'.join(statements + [next_steps])
    if not ai:
        return deterministic
    # AI sees formatted, finite summaries and authored decisions, never fitted objects.
    from utils.numerical_formatting import display_rows
    context = {'response': str(prepared['outcome']), 'effect_types': prepared.get('effects', dict.fromkeys(prepared.get('factors', []), 'Fixed')),
               'selection_mode': mode, 'model': 'Gaussian REML variance-components model' if random else 'OLS Type II Factorial ANOVA',
               'interactions': list(map(list, prepared['interactions'])),
               'model_structure': {'formula': prepared.get('formula'), 'parents': prepared.get('parents', {}),
                                   'components': prepared.get('components', [])},
               'validated_columns': list(table.columns),
               'validated_results': display_rows(table, INTERACTION_TYPES, alpha=result.get('alpha', .05)),
               'contrast_columns': list(contrasts[0]) if contrasts else [],
               'contrasts': display_rows(pd.DataFrame(contrasts), INTERACTION_TYPES) if contrasts else [],
               'diagnostics': diagnostics or [], 'deterministic_findings': deterministic,
               'variance_columns': list(result['variance'].columns) if random else [],
               'variance_components': display_rows(result['variance'], {**INTERACTION_TYPES, 'Standard deviation': 'sd'}) if random else [],
               'fixed_columns': list(result['fixed'].columns) if random else [],
               'fixed_coefficients': display_rows(result['fixed'], {**INTERACTION_TYPES, 'Estimate': 'coefficient', 'SE': 'se',
                    'Lower CI': 'ci', 'Upper CI': 'ci', 'p-value (asymptotic)': 'p', 'Wald z (asymptotic)': 'z'}) if random else [],
               'analytical_objective': st.session_state.get('fac_objective'),
               'available_follow_up': next_steps, 'limitations': 'No random-component p-values or confidence intervals; mixed fixed tests are asymptotic only.' if random else 'Classical OLS assumptions; Holm family is confined to this fitted model.'}
    digest = hashlib.sha256((json.dumps(context, sort_keys=True, default=str) + repr(current_settings())).encode()).hexdigest()
    with st.expander('🤖 AI Interaction Interpretation'):
        if st.button('Generate interpretation', key='fac_action_ai_generate'):
            try:
                from utils.ai_interpreter import interpret_factorial_interactions
                with st.spinner('Explaining the validated interaction findings…'):
                    text = interpret_factorial_interactions(context)
                st.session_state['fac_interaction_ai'] = {'context': digest, 'text': text}
            except Exception:
                st.session_state.pop('fac_interaction_ai', None)
                presentation.info('AI guidance is unavailable. The deterministic findings above remain available.')
        saved = st.session_state.get('fac_interaction_ai', {})
        if saved.get('context') == digest:
            presentation.write(saved['text'])
            presentation.caption('AI explanation is advisory; calculations, assessments and limitations are determined by the statistical engine.')
            return deterministic + '\n' + saved['text']
    return deterministic


def _confirm_reduction(remaining):
    st.session_state['fac_interaction_mode'] = MANUAL
    st.session_state['fac_interactions'] = remaining
    st.session_state['fac_interaction_choices'] = remaining or [NO_INTERACTION]
    st.session_state.pop('fac_applied_configuration', None)
    st.session_state.pop('fac_interaction_ai', None)
    st.session_state.pop('report_latest', None)


def refinement(prepared, result, *, random=False, context=''):
    pairs = prepared['interactions']
    if not pairs:
        return
    with st.expander('Review interaction terms'):
        st.caption('Removing an interaction assumes its pattern is unnecessary in the revised model. All main effects remain. Refinement after exploring results is exploratory; it does not confirm a predefined hypothesis.')
        if st.session_state.get('fac_refinement_context') != context:
            st.session_state.pop('fac_refinement_remove', None)
            st.session_state['fac_refinement_context'] = context
        removed = st.multiselect('Interactions to consider removing', pairs, key='fac_refinement_remove',
                                 format_func=lambda p: ' × '.join(map(str, p)))
        if not removed:
            return
        remaining = [p for p in pairs if p not in removed]
        if random:
            st.caption('A calibrated random-component comparison is unavailable. Review descriptive variance magnitudes and Diagnostics; no ordinary chi-square likelihood-ratio test or OLS F test is substituted. REML likelihoods with different fixed structures are not compared. Confirm only a scientifically justified revision; it will fit a separate model.')
        else:
            try:
                comparison = compare_fixed_reduction(prepared, result, removed)
                presentation.dataframe(pd.DataFrame([{k: v for k, v in comparison.items() if not isinstance(v, list)}]),
                                       result_types={**INFERENCE, 'DF removed': 'df', 'Full residual DF': 'df', 'Reduced residual DF': 'df'}, hide_index=True)
                presentation.caption('Nested OLS F comparison on identical complete-case observations, assuming independent errors with common variance. This comparison is exploratory after selection and is not a universal elimination rule. Neither statistical significance nor its absence establishes practical importance.')
            except (ValueError, np.linalg.LinAlgError) as exc:
                st.info('Comparison unavailable: ' + str(exc))
        st.button('Confirm removal and refit', key='fac_action_refine_confirm',
                  on_click=_confirm_reduction, args=(remaining,))
