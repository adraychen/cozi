"""Study-design facts, user confirmations and model eligibility are distinct."""
from tests.factorial_selection_helper import interaction_selection

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from utils.factorial_study_design import (SAME, UNSURE, SEPARATE, SEQUENCE, SHARED,
    detect_relationships, assignment_parent, assignment_language, collection_language,
    time_evidence, check_time_column, collection_requirement, study_context, factor_key,
    exploratory_result)
from utils.anova_effects import prepare_random, fit_random
from utils.guided_diagnostics import diagnostic_status

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = {'Machine': 'Fixed', 'Operator': 'Random'}


def productivity():
    return pd.read_csv(ROOT / 'data/samples/Productivity.csv')


class StudyDesignEvidenceTests(unittest.TestCase):
    def test_productivity_observable_facts_not_measurement_process(self):
        observed = detect_relationships(productivity(), 'Score', EFFECTS)
        self.assertEqual(observed['n'], 54)
        self.assertEqual(observed['levels'], {'Machine': 3, 'Operator': 6})
        self.assertEqual(observed['observed_cells'], 18)
        self.assertEqual((observed['minimum_n'], observed['maximum_n']), (3, 3))
        relation = observed['relationships'][0]
        self.assertEqual(relation['suggestion'], SAME)
        self.assertEqual(relation['comparisons'][0]['labels_across_groups'], 6)
        self.assertEqual(relation['comparisons'][0]['empty_pairs'], 0)
        self.assertNotIn('collection', observed)
        self.assertEqual(time_evidence(productivity(), ['Score', *EFFECTS]), [])
        nullable = productivity()
        nullable['Score'] = nullable.Score.astype('Float64')
        nullable.loc[0, 'Score'] = pd.NA
        self.assertEqual(detect_relationships(nullable, 'Score', EFFECTS)['n'], 53)

    def test_nested_unique_labels_suggest_but_do_not_confirm_ownership(self):
        df = productivity()
        df['Operator'] = df.Machine + '-' + df.Operator.astype(str)
        relation = detect_relationships(df, 'Score', EFFECTS)['relationships'][0]
        self.assertEqual(relation['levels'], 18)
        self.assertEqual(relation['suggestion'], 'within:0')
        self.assertEqual(relation['comparisons'][0]['pattern'], 'labels confined to one group')
        self.assertEqual(assignment_parent('within:0', 'Operator', EFFECTS), 'Machine')
        self.assertIsNone(assignment_parent(SAME, 'Operator', EFFECTS))
        with self.assertRaises(ValueError):
            assignment_parent(UNSURE, 'Operator', EFFECTS)

    def test_reused_local_labels_are_observationally_indistinguishable(self):
        # Same table can mean one person working across machines OR local IDs.
        df = productivity()
        evidence = detect_relationships(df, 'Score', EFFECTS)
        self.assertEqual(evidence['relationships'][0]['suggestion'], SAME)
        crossed = prepare_random(df, 'Score', EFFECTS)
        nested = prepare_random(df, 'Score', EFFECTS, parents={'Operator':'Machine'})
        self.assertEqual(crossed['groups'].Groups.tolist(), [6])
        self.assertEqual(nested['groups'].Groups.tolist(), [18])
        with self.assertRaisesRegex(ValueError, 'confounded'):
            prepare_random(df, 'Score', EFFECTS, [('Machine','Operator')], {'Operator':'Machine'})

    def test_partial_crossing_missing_and_sparse_pairs(self):
        df = productivity()
        partial = df.loc[~((df.Machine == 'C') & (df.Operator == 6))]
        relation = detect_relationships(partial, 'Score', EFFECTS)['relationships'][0]
        self.assertIsNone(relation['suggestion'])
        self.assertEqual(relation['comparisons'][0]['pattern'], 'partially crossed labels')
        self.assertEqual(relation['comparisons'][0]['empty_pairs'], 1)
        prepared = prepare_random(partial, 'Score', EFFECTS, [('Machine','Operator')])
        self.assertEqual(prepared['groups'].Groups.tolist(), [6, 17])
        self.assertEqual(prepared['covariance_rank'], 3)
        sparse = df.loc[~((df.Machine == 'A') & (df.Operator == 1)) | (df.index == 0)]
        self.assertEqual(detect_relationships(sparse, 'Score', EFFECTS)['singleton_cells'], 1)

    def test_multiple_possible_owners_are_ambiguous(self):
        df = productivity()
        df['Site'] = df.Machine
        df['Operator'] = df.Machine + '-' + df.Operator.astype(str)
        relation = detect_relationships(df, 'Score', {**EFFECTS, 'Site':'Fixed'})['relationships'][0]
        self.assertIsNone(relation['suggestion'])

    def test_time_evidence_does_not_use_csv_order(self):
        df = productivity().assign(Sequence=np.arange(54), VisitTime=['2026-01-01', '2026-01-02']*27,
                                   EmptyTime=None)
        detected = {row['Column']: row for row in time_evidence(df, ['Score', *EFFECTS])}
        self.assertTrue(detected['Sequence']['Usable'])
        self.assertTrue(detected['VisitTime']['Usable'])
        self.assertFalse(detected['EmptyTime']['Usable'])
        self.assertFalse(check_time_column(pd.Series([1, 1, None]))['Usable'])
        self.assertEqual(time_evidence(productivity().iloc[::-1], ['Score', *EFFECTS]), [])

    def test_collection_mapping_and_unconfirmed_assumption(self):
        self.assertEqual(collection_requirement(SEPARATE), {'fit':True, 'inference':True, 'severity':'ok'})
        self.assertEqual(collection_requirement(UNSURE), {'fit':True, 'inference':False, 'severity':'unresolved'})
        self.assertEqual(collection_requirement(SEQUENCE, 'yes')['severity'], 'blocking')
        self.assertFalse(collection_requirement(SEQUENCE)['fit'])
        self.assertTrue(collection_requirement(SEQUENCE, 'no')['inference'])
        self.assertFalse(collection_requirement(SHARED)['fit'])
        self.assertTrue(collection_requirement(SHARED, shared_confirmed=True)['fit'])

    def test_reusable_context_and_language(self):
        q, _, labels = assignment_language('Operator', EFFECTS)
        self.assertEqual(q, 'How are operators assigned to machines?')
        self.assertEqual(labels[SAME], 'The same operators work on different machines.')
        self.assertIn('technicians', assignment_language('Technician', {'Technician':'Random','Production Line':'Fixed'})[0])
        self.assertIn('Separate visits', collection_language(['Patient','Hospital'])[0][SEPARATE])
        first = study_context(productivity(), 'Score', EFFECTS, [('Machine','Operator')], {}, 'one')
        self.assertEqual(first, study_context(productivity(), 'Score', EFFECTS, [], {}, 'one'))
        self.assertNotEqual(first, study_context(productivity(), 'Score', EFFECTS, [('Machine','Operator')], {}, 'two'))

    def test_exploratory_copy_withholds_inference_without_mutating_fit(self):
        result = fit_random(prepare_random(productivity(), 'Score', EFFECTS, [('Machine','Operator')]))
        self.assertTrue(result['inference'])
        exploratory = exploratory_result(result)
        self.assertFalse(exploratory['inference'])
        self.assertEqual(list(exploratory['fixed']), ['Coefficient','Estimate'])
        self.assertNotIn('p-value', str(exploratory['fixed_interactions']))
        self.assertIn('p-value (asymptotic)', result['fixed'])


SCRIPT = '''
import streamlit as st
from tests.test_factorial_study_design import productivity
from utils.factorial_anova_ui import render_factorial_anova
df = productivity()
if st.session_state.get('include_batch'):
    df['Batch'] = [str(i % 3) for i in range(len(df))]
    # A synthetic batch fixture with a real estimable shared-source effect;
    # the unmodified CSV remains the reference in all Productivity tests.
    df['Score'] += df.Batch.map({'0':-5., '1':0., '2':5.})
if st.session_state.get('include_time'):
    df['Sequence'] = range(len(df))
if st.session_state.get('change_data'):
    df.loc[0, 'Score'] += .1
st.session_state.setdefault('fac_effects_context', (st.session_state.get('file_id'), 'Score'))
st.session_state.setdefault('fac_effects', {'Machine':'Fixed','Operator':'Random'})
factors = ['Machine','Operator']
if st.session_state.get('remove_machine'):
    df['Site'] = df.Machine
    factors = ['Site','Operator']
render_factorial_anova(df, ['Score'], factors)
'''


class StudyDesignUITests(unittest.TestCase):
    def app(self, **state):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string(SCRIPT)
        for k, v in state.items():
            app.session_state[k] = v
        app.run(timeout=60)
        self.assertFalse(app.exception)
        return app

    def confirm(self, app):
        app.button(key='fac_action_design_confirm_' + factor_key('Operator')).click().run(timeout=60)
        self.assertFalse(app.exception)

    def collect(self, app, value):
        app.radio(key='fac_design_collection').set_value(value).run(timeout=60)
        self.assertFalse(app.exception)

    def test_productivity_progressive_confirmation_and_mapping(self):
        with patch('utils.anova_effects_ui._fit') as fit:
            app = self.app()
            fit.assert_not_called()
            assignment = app.selectbox(key='fac_design_assignment_' + factor_key('Operator'))
            self.assertEqual(assignment.value, SAME)
            self.assertFalse(any(r.key == 'fac_design_collection' for r in app.radio))
            self.assertNotIn('report_latest', app.session_state)
        self.confirm(app)
        self.assertIsNone(app.radio(key='fac_design_collection').value)
        self.collect(app, SEPARATE)
        settings = app.session_state['report_latest']['settings']
        self.assertEqual(settings['effects'], EFFECTS)
        self.assertEqual(settings['parents'], {})
        self.assertEqual(settings['interactions'], [['Machine','Operator']])
        self.assertTrue(settings['study_design']['assignments'][0]['confirmed'])
        self.assertEqual(settings['study_design']['collection'], SEPARATE)
        self.assertTrue(any('54 usable observations' in m.value for m in app.markdown))
        app.run(timeout=60)
        self.assertEqual(app.radio(key='fac_design_collection').value, SEPARATE)

    def test_unknown_collection_exploration_and_cached_assumption_change(self):
        from utils.anova_effects_ui import _fit
        _fit.clear()
        with patch('utils.anova_effects_ui.fit_random', wraps=fit_random) as fitting:
            app = self.app()
            self.confirm(app)
            self.collect(app, UNSURE)
            self.assertEqual(fitting.call_count, 0)
            self.assertFalse(app.tabs)
            self.assertNotIn('report_latest', app.session_state)
            self.collect(app, SEPARATE)
            app.button(key='report_add_fac_random').click().run(timeout=60)
            snapshot = deepcopy(app.session_state['cozy_report']['blocks'])
            self.collect(app, UNSURE)
            self.assertFalse(app.tabs)
            self.collect(app, SEPARATE)
            self.assertEqual(fitting.call_count, 1, 'An assumption-only change must reuse the fit')
            self.assertTrue(any('p-value (asymptotic)' in t.value for t in app.dataframe))
            self.assertEqual(app.session_state['cozy_report']['blocks'], snapshot)
            self.assertIn('separate', str(snapshot))

    def test_unknown_identity_never_fits_and_changed_identity_needs_refit(self):
        app = self.app()
        key = 'fac_design_assignment_' + factor_key('Operator')
        with patch('utils.anova_effects_ui._fit') as fit:
            app.selectbox(key=key).select(UNSURE).run(timeout=60)
            fit.assert_not_called()
            self.assertNotIn('report_latest', app.session_state)
            self.assertFalse(app.tabs)
        app.selectbox(key=key).select(SAME).run(timeout=60)
        self.confirm(app)
        self.collect(app, SEPARATE)
        app.button(key='report_add_fac_random').click().run(timeout=60)
        snapshot = deepcopy(app.session_state['cozy_report']['blocks'])
        app.selectbox(key=key).select('within:0').run(timeout=60)
        self.assertNotIn('report_latest', app.session_state)
        self.confirm(app)
        self.assertFalse(app.exception)
        self.assertTrue(app.tabs[0].error)
        self.assertTrue(any('confounded' in m.value for m in app.tabs[1].markdown))
        self.assertEqual(app.session_state['cozy_report']['blocks'], snapshot)

    def test_sequential_missing_order_column_and_known_dependence(self):
        app = self.app()
        self.confirm(app)
        self.collect(app, SEQUENCE)
        self.assertEqual(app.selectbox(key='fac_design_time_column').options[0], 'No known time or sequence column')
        app.radio(key='fac_design_sequence_connection').set_value('yes').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any(e.label == 'Model issue' for e in app.expander))
        self.assertNotIn('report_latest', app.session_state)
        self.assertTrue(any('Record the measurement order' in m.value for m in app.markdown))
        self.assertFalse(any('fac_design_shared_column' == s.key for s in app.selectbox))

    def test_sequential_with_time_still_requires_process_knowledge(self):
        app = self.app(include_time=True)
        self.confirm(app)
        self.collect(app, SEQUENCE)
        app.selectbox(key='fac_design_time_column').select('Sequence').run(timeout=60)
        app.radio(key='fac_design_sequence_connection').set_value(UNSURE).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs)
        self.assertNotIn('report_latest', app.session_state)
        self.assertTrue(any(e.label == 'Review advised' for e in app.expander))

    def test_shared_group_requires_confirmation_and_existing_engine(self):
        app = self.app(include_batch=True)
        self.confirm(app)
        self.collect(app, SHARED)
        app.selectbox(key='fac_design_shared_column').select('Batch').run(timeout=60)
        self.assertNotIn('report_latest', app.session_state)
        app.checkbox(key='fac_design_shared_identity').check().run(timeout=60)
        self.assertNotIn('report_latest', app.session_state)
        app.button(key='fac_action_design_add_group').click().run(timeout=60)
        self.assertFalse(app.exception)
        settings = app.session_state['report_latest']['settings']
        self.assertEqual(settings['selected_effects'], EFFECTS)
        self.assertEqual(settings['effects']['Batch'], 'Random')
        self.assertEqual(settings['interactions'], [['Machine','Operator']])
        self.assertTrue(settings['study_design']['shared_identity_confirmed'])

    def test_shared_fixed_group_is_not_silently_ignored(self):
        app = self.app()
        self.confirm(app)
        self.collect(app, SHARED)
        app.selectbox(key='fac_design_shared_column').select('Machine').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any(e.label == 'Model issue' for e in app.expander))
        self.assertNotIn('report_latest', app.session_state)
        self.assertTrue(any('Fixed mean differences' in m.value for m in app.markdown))

    def test_data_dataset_and_factor_changes_clear_confirmations(self):
        app = self.app()
        self.confirm(app)
        self.collect(app, SEPARATE)
        before = app.session_state['fac_design_context']
        app.session_state['change_data'] = True
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotEqual(before, app.session_state['fac_design_context'])
        self.assertNotIn('report_latest', app.session_state)
        self.assertFalse(any(r.key == 'fac_design_collection' for r in app.radio))
        self.confirm(app)
        self.collect(app, SEPARATE)
        app.session_state['file_id'] = 'unrelated'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotIn('fac_design_confirmed_' + factor_key('Operator'), app.session_state)

    def test_guidance_does_not_depend_on_ai_availability(self):
        with patch('utils.ai_interpreter._get_llm', side_effect=RuntimeError('offline')) as llm:
            app = self.app()
            self.confirm(app)
            self.collect(app, SEPARATE)
            self.assertIn('report_latest', app.session_state)
            llm.assert_not_called()

    def test_factor_change_invalidates_identity_before_model_apply(self):
        app = self.app()
        self.confirm(app)
        self.collect(app, SEPARATE)
        app.session_state['remove_machine'] = True
        app.run(timeout=60)
        self.assertNotIn('report_latest', app.session_state)
        self.assertNotIn('fac_design_confirmed_' + factor_key('Operator'), app.session_state)
        self.assertFalse(app.exception)
        self.assertFalse(any(r.key == 'fac_design_collection' for r in app.radio))

    def test_confirmed_nested_group_conflicts_with_global_shared_identity(self):
        app = self.app()
        from utils.factorial_interactions_ui import NO_INTERACTION
        interaction_selection(app).set_value([NO_INTERACTION]).run(timeout=60)
        key = 'fac_design_assignment_' + factor_key('Operator')
        app.selectbox(key=key).select('within:0').run(timeout=60)
        self.confirm(app)
        self.collect(app, SHARED)
        app.selectbox(key='fac_design_shared_column').select('Operator').run(timeout=60)
        app.checkbox(key='fac_design_shared_identity').check().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any(e.label == 'Model issue' for e in app.expander))
        self.assertTrue(any('cannot both configure' in m.value for m in app.markdown))
        self.assertNotIn('report_latest', app.session_state)


if __name__ == '__main__':
    unittest.main()
