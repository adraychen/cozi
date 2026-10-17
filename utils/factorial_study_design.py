"""Observable study-design evidence and explicit mappings, without fitting."""
import hashlib
import re
from math import prod
import numpy as np
import pandas as pd
from utils.factorial_guidance import analysis_context
from utils.guided_diagnostics import Diagnostic

SAME = 'same'
UNSURE = 'unknown'
SEPARATE = 'separate'
SEQUENCE = 'sequence'
SHARED = 'shared'


def study_context(df, outcome, effects, interactions, metadata, dataset_id):
    # Alpha/formatting/selection mode do not change who or what was measured.
    # Identity and collection facts survive interaction/alpha changes. The
    # statistical engine still validates and refits each new term structure.
    return analysis_context(df, outcome, list(effects), (), effects, None,
                            {'metadata': metadata, 'dataset_id': dataset_id})


def factor_key(factor):
    return hashlib.sha256(str(factor).encode()).hexdigest()[:12]


def sync_study_state(state, context):
    """Invalidate before pending-model/unsupported-selection early returns too."""
    if state.get('fac_design_context') != context:
        for key in list(state):
            if key.startswith('fac_design_'):
                del state[key]
        state['fac_design_context'] = context
        state.pop('fac_interaction_ai', None)
        state.pop('report_latest', None)


def clean_design(df, outcome, factors):
    selected = df[[outcome, *factors]]
    mask = selected.notna().all(axis=1) & np.isfinite(selected[outcome].to_numpy(dtype=float, na_value=np.nan))
    return df.loc[mask]


def detect_relationships(df, outcome, effects):
    raw = clean_design(df, outcome, list(effects))
    levels = {f: int(raw[f].nunique()) for f in effects}
    cells = raw.groupby(list(effects), observed=True).size()
    relationships = []
    for factor, effect in effects.items():
        if effect != 'Random':
            continue
        comparisons = []
        for other in effects:
            if other == factor:
                continue
            counts = raw.groupby([factor, other], observed=True).size()
            coverage = raw.groupby(factor, observed=True)[other].nunique()
            possible = levels[factor] * levels[other]
            full = possible > 0 and len(counts) == possible
            confined = len(coverage) > 0 and coverage.le(1).all() and levels[other] >= 2
            pattern = ('fully crossed labels' if full else 'labels confined to one group' if confined
                       else 'partially crossed labels' if coverage.gt(1).any() else 'ambiguous')
            comparisons.append({'other': other, 'levels': levels[other], 'observed_pairs': len(counts),
                'possible_pairs': possible, 'empty_pairs': possible-len(counts),
                'singleton_pairs': int(counts.eq(1).sum()),
                'minimum_n': int(counts.min()) if len(counts) else 0,
                'maximum_n': int(counts.max()) if len(counts) else 0,
                'labels_across_groups': int(coverage.gt(1).sum()), 'pattern': pattern})
        confined_to = [c['other'] for c in comparisons if c['pattern'] == 'labels confined to one group']
        suggestion = None
        if levels[factor] >= 2 and comparisons:
            if all(c['pattern'] == 'fully crossed labels' and c['levels'] >= 2 for c in comparisons):
                suggestion = SAME
            elif len(confined_to) == 1 and all(c['pattern'] in ('labels confined to one group', 'fully crossed labels')
                                               for c in comparisons):
                suggestion = 'within:' + str(list(effects).index(confined_to[0]))
        relationships.append({'factor': factor, 'levels': levels[factor], 'comparisons': comparisons,
                              'suggestion': suggestion})
    return {'n': len(raw), 'excluded': len(df)-len(raw), 'levels': levels,
            'observed_cells': len(cells), 'possible_cells': prod(levels.values()),
            'minimum_n': int(cells.min()) if len(cells) else 0,
            'maximum_n': int(cells.max()) if len(cells) else 0,
            'singleton_cells': int(cells.eq(1).sum()), 'relationships': relationships}


def assignment_parent(choice, factor, effects):
    if choice == SAME:
        return None
    if not isinstance(choice, str) or not choice.startswith('within:'):
        raise ValueError('Confirm the assignment before selecting a model structure.')
    try:
        parent = list(effects)[int(choice.split(':', 1)[1])]
    except (ValueError, IndexError) as exc:
        raise ValueError('The assignment no longer matches the selected factors.') from exc
    if parent == factor or effects[factor] != 'Random':
        raise ValueError('Only a random factor can be assigned to a different selected factor.')
    return parent


_NOUNS = {'operator': ('operators', 'person'), 'machine': ('machines', 'machine'),
    'technician': ('technicians', 'person'), 'production line': ('production lines', 'line'),
    'inspector': ('inspectors', 'person'), 'laboratory': ('laboratories', 'laboratory'),
    'employee': ('employees', 'person'), 'department': ('departments', 'department'),
    'patient': ('patients', 'person'), 'hospital': ('hospitals', 'hospital'),
    'student': ('students', 'person'), 'teacher': ('teachers', 'person'),
    'classroom': ('classrooms', 'classroom'), 'store': ('stores', 'store'),
    'region': ('regions', 'region'), 'batch': ('batches', 'batch')}


def assignment_language(factor, effects):
    """Business nouns when known; precise quoted column names otherwise."""
    others = [f for f in effects if f != factor]
    noun = _NOUNS.get(str(factor).casefold())
    simple = len(others) == 1 and noun and str(others[0]).casefold() in _NOUNS
    options = [SAME] + ['within:' + str(list(effects).index(f)) for f in others] + [UNSURE]
    if simple:
        other = others[0]
        plural = _NOUNS[str(other).casefold()][0]
        question = f'How are {noun[0]} assigned to {plural}?'
        same = f'The same {noun[0]} work on different {plural}.' if noun[1] == 'person' and str(other).casefold() in ('machine', 'production line') else f'The same {noun[0]} appear across different {plural}.'
        labels = {SAME: same, options[1]: f'Each {str(other).lower()} has its own {noun[0]}.', UNSURE: "I'm not sure."}
    else:
        question = f'How does {factor} relate to the other selected factors?'
        labels = {SAME: f'The same {factor} labels identify the same people or units across the selected factors.',
                  UNSURE: "I'm not sure."}
        labels.update({'within:' + str(list(effects).index(f)):
                       f'Each {f} has its own {factor} people or units.' for f in others})
    return question, options, labels


def collection_language(factors):
    words = ' '.join(map(str, factors)).casefold()
    if any(w in words for w in ('machine', 'operator', 'production', 'factory')):
        separate, unit, examples = 'Separate production runs', 'run', 'machine warm-up, operator fatigue, or changing conditions'
        shared = 'production batch, material lot, or production session'
    elif any(w in words for w in ('patient', 'hospital', 'visit')):
        separate, unit, examples = 'Separate visits or measurements', 'visit or measurement', 'changes in treatment or health over time'
        shared = 'patient, treatment session, or clinic visit'
    elif any(w in words for w in ('student', 'teacher', 'classroom', 'school')):
        separate, unit, examples = 'Separate observations', 'observation', 'learning, fatigue, or changing conditions over time'
        shared = 'student, classroom session, or assessment occasion'
    else:
        separate, unit, examples = 'Separate measurements', 'measurement', 'changing conditions over time'
        shared = 'session, shared source, or reporting period'
    labels = {SEPARATE: separate, SEQUENCE: 'Measurements taken in sequence or over time',
              SHARED: 'Measurements share another common source', UNSURE: "I'm not sure"}
    descriptions = {SEPARATE: f'Each measurement comes from a separate {unit}. No additional relationship is expected beyond the selected factors.',
                    SEQUENCE: f'Measurements taken one after another may be affected by {examples}.',
                    SHARED: f'For example, several measurements come from the same {shared}.',
                    UNSURE: 'Cozy cannot determine how the measurements were collected from the dataset alone. Clarify the collection process to generate the model summary and diagnostics.'}
    return labels, descriptions


def time_evidence(df, excluded=(), metadata=None):
    rows = []
    for col in df:
        if col in excluded:
            continue
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', str(col)).casefold()
        subtype = (metadata or {}).get(col, {}).get('variable_subtype')
        temporal = pd.api.types.is_datetime64_any_dtype(df[col]) or subtype in ('Date', 'DateTime', 'Year-Month')
        named = bool(set(re.split(r'[^a-z0-9]+', name)) & {'time', 'date', 'timestamp', 'sequence', 'order', 'run', 'visit'})
        if not temporal and not named:
            continue
        rows.append({'Column': col, **check_time_column(df[col])})
    return rows


def check_time_column(series):
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    elif pd.api.types.is_datetime64_any_dtype(series):
        values = series
    else:
        values = pd.to_datetime(series, errors='coerce', format='mixed')
    usable = values.notna().sum() >= 2 and values.nunique() >= 2
    return {'Usable': bool(usable), 'Missing or invalid': int(values.isna().sum()), 'Distinct values': int(values.nunique())}


def collection_requirement(choice, sequence_connection=None, shared_confirmed=False):
    """Eligibility is a user assumption; never proof of residual independence."""
    if choice == SEPARATE or choice == SHARED and shared_confirmed or choice == SEQUENCE and sequence_connection == 'no':
        return {'fit': True, 'inference': True, 'severity': 'ok'}
    if choice == UNSURE or choice == SEQUENCE and sequence_connection == UNSURE:
        return {'fit': True, 'inference': False, 'severity': 'unresolved'}
    if choice == SEQUENCE and sequence_connection == 'yes':
        return {'fit': False, 'inference': False, 'severity': 'blocking'}
    return {'fit': False, 'inference': False, 'severity': 'unresolved'}


def design_diagnostic(key, severity, title, found, why, action):
    return Diagnostic(key, 'study_design', severity, title, found, why, action)


def exploratory_result(result):
    """Reuse estimates but withhold model-based inferential conclusions."""
    return {**result, 'inference': False, 'fixed_interactions': {},
            'fixed': result['fixed'][['Coefficient', 'Estimate']].copy(),
            'study_design_inference': 'Withheld: measurement collection is unconfirmed.'}
