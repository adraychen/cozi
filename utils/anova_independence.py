"""Structural dependence screening; absence of evidence never proves independence."""
import re
import warnings
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from utils.temporal import parse_temporal

QUALIFICATION = ('No diagnostic can prove independence. Welch and Games–Howell assume independent '
                 'observations within and across groups; study design remains decisive.')


def assess_independence(df, prepared, metadata=None):
    metadata = metadata or {}
    frame = df.reset_index(drop=True).iloc[prepared['clean'].index]
    evidence, serial, needs = [], [], False
    complete_unique_id = False
    for column in frame:
        if column == prepared['outcome']:
            continue
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', str(column)).lower()
        tokens = set(re.split(r'[^a-z0-9]+', name))
        entity = bool(tokens & {'subject', 'participant', 'patient', 'respondent', 'person', 'customer',
                                'student', 'employee', 'animal', 'entity', 'device', 'machine'})
        identifier = entity or bool(tokens & {'id', 'identifier', 'uuid'})
        cluster = bool(tokens & {'batch', 'cluster', 'household', 'family', 'school', 'classroom', 'site', 'clinic', 'facility'})
        if not identifier and not cluster:
            continue
        values = frame[column].dropna()
        repeats = int(values.duplicated(keep=False).sum())
        distinct = int(values.nunique())
        if identifier and repeats:
            cross = int(frame.dropna(subset=[column]).groupby(column, observed=True)[prepared['category']].nunique().gt(1).sum())
            evidence.append({'Column': str(column), 'Indicator': 'Repeated entity/identifier',
                             'Detail': f'{repeats} rows have repeated labels; {cross} identifiers occur across groups. Column meaning needs clarification; labels alone do not prove repeated measurements.'})
            needs = True
        elif identifier and len(values) == len(frame) and len(frame):
            complete_unique_id = True
            evidence.append({'Column': str(column), 'Indicator': 'Unique candidate identifier',
                             'Detail': 'One distinct label per analyzed row. This does not establish that rows are independent entities.'})
        if cluster and repeats and distinct >= 2:
            evidence.append({'Column': str(column), 'Indicator': 'Possible clustering/batch structure',
                             'Detail': f'{distinct} labels recur across {len(values)} rows. Shared labels suggest a structure to review, not proof of dependence.'})
            needs = True
        if identifier and len(values) < len(frame):
            evidence.append({'Column': str(column), 'Indicator': 'Incomplete identifier coverage',
                             'Detail': f'{len(frame)-len(values)} analyzed rows lack an identifier; structural screening is incomplete.'})
    times = []
    for column in frame:
        if column in (prepared['category'], prepared['outcome']):
            continue
        meta = metadata.get(column, {})
        subtype = meta.get('variable_subtype')
        if subtype in ('Date', 'DateTime', 'Year-Month'):
            parsed = parse_temporal(frame[column], subtype)
        elif pd.api.types.is_datetime64_any_dtype(frame[column]):
            parsed = pd.to_datetime(frame[column], errors='coerce', utc=True)
        elif re.search(r'date|timestamp|time', str(column), re.I) and not pd.api.types.is_numeric_dtype(frame[column]):
            parsed = parse_temporal(frame[column], 'DateTime')
        else:
            continue
        if parsed.notna().sum() >= 3:
            times.append((column, parsed))
    if len(times) == 1:
        column, parsed = times[0]
        evidence.append({'Column': str(column), 'Indicator': 'Chronological information',
                         'Detail': 'Diagnostics use chronological order within groups, never arbitrary CSV row order. Time alone does not prove dependence.'})
        time_key = '_cozy_anova_chronology'
        while time_key in frame:
            time_key += '_'
        for label, group in frame.assign(**{time_key: parsed}).groupby(prepared['category'], observed=True, sort=False):
            group = group.sort_values(time_key)
            times_g = group[time_key]
            n = len(group)
            reason = None
            if n < 12 or times_g.isna().any():
                reason = 'At least 12 complete chronological observations are needed for this screen.'
            elif times_g.duplicated().any():
                reason = 'Tied times do not define a unique within-group sequence.'
            else:
                gaps = times_g.diff().dropna().dt.total_seconds().to_numpy()
                if len(gaps) and not np.allclose(gaps, gaps[0], rtol=.01, atol=1e-6):
                    reason = 'Irregular spacing: discrete-lag serial tests were not applied.'
            if reason:
                serial.append({'Group': str(label), 'Time column': str(column), 'N': n, 'Assessment': reason})
                continue
            values = group[prepared['outcome']].to_numpy(dtype=float)
            sd = np.std(values, ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                serial.append({'Group': str(label), 'Time column': str(column), 'N': n, 'Assessment': 'Nonconstant finite residuals are required.'})
                continue
            residual = (values-values.mean())/sd
            lag = min(5, n//5)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                lb = acorr_ljungbox(residual, lags=[lag], model_df=0, return_df=True).iloc[0]
                corr = float(np.corrcoef(residual[:-1], residual[1:])[0,1])
                dw = float(durbin_watson(residual))
            serial.append({'Group': str(label), 'Time column': str(column), 'N': n, 'Lags': lag,
                           'Lag-1 correlation': corr, 'Durbin–Watson': dw,
                           'Ljung–Box p-value': float(lb.lb_pvalue),
                           'Assessment': 'Exploratory serial screen; a nonsignificant result does not establish independence.'})
        tested = [s for s in serial if 'Ljung–Box p-value' in s]
        for row in tested:
            adjusted = min(1., row['Ljung–Box p-value']*len(tested))
            row['Screen-adjusted p-value'] = adjusted
            if np.isfinite(adjusted) and adjusted < .05 and abs(row['Lag-1 correlation']) >= .4:
                row['Assessment'] = 'Potential serial dependence: adjusted p < 0.05 and |lag-1 correlation| ≥ 0.4 (screening rule).'
                evidence.append({'Column': str(column), 'Indicator': 'Potential serial dependence',
                                 'Detail': f'Group {row["Group"]} shows serial structure. Review a repeated-measures, mixed-effects or time-aware model.'})
                needs = True
    elif len(times) > 1:
        evidence.append({'Column': ', '.join(str(c) for c,_ in times), 'Indicator': 'Ambiguous chronological information',
                         'Detail': 'Multiple time columns: no sequence was chosen automatically and no serial test was run.'})
    state = ('Potential dependence detected' if needs else
             'No dependence indicators detected' if complete_unique_id else 'Insufficient information')
    return {'state': state, 'requires_clarification': needs,
            'evidence': pd.DataFrame(evidence, columns=['Column', 'Indicator', 'Detail']),
            'serial': pd.DataFrame(serial), 'qualification': QUALIFICATION}
