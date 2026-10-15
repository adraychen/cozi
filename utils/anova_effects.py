"""Gaussian random-intercept ANOVA; explicit structures, REML, no boundary tests."""
from itertools import combinations
import hashlib
import warnings
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm
from statsmodels.regression.mixed_linear_model import MixedLM


def model_family(effects):
    random = sum(v == 'Random' for v in effects.values())
    if not random:
        return "Welch's One-way ANOVA" if len(effects) == 1 else 'Fixed-effects Factorial ANOVA'
    if len(effects) == 1:
        return 'One-factor random-effects model'
    return 'Random-effects Factorial Model' if random == len(effects) else 'Mixed-effects Factorial Model'


def sync_effects(previous, factors):
    return {f: previous.get(f, 'Fixed') for f in factors}


def prune_effect_state(state, factors, prefix):
    """Also called when Comparison makes the ANOVA method temporarily unavailable."""
    base = prefix + '_effects'
    previous = state.get(base, {})
    removed = set(previous)-set(factors)
    for factor in removed:
        state.pop(base + '_' + hashlib.sha256(str(factor).encode()).hexdigest()[:16], None)
    if removed:
        state[base] = {f: v for f, v in previous.items() if f in factors}
        for key in list(state):
            if key.startswith(prefix + '_random_'):
                del state[key]


def prepare_random(df, outcome, effects, interactions=(), parents=None):
    factors = list(effects)
    parents = parents or {}
    if not factors or outcome in factors or not df.columns.is_unique:
        raise ValueError('Choose distinct factors and a numerical response with unique column names.')
    if any(v not in ('Fixed', 'Random') for v in effects.values()) or 'Random' not in effects.values():
        raise ValueError('This engine requires at least one Random factor.')
    if not pd.api.types.is_numeric_dtype(df[outcome]) or pd.api.types.is_bool_dtype(df[outcome]):
        raise ValueError('The response must be numerical.')
    raw = df[[outcome] + factors].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(raw)
    if n < 4 or n > 1500:
        raise ValueError('Random/mixed models currently support 4–1,500 complete observations; the upper limit bounds covariance validation memory.')
    if not np.isfinite(raw[outcome].to_numpy(dtype=float)).all():
        raise ValueError('The response contains unsupported nonfinite values.')
    with np.errstate(over='ignore', invalid='ignore'):
        response_variance = np.var(raw[outcome].to_numpy(dtype=float))
    if not np.isfinite(response_variance) or response_variance <= 0:
        raise ValueError('The response must vary on a finite numerical scale. Rescale extreme values before fitting.')
    names = {f: f'f{i}' for i, f in enumerate(factors)}
    data = pd.DataFrame({'y': raw[outcome].to_numpy(dtype=float)})
    for f in factors:
        data[names[f]] = pd.factorize(raw[f], sort=False)[0]
        if raw[f].nunique() < 2:
            raise ValueError(f'{f} needs at least two observed levels.')
    def ancestry(f, trail=()):
        if f in trail:
            raise ValueError('The group ownership answers form a cycle. Review which group belongs to which parent.')
        parent = parents.get(f)
        if parent is None:
            return [f]
        if parent not in factors or effects[f] != 'Random':
            raise ValueError('Only a selected Random factor can have a selected parent.')
        return ancestry(parent, trail + (f,)) + [f]
    fixed = [f'C({names[f]})' for f in factors if effects[f] == 'Fixed']
    components = {}
    for f in factors:
        if effects[f] == 'Random':
            components[str(f)] = ancestry(f)
    for pair in interactions:
        if len(pair) != 2 or len(set(pair)) != 2 or any(f not in effects for f in pair):
            raise ValueError('Interactions must contain two distinct selected factors.')
        if all(effects[f] == 'Fixed' for f in pair):
            fixed.append(':'.join(f'C({names[f]})' for f in pair))
        else:
            components[' × '.join(map(str, pair))] = list(dict.fromkeys(sum((ancestry(f) for f in pair), [])))
    if len(components) > 12:
        raise ValueError('At most 12 random components are supported in one model; reduce the requested structure only when justified by the design.')
    X = dmatrix(' + '.join(fixed) or '1', data, return_type='dataframe')
    if np.linalg.matrix_rank(X) != X.shape[1] or n <= X.shape[1] + len(components):
        raise ValueError('Fixed effects are confounded or leave too little residual information. Remove redundant terms or collect more observations.')
    # Identifiability of covariance components in the REML residual subspace.
    q, _ = np.linalg.qr(np.asarray(X), mode='reduced')
    kernels, vc, groups = [], {}, []
    for i, (label, columns) in enumerate(components.items()):
        codes = pd.factorize(pd.MultiIndex.from_frame(data[[names[c] for c in columns]]))[0]
        counts = pd.Series(codes).value_counts()
        if len(counts) < 2 or len(counts) > 200:
            raise ValueError(f'{label}: supported range is 2–200 groups per variance component.')
        z = np.eye(len(counts))[codes]
        z = z - q @ (q.T @ z)
        kernels.append(z @ z.T)
        data[f'v{i}'] = codes.astype(str)
        vc[f'v{i}'] = '0 + C(v%d)' % i
        groups.append({'Component': label, 'Groups': len(counts), 'Minimum N': int(counts.min()), 'Maximum N': int(counts.max())})
    if sum(g['Groups'] for g in groups) > 500:
        raise ValueError('This model exceeds the current limit of 500 random group coefficients.')
    kernels.append(np.eye(n) - q @ q.T)
    gram = np.array([[np.sum(a*b) for b in kernels] for a in kernels])
    scale = np.sqrt(np.diag(gram))
    if np.any(scale < 1e-8) or np.linalg.matrix_rank(gram / np.outer(scale, scale), tol=1e-8) < len(kernels):
        raise ValueError('Variance components are confounded with each other, fixed effects, or residual variation. Remove redundant interactions/groupings or collect replicated observations. A nested child × parent duplicates the child grouping.')
    cells = raw.groupby(factors, observed=True).size()
    possible = int(np.prod([raw[f].nunique() for f in factors], dtype=object))
    relationships = []
    for a, b in combinations(factors, 2):
        relationships.append({'Factors': f'{a} / {b}', 'Observed pairs': len(raw[[a,b]].drop_duplicates()),
            'Possible pairs': raw[a].nunique()*raw[b].nunique(),
            'Each first label occurs under one second label': bool(raw.groupby(a, observed=True)[b].nunique().le(1).all())})
    return {'data': data, 'raw': raw, 'formula': 'y ~ ' + (' + '.join(fixed) or '1'), 'vc': vc,
            'groups': pd.DataFrame(groups), 'names': names, 'components': list(components),
            'effects': dict(effects), 'parents': dict(parents), 'n': n, 'excluded': len(df)-n,
            'empty_cells': possible-len(cells), 'balanced': cells.nunique() == 1 and len(cells) == possible,
            'relationships': pd.DataFrame(relationships), 'outcome': outcome}


def fit_random(prepared, alpha=.05):
    if not 0 < alpha < 1:
        raise ValueError('Alpha must be between zero and one.')
    data = prepared['data']
    model = MixedLM.from_formula(prepared['formula'], groups=np.ones(len(data)),
                                re_formula='0', vc_formula=prepared['vc'], data=data)
    candidates = []
    for method in ('lbfgs', 'powell'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                candidate = model.fit(reml=True, method=method, maxiter=400, disp=False)
                if candidate.converged and np.isfinite(candidate.llf):
                    candidates.append((candidate, list(caught)))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
    if not candidates:
        raise ValueError('The requested model did not converge to a finite likelihood. Estimates are withheld; review the structure, replication and response scale. No terms were removed.')
    fit, caught = max(candidates, key=lambda item: item[0].llf)
    values = np.r_[fit.vcomp, fit.scale]
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError('The fitted variances are invalid; results are withheld.')
    # statsmodels sorts VC names; map explicitly (v10 sorts before v2).
    labels = dict(zip(prepared['vc'], prepared['components']))
    labels = [labels[name] for name in model.exog_vc.names] + ['Residual (within groups)']
    variance = pd.DataFrame({'Component': labels, 'Variance': values, 'Standard deviation': np.sqrt(values),
                             'Percentage of modeled variance': 100*values/values.sum()})
    notes = list(dict.fromkeys(str(w.message) for w in caught))
    near_zero = bool(np.any(values < values.sum()*1e-6))
    hessian, singular = model.hessian(fit.params_object)
    healthy = not singular and np.isfinite(hessian).all() and np.linalg.eigvalsh(-hessian).min() > 0
    if not healthy:
        raise ValueError('Singular or non-positive-definite information matrix despite optimizer convergence. Variances and inference are withheld; review boundary components, replication and model structure.')
    few = bool(prepared['groups']['Groups'].min() < 5)
    inference = healthy and not near_zero and not few
    if near_zero:
        notes.append('Near-zero variance: boundary/singular fit. Treat variance estimates cautiously; fixed-effect inference is withheld.')
    if not healthy:
        notes.append('The information matrix is not positive definite. Estimates are exploratory; fixed-effect inference is withheld.')
    if few:
        notes.append('Fewer than five groups in a random component: variance estimates are unstable; large-sample fixed-effect inference is withheld.')
    labels_fixed = list(fit.fe_params.index)
    for f, internal in prepared['names'].items():
        levels = list(pd.unique(prepared['raw'][f]))
        for index in range(len(levels)-1, -1, -1):
            labels_fixed = [s.replace(f'C({internal})[T.{index}]', f'{f}[{levels[index]}]') for s in labels_fixed]
    fixed = pd.DataFrame({'Coefficient': labels_fixed, 'Estimate': np.asarray(fit.fe_params)})
    if inference:
        se = np.asarray(fit.bse_fe)
        if not np.isfinite(se).all() or np.any(se <= 0):
            raise ValueError('Fixed-effect covariance is invalid; results are withheld.')
        z = fixed['Estimate'].to_numpy()/se
        fixed['SE'] = se
        fixed['Lower CI'] = fixed['Estimate']-norm.ppf(1-alpha/2)*se
        fixed['Upper CI'] = fixed['Estimate']+norm.ppf(1-alpha/2)*se
        fixed['Wald z (asymptotic)'] = z
        fixed['p-value (asymptotic)'] = 2*norm.sf(abs(z))
    try:
        fitted, residuals = np.asarray(fit.fittedvalues), np.asarray(fit.resid)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError('The singular random-effect covariance prevents reliable fitted values; results are withheld.') from exc
    if not np.isfinite(fitted).all() or not np.isfinite(residuals).all():
        raise ValueError('Nonfinite fitted values or residuals; results are withheld.')
    return {'converged': True, 'information_positive_definite': bool(healthy), 'near_boundary': near_zero, 'variance': variance, 'fixed': fixed, 'total_variance': float(values.sum()),
            'icc': float(values[0]/values.sum()) if len(values) == 2 else None,
            'inference': inference, 'warnings': notes, 'method': 'REML', 'n': prepared['n'],
            'fitted': fitted, 'residuals': residuals}
