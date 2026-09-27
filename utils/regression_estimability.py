"""Actionable, non-mutating diagnostics for an unidentifiable OLS design."""
import numpy as np
from utils.regression_design import dependencies, label


class RegressionDesignError(ValueError):
    def __init__(self, message, details, n, coefficients, rank):
        super().__init__(message)
        self.details = details
        self.n = n
        self.coefficients = coefficients
        self.rank = rank


def explain_singularity(x, names, indices, active):
    """Propose one hierarchy-preserving full-rank subset on the same rows.

    Column scaling makes dependency screening less sensitive to physical units.
    Whole model terms are kept or rejected, never individual dummy coefficients.
    This is a structural diagnostic, not statistical variable selection.
    """
    magnitudes = np.max(np.abs(x), axis=0)
    scaled = x / np.where(magnitudes > 0, magnitudes, 1)
    norms = np.linalg.norm(scaled, axis=0)
    scaled /= np.where(norms > 0, norms, 1)
    rank = int(np.linalg.matrix_rank(scaled))
    rows = []
    if rank == x.shape[1]:
        numeric_columns = sorted(range(1, x.shape[1]), key=lambda i: magnitudes[i], reverse=True)
        for i in numeric_columns[:5]:
            rows.append({'Term': names[i], 'Issue': f'Largest absolute generated value: {magnitudes[i]:.4g}',
                         'Suggested action': 'Rescale the numerical predictor(s) used in this term, then retry.'})
        return RegressionDesignError(
            'The model is numerically singular at the current scales, but no exact redundancy was detected after column scaling. Rescale numerical predictors (for example, dollars to thousands of dollars) before removing terms.',
            rows, len(x), x.shape[1], rank)

    kept_columns, kept_blocks = [0], set()
    # Favor main effects, then lower powers, then interactions. This preserves
    # hierarchy and provides a reproducible suggestion, not a unique solution.
    ordered = sorted(active, key=lambda b: (0 if b[0] == 'main' else 1 if b[0] == 'power' else 2,
                                           b[2] if b[0] == 'power' else 0))
    for block in ordered:
        columns = indices[block]
        missing = dependencies(block) - kept_blocks
        candidate = kept_columns + columns
        if missing:
            issue = 'Requires another term suggested for removal: ' + ', '.join(sorted(label(b) for b in missing))
        elif np.linalg.matrix_rank(scaled[:, candidate]) == len(candidate):
            kept_columns = candidate
            kept_blocks.add(block)
            continue
        else:
            zero = [names[i] for i in columns if magnitudes[i] == 0]
            constant = [names[i] for i in columns if np.ptp(x[:, i]) == 0 and magnitudes[i] != 0]
            if zero:
                issue = 'Generated columns contain only zeros on complete cases: ' + '; '.join(zero)
            elif constant:
                issue = 'Generated columns are constant on complete cases: ' + '; '.join(constant)
            else:
                # Show the involved basis columns without claiming a unique cause.
                basis = scaled[:, kept_columns]
                related = set()
                for i in columns:
                    weights = np.linalg.lstsq(basis, scaled[:, i], rcond=None)[0]
                    if np.linalg.norm(basis @ weights - scaled[:, i]) < 1e-10:
                        related.update(names[j] for j, w in zip(kept_columns, weights) if abs(w) > 1e-8)
                issue = ('Overlaps with retained columns: ' + '; '.join(sorted(related)) if related else
                         'Some generated coefficients cannot be estimated independently of the retained terms.')
        action = ('Uncheck this predictor under Predictors (X), along with its interactions and polynomial terms.' if block[0] == 'main' else
                  'Uncheck this term under Interaction terms.' if block[0] == 'interaction' else
                  'Uncheck this polynomial term and any higher powers of the same predictor.')
        rows.append({'Term': label(block), 'Issue': issue, 'Suggested action': action})
    return RegressionDesignError(
        'Some selected terms cannot be estimated separately from these data. The table identifies one possible set of terms to remove while retaining main effects where possible.',
        rows, len(x), x.shape[1], rank)
