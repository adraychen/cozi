from utils.numerical_formatting import format_statistic as fmt
"""Lightweight predictive controls; no sklearn or SHAP imports."""
import re
import numpy as np
import pandas as pd

PARAM_GRID = {
    "model__n_estimators":     [300],
    "model__max_depth":        [10, 15],
    "model__min_samples_split":[5, 10],
    "model__min_samples_leaf": [3, 5],
}

CV_FOLDS          = 5

TOTAL_COMBOS      = 8    # 1 × 2 × 2 × 2

TOTAL_FITS        = TOTAL_COMBOS * CV_FOLDS   # 40

_TARGET_DISCRETE_THRESHOLD = 15

def detect_target_type(series: pd.Series, col_type: str, subtype=None) -> str:
    """Route typed targets: numerical regression, categorical classification,
    or temporal guidance. Numeric class codes require a categorical override.
    """
    from utils.predictive_routing import is_temporal_target
    if is_temporal_target(series, col_type):
        return 'temporal'
    if subtype is not None:
        from utils.variable_metadata import metadata_target_type
        return metadata_target_type(series, col_type, subtype)
    if col_type == "Numerical":
        return "continuous_numerical"
    if series.dropna().nunique() == 2:
        return "binary"
    return "multiclass"

_ID_KEYWORDS = {"id", "key", "index", "code", "uuid", "ref", "pk"}

def detect_id_columns(df: pd.DataFrame, columns: list) -> list:
    """
    Return list of (col_name, reason) for columns that look like identifiers.
    - Name check: keyword must be a whole word in the column name (split by _ - space)
    - Uniqueness check: integer columns only (floats are almost always continuous variables)
    """
    suspects = []
    n_rows   = len(df)

    for col in columns:
        reasons  = []

        # Word-boundary keyword check
        col_words = set(re.split(r"[_\-\s]+", col.lower()))
        col_words.add(col.lower())          # also check the full name
        if any(kw in col_words for kw in _ID_KEYWORDS):
            reasons.append("column name suggests an identifier")

        # Uniqueness check — integers only; floats are continuous by nature
        if pd.api.types.is_integer_dtype(df[col]):
            n_unique = df[col].nunique()
            if n_unique == n_rows:
                reasons.append("every value is unique")
            elif n_rows > 10 and n_unique / n_rows > 0.95:
                reasons.append(f"{fmt(n_unique / n_rows * 100, 'percentage')} of values are unique")

        if reasons:
            suspects.append((col, " and ".join(reasons)))

    return suspects

def estimate_training_time(n_rows: int) -> str:
    if n_rows < 1_000:
        return "~1–2 minutes"
    elif n_rows < 10_000:
        return "~3–6 minutes"
    elif n_rows < 50_000:
        return "~8–15 minutes"
    else:
        return "~20–35 minutes"

def get_model_explanation(target_type: str) -> str:
    if target_type == "continuous_numerical":
        return (
            "Your target column contains continuous numbers — this is a **regression problem**.\n\n"
            "The app will run both **Lasso Regression** and **Random Forest Regressor**, "
            "compare their performance, and automatically identify the better model based on Test R².\n\n"
            "**Lasso Regression** — a linear model that applies regularisation to shrink "
            "less important coefficients toward zero. Fast, interpretable, and effective "
            "when the relationship between predictors and target is roughly linear.\n\n"
            "**Random Forest Regressor** — an ensemble of decision trees that captures "
            "non-linear patterns. More flexible but harder to interpret than Lasso."
        )
    elif target_type == "discrete_numerical":
        return (
            "Your target column contains whole numbers with a limited set of unique values "
            "— this is a **multi-class classification problem**.\n\n"
            "The app will run both **Multinomial Logistic Regression** and **Random Forest Classifier**, "
            "compare their performance, and automatically identify the better model based on weighted Test F1 Score.\n\n"
            "**Multinomial Logistic Regression** — a linear model extended to multiple classes. "
            "Interpretable via per-class coefficients and odds ratios. "
            "Effective when class boundaries are roughly linear.\n\n"
            "**Random Forest Classifier** — captures non-linear boundaries between classes. "
            "More flexible but harder to interpret than Logistic Regression."
        )
    elif target_type == "binary":
        return (
            "Your target column has exactly 2 unique values — this is a "
            "**binary classification problem**.\n\n"
            "The app will run both **Logistic Regression** and **Random Forest Classifier**, "
            "compare their performance, and automatically identify the better model based on Test F1 Score.\n\n"
            "**Logistic Regression** — a linear model that estimates the probability of each class. "
            "Fast, interpretable, and coefficients can be converted to odds ratios for plain-English explanation. "
            "Effective when the boundary between classes is roughly linear.\n\n"
            "**Random Forest Classifier** — an ensemble of decision trees that captures "
            "non-linear boundaries. More flexible but harder to interpret than Logistic Regression."
        )
    else:  # multiclass categorical
        return (
            "Your target column has multiple categories — this is a "
            "**multi-class classification problem**.\n\n"
            "The app will run both **Multinomial Logistic Regression** and **Random Forest Classifier**, "
            "compare their performance, and automatically identify the better model based on weighted Test F1 Score.\n\n"
            "**Multinomial Logistic Regression** — a linear model extended to multiple classes. "
            "Interpretable via per-class coefficients and odds ratios. "
            "Effective when class boundaries are roughly linear.\n\n"
            "**Random Forest Classifier** — captures non-linear boundaries between classes. "
            "More flexible but harder to interpret than Logistic Regression."
        )

def interpret_learning_curve(train_scores: np.ndarray, val_scores: np.ndarray) -> str:
    train_final = train_scores.mean(axis=1)[-1]
    val_final   = val_scores.mean(axis=1)[-1]
    gap         = train_final - val_final

    if val_final < 0.4:
        return (
            "⚠️ **Underfitting** — the model performs poorly on both training and validation data. "
            "Consider adding more predictor columns or reviewing data quality."
        )
    if gap > 0.2:
        return (
            "⚠️ **Overfitting** — the model performs much better on training data than on unseen data. "
            "The current hyperparameters are controlling this, but consider removing noisy predictors."
        )
    if gap > 0.1:
        return (
            "ℹ️ **Mild overfitting** — training score is slightly higher than validation. "
            "This is normal for Random Forest. The model generalises reasonably well."
        )
    return (
        "✅ **Good generalisation** — training and validation scores are close. "
        "The model is learning patterns that apply to unseen data."
    )

def interpret_r2(r2: float) -> str:
    if r2 >= 0.85:
        return f"✅ **Strong fit** (R² = {fmt(r2, 'r2')}) — the model explains {fmt(r2*100, 'percentage')} of the variation in the target column."
    elif r2 >= 0.65:
        return f"ℹ️ **Moderate fit** (R² = {fmt(r2, 'r2')}) — the model captures most patterns but leaves room for improvement."
    else:
        return f"⚠️ **Weak fit** (R² = {fmt(r2, 'r2')}) — consider reviewing your predictor selection or data quality."

PARAM_GRID_CLASSIFIER = {
    "model__n_estimators":     [300],
    "model__max_depth":        [10, 15],
    "model__min_samples_split":[5, 10],
    "model__min_samples_leaf": [3, 5],
    "model__max_features":     ["sqrt", "log2"],
}

TOTAL_COMBOS_CLASSIFIER = 16   # 1×2×2×2×2

TOTAL_FITS_CLASSIFIER   = TOTAL_COMBOS_CLASSIFIER * CV_FOLDS   # 80

def interpret_classifier_f1(f1: float) -> str:
    if f1 >= 0.85:
        return f"✅ **Strong F1 Score** ({f1:.2f}) — the model balances precision and recall well."
    elif f1 >= 0.65:
        return f"ℹ️ **Moderate F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
    return f"⚠️ **Low F1 Score** ({f1:.2f}) — consider reviewing predictor selection or addressing class imbalance."

def interpret_learning_curve_classifier(
    train_scores: np.ndarray,
    val_scores:   np.ndarray,
) -> str:
    train_final = train_scores.mean(axis=1)[-1]
    val_final   = val_scores.mean(axis=1)[-1]
    gap         = train_final - val_final

    if val_final < 0.55:
        return (
            "⚠️ **Underfitting** — the model struggles on both training and validation data. "
            "Consider adding more predictor columns."
        )
    if gap > 0.15:
        return (
            "⚠️ **Overfitting** — training accuracy is much higher than validation. "
            "The model memorises rather than generalises."
        )
    if gap > 0.08:
        return (
            "ℹ️ **Mild overfitting** — slight gap between training and validation. "
            "This is normal for Random Forest."
        )
    return "✅ **Good generalisation** — training and validation accuracy are close and stable."

def interpret_multiclass_f1(f1: float) -> str:
    if f1 >= 0.85:
        return f"✅ **Strong weighted F1 Score** ({f1:.2f}) — the model classifies all categories well."
    elif f1 >= 0.65:
        return f"ℹ️ **Moderate weighted F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
    return f"⚠️ **Low weighted F1 Score** ({f1:.2f}) — some categories may be difficult to distinguish. Review class distribution and predictor selection."

LASSO_PARAM_GRID  = {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}

LASSO_TOTAL_FITS  = len(LASSO_PARAM_GRID["model__alpha"]) * CV_FOLDS   # 25

COMPARISON_TOTAL_FITS = LASSO_TOTAL_FITS + TOTAL_FITS                  # 65

def estimate_training_time_comparison(n_rows: int) -> str:
    """Longer estimate for two-model comparison (65 total fits)."""
    if n_rows < 1_000:
        return "~2–4 minutes"
    elif n_rows < 10_000:
        return "~5–12 minutes"
    elif n_rows < 50_000:
        return "~15–30 minutes"
    else:
        return "~35–60 minutes"

LOGISTIC_PARAM_GRID = {
    "model__penalty": ["l1", "l2"],
    "model__C":       [0.01, 0.1, 1, 10],
}

LOGISTIC_TOTAL_FITS          = 8 * CV_FOLDS                              # 40

BINARY_COMPARISON_TOTAL_FITS = LOGISTIC_TOTAL_FITS + TOTAL_FITS_CLASSIFIER  # 120

def estimate_binary_comparison_time(n_rows: int) -> str:
    """Time estimate for two-model binary comparison (120 total fits)."""
    if n_rows < 1_000:
        return "~3–6 minutes"
    elif n_rows < 10_000:
        return "~8–18 minutes"
    elif n_rows < 50_000:
        return "~20–45 minutes"
    else:
        return "~50–90 minutes"

MULTICLASS_COMPARISON_TOTAL_FITS = LOGISTIC_TOTAL_FITS + TOTAL_FITS_CLASSIFIER  # 120

def estimate_multiclass_comparison_time(n_rows: int) -> str:
    """Time estimate for multiclass comparison (120 fits; saga solver can be slower)."""
    if n_rows < 1_000:
        return "~4–8 minutes"
    elif n_rows < 10_000:
        return "~10–20 minutes"
    elif n_rows < 50_000:
        return "~25–50 minutes"
    else:
        return "~60–100 minutes"
