import re
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── Confirmed hyperparameter grid ─────────────────────────────────────────────
PARAM_GRID = {
    "model__n_estimators":     [300],
    "model__max_depth":        [10, 15],
    "model__min_samples_split":[5, 10],
    "model__min_samples_leaf": [3, 5],
}
CV_FOLDS          = 5
TOTAL_COMBOS      = 8    # 1 × 2 × 2 × 2
TOTAL_FITS        = TOTAL_COMBOS * CV_FOLDS   # 40


# ── Target type detection ─────────────────────────────────────────────────────

def detect_target_type(series: pd.Series, col_type: str) -> str:
    """
    Return 'numerical', 'binary', or 'multiclass'.
    Binary check runs first — a numerical column with exactly 2 unique values
    (e.g. 0/1) is a classification problem, not regression.
    """
    unique_vals = series.dropna().nunique()
    if unique_vals == 2:
        return "binary"
    if col_type == "Numerical":
        return "numerical"
    return "multiclass"


# ── ID column detection ───────────────────────────────────────────────────────

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
                reasons.append(f"{n_unique / n_rows * 100:.0f}% of values are unique")

        if reasons:
            suspects.append((col, " and ".join(reasons)))

    return suspects


# ── Training time estimate ────────────────────────────────────────────────────

def estimate_training_time(n_rows: int) -> str:
    if n_rows < 1_000:
        return "~1–2 minutes"
    elif n_rows < 10_000:
        return "~3–6 minutes"
    elif n_rows < 50_000:
        return "~8–15 minutes"
    else:
        return "~20–35 minutes"


# ── Model recommendation text ─────────────────────────────────────────────────

def get_model_explanation(target_type: str) -> str:
    if target_type == "numerical":
        return (
            "Your target column contains continuous numbers — this is a **regression problem**.\n\n"
            "**Recommended: Random Forest Regressor**\n"
            "Builds an ensemble of decision trees and averages their predictions. "
            "It handles non-linear relationships, requires no feature scaling, and is "
            "robust to outliers and missing values — making it a strong general-purpose choice.\n\n"
            "**Alternative: Linear Regression** *(coming soon)*\n"
            "Faster and more interpretable, but assumes a straight-line relationship "
            "between predictors and target. Best when that assumption holds."
        )
    elif target_type == "binary":
        return (
            "Your target column has exactly 2 unique values — this is a **binary classification problem**.\n\n"
            "**Recommended: Random Forest Classifier**\n"
            "Handles non-linear decision boundaries well and is robust to imbalanced classes. "
            "No feature scaling required.\n\n"
            "**Alternative: Logistic Regression** *(coming soon)*\n"
            "Simpler and highly interpretable. Works well when the boundary between classes is roughly linear."
        )
    else:
        return (
            "Your target column has multiple categories — this is a **multi-class classification problem**.\n\n"
            "**Recommended: Random Forest Classifier**\n"
            "Natively supports multiple classes without modification. "
            "Handles complex patterns between predictors and categories.\n\n"
            "**Alternative: ANOVA** *(coming soon)*\n"
            "Tests whether each predictor differs significantly across categories. "
            "Best used as a statistical test rather than a predictive model."
        )


# ── Learning curve interpretation ─────────────────────────────────────────────

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


# ── R² interpretation ─────────────────────────────────────────────────────────

def interpret_r2(r2: float) -> str:
    if r2 >= 0.85:
        return f"✅ **Strong fit** (R² = {r2:.2f}) — the model explains {r2*100:.0f}% of the variation in the target column."
    elif r2 >= 0.65:
        return f"ℹ️ **Moderate fit** (R² = {r2:.2f}) — the model captures most patterns but leaves room for improvement."
    else:
        return f"⚠️ **Weak fit** (R² = {r2:.2f}) — consider reviewing your predictor selection or data quality."


# ── Main training function ────────────────────────────────────────────────────

def run_random_forest_regressor(
    df: pd.DataFrame,
    target: str,
    predictors: list,
) -> dict:
    """
    Full pipeline:
    1. 80/20 train-test split
    2. Preprocessing (median imputation for numerical, mode + one-hot for categorical)
    3. GridSearchCV with PARAM_GRID and CV_FOLDS
    4. Test metrics: R², RMSE
    5. CV fold scores (extracted from GridSearchCV — no re-fitting)
    6. Feature importance
    7. Learning curve (4 training-size checkpoints)
    8. SHAP summary (up to 1,000 test rows)
    """
    X = df[predictors].copy()
    y = df[target].copy()

    # Drop rows where target is missing
    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify predictor column types
    numeric_features     = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(transformers)
    pipeline     = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline, PARAM_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model  = grid_search.best_estimator_
    best_index  = grid_search.best_index_

    # Test metrics
    y_pred = best_model.predict(X_test)
    r2     = float(r2_score(y_test, y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # CV fold scores (no re-fitting — extracted from GridSearchCV)
    cv_fold_scores = np.array([
        grid_search.cv_results_[f"split{i}_test_score"][best_index]
        for i in range(CV_FOLDS)
    ])

    # Feature importance
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importances   = best_model.named_steps["model"].feature_importances_
    feat_imp      = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # Learning curve (4 checkpoints, no extra CV fits beyond GridSearch)
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=5, scoring="r2", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 4),
    )

    # SHAP summary (sample up to 1,000 rows for speed)
    preprocessor_fitted = best_model.named_steps["preprocessor"]
    model_fitted        = best_model.named_steps["model"]
    X_test_transformed  = preprocessor_fitted.transform(X_test)
    n_sample            = min(1000, X_test_transformed.shape[0])
    X_sample            = X_test_transformed[:n_sample]

    explainer   = shap.TreeExplainer(model_fitted)
    shap_values = explainer.shap_values(X_sample)

    return {
        "best_params":    grid_search.best_params_,
        "r2":             r2,
        "rmse":           rmse,
        "cv_fold_scores": cv_fold_scores,
        "cv_mean":        float(cv_fold_scores.mean()),
        "cv_std":         float(cv_fold_scores.std()),
        "feat_imp":       feat_imp,
        "train_sizes":    train_sizes,
        "train_scores":   train_scores,
        "val_scores":     val_scores,
        "shap_values":    shap_values,
        "X_sample":       X_sample,
        "feature_names":  feature_names,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
    }


# ── Binary classifier constants ───────────────────────────────────────────────

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


def run_random_forest_classifier(
    df: pd.DataFrame,
    target: str,
    predictors: list,
) -> dict:
    """
    Full binary classification pipeline:
    1. Stratified 80/20 train-test split
    2. Preprocessing (median imputation for numerical, mode + one-hot for categorical)
    3. GridSearchCV with PARAM_GRID_CLASSIFIER, scoring=f1, cv=5
    4. Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve
    5. CV fold scores extracted from GridSearchCV
    6. Feature importance, learning curve (accuracy), SHAP summary (positive class)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, roc_curve,
    )

    X = df[predictors].copy()
    y = df[target].copy()

    mask = y.notna()
    X, y = X[mask], y[mask]

    class_labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    numeric_features     = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(transformers)
    pipeline     = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced",
        )),
    ])

    grid_search = GridSearchCV(
        pipeline, PARAM_GRID_CLASSIFIER, cv=CV_FOLDS, scoring="f1", n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_index = grid_search.best_index_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc   = float(roc_auc_score(y_test, y_prob))
    cm        = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    cv_fold_scores = np.array([
        grid_search.cv_results_[f"split{i}_test_score"][best_index]
        for i in range(CV_FOLDS)
    ])

    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importances   = best_model.named_steps["model"].feature_importances_
    feat_imp      = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=5, scoring="accuracy", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5),
    )

    preprocessor_fitted = best_model.named_steps["preprocessor"]
    model_fitted        = best_model.named_steps["model"]
    X_test_transformed  = preprocessor_fitted.transform(X_test)
    n_sample            = min(1000, X_test_transformed.shape[0])
    X_sample            = X_test_transformed[:n_sample]

    explainer   = shap.TreeExplainer(model_fitted)
    shap_values = explainer.shap_values(X_sample)

    # Handle both SHAP 3-D (>=0.40) and list-of-arrays (older) formats
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
    else:
        shap_vals_class1 = shap_values[:, :, 1]

    return {
        "best_params":       grid_search.best_params_,
        "accuracy":          accuracy,
        "precision":         precision,
        "recall":            recall,
        "f1":                f1,
        "roc_auc":           roc_auc,
        "cm":                cm,
        "fpr":               fpr,
        "tpr":               tpr,
        "cv_fold_scores":    cv_fold_scores,
        "cv_mean":           float(cv_fold_scores.mean()),
        "cv_std":            float(cv_fold_scores.std()),
        "feat_imp":          feat_imp,
        "train_sizes":       train_sizes,
        "train_scores":      train_scores,
        "val_scores":        val_scores,
        "shap_vals_class1":  shap_vals_class1,
        "X_sample":          X_sample,
        "feature_names":     feature_names,
        "class_labels":      class_labels,
        "n_train":           len(X_train),
        "n_test":            len(X_test),
    }
