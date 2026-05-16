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


# Threshold for discrete vs continuous numerical target.
# Consistent with DISCRETE_THRESHOLD in utils/charts.py.
_TARGET_DISCRETE_THRESHOLD = 15


def detect_target_type(series: pd.Series, col_type: str) -> str:
    """
    Return one of four target types:
    - 'binary'              : exactly 2 unique values (any dtype)
    - 'discrete_numerical'  : numerical, 3–14 unique values → treat as multi-class
    - 'continuous_numerical': numerical, 15+ unique values → treat as regression
    - 'multiclass'          : categorical, 3+ unique values
    """
    unique_vals = series.dropna().nunique()
    if unique_vals == 2:
        return "binary"
    if col_type == "Numerical":
        return (
            "discrete_numerical"
            if unique_vals < _TARGET_DISCRETE_THRESHOLD
            else "continuous_numerical"
        )
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


# ── Multi-class classifier ────────────────────────────────────────────────────

def interpret_multiclass_f1(f1: float) -> str:
    if f1 >= 0.85:
        return f"✅ **Strong weighted F1 Score** ({f1:.2f}) — the model classifies all categories well."
    elif f1 >= 0.65:
        return f"ℹ️ **Moderate weighted F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
    return f"⚠️ **Low weighted F1 Score** ({f1:.2f}) — some categories may be difficult to distinguish. Review class distribution and predictor selection."


def run_random_forest_multiclass_classifier(
    df: pd.DataFrame,
    target: str,
    predictors: list,
) -> dict:
    """
    Full multi-class classification pipeline:
    1. Stratified 80/20 train-test split
    2. Preprocessing (median imputation for numerical, mode + one-hot for categorical)
    3. GridSearchCV with PARAM_GRID_CLASSIFIER, scoring=f1_weighted, cv=5
    4. Metrics: accuracy, precision/recall/F1 (weighted), ROC-AUC (OvR weighted)
    5. Confusion matrix, CV fold scores, feature importance, learning curve
    6. SHAP summary — all class SHAP values stored; user selects class in UI
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

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
        pipeline, PARAM_GRID_CLASSIFIER, cv=CV_FOLDS,
        scoring="f1_weighted", n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_index = grid_search.best_index_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_test, y_pred,    average="weighted", zero_division=0))
    f1        = float(f1_score(y_test, y_pred,        average="weighted", zero_division=0))

    # ROC-AUC: OvR with weighted average
    y_test_bin = label_binarize(y_test, classes=class_labels)
    roc_auc    = float(roc_auc_score(
        y_test_bin, y_prob, multi_class="ovr", average="weighted",
    ))

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

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
    # shap_values: 3D array (n_samples, n_features, n_classes) in SHAP >= 0.40
    # or list of 2D arrays in older SHAP

    return {
        "best_params":    grid_search.best_params_,
        "accuracy":       accuracy,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "roc_auc":        roc_auc,
        "cm":             cm,
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
        "class_labels":   [str(c) for c in class_labels],
        "n_train":        len(X_train),
        "n_test":         len(X_test),
    }


# ── Regression comparison: Lasso vs Random Forest ─────────────────────────────

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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


def _build_preprocessor(numeric_features, categorical_features, scale_numeric=False):
    """Build a ColumnTransformer; optionally add StandardScaler for Lasso."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline(num_steps), numeric_features))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers)


def _model_metrics(model, X_train, y_train, X_test, y_test, grid):
    """Compute train/test R², MAE, RMSE and CV scores from GridSearchCV."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    best_idx   = grid.best_index_
    cv_scores  = np.array([
        grid.cv_results_[f"split{i}_test_score"][best_idx]
        for i in range(CV_FOLDS)
    ])
    return {
        "train_r2":   float(r2_score(y_train, train_pred)),
        "test_r2":    float(r2_score(y_test,  test_pred)),
        "train_mae":  float(mean_absolute_error(y_train, train_pred)),
        "test_mae":   float(mean_absolute_error(y_test,  test_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_test,  test_pred))),
        "cv_mean":    float(cv_scores.mean()),
        "cv_std":     float(cv_scores.std()),
    }


def run_regression_comparison(
    df: pd.DataFrame,
    target: str,
    predictors: list,
) -> dict:
    """
    Run Lasso and Random Forest Regressor side-by-side.
    Determines winner by Test R².
    Returns comparison table + winner's learning curve, importance, and SHAP.
    """
    X = df[predictors].copy()
    y = df[target].copy()
    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    num_feats = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # ── Lasso (with scaling) ──────────────────────────────────────────────────
    lasso_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats, scale_numeric=True)),
        ("model", Lasso(max_iter=10_000)),
    ])
    lasso_grid = GridSearchCV(
        lasso_pipeline, LASSO_PARAM_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    lasso_grid.fit(X_train, y_train)
    best_lasso   = lasso_grid.best_estimator_
    lasso_m      = _model_metrics(best_lasso, X_train, y_train, X_test, y_test, lasso_grid)

    # ── Random Forest (no scaling) ────────────────────────────────────────────
    rf_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats, scale_numeric=False)),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    rf_grid = GridSearchCV(
        rf_pipeline, PARAM_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_m    = _model_metrics(best_rf, X_train, y_train, X_test, y_test, rf_grid)

    # ── Winner ────────────────────────────────────────────────────────────────
    winner       = "Random Forest" if rf_m["test_r2"] >= lasso_m["test_r2"] else "Lasso"
    best_model   = best_rf   if winner == "Random Forest" else best_lasso
    winner_m     = rf_m      if winner == "Random Forest" else lasso_m

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison_df = pd.DataFrame({
        "Metric": ["Train R²","Test R²","Train MAE","Test MAE","Train RMSE","Test RMSE","CV Mean R²"],
        "Lasso":         [round(lasso_m[k], 3) for k in
                          ("train_r2","test_r2","train_mae","test_mae","train_rmse","test_rmse","cv_mean")],
        "Random Forest": [round(rf_m[k], 3)    for k in
                          ("train_r2","test_r2","train_mae","test_mae","train_rmse","test_rmse","cv_mean")],
    })

    # ── Learning curve (winner) ───────────────────────────────────────────────
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=5, scoring="r2", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 4),
    )

    # ── Feature importance / coefficients (winner) ────────────────────────────
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    if winner == "Random Forest":
        importances  = best_model.named_steps["model"].feature_importances_
        feat_imp     = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        coefficients = None
    else:
        coef         = best_model.named_steps["model"].coef_
        coef_series  = pd.Series(coef, index=feature_names)
        coef_series  = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
        feat_imp     = None
        coefficients = coef_series

    # ── SHAP (winner, up to 1000 test rows) ───────────────────────────────────
    preprocessor_fitted = best_model.named_steps["preprocessor"]
    model_fitted        = best_model.named_steps["model"]
    X_test_transformed  = preprocessor_fitted.transform(X_test)
    n_sample            = min(1000, X_test_transformed.shape[0])
    X_sample            = X_test_transformed[:n_sample]

    if winner == "Random Forest":
        explainer   = shap.TreeExplainer(model_fitted)
        shap_values = explainer.shap_values(X_sample)
        shap_type   = "tree"
    else:
        explainer   = shap.Explainer(model_fitted, X_sample)
        shap_expl   = explainer(X_sample)
        shap_values = shap_expl.values   # numpy array from Explanation object
        shap_type   = "linear"

    return {
        "_model_type":    "regression_comparison",
        "comparison_df":  comparison_df,
        "winner":         winner,
        "lasso_params":   lasso_grid.best_params_,
        "rf_params":      rf_grid.best_params_,
        "lasso_metrics":  lasso_m,
        "rf_metrics":     rf_m,
        "winner_metrics": winner_m,
        "feat_imp":       feat_imp,
        "coefficients":   coefficients,
        "train_sizes":    train_sizes,
        "train_scores":   train_scores,
        "val_scores":     val_scores,
        "shap_values":    shap_values,
        "shap_type":      shap_type,
        "X_sample":       X_sample,
        "feature_names":  feature_names,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
    }


# ── Binary classification comparison: Logistic Regression vs Random Forest ────

from sklearn.linear_model import LogisticRegression as _LR

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


def run_binary_classification_comparison(
    df: pd.DataFrame,
    target: str,
    predictors: list,
) -> dict:
    """
    Run Logistic Regression and Random Forest Classifier side-by-side.
    Determines winner by Test F1 Score.
    Returns comparison table + winner's learning curve, importance/coefficients,
    odds ratio (if LR wins), and SHAP.
    """
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

    num_feats = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # ── Logistic Regression (with scaling) ───────────────────────────────────
    lr_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats, scale_numeric=True)),
        ("model", _LR(max_iter=5000, class_weight="balanced", solver="liblinear")),
    ])
    lr_grid = GridSearchCV(
        lr_pipeline, LOGISTIC_PARAM_GRID, cv=CV_FOLDS, scoring="f1", n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_

    # ── Random Forest Classifier (without scaling) ────────────────────────────
    from sklearn.ensemble import RandomForestClassifier as _RFC
    rf_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats, scale_numeric=False)),
        ("model", _RFC(random_state=42, n_jobs=-1, class_weight="balanced")),
    ])
    rf_grid = GridSearchCV(
        rf_pipeline, PARAM_GRID_CLASSIFIER, cv=CV_FOLDS, scoring="f1", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # ── Metrics helper ────────────────────────────────────────────────────────
    def _clf_metrics(model, grid):
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        y_te_prob = model.predict_proba(X_test)[:, 1]
        best_idx  = grid.best_index_
        cv_scores = np.array([
            grid.cv_results_[f"split{i}_test_score"][best_idx] for i in range(CV_FOLDS)
        ])
        return {
            "train_accuracy":  float(accuracy_score(y_train, y_tr_pred)),
            "test_accuracy":   float(accuracy_score(y_test,  y_te_pred)),
            "train_precision": float(precision_score(y_train, y_tr_pred, zero_division=0)),
            "test_precision":  float(precision_score(y_test,  y_te_pred, zero_division=0)),
            "train_recall":    float(recall_score(y_train, y_tr_pred, zero_division=0)),
            "test_recall":     float(recall_score(y_test,  y_te_pred, zero_division=0)),
            "train_f1":        float(f1_score(y_train, y_tr_pred, zero_division=0)),
            "test_f1":         float(f1_score(y_test,  y_te_pred, zero_division=0)),
            "roc_auc":         float(roc_auc_score(y_test, y_te_prob)),
            "cv_mean":         float(cv_scores.mean()),
            "cv_std":          float(cv_scores.std()),
        }

    lr_m = _clf_metrics(best_lr, lr_grid)
    rf_m = _clf_metrics(best_rf, rf_grid)

    # ── Winner by Test F1 ─────────────────────────────────────────────────────
    winner     = "Random Forest" if rf_m["test_f1"] >= lr_m["test_f1"] else "Logistic Regression"
    best_model = best_rf   if winner == "Random Forest" else best_lr
    winner_m   = rf_m      if winner == "Random Forest" else lr_m

    # Winner confusion matrix + ROC curve
    y_te_pred_w  = best_model.predict(X_test)
    y_te_prob_w  = best_model.predict_proba(X_test)[:, 1]
    cm           = confusion_matrix(y_test, y_te_pred_w, labels=class_labels)
    fpr, tpr, _  = roc_curve(y_test, y_te_prob_w)

    # ── Comparison table (all metrics higher-better) ──────────────────────────
    metric_keys = [
        ("Train Accuracy",  "train_accuracy"),
        ("Test Accuracy",   "test_accuracy"),
        ("Train Precision", "train_precision"),
        ("Test Precision",  "test_precision"),
        ("Train Recall",    "train_recall"),
        ("Test Recall",     "test_recall"),
        ("Train F1",        "train_f1"),
        ("Test F1",         "test_f1"),
        ("ROC-AUC",         "roc_auc"),
        ("CV Mean F1",      "cv_mean"),
    ]
    comparison_df = pd.DataFrame({
        "Metric":               [m[0] for m in metric_keys],
        "Logistic Regression":  [round(lr_m[m[1]], 3) for m in metric_keys],
        "Random Forest":        [round(rf_m[m[1]], 3) for m in metric_keys],
    })

    # ── Learning curve (winner, F1 scoring) ──────────────────────────────────
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=5, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 4),
    )

    # ── Feature importance or coefficients ────────────────────────────────────
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    if winner == "Random Forest":
        importances  = best_model.named_steps["model"].feature_importances_
        feat_imp     = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        coefficients = None
        odds_ratio_df = None
    else:
        coef         = best_model.named_steps["model"].coef_[0]
        coef_series  = pd.Series(coef, index=feature_names)
        coef_series  = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
        feat_imp     = None
        coefficients = coef_series
        # Odds ratio table (top 20 by absolute coefficient)
        top20        = coef_series.head(20)
        clean_labels = [n.split("__", 1)[-1] for n in top20.index]
        odds_ratio_df = pd.DataFrame({
            "Feature":     clean_labels,
            "Coefficient": top20.values.round(4),
            "Odds Ratio":  np.exp(top20.values).round(4),
        })

    # ── SHAP ──────────────────────────────────────────────────────────────────
    preprocessor_fitted = best_model.named_steps["preprocessor"]
    model_fitted        = best_model.named_steps["model"]
    X_test_transformed  = preprocessor_fitted.transform(X_test)
    n_sample            = min(1000, X_test_transformed.shape[0])
    X_sample            = X_test_transformed[:n_sample]

    if winner == "Random Forest":
        explainer   = shap.TreeExplainer(model_fitted)
        sv          = explainer.shap_values(X_sample)
        shap_display = sv[:, :, 1] if not isinstance(sv, list) else sv[1]
        shap_type    = "tree_binary"
    else:
        explainer    = shap.Explainer(model_fitted, X_sample)
        shap_expl    = explainer(X_sample)
        shap_display = shap_expl.values
        shap_type    = "linear"

    return {
        "_model_type":    "binary_comparison",
        "comparison_df":  comparison_df,
        "winner":         winner,
        "lr_params":      lr_grid.best_params_,
        "rf_params":      rf_grid.best_params_,
        "lr_metrics":     lr_m,
        "rf_metrics":     rf_m,
        "winner_metrics": winner_m,
        "cm":             cm,
        "fpr":            fpr,
        "tpr":            tpr,
        "class_labels":   [str(c) for c in class_labels],
        "feat_imp":       feat_imp,
        "coefficients":   coefficients,
        "odds_ratio_df":  odds_ratio_df,
        "train_sizes":    train_sizes,
        "train_scores":   train_scores,
        "val_scores":     val_scores,
        "shap_display":   shap_display,
        "shap_type":      shap_type,
        "X_sample":       X_sample,
        "feature_names":  feature_names,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
    }


# ── Multi-class comparison: Multinomial Logistic Regression vs Random Forest ──

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


def run_multiclass_classification_comparison(
    df: pd.DataFrame,
    target: str,
    predictors: list,
) -> dict:
    """
    Run Multinomial Logistic Regression and RF Classifier side-by-side.
    Determines winner by weighted Test F1 Score.
    LR wins: per-class coefficients, per-class odds ratios, SHAP (class selector).
    RF wins: feature importance, SHAP (class selector).
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    from sklearn.preprocessing import label_binarize

    X = df[predictors].copy()
    y = df[target].copy()
    mask = y.notna()
    X, y = X[mask], y[mask]

    class_labels     = sorted(y.unique().tolist())
    class_labels_str = [str(c) for c in class_labels]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    num_feats = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # ── Multinomial Logistic Regression (with scaling) ────────────────────────
    lr_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats, scale_numeric=True)),
        ("model", _LR(
            max_iter=5000, class_weight="balanced", solver="saga",
        )),
    ])
    lr_grid = GridSearchCV(
        lr_pipeline, LOGISTIC_PARAM_GRID, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_

    # ── Random Forest Classifier (without scaling) ────────────────────────────
    from sklearn.ensemble import RandomForestClassifier as _RFC2
    rf_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats, scale_numeric=False)),
        ("model", _RFC2(random_state=42, n_jobs=-1, class_weight="balanced")),
    ])
    rf_grid = GridSearchCV(
        rf_pipeline, PARAM_GRID_CLASSIFIER, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # ── Metrics helper ────────────────────────────────────────────────────────
    y_test_bin = label_binarize(y_test, classes=class_labels)

    def _mc_metrics(model, grid):
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        y_te_prob = model.predict_proba(X_test)
        best_idx  = grid.best_index_
        cv_scores = np.array([
            grid.cv_results_[f"split{i}_test_score"][best_idx] for i in range(CV_FOLDS)
        ])
        roc_auc = float(roc_auc_score(
            y_test_bin, y_te_prob, multi_class="ovr", average="weighted",
        ))
        kw = dict(average="weighted", zero_division=0)
        return {
            "train_accuracy":  float(accuracy_score(y_train, y_tr_pred)),
            "test_accuracy":   float(accuracy_score(y_test,  y_te_pred)),
            "train_precision": float(precision_score(y_train, y_tr_pred, **kw)),
            "test_precision":  float(precision_score(y_test,  y_te_pred, **kw)),
            "train_recall":    float(recall_score(y_train, y_tr_pred, **kw)),
            "test_recall":     float(recall_score(y_test,  y_te_pred, **kw)),
            "train_f1":        float(f1_score(y_train, y_tr_pred, **kw)),
            "test_f1":         float(f1_score(y_test,  y_te_pred, **kw)),
            "roc_auc":         roc_auc,
            "cv_mean":         float(cv_scores.mean()),
            "cv_std":          float(cv_scores.std()),
        }

    lr_m = _mc_metrics(best_lr, lr_grid)
    rf_m = _mc_metrics(best_rf, rf_grid)

    # ── Winner by weighted Test F1 ────────────────────────────────────────────
    winner     = "Random Forest" if rf_m["test_f1"] >= lr_m["test_f1"] else "Multinomial Logistic Regression"
    best_model = best_rf if winner == "Random Forest" else best_lr
    winner_m   = rf_m   if winner == "Random Forest" else lr_m

    y_te_pred_w = best_model.predict(X_test)
    cm          = confusion_matrix(y_test, y_te_pred_w, labels=class_labels)

    # ── Comparison table ──────────────────────────────────────────────────────
    metric_keys = [
        ("Train Accuracy",             "train_accuracy"),
        ("Test Accuracy",              "test_accuracy"),
        ("Train Precision (weighted)", "train_precision"),
        ("Test Precision (weighted)",  "test_precision"),
        ("Train Recall (weighted)",    "train_recall"),
        ("Test Recall (weighted)",     "test_recall"),
        ("Train F1 (weighted)",        "train_f1"),
        ("Test F1 (weighted)",         "test_f1"),
        ("Weighted ROC-AUC",           "roc_auc"),
        ("CV Mean F1 (weighted)",      "cv_mean"),
    ]
    comparison_df = pd.DataFrame({
        "Metric":                          [m[0] for m in metric_keys],
        "Multinomial Logistic Regression": [round(lr_m[m[1]], 3) for m in metric_keys],
        "Random Forest":                   [round(rf_m[m[1]], 3) for m in metric_keys],
    })

    # ── Learning curve (f1_weighted) ──────────────────────────────────────────
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=5, scoring="f1_weighted", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 4),
    )

    # ── Feature importance / per-class coefficients ───────────────────────────
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    if winner == "Random Forest":
        importances            = best_model.named_steps["model"].feature_importances_
        feat_imp               = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        coefficients_per_class = None
        odds_ratio_per_class   = None
    else:
        coef_matrix            = best_model.named_steps["model"].coef_  # (n_classes, n_features)
        feat_imp               = None
        coefficients_per_class = {}
        odds_ratio_per_class   = {}
        for i, cls in enumerate(class_labels_str):
            coef_s = pd.Series(coef_matrix[i], index=feature_names)
            coef_s = coef_s.reindex(coef_s.abs().sort_values(ascending=False).index)
            coefficients_per_class[cls] = coef_s
            top20  = coef_s.head(20)
            clean  = [n.split("__", 1)[-1] for n in top20.index]
            odds_ratio_per_class[cls] = pd.DataFrame({
                "Feature":     clean,
                "Coefficient": top20.values.round(4),
                "Odds Ratio":  np.exp(top20.values).round(4),
            })

    # ── SHAP ──────────────────────────────────────────────────────────────────
    preprocessor_fitted = best_model.named_steps["preprocessor"]
    model_fitted        = best_model.named_steps["model"]
    X_test_transformed  = preprocessor_fitted.transform(X_test)
    n_sample            = min(1000, X_test_transformed.shape[0])
    X_sample            = X_test_transformed[:n_sample]

    if winner == "Random Forest":
        explainer   = shap.TreeExplainer(model_fitted)
        shap_values = explainer.shap_values(X_sample)
        shap_type   = "tree_multiclass"
    else:
        explainer   = shap.Explainer(model_fitted, X_sample)
        shap_expl   = explainer(X_sample)
        shap_values = shap_expl.values   # 3D: (n_samples, n_features, n_classes)
        shap_type   = "linear_multiclass"

    return {
        "_model_type":              "multiclass_comparison",
        "comparison_df":            comparison_df,
        "winner":                   winner,
        "lr_params":                lr_grid.best_params_,
        "rf_params":                rf_grid.best_params_,
        "lr_metrics":               lr_m,
        "rf_metrics":               rf_m,
        "winner_metrics":           winner_m,
        "cm":                       cm,
        "class_labels":             class_labels_str,
        "feat_imp":                 feat_imp,
        "coefficients_per_class":   coefficients_per_class,
        "odds_ratio_per_class":     odds_ratio_per_class,
        "train_sizes":              train_sizes,
        "train_scores":             train_scores,
        "val_scores":               val_scores,
        "shap_values":              shap_values,
        "shap_type":                shap_type,
        "X_sample":                 X_sample,
        "feature_names":            feature_names,
        "n_train":                  len(X_train),
        "n_test":                   len(X_test),
    }
