"""
Classifier definitions and imbalance-handling utilities for sentiment polarity.

Covers the majority-class baseline plus three classical models — Multinomial
Naive Bayes, Logistic Regression and ``sklearn.ensemble.GradientBoostingClassifier`` —
with a consistent interface for comparison (RQ1).

Imbalance handling
------------------
``LogisticRegression`` accepts ``class_weight='balanced'`` directly.
``MultinomialNB`` does **not**; for it, imbalance is handled at the data level
via :func:`resample_balanced` (random over-/under-sampling). Scikit-learn's
``GradientBoostingClassifier`` has no class-weight argument, so balanced runs
use sample weights inside a small explicit wrapper. To keep runtime tractable,
gradient boosting is trained on a deterministic stratified sample; this is
reported in the results and paper rather than hidden behind an automatic
implementation fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample

RANDOM_STATE = 42
SKLEARN_GB_MAX_TRAIN_SAMPLES = 12000
SKLEARN_GB_N_ESTIMATORS = 40
SKLEARN_GB_LEARNING_RATE = 0.05
SKLEARN_GB_MAX_DEPTH = 3

NB_ALPHA_GRID = [0.1, 0.5, 1.0, 2.0]
LR_C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
LR_CLASS_WEIGHT_GRID = [None, "balanced"]
GB_N_ESTIMATORS_GRID = [40, 80]
GB_LEARNING_RATE_GRID = [0.05, 0.1]
GB_MAX_DEPTH_GRID = [2, 3]
GB_CLASS_WEIGHT_GRID = [None, "balanced"]


def get_majority_baseline() -> DummyClassifier:
    """Always predicts the most frequent class — the floor every model must beat."""
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)


def get_naive_bayes(alpha: float = 1.0) -> MultinomialNB:
    """
    Multinomial Naive Bayes.

    Note: NB has no ``class_weight``; balance the data with
    :func:`resample_balanced` before fitting if imbalance handling is desired.
    """
    return MultinomialNB(alpha=alpha)


def get_logistic_regression(
    class_weight: str | None = "balanced",
    C: float = 1.0,
) -> LogisticRegression:
    """Logistic Regression with L2 regularisation and optional balanced weighting."""
    return LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def get_gradient_boosting(
    class_weight: str | None = "balanced",
    n_estimators: int = SKLEARN_GB_N_ESTIMATORS,
    learning_rate: float = SKLEARN_GB_LEARNING_RATE,
    max_depth: int = SKLEARN_GB_MAX_DEPTH,
    max_train_samples: int = SKLEARN_GB_MAX_TRAIN_SAMPLES,
):
    """
    ``sklearn.ensemble.GradientBoostingClassifier`` used in the final experiment.

    This deliberately does not auto-switch to another implementation. Alternative
    gradient-boosting variants should be separate, explicitly named experiments.
    """
    return SklearnGradientBoosting(
        class_weight=class_weight,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_train_samples=max_train_samples,
    )


class SklearnGradientBoosting:
    """Small wrapper adding balanced sample weights to sklearn GB."""

    def __init__(
        self,
        class_weight: str | None = "balanced",
        n_estimators: int = SKLEARN_GB_N_ESTIMATORS,
        learning_rate: float = SKLEARN_GB_LEARNING_RATE,
        max_depth: int = SKLEARN_GB_MAX_DEPTH,
        max_train_samples: int = SKLEARN_GB_MAX_TRAIN_SAMPLES,
    ):
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_train_samples = max_train_samples
        self.fit_n_samples_ = None
        self.fit_sampled_ = False
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )

    def fit(self, X, y):
        y = np.asarray(y)
        if len(y) > self.max_train_samples:
            rng = np.random.RandomState(RANDOM_STATE)
            keep = []
            classes, counts = np.unique(y, return_counts=True)
            for cls, count in zip(classes, counts):
                cls_idx = np.where(y == cls)[0]
                n_cls = max(1, int(round(self.max_train_samples * count / len(y))))
                n_cls = min(n_cls, len(cls_idx))
                keep.extend(rng.choice(cls_idx, size=n_cls, replace=False))
            keep = np.array(sorted(keep))
            X = X[keep]
            y = y[keep]
            self.fit_sampled_ = True
        else:
            self.fit_sampled_ = False
        self.fit_n_samples_ = len(y)
        sample_weight = None
        if self.class_weight == "balanced":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "class_weight": self.class_weight,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_train_samples": self.max_train_samples,
            "random_state": RANDOM_STATE,
        }


def get_all_models(class_weight: str | None = "balanced", include_baseline: bool = True) -> dict:
    """
    Return the classifiers used in RQ1.

    With ``class_weight`` set, LR and GB are weighted; NB is returned unweighted
    (handle its imbalance at the data level). Set ``class_weight=None`` to get the
    no-handling configuration used to demonstrate the majority-class collapse.
    """
    models: dict[str, object] = {}
    if include_baseline:
        models["majority_baseline"] = get_majority_baseline()
    models["naive_bayes"] = get_naive_bayes()
    models["logistic_regression"] = get_logistic_regression(class_weight=class_weight)
    models["gradient_boosting"] = get_gradient_boosting(class_weight=class_weight)
    return models


def resample_balanced(
    X,
    y,
    strategy: str = "undersample",
    random_state: int = RANDOM_STATE,
):
    """
    Balance classes by random resampling on a (sparse) feature matrix.

    Parameters
    ----------
    strategy : {'undersample', 'oversample'}
        ``undersample`` draws every class down to the size of the smallest
        class; ``oversample`` draws every class up to the largest. Undersampling
        is the default — it avoids duplicating rows and keeps training fast,
        which matters for the corpus sizes here.

    Returns
    -------
    (X_resampled, y_resampled)
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.min() if strategy == "undersample" else counts.max()

    parts_X, parts_y = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        replace = strategy == "oversample" and len(idx) < target
        chosen = resample(
            idx,
            replace=replace,
            n_samples=target,
            random_state=random_state,
        )
        parts_X.append(X[chosen])
        parts_y.append(y[chosen])

    from scipy.sparse import issparse, vstack

    X_res = vstack(parts_X) if issparse(X) else np.vstack(parts_X)
    y_res = np.concatenate(parts_y)

    # Shuffle so classes are not block-ordered.
    order = np.random.RandomState(random_state).permutation(len(y_res))
    return X_res[order], y_res[order]


def select_best_config(candidates: list[dict], minority_label: str) -> pd.DataFrame:
    """
    Rank candidate configurations evaluated on the validation split.

    Each candidate dict must contain ``name``, ``f1_macro`` and
    ``recall_<minority_label>``. Primary criterion is validation macro-F1
    (descending); minority-class recall is the transparent tie-breaker. The
    final test set must never appear in ``candidates`` — selection happens
    before it is touched (see notebooks/03_modeling_rq1.ipynb, Section on
    validation-only selection).
    """
    df = pd.DataFrame(candidates)
    recall_col = f"recall_{minority_label}"
    df = df.sort_values(["f1_macro", recall_col], ascending=[False, False]).reset_index(drop=True)
    df["selected"] = False
    if len(df):
        df.loc[0, "selected"] = True
    return df
