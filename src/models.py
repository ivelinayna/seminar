"""
Classifier definitions and imbalance-handling utilities for sentiment polarity.

Covers the majority-class baseline plus three classical models — Multinomial
Naive Bayes, Logistic Regression and Gradient Boosting (LightGBM) — with a
consistent interface for fair comparison (RQ1).

Imbalance handling
------------------
``LogisticRegression`` and ``LGBMClassifier`` accept ``class_weight='balanced'``
directly. ``MultinomialNB`` does **not**; for it, imbalance is handled at the
data level via :func:`resample_balanced` (random over-/under-sampling). The
original code accepted a ``class_weight`` argument for NB and silently ignored
it — that footgun is removed here.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def get_majority_baseline() -> DummyClassifier:
    """Always predicts the most frequent class — the floor every model must beat."""
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
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
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
):
    """LightGBM gradient boosting classifier."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


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
    if LIGHTGBM_AVAILABLE:
        models["gradient_boosting"] = get_gradient_boosting(class_weight=class_weight)
    return models


# --------------------------------------------------------------------------- #
# Data-level imbalance handling (for models without class_weight, e.g. NB)
# --------------------------------------------------------------------------- #
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
