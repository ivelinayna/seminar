"""
Classifier definitions and training utilities for sentiment polarity.

Covers Naive Bayes, Logistic Regression, and Gradient Boosting (LightGBM)
with consistent interfaces for fair comparison (RQ1).
"""

from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

RANDOM_STATE = 42


def get_naive_bayes(class_weight: str | None = None) -> MultinomialNB:
    """
    Multinomial Naive Bayes baseline.

    Note: MultinomialNB does not natively support class_weight; imbalance
    handling is therefore done via resampling at the data level for this model.
    """
    return MultinomialNB()


def get_logistic_regression(
    class_weight: str | None = "balanced",
    C: float = 1.0,
) -> LogisticRegression:
    """Logistic Regression with L2 regularization and balanced class weighting."""
    return LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_gradient_boosting(
    class_weight: str | None = "balanced",
    n_estimators: int = 200,
    learning_rate: float = 0.05,
):
    """LightGBM gradient boosting classifier."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def get_all_models(class_weight: str | None = "balanced") -> dict:
    """Return all three classifiers in a single dict for iteration."""
    return {
        "naive_bayes": get_naive_bayes(),
        "logistic_regression": get_logistic_regression(class_weight=class_weight),
        "gradient_boosting": get_gradient_boosting(class_weight=class_weight),
    }
