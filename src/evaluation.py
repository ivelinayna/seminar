"""
Evaluation utilities for sentiment classifiers.

Provides accuracy, F1 (macro and per-class), confusion matrix, and error
analysis helpers — including identification of hardest-to-classify reviews
which is central to RQ1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list | None = None,
) -> dict:
    """Return a dict of evaluation metrics for a single model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=labels).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Compile a comparison table across multiple models.

    Parameters
    ----------
    results : dict
        Mapping of model name -> evaluation dict from evaluate_model().
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        })
    return pd.DataFrame(rows).set_index("model").round(4)


def find_hardest_examples(
    X_text: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n: int = 20,
) -> pd.DataFrame:
    """
    Return the n most confidently wrong predictions for error analysis (RQ1).

    If probabilities are available, examples are ranked by how confidently
    the model made the wrong prediction.
    """
    df = pd.DataFrame({"text": X_text.values, "y_true": y_true, "y_pred": y_pred})
    df["correct"] = df["y_true"] == df["y_pred"]
    wrong = df[~df["correct"]].copy()
    if y_proba is not None:
        wrong["confidence"] = y_proba.max(axis=1)[~df["correct"].values]
        wrong = wrong.sort_values("confidence", ascending=False)
    return wrong.head(n)
