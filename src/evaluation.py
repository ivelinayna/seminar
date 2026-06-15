"""
Evaluation utilities for sentiment classifiers.

Accuracy, macro/weighted/per-class F1, confusion matrices and error analysis
(hardest-to-classify reviews — central to RQ1), plus helpers to plot confusion
matrices and export comparison tables to LaTeX for the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
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
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "labels": list(labels) if labels is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Compile a comparison table across models.

    Includes per-class recall columns, since the whole point under class
    imbalance is to see how each model treats the minority class — the headline
    accuracy hides exactly that.
    """
    rows = []
    for name, m in results.items():
        row = {
            "model": name,
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "recall_macro": m["recall_macro"],
        }
        if m.get("labels") and m.get("recall_per_class"):
            for lbl, rec in zip(m["labels"], m["recall_per_class"]):
                row[f"recall_{lbl}"] = rec
        rows.append(row)
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

    With probabilities available, examples are ranked by how confidently the
    model made the *wrong* call — those are the most informative failures.
    """
    df = pd.DataFrame(
        {"text": np.asarray(X_text), "y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}
    )
    df["correct"] = df["y_true"] == df["y_pred"]
    wrong = df[~df["correct"]].copy()
    if y_proba is not None:
        wrong["confidence"] = np.asarray(y_proba).max(axis=1)[~df["correct"].values]
        wrong = wrong.sort_values("confidence", ascending=False)
    return wrong.head(n)


# --------------------------------------------------------------------------- #
# Plotting & export
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm,
    labels: list,
    title: str = "Confusion matrix",
    normalize: bool = True,
    ax: plt.Axes | None = None,
    cmap: str = "Blues",
):
    """Plot a confusion matrix (row-normalised by default) with cell annotations."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        with np.errstate(all="ignore"):
            cm_display = cm / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = lambda v: f"{v:.0%}"
    else:
        cm_display = cm
        fmt = lambda v: f"{int(v):,}"

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8 + 0.4 * len(labels), 4.2 + 0.3 * len(labels)))
    im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=cm_display.max() or 1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontweight="bold")
    thresh = (cm_display.max() or 1) / 2
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            ax.text(
                j, i, fmt(cm_display[i, j]),
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=11,
            )
    return ax


def results_to_latex(
    df: pd.DataFrame,
    path: str | Path | None = None,
    caption: str = "Model comparison",
    label: str = "tab:model_comparison",
    float_format: str = "%.3f",
) -> str:
    """Render a comparison DataFrame as a LaTeX table; optionally write to disk."""
    latex = df.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        bold_rows=True,
        column_format="l" + "r" * df.shape[1],
    )
    if path is not None:
        Path(path).write_text(latex, encoding="utf-8")
    return latex
