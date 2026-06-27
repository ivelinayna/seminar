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


def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    One replication of a **class-stratified** bootstrap resample: within each
    true-class stratum, resample that stratum's own rows with replacement (same
    stratum size every time), then concatenate. This fixes each class's count
    across replications — unlike a plain ``randint(0, n, size=n)`` resample of
    the whole test set, where class proportions also vary by chance — which is
    what "stratified by true class" means throughout this module's docstrings
    and the paper's methodology section.
    """
    idx_parts = []
    for cls in np.unique(y_true):
        cls_idx = np.where(y_true == cls)[0]
        idx_parts.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    return np.concatenate(idx_parts)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "accuracy",
    labels: list | None = None,
    n_replications: int = 2000,
    random_state: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    Non-parametric bootstrap CI for a single metric on one fixed test set.

    This is a **class-stratified** bootstrap: each replication resamples with
    replacement *within* each true-class stratum (preserving that stratum's
    exact size), then concatenates — so class proportions are fixed across
    replications rather than also varying by chance, which is what
    "stratified by true class" means here. ``metric`` is one of
    ``'accuracy'``, ``'f1_macro'``, or ``'recall_<label>'`` (per-class recall,
    e.g. ``'recall_positive'``).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(random_state)

    def _score(yt, yp) -> float:
        if metric == "accuracy":
            return accuracy_score(yt, yp)
        if metric == "f1_macro":
            return f1_score(yt, yp, average="macro", zero_division=0, labels=labels)
        if metric.startswith("recall_"):
            cls = metric[len("recall_"):]
            mask = yt == cls
            if mask.sum() == 0:
                return np.nan
            return float(np.mean(yp[mask] == yt[mask]))
        raise ValueError(f"Unknown metric: {metric}")

    point_estimate = _score(y_true, y_pred)
    samples = np.empty(n_replications)
    for i in range(n_replications):
        idx = _stratified_bootstrap_indices(y_true, rng)
        samples[i] = _score(y_true[idx], y_pred[idx])
    samples = samples[~np.isnan(samples)]
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lo, hi = np.percentile(samples, [lo_pct, hi_pct])
    return {
        "metric": metric,
        "point_estimate": float(point_estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_replications": n_replications,
        "random_state": random_state,
    }


def bootstrap_classification_report_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    n_replications: int = 2000,
    random_state: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Bootstrap CIs for accuracy, macro-F1 and per-true-class recall in one pass.

    Resamples once per replication and computes all metrics on the same
    resampled indices (cheaper and more consistent than calling
    :func:`bootstrap_metric_ci` repeatedly). Resampling is **class-stratified**
    (see :func:`_stratified_bootstrap_indices`): each replication resamples
    within each true-class stratum at that stratum's original size, so class
    proportions are fixed across replications rather than also varying by
    chance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(random_state)

    metric_names = ["accuracy", "f1_macro"] + [f"recall_{lbl}" for lbl in labels]
    samples = {m: np.empty(n_replications) for m in metric_names}

    def _row(yt, yp) -> dict:
        row = {
            "accuracy": accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0, labels=labels),
        }
        for lbl, rec in zip(labels, recall_score(yt, yp, average=None, labels=labels, zero_division=0)):
            row[f"recall_{lbl}"] = rec
        return row

    point = _row(y_true, y_pred)
    for i in range(n_replications):
        idx = _stratified_bootstrap_indices(y_true, rng)
        row = _row(y_true[idx], y_pred[idx])
        for m in metric_names:
            samples[m][i] = row[m]

    lo_pct, hi_pct = 2.5, 97.5
    if ci != 0.95:
        lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    rows = []
    for m in metric_names:
        lo, hi = np.percentile(samples[m], [lo_pct, hi_pct])
        rows.append({
            "metric": m,
            "point_estimate": round(point[m], 4),
            "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4),
        })
    return pd.DataFrame(rows)


def paired_bootstrap_macro_f1_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    labels: list | None = None,
    n_replications: int = 2000,
    random_state: int = 42,
    ci: float = 0.95,
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> dict:
    """
    Paired bootstrap CI for the macro-F1 *difference* between two models
    evaluated on the **same** fixed test set (e.g. unigram vs. uni+bigram
    features). Both models' predictions are resampled with the same
    **class-stratified** indices in every replication (see
    :func:`_stratified_bootstrap_indices`), so the comparison is paired and
    class proportions are fixed across replications.

    If the CI of the difference includes 0, report the gain as a "small
    observed gain" rather than a confirmed/significant improvement.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    rng = np.random.RandomState(random_state)

    def _f1(yt, yp) -> float:
        return f1_score(yt, yp, average="macro", zero_division=0, labels=labels)

    point_a = _f1(y_true, y_pred_a)
    point_b = _f1(y_true, y_pred_b)
    point_diff = point_b - point_a

    diffs = np.empty(n_replications)
    for i in range(n_replications):
        idx = _stratified_bootstrap_indices(y_true, rng)
        diffs[i] = _f1(y_true[idx], y_pred_b[idx]) - _f1(y_true[idx], y_pred_a[idx])

    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lo, hi = np.percentile(diffs, [lo_pct, hi_pct])
    ci_excludes_zero = (lo > 0) or (hi < 0)
    return {
        f"f1_macro_{name_a}": float(point_a),
        f"f1_macro_{name_b}": float(point_b),
        "diff": float(point_diff),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_excludes_zero": bool(ci_excludes_zero),
        "interpretation": (
            "stable difference under this holdout" if ci_excludes_zero else "small observed gain (CI includes 0)"
        ),
        "n_replications": n_replications,
        "random_state": random_state,
    }


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
