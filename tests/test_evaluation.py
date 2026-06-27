import numpy as np

from src import evaluation


def _toy_labels(n=200, seed=0):
    rng = np.random.RandomState(seed)
    y_true = rng.choice(["positive", "negative"], size=n, p=[0.7, 0.3])
    # predictions agree most of the time, with some noise
    flip = rng.rand(n) < 0.15
    y_pred = np.where(flip, np.where(y_true == "positive", "negative", "positive"), y_true)
    return y_true, y_pred


def test_bootstrap_metric_ci_reproducible_with_fixed_seed():
    y_true, y_pred = _toy_labels()
    r1 = evaluation.bootstrap_metric_ci(y_true, y_pred, metric="accuracy", n_replications=200, random_state=42)
    r2 = evaluation.bootstrap_metric_ci(y_true, y_pred, metric="accuracy", n_replications=200, random_state=42)
    assert r1 == r2


def test_bootstrap_metric_ci_different_seed_can_differ():
    y_true, y_pred = _toy_labels()
    r1 = evaluation.bootstrap_metric_ci(y_true, y_pred, metric="accuracy", n_replications=200, random_state=1)
    r2 = evaluation.bootstrap_metric_ci(y_true, y_pred, metric="accuracy", n_replications=200, random_state=2)
    assert r1["ci_low"] != r2["ci_low"] or r1["ci_high"] != r2["ci_high"]


def test_bootstrap_ci_contains_point_estimate():
    y_true, y_pred = _toy_labels()
    r = evaluation.bootstrap_metric_ci(y_true, y_pred, metric="f1_macro", n_replications=300, random_state=42)
    assert r["ci_low"] <= r["point_estimate"] <= r["ci_high"]


def test_paired_bootstrap_zero_diff_when_predictions_identical():
    y_true, y_pred = _toy_labels()
    result = evaluation.paired_bootstrap_macro_f1_diff(
        y_true, y_pred, y_pred, labels=["positive", "negative"], n_replications=200, random_state=42
    )
    assert result["diff"] == 0.0
    assert result["ci_excludes_zero"] is False


def test_bootstrap_classification_report_ci_reproducible():
    y_true, y_pred = _toy_labels()
    df1 = evaluation.bootstrap_classification_report_ci(y_true, y_pred, labels=["positive", "negative"], n_replications=150, random_state=7)
    df2 = evaluation.bootstrap_classification_report_ci(y_true, y_pred, labels=["positive", "negative"], n_replications=150, random_state=7)
    assert df1.equals(df2)


def test_stratified_bootstrap_indices_preserves_class_counts():
    y_true, _ = _toy_labels(n=200, seed=0)
    rng = np.random.RandomState(0)
    n_pos = int((y_true == "positive").sum())
    n_neg = int((y_true == "negative").sum())
    for _ in range(20):
        idx = evaluation._stratified_bootstrap_indices(y_true, rng)
        resampled = y_true[idx]
        # Stratified: each class's count in the resample exactly matches its
        # original stratum size, every single replication -- a plain
        # randint(0, n, size=n) resample would NOT guarantee this.
        assert (resampled == "positive").sum() == n_pos
        assert (resampled == "negative").sum() == n_neg
        assert len(idx) == len(y_true)
