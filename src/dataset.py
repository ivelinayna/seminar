"""
Dataset loading, sentiment labelling and train/test splitting.

The EDA established two decisions that this module implements:

* **Sentiment framing** is derived from star ratings in two ways — binary
  (1-2 = negative, 4-5 = positive, 3 dropped) and ternary (3 = neutral) — so
  both can be reported, with binary leading.
* **The split is chronological**, not random, to mirror deployment (score new
  reviews from patterns learned on older ones). It is performed *within each
  category* so that both categories appear in train and test, keeping the
  cross-category comparison in RQ2 valid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CATEGORIES = {
    "Subscription_Boxes": "Subscription_Boxes.jsonl",
    "Magazine_Subscriptions": "Magazine_Subscriptions.jsonl",
}


def map_binary(rating: float) -> str | None:
    """1-2 -> negative, 4-5 -> positive, 3 -> None (dropped)."""
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return None


def map_ternary(rating: float) -> str:
    """1-2 -> negative, 3 -> neutral, 4-5 -> positive."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def load_reviews(raw_dir: str | Path) -> pd.DataFrame:
    """Load both categories, tag category, parse dates and add sentiment labels."""
    raw_dir = Path(raw_dir)
    frames = []
    for category, fname in CATEGORIES.items():
        part = pd.read_json(raw_dir / fname, lines=True)
        part["category"] = category
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["text"] = df["text"].fillna("")
    df["review_length"] = df["text"].str.split().str.len()
    df["sentiment_binary"] = df["rating"].apply(map_binary)
    df["sentiment_ternary"] = df["rating"].apply(map_ternary)
    return df


def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    by_category: bool = True,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically: oldest ``1 - test_size`` for train, newest for test.

    With ``by_category=True`` the cut is applied within each category and the
    pieces are concatenated, guaranteeing both categories in both splits.
    """
    def _split_one(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = frame.sort_values(time_col, kind="stable")
        cut = int(len(frame) * (1 - test_size))
        return frame.iloc[:cut], frame.iloc[cut:]

    if by_category:
        train_parts, test_parts = [], []
        for _, frame in df.groupby("category", sort=False):
            tr, te = _split_one(frame)
            train_parts.append(tr)
            test_parts.append(te)
        train = pd.concat(train_parts).sort_values(time_col, kind="stable")
        test = pd.concat(test_parts).sort_values(time_col, kind="stable")
    else:
        train, test = _split_one(df)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def chronological_split_three_way(
    df: pd.DataFrame,
    dev_train_size: float = 0.64,
    validation_size: float = 0.16,
    by_category: bool = True,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into dev-train / validation / final-test.

    Default 64% / 16% / 20% (so dev-train + validation together reproduce the
    oldest 80% used by :func:`chronological_split`). The final 20% is reserved
    for a single evaluation after all feature, imbalance-handling and
    hyperparameter decisions have been made on the validation slice — it must
    not be inspected before that point (see notebooks/03_modeling_rq1.ipynb).
    """

    def _split_one(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        frame = frame.sort_values(time_col, kind="stable")
        n = len(frame)
        cut_dev = int(n * dev_train_size)
        cut_val = int(n * (dev_train_size + validation_size))
        return frame.iloc[:cut_dev], frame.iloc[cut_dev:cut_val], frame.iloc[cut_val:]

    if by_category:
        dev_parts, val_parts, test_parts = [], [], []
        for _, frame in df.groupby("category", sort=False):
            dv, va, te = _split_one(frame)
            dev_parts.append(dv)
            val_parts.append(va)
            test_parts.append(te)
        dev = pd.concat(dev_parts).sort_values(time_col, kind="stable")
        val = pd.concat(val_parts).sort_values(time_col, kind="stable")
        test = pd.concat(test_parts).sort_values(time_col, kind="stable")
    else:
        dev, val, test = _split_one(df)
    return dev.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def assert_no_temporal_overlap(
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str = "timestamp",
    category_col: str | None = "category",
) -> None:
    """
    Raise if dev-train/validation/test overlap in time.

    Checked **per category** when ``category_col`` is given, since the split is
    performed within each category — two categories can legitimately have
    different absolute date ranges.
    """
    def _check(dev_g, val_g, test_g, label: str) -> None:
        if len(dev_g) and len(val_g) and dev_g[time_col].max() > val_g[time_col].min():
            raise AssertionError(f"dev-train overlaps validation in time ({label})")
        if len(val_g) and len(test_g) and val_g[time_col].max() > test_g[time_col].min():
            raise AssertionError(f"validation overlaps final test in time ({label})")

    if category_col is None:
        _check(dev_df, val_df, test_df, label="all")
        return
    categories = set(dev_df[category_col]) | set(val_df[category_col]) | set(test_df[category_col])
    for cat in categories:
        _check(
            dev_df[dev_df[category_col] == cat],
            val_df[val_df[category_col] == cat],
            test_df[test_df[category_col] == cat],
            label=cat,
        )


def deterministic_stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    n: int = 12000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Deterministic, label-stratified sample of ``n`` rows (proportional
    per-class allocation), used for the matched-sample fair model-family
    comparison in RQ1 — the same sampled rows are reused for every model so
    that NB, LR and Gradient Boosting train on an identical training set.
    """
    frac = n / len(df)
    parts = []
    for _, group in df.groupby(label_col, sort=False):
        k = max(1, round(len(group) * frac))
        k = min(k, len(group))
        parts.append(group.sample(n=k, random_state=random_state))
    sample = pd.concat(parts)
    return sample.sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_task(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    text_col: str = "clean_text",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows with no label (e.g. 3-star reviews under binary framing)."""
    tr = train_df[train_df[label_col].notna()].copy()
    te = test_df[test_df[label_col].notna()].copy()
    return tr, te
