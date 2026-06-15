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
