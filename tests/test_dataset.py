import pandas as pd
import pytest

from src import dataset


def test_map_binary_thresholds():
    assert dataset.map_binary(1) == "negative"
    assert dataset.map_binary(2) == "negative"
    assert dataset.map_binary(3) is None
    assert dataset.map_binary(4) == "positive"
    assert dataset.map_binary(5) == "positive"


def test_map_ternary_thresholds():
    assert dataset.map_ternary(1) == "negative"
    assert dataset.map_ternary(2) == "negative"
    assert dataset.map_ternary(3) == "neutral"
    assert dataset.map_ternary(4) == "positive"
    assert dataset.map_ternary(5) == "positive"


def _toy_df(n_per_category=100):
    rows = []
    for cat in ["A", "B"]:
        for t in range(n_per_category):
            rows.append({"category": cat, "timestamp": t})
    return pd.DataFrame(rows)


def test_chronological_split_three_way_proportions():
    df = _toy_df(100)
    dev, val, test = dataset.chronological_split_three_way(df)
    # 64/16/20 per category (200 rows total -> 128/32/40)
    assert len(dev) == 128
    assert len(val) == 32
    assert len(test) == 40
    # dev+val reproduces the legacy oldest-80% split
    train80, _ = dataset.chronological_split(df, test_size=0.2, by_category=True)
    assert len(dev) + len(val) == len(train80)


def test_chronological_split_three_way_no_temporal_overlap():
    df = _toy_df(100)
    dev, val, test = dataset.chronological_split_three_way(df)
    dataset.assert_no_temporal_overlap(dev, val, test)
    for cat in ["A", "B"]:
        d = dev[dev["category"] == cat]["timestamp"]
        v = val[val["category"] == cat]["timestamp"]
        te = test[test["category"] == cat]["timestamp"]
        assert d.max() <= v.min()
        assert v.max() <= te.min()


def test_assert_no_temporal_overlap_raises_on_overlap():
    dev = pd.DataFrame({"category": ["A"], "timestamp": [10]})
    val = pd.DataFrame({"category": ["A"], "timestamp": [5]})  # overlaps: before dev's max
    test = pd.DataFrame({"category": ["A"], "timestamp": [20]})
    with pytest.raises(AssertionError):
        dataset.assert_no_temporal_overlap(dev, val, test)


def test_prepare_task_drops_unlabelled_rows():
    train_df = pd.DataFrame({"sentiment_binary": ["positive", None, "negative"], "clean_text": ["a", "b", "c"]})
    test_df = pd.DataFrame({"sentiment_binary": [None, "positive"], "clean_text": ["d", "e"]})
    tr, te = dataset.prepare_task(train_df, test_df, "sentiment_binary")
    assert tr["sentiment_binary"].notna().all()
    assert te["sentiment_binary"].notna().all()
    assert len(tr) == 2
    assert len(te) == 1
