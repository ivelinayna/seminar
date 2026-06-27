from src import models


def test_select_best_config_picks_highest_macro_f1():
    candidates = [
        {"name": "a", "f1_macro": 0.70, "recall_neutral": 0.50},
        {"name": "b", "f1_macro": 0.85, "recall_neutral": 0.10},
        {"name": "c", "f1_macro": 0.85, "recall_neutral": 0.20},  # ties b on f1, wins on recall tie-break
    ]
    ranked = models.select_best_config(candidates, minority_label="neutral")
    assert ranked.iloc[0]["name"] == "c"
    assert bool(ranked.iloc[0]["selected"]) is True
    assert ranked["selected"].sum() == 1


def test_resample_balanced_undersample_equalizes_classes():
    import numpy as np
    X = np.arange(20).reshape(20, 1)
    y = np.array(["a"] * 15 + ["b"] * 5)
    X_res, y_res = models.resample_balanced(X, y, strategy="undersample", random_state=42)
    counts = {cls: int((y_res == cls).sum()) for cls in ["a", "b"]}
    assert counts["a"] == counts["b"] == 5


def test_gradient_boosting_caps_training_sample():
    import numpy as np
    gb = models.get_gradient_boosting(max_train_samples=10, n_estimators=5)
    X = np.random.RandomState(0).rand(50, 3)
    y = np.array(["pos", "neg"] * 25)
    gb.fit(X, y)
    assert gb.fit_n_samples_ == 10
    assert gb.fit_sampled_ is True
