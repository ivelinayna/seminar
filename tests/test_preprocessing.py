from src import preprocessing


def test_negation_survives_stopword_removal():
    cleaned = preprocessing.preprocess_review("This is not good at all")
    tokens = cleaned.split()
    assert any(t in ("not", "no", "never") for t in tokens), f"negation cue missing from: {cleaned!r}"


def test_negation_survives_contraction():
    for raw in [
        "I don't like it",
        "It isn't good",
        "It wasn't worth the money",
        "I can't recommend it",
        "She won't buy this again",
    ]:
        cleaned = preprocessing.preprocess_review(raw)
        assert "not" in cleaned.split(), f"negation cue missing from: {cleaned!r} (input: {raw!r})"


def test_clean_text_strips_html_and_urls():
    raw = "<b>Great</b> product! Check http://example.com for more."
    cleaned = preprocessing.clean_text(raw)
    assert "<b>" not in cleaned
    assert "http" not in cleaned
    assert "great" in cleaned


def test_preprocess_corpus_matches_single_review(monkeypatch=None):
    texts = ["I love this magazine", "Terrible, would not recommend"]
    batch = preprocessing.preprocess_corpus(texts)
    singles = [preprocessing.preprocess_review(t) for t in texts]
    assert batch == singles
