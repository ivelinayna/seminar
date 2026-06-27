from src import features


def test_tfidf_fit_only_on_train():
    train_texts = ["great product love it", "terrible waste of money"]
    test_texts = ["unseenword appears only here", "great product"]
    Xtr, Xte, vec = features.fit_transform_split(train_texts, test_texts, ngram_range=(1, 1), max_features=100, min_df=1)
    vocab = set(vec.get_feature_names_out())
    # A token that appears ONLY in the test split must not have entered the vocabulary.
    assert "unseenword" not in vocab
    assert Xtr.shape[0] == len(train_texts)
    assert Xte.shape[0] == len(test_texts)
    assert Xtr.shape[1] == Xte.shape[1] == len(vocab)


def test_feature_configs_define_unigram_and_bigram():
    assert features.FEATURE_CONFIGS["tfidf_unigram"]["ngram_range"] == (1, 1)
    assert features.FEATURE_CONFIGS["tfidf_uni_bigram"]["ngram_range"] == (1, 2)
