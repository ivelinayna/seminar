"""
Feature extraction for sentiment classification.

TF-IDF vectorisation with configurable n-gram ranges so that unigram and
unigram+bigram feature spaces can be compared (one of the RQ1 sub-questions).

The vectoriser is always fit on the **training split only** and then applied to
the test split, to avoid leaking test-set vocabulary statistics into training.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int = 20000,
    min_df: int = 5,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """
    Construct a TF-IDF vectoriser with sensible defaults.

    Parameters
    ----------
    ngram_range : tuple
        ``(1, 1)`` for unigrams, ``(1, 2)`` for unigrams + bigrams.
    max_features : int
        Cap on vocabulary size to keep models tractable.
    min_df : int
        Minimum document frequency; filters out very rare terms.
    max_df : float
        Maximum document frequency; filters out overly common terms.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


# Named feature configurations compared in RQ1.
FEATURE_CONFIGS: dict[str, dict] = {
    "tfidf_unigram": {"ngram_range": (1, 1)},
    "tfidf_uni_bigram": {"ngram_range": (1, 2)},
}


def fit_transform_split(train_texts, test_texts, **vectorizer_kwargs):
    """
    Fit a TF-IDF vectoriser on the training texts and transform both splits.

    Returns
    -------
    (X_train, X_test, vectorizer)
    """
    vectorizer = build_tfidf_vectorizer(**vectorizer_kwargs)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer
