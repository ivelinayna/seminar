"""
Feature extraction for sentiment classification.

Implements TF-IDF vectorization with configurable n-gram ranges to compare
unigram and bigram features (relevant for RQ1).
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
    Construct a TF-IDF vectorizer with sensible defaults.

    Parameters
    ----------
    ngram_range : tuple
        (1, 1) for unigrams only, (1, 2) for unigrams + bigrams.
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
