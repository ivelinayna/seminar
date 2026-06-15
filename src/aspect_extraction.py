"""
Aspect extraction and sentence-level sentiment scoring for RQ2.

Two extraction strategies are compared (per the interim-presentation plan):

* **Keyword-based:** a curated domain lexicon (delivery, packaging, quality,
  price, content, customer service, billing/renewal). Interpretable and fast.
* **Noun-phrase-based:** spaCy noun-chunks (or an NLTK fallback when the model
  is absent), aggregated by frequency — a data-driven view of what reviewers
  actually talk about.

Sentiment attribution — the methodological core
----------------------------------------------
Sentiment is scored **per sentence with VADER**, a lexicon-based scorer, on the
sentences that mention each aspect, then aggregated per review and per category.
This is deliberate: the RQ1 classifier is trained on *whole reviews*, so applying
it to individual *sentences* is an out-of-distribution use (sentences are much
shorter, with a different feature distribution). VADER needs no training and
operates naturally at the sentence level, so it is the primary method. The
document classifier is retained only as a **robustness check**
(:func:`aspect_sentiment_classifier`), and agreement between the two is reported
rather than either being assumed correct.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .nlp_engine import get_parser_nlp
from .preprocessing import split_into_sentences, split_corpus_into_sentences, preprocess_review

# --------------------------------------------------------------------------- #
# Aspect lexicon
# --------------------------------------------------------------------------- #
# Extended beyond the original five: the EDA log-odds analysis surfaced
# billing/renewal vocabulary (cancel, refund, charged) as a major theme for
# subscription products, so it gets its own aspect.
DEFAULT_ASPECT_KEYWORDS: dict[str, list[str]] = {
    "delivery": ["delivery", "deliver", "shipping", "shipped", "arrived", "arrive",
                 "delayed", "late", "fast", "shipment", "courier"],
    "packaging": ["packaging", "package", "packaged", "box", "wrapped", "damaged",
                  "sealed", "broken", "crushed"],
    "quality": ["quality", "build", "material", "durable", "cheap", "sturdy",
                "flimsy", "well made", "poorly made"],
    "price": ["price", "priced", "value", "expensive", "overpriced", "worth",
              "affordable", "cost", "pricey"],
    "content": ["content", "selection", "variety", "curated", "curation", "items",
                "products", "magazine", "issue", "article", "articles"],
    "customer_service": ["service", "support", "response", "helpful", "rude",
                         "representative", "contacted", "email"],
    "billing": ["cancel", "cancelled", "cancellation", "refund", "refunded",
                "charged", "charge", "billing", "subscription", "renew", "renewal",
                "auto"],
}

_ANALYZER = SentimentIntensityAnalyzer()

# VADER convention for mapping the compound score to a label.
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05


# --------------------------------------------------------------------------- #
# Aspect extraction
# --------------------------------------------------------------------------- #
def extract_aspects_keyword(text: str, lexicon: dict[str, list[str]] | None = None) -> list[str]:
    """Return aspect categories whose keywords appear anywhere in the text."""
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    text_lower = text.lower()
    return [a for a, kws in lexicon.items() if any(kw in text_lower for kw in kws)]


def sentence_mentions_aspect(sentence_lower: str, keywords: list[str]) -> bool:
    """Whether a (lowercased) sentence mentions any keyword of an aspect."""
    return any(kw in sentence_lower for kw in keywords)


# --- noun-phrase extraction (spaCy if available, NLTK fallback otherwise) --- #
_NLTK_READY = False
_NP_CHUNKER = None
_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _ensure_nltk():
    global _NLTK_READY, _NP_CHUNKER
    if _NLTK_READY:
        return
    import nltk
    from nltk import RegexpParser
    grammar = r"NP: {<DT>?<JJ.*>*<NN.*>+}"
    _NP_CHUNKER = RegexpParser(grammar)
    _NLTK_READY = True


def _noun_phrases_nltk(text: str, min_length: int) -> list[str]:
    import nltk
    _ensure_nltk()
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []
    tagged = nltk.pos_tag(tokens)
    tree = _NP_CHUNKER.parse(tagged)
    phrases = []
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        phrase = " ".join(w for w, _ in subtree.leaves()).lower().strip()
        if len(phrase) >= min_length:
            phrases.append(phrase)
    return phrases


def extract_noun_phrases(text: str, min_length: int = 2) -> list[str]:
    """
    Extract candidate-aspect noun phrases from a single review.

    Uses spaCy ``noun_chunks`` when ``en_core_web_sm`` is available; otherwise
    falls back to an NLTK POS-tagger + regex NP chunker so the analysis still
    runs end-to-end.
    """
    nlp = get_parser_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if len(chunk.text.strip()) >= min_length
        ]
    return _noun_phrases_nltk(text, min_length)


def top_noun_phrases(texts, n: int = 50, drop_pronouns: bool = True) -> list[tuple[str, int]]:
    """Aggregate noun phrases across a corpus and return the most frequent ones."""
    stop_np = {"it", "i", "they", "you", "we", "he", "she", "this", "that",
               "these", "those", "me", "them", "us", "him", "her"}
    counter: Counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        for np_ in extract_noun_phrases(text):
            if drop_pronouns and np_ in stop_np:
                continue
            counter.update([np_])
    return counter.most_common(n)


# --------------------------------------------------------------------------- #
# Sentence-level sentiment — VADER (primary method)
# --------------------------------------------------------------------------- #
def vader_compound(sentence: str) -> float:
    """VADER compound polarity score in [-1, 1] for one sentence."""
    return _ANALYZER.polarity_scores(sentence)["compound"]


def vader_label(compound: float) -> str:
    """Map a compound score to positive / neutral / negative."""
    if compound >= VADER_POS_THRESHOLD:
        return "positive"
    if compound <= VADER_NEG_THRESHOLD:
        return "negative"
    return "neutral"


def aspect_sentiment_vader(
    review_text: str,
    lexicon: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """
    Per-aspect sentiment for one review using VADER on aspect-mentioning sentences.

    Returns a dict ``{aspect: {'compound': float, 'label': str, 'n_sentences': int}}``
    containing only the aspects actually mentioned in the review.
    """
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    sentences = split_into_sentences(review_text)
    return _aspects_from_sentences(sentences, lexicon)


def _aspects_from_sentences(sentences, lexicon) -> dict[str, dict]:
    bucket: dict[str, list[float]] = {}
    for sent in sentences:
        s_low = sent.lower()
        score = None
        for aspect, kws in lexicon.items():
            if sentence_mentions_aspect(s_low, kws):
                if score is None:
                    score = vader_compound(sent)
                bucket.setdefault(aspect, []).append(score)
    return {
        aspect: {
            "compound": float(np.mean(scores)),
            "label": vader_label(float(np.mean(scores))),
            "n_sentences": len(scores),
        }
        for aspect, scores in bucket.items()
    }


def build_aspect_sentiment_table(
    texts,
    categories,
    lexicon: dict[str, list[str]] | None = None,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Corpus-level long table of per-review, per-aspect VADER sentiment.

    One row per (review, mentioned aspect). Columns:
    ``review_idx, category, aspect, compound, label, n_sentences``.

    Sentence splitting is batched via ``nlp.pipe`` for speed.
    """
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    texts = list(texts)
    categories = list(categories)
    sentences_per_review = split_corpus_into_sentences(texts, batch_size=batch_size)

    rows = []
    for idx, (sents, cat) in enumerate(zip(sentences_per_review, categories)):
        for aspect, info in _aspects_from_sentences(sents, lexicon).items():
            rows.append({
                "review_idx": idx,
                "category": cat,
                "aspect": aspect,
                "compound": info["compound"],
                "label": info["label"],
                "n_sentences": info["n_sentences"],
            })
    return pd.DataFrame(rows)


def aggregate_by_category(long_df: pd.DataFrame, n_reviews_per_category: dict | None = None) -> pd.DataFrame:
    """
    Aggregate the long aspect table to per-(category, aspect) summary statistics.

    Columns: ``mention_count, mention_rate, mean_compound, pct_positive,
    pct_neutral, pct_negative``. ``mention_rate`` is the share of reviews in the
    category that mention the aspect (requires ``n_reviews_per_category``).
    """
    grp = long_df.groupby(["category", "aspect"])
    out = grp.agg(
        mention_count=("compound", "size"),
        mean_compound=("compound", "mean"),
    )
    label_share = (
        long_df.groupby(["category", "aspect"])["label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    for lbl in ["positive", "neutral", "negative"]:
        out[f"pct_{lbl}"] = label_share.get(lbl, 0.0)

    if n_reviews_per_category:
        out["mention_rate"] = [
            cnt / n_reviews_per_category.get(cat, np.nan)
            for (cat, _asp), cnt in out["mention_count"].items()
        ]
    return out.round(4)


# --------------------------------------------------------------------------- #
# Robustness check — RQ1 classifier applied to aspect sentences
# --------------------------------------------------------------------------- #
def aspect_sentiment_classifier(
    review_text: str,
    aspects: list[str],
    classifier,
    vectorizer,
    aspect_lexicon: dict[str, list[str]] | None = None,
) -> dict[str, str | None]:
    """
    Robustness check: predict aspect sentiment with the RQ1 document classifier.

    For each aspect, the classifier labels every mentioning sentence and a
    majority vote is taken. This is **not** the primary method — see the module
    docstring — but lets us quantify how far a document-trained model drifts when
    pushed down to the sentence level.
    """
    if aspect_lexicon is None:
        aspect_lexicon = DEFAULT_ASPECT_KEYWORDS
    sentences = split_into_sentences(review_text)

    result: dict[str, str | None] = {}
    for aspect in aspects:
        kws = aspect_lexicon.get(aspect, [aspect])
        relevant = [s for s in sentences if sentence_mentions_aspect(s.lower(), kws)]
        if not relevant:
            result[aspect] = None
            continue
        # Preprocess sentences the same way the classifier's training data was
        # (lemmatised clean_text), otherwise the vectoriser sees out-of-vocab raw text.
        processed = [preprocess_review(s) for s in relevant]
        preds = classifier.predict(vectorizer.transform(processed))
        result[aspect] = Counter(preds).most_common(1)[0][0]
    return result


def compare_vader_vs_classifier(
    texts,
    classifier,
    vectorizer,
    lexicon: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Agreement between VADER and the classifier on per-aspect sentiment.

    Run on a sample (e.g. the test split) for tractability. Returns one row per
    (review, aspect) with both labels and an ``agree`` flag. VADER's neutral
    label is mapped to the nearest polarity for a fair binary/ternary comparison
    against the classifier where needed by the caller.
    """
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    rows = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            continue
        v = aspect_sentiment_vader(text, lexicon)
        if not v:
            continue
        aspects = list(v.keys())
        c = aspect_sentiment_classifier(text, aspects, classifier, vectorizer, lexicon)
        for aspect in aspects:
            rows.append({
                "review_idx": idx,
                "aspect": aspect,
                "vader_label": v[aspect]["label"],
                "clf_label": c.get(aspect),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["agree"] = df["vader_label"] == df["clf_label"]
    return df
