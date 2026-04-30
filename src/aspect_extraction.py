"""
Aspect extraction and sentence-level sentiment scoring for RQ2.

Implements two extraction strategies:
- Keyword-based: predefined domain lexicon (delivery, packaging, quality, etc.)
- Noun-phrase-based: spaCy POS tagging + frequency filtering

Sentence-level sentiment is attributed using the trained classifier from RQ1
(applied to sentences containing each aspect), addressing the methodological
concern raised by the supervisor about document-vs-aspect-level sentiment.
"""

from __future__ import annotations

from collections import Counter
import spacy

_NLP = spacy.load("en_core_web_sm")

# Initial keyword lexicon — extend based on EDA findings
DEFAULT_ASPECT_KEYWORDS = {
    "delivery": ["delivery", "shipping", "shipped", "arrived", "delayed", "fast"],
    "packaging": ["packaging", "package", "box", "wrapped", "damaged", "sealed"],
    "quality": ["quality", "build", "material", "durable", "cheap", "sturdy"],
    "price": ["price", "value", "expensive", "cheap", "worth", "affordable"],
    "customer_service": ["service", "support", "response", "helpful", "rude"],
}


def extract_aspects_keyword(text: str, lexicon: dict[str, list[str]] = None) -> list[str]:
    """Return aspect categories whose keywords appear in the text."""
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    text_lower = text.lower()
    found = []
    for aspect, keywords in lexicon.items():
        if any(kw in text_lower for kw in keywords):
            found.append(aspect)
    return found


def extract_noun_phrases(text: str, min_length: int = 2) -> list[str]:
    """Extract noun phrases as candidate aspects (alternative to keyword lookup)."""
    doc = _NLP(text)
    phrases = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if len(phrase.split()) >= 1 and len(phrase) >= min_length:
            phrases.append(phrase)
    return phrases


def top_noun_phrases(texts, n: int = 50) -> list[tuple[str, int]]:
    """Aggregate noun phrases across a corpus and return the most frequent."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(extract_noun_phrases(text))
    return counter.most_common(n)


def attribute_sentiment_per_aspect(
    review_text: str,
    aspects: list[str],
    classifier,
    vectorizer,
    aspect_lexicon: dict[str, list[str]] = None,
) -> dict[str, str | None]:
    """
    For each aspect, find sentences that mention it and predict their sentiment.

    Returns a dict mapping aspect -> predicted sentiment label (or None if
    no sentence in the review mentions that aspect).

    This addresses the supervisor's concern: instead of attributing the
    document-level sentiment to all aspects, sentiment is computed only on
    sentences that actually mention each aspect.
    """
    if aspect_lexicon is None:
        aspect_lexicon = DEFAULT_ASPECT_KEYWORDS

    doc = _NLP(review_text)
    sentences = [sent.text for sent in doc.sents]

    aspect_sentiments: dict[str, str | None] = {}
    for aspect in aspects:
        keywords = aspect_lexicon.get(aspect, [aspect])
        relevant = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        if not relevant:
            aspect_sentiments[aspect] = None
            continue
        X = vectorizer.transform(relevant)
        preds = classifier.predict(X)
        # Aggregate: majority vote across mentioning sentences
        from collections import Counter as _Counter
        most_common = _Counter(preds).most_common(1)[0][0]
        aspect_sentiments[aspect] = most_common

    return aspect_sentiments
