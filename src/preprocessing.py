"""
Text preprocessing utilities for Amazon review data.

Covers HTML/URL stripping, lowercasing, tokenization, stopword removal and
lemmatization. The same functions are applied to train and evaluation data so
that the feature distribution is consistent across splits.

Performance
-----------
For corpus-scale work, prefer :func:`preprocess_corpus`, which uses spaCy's
``nlp.pipe`` for batched processing — roughly an order of magnitude faster than
calling :func:`preprocess_review` in a Python loop.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence
from html import unescape

from .nlp_engine import get_light_nlp

# Compile regex patterns once.
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_MULTIPLE_SPACES = re.compile(r"\s+")

# _NON_ALPHA strips apostrophes before tokenization, so contracted negations
# must be expanded to their full word form first (e.g. "don't" -> "do not"),
# or the "not" cue is lost ("don't" would otherwise become "don t").
_APOSTROPHE = "['’]"
_WONT = re.compile(r"\bwon" + _APOSTROPHE + r"t\b")
_CANT_OR_CANNOT = re.compile(r"\b(can" + _APOSTROPHE + r"t|cannot)\b")
_GENERIC_NT = re.compile(r"\b(\w+)n" + _APOSTROPHE + r"t\b")


def _expand_negation_contractions(text: str) -> str:
    """Expand contracted negations (won't, can't, don't, isn't, ...) to full words."""
    text = _WONT.sub("will not", text)
    text = _CANT_OR_CANNOT.sub("can not", text)
    # Generic "n't" -> " not" covers don't, isn't, wasn't, doesn't, didn't,
    # haven't, hadn't, shouldn't, wouldn't, couldn't, hasn't, weren't, aren't.
    text = _GENERIC_NT.sub(lambda m: f"{m.group(1)} not", text)
    return text


# Keep simple negation cues even when stopwords are removed. They are central
# sentiment features and allow bigrams such as "not worth" to survive.
NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "nor",
    "neither",
    "cannot",
}


def strip_html_urls(text: str) -> str:
    """Remove HTML tags/entities and URLs while preserving ordinary punctuation."""
    if not isinstance(text, str):
        return ""
    text = unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    return _MULTIPLE_SPACES.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Remove HTML tags, URLs and non-alphabetic characters; normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = strip_html_urls(text).lower()
    text = _expand_negation_contractions(text)
    text = _NON_ALPHA.sub(" ", text)
    text = _MULTIPLE_SPACES.sub(" ", text).strip()
    return text


def _lemmatize_doc(doc, remove_stopwords: bool, min_token_length: int) -> list[str]:
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        token_lower = token.text.strip().lower()
        lemma = token.lemma_.strip().lower() or token_lower
        if remove_stopwords and token.is_stop and token_lower not in NEGATION_TERMS and lemma not in NEGATION_TERMS:
            continue
        if len(lemma) < min_token_length:
            continue
        tokens.append(lemma)
    return tokens


def tokenize_and_lemmatize(
    text: str,
    remove_stopwords: bool = True,
    min_token_length: int = 2,
) -> list[str]:
    """Tokenise, lemmatise and optionally drop stopwords for a single string."""
    nlp = get_light_nlp()
    return _lemmatize_doc(nlp(text), remove_stopwords, min_token_length)


def preprocess_review(text: str) -> str:
    """Full pipeline for one review: clean -> tokenize/lemmatize -> rejoin."""
    cleaned = clean_text(text)
    return " ".join(tokenize_and_lemmatize(cleaned))


def preprocess_corpus(
    texts: Sequence[str],
    remove_stopwords: bool = True,
    min_token_length: int = 2,
    batch_size: int = 1000,
    n_process: int = 1,
) -> list[str]:
    """
    Batched preprocessing for a whole corpus (clean -> lemmatize -> rejoin).

    Cleaning is applied first (cheap, vectorised over the list), then spaCy
    processes the cleaned strings with ``nlp.pipe`` for speed.
    """
    cleaned = [clean_text(t) for t in texts]
    nlp = get_light_nlp()
    out: list[str] = []
    for doc in nlp.pipe(cleaned, batch_size=batch_size, n_process=n_process):
        out.append(" ".join(_lemmatize_doc(doc, remove_stopwords, min_token_length)))
    return out


# Backwards-compatible alias for the original API.
def preprocess_batch(texts: Iterable[str]) -> list[str]:
    """Apply preprocessing to a batch of reviews (delegates to preprocess_corpus)."""
    return preprocess_corpus(list(texts))


def split_into_sentences(text: str) -> list[str]:
    """
    Split one review into sentences for sentence-level aspect analysis (RQ2).

    Uses the cached light pipeline (with a ``sentencizer``); does **not** reload
    spaCy on every call as the original implementation did.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text = strip_html_urls(text)
    nlp = get_light_nlp()
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]


def split_corpus_into_sentences(
    texts: Sequence[str],
    batch_size: int = 1000,
) -> list[list[str]]:
    """Batched sentence splitting for a corpus; returns one list per document."""
    nlp = get_light_nlp()
    results: list[list[str]] = []
    for doc in nlp.pipe((strip_html_urls(t) if isinstance(t, str) else "" for t in texts),
                        batch_size=batch_size):
        results.append([s.text.strip() for s in doc.sents if s.text.strip()])
    return results
