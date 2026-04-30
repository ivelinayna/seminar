"""
Text preprocessing utilities for Amazon review data.

Covers tokenization, lowercasing, stopword removal, and lemmatization.
Functions are designed to be applied consistently across train and evaluation
data to avoid distribution mismatch.
"""

from __future__ import annotations

import re
from typing import Iterable

import spacy

# Load spaCy model once at module import
_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Compile regex patterns once
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_MULTIPLE_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove HTML tags, URLs, and non-alphabetic characters; normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    text = _MULTIPLE_SPACES.sub(" ", text).strip().lower()
    return text


def tokenize_and_lemmatize(
    text: str,
    remove_stopwords: bool = True,
    min_token_length: int = 2,
) -> list[str]:
    """Tokenize, lemmatize, and optionally filter stopwords."""
    doc = _NLP(text)
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if remove_stopwords and token.is_stop:
            continue
        lemma = token.lemma_.strip().lower()
        if len(lemma) < min_token_length:
            continue
        tokens.append(lemma)
    return tokens


def preprocess_review(text: str) -> str:
    """Full preprocessing pipeline for a single review: clean -> tokenize -> rejoin."""
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


def preprocess_batch(texts: Iterable[str]) -> list[str]:
    """Apply preprocessing to a batch of reviews."""
    return [preprocess_review(t) for t in texts]


def split_into_sentences(text: str) -> list[str]:
    """Split a review into sentences for sentence-level aspect analysis (RQ2)."""
    nlp_with_parser = spacy.load("en_core_web_sm", disable=["ner"])
    doc = nlp_with_parser(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
