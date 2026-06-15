"""
Centralized spaCy pipeline management.

Why this module exists
----------------------
The original modules called ``spacy.load(...)`` on every function call, which
re-loads the model from disk for each review — prohibitively slow over ~87K
reviews. Here the pipelines are loaded once and cached.

Model vs. fallback
------------------
The project is built around ``en_core_web_sm``. If that model is not installed,
the module falls back to a blank English pipeline with a rule-based
``sentencizer`` and a lookup ``lemmatizer`` (requires ``spacy-lookups-data``).
The fallback keeps the whole pipeline runnable; lemmas are slightly less
accurate and noun-phrase chunking is delegated to an NLTK tagger. A one-line
notice is printed once so the runtime backend is never a silent surprise.
"""

from __future__ import annotations

import warnings

import spacy
from spacy.language import Language

MODEL_NAME = "en_core_web_sm"

# Module-level flags so callers can branch on what is actually available.
SPACY_MODEL_AVAILABLE: bool = False

_nlp_light: Language | None = None  # cleaning / lemmatization / sentence split
_nlp_parser: Language | None = None  # noun-phrase chunking (needs a parser)
_notified = False


def _notify_fallback() -> None:
    global _notified
    if not _notified:
        print(
            f"[nlp_engine] spaCy model '{MODEL_NAME}' not found — using a blank "
            "English pipeline (sentencizer + lookup lemmatizer). Results are valid; "
            "lemmas are slightly less accurate. Install the model with:\n"
            f"    python -m spacy download {MODEL_NAME}"
        )
        _notified = True


def _build_fallback_light() -> Language:
    """Blank pipeline: sentence boundaries + lookup-based lemmas."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    try:
        nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        nlp.initialize()
    except Exception:
        # spacy-lookups-data missing: lemmatizer falls back to identity.
        warnings.warn(
            "Lookup lemmatizer unavailable (install 'spacy-lookups-data'); "
            "tokens will not be lemmatized in fallback mode."
        )
    return nlp


def get_light_nlp() -> Language:
    """
    Pipeline for cleaning, lemmatization and sentence splitting.

    Parser and NER are disabled for speed; a ``sentencizer`` is added so that
    ``doc.sents`` works without the dependency parser.
    """
    global _nlp_light, SPACY_MODEL_AVAILABLE
    if _nlp_light is not None:
        return _nlp_light
    try:
        nlp = spacy.load(MODEL_NAME, disable=["parser", "ner"])
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        SPACY_MODEL_AVAILABLE = True
    except OSError:
        _notify_fallback()
        nlp = _build_fallback_light()
    _nlp_light = nlp
    return _nlp_light


def get_parser_nlp() -> Language | None:
    """
    Pipeline with a parser for ``noun_chunks`` (used by noun-phrase extraction).

    Returns ``None`` if the model is unavailable, signalling callers to use the
    NLTK-based noun-phrase fallback instead.
    """
    global _nlp_parser, SPACY_MODEL_AVAILABLE
    if _nlp_parser is not None:
        return _nlp_parser
    try:
        nlp = spacy.load(MODEL_NAME, disable=["ner"])
        SPACY_MODEL_AVAILABLE = True
        _nlp_parser = nlp
        return _nlp_parser
    except OSError:
        _notify_fallback()
        return None


def model_available() -> bool:
    """True if the full ``en_core_web_sm`` model backs the pipelines."""
    get_light_nlp()
    return SPACY_MODEL_AVAILABLE
