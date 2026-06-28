"""
Aspect extraction and sentence-level sentiment scoring for RQ2.

Two extraction views are used:

* **Keyword-based:** a curated domain lexicon (delivery, packaging, quality,
  price, content, customer service, billing/renewal, advertising). Interpretable
  and fast.
* **Noun-phrase-based:** spaCy noun-chunks (or an NLTK fallback when the model
  is absent), aggregated by frequency as an exploratory candidate-discovery
  view and cross-check, not as the final aspect classifier.

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
from functools import lru_cache

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .nlp_engine import get_parser_nlp
from .preprocessing import split_into_sentences, split_corpus_into_sentences, preprocess_review, strip_html_urls


DEFAULT_ASPECT_KEYWORDS: dict[str, list[str]] = {
    "delivery": ["delivery", "deliver", "delivered", "shipping", "shipped",
                 "shipment", "courier", "arrive", "arrived", "arrival"],
    "packaging": ["packaging", "package", "packages", "packaged", "wrapping",
                  "wrapped", "wrapper", "mailer", "envelope", "container"],
    "quality": ["quality", "build", "construction", "workmanship", "fabric",
                "ingredient", "ingredients"],
    "price": ["price", "prices", "priced", "cost", "costs", "fee", "fees",
              "money", "dollar", "dollars"],
    "content": ["content", "selection", "variety", "curated", "curation",
                "item", "items", "product", "products", "issue", "issues",
                "article", "articles", "recipe", "recipes", "story", "stories",
                "editorial", "edition", "editions", "toy", "toys", "treat",
                "treats"],
    "customer_service": ["customer service", "customer support", "support team",
                         "service representative", "service representatives",
                         "customer representative", "customer representatives",
                         "support agent", "support agents",
                         "customer care", "contacted customer service",
                         "contacted support"],
    "billing": ["cancel", "canceled", "cancelled", "cancellation", "unsubscribe",
                "refund", "refunded", "charge", "charged", "charges", "billing",
                "bill", "billed", "payment", "payments", "invoice", "renew",
                "renewed", "renewal", "auto renew", "auto-renew",
                "automatic renewal", "automatically renewed"],
    "advertising": ["ad", "ads", "advertisement", "advertisements",
                    "advertising", "advertise", "advertised", "sponsored",
                    "commercial", "commercials", "promo", "promos"],
}

_ANALYZER = SentimentIntensityAnalyzer()

VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05


def _keyword_pattern(keyword: str) -> re.Pattern:
    """
    Compile a keyword/phrase matcher with word boundaries.

    This avoids the main false-positive problem of substring matching (e.g.
    matching ``ad`` inside unrelated words) while still allowing flexible
    whitespace inside multi-word phrases such as ``auto renew``.
    """
    parts = [re.escape(part) for part in keyword.lower().split()]
    pattern = r"\b" + r"\s+".join(parts) + r"\b"
    return re.compile(pattern)


@lru_cache(maxsize=256)
def _cached_keyword_pattern(keyword: str) -> re.Pattern:
    return _keyword_pattern(keyword)


def extract_aspects_keyword(text: str, lexicon: dict[str, list[str]] | None = None) -> list[str]:
    """Return aspect categories whose keywords appear anywhere in the text."""
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    text_lower = strip_html_urls(text).lower()
    return [a for a, kws in lexicon.items() if sentence_mentions_aspect(text_lower, kws)]


def sentence_mentions_aspect(sentence_lower: str, keywords: list[str]) -> bool:
    """Whether a lowercased sentence mentions any keyword/phrase as a token."""
    return any(_cached_keyword_pattern(kw).search(sentence_lower) for kw in keywords)


def matched_aspect_keywords(sentence_lower: str, keywords: list[str]) -> list[str]:
    """Return the concrete keywords/phrases matched in a sentence."""
    return [kw for kw in keywords if _cached_keyword_pattern(kw).search(sentence_lower)]


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
    text = strip_html_urls(text)
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
               "these", "those", "me", "them", "us", "him", "her", "what",
               "who", "which", "all", "some", "something", "everything",
               "anything", "nothing", "lots", "a lot"}
    counter: Counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        for np_ in extract_noun_phrases(text):
            if drop_pronouns and np_ in stop_np:
                continue
            counter.update([np_])
    return counter.most_common(n)


def noun_phrase_aspect_overlap(
    noun_phrase_counts: list[tuple[str, int]],
    lexicon: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Map frequent noun phrases back to keyword aspects for a transparent
    keyword-vs.-noun-phrase comparison.
    """
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    rows = []
    for phrase, count in noun_phrase_counts:
        phrase_lower = phrase.lower()
        matched = [
            aspect for aspect, kws in lexicon.items()
            if sentence_mentions_aspect(phrase_lower, kws)
        ]
        rows.append({
            "noun_phrase": phrase,
            "count": count,
            "matched_aspects": ";".join(matched) if matched else "",
        })
    return pd.DataFrame(rows)


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


HIGH_COVERAGE_MENTION_RATE_THRESHOLD = 0.05


def is_high_coverage(mention_rate: float, threshold: float = HIGH_COVERAGE_MENTION_RATE_THRESHOLD) -> bool:
    """Whether an aspect's mention rate clears the high-coverage threshold."""
    return mention_rate >= threshold


def billing_keyword_vader_valence(lexicon: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """VADER compound score of each billing keyword scored in isolation."""
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    return pd.DataFrame([
        {"keyword": kw, "vader_compound_isolated": vader_compound(kw)}
        for kw in lexicon["billing"]
    ])


def mask_keyword_in_sentence(sentence: str, keyword: str) -> str:
    """Replace a matched aspect keyword/phrase with a neutral placeholder token."""
    return _cached_keyword_pattern(keyword).sub("something", sentence)


def aspect_sentiment_vader_masked(
    review_text: str,
    lexicon: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """
    Lexicon-sensitivity counterpart to :func:`aspect_sentiment_vader`: before
    scoring a sentence with VADER, every matched aspect keyword is masked out
    (replaced with a neutral placeholder). If sentiment is largely driven by
    the matched keyword itself (e.g. "cancel", "refund") rather than the
    surrounding context, masking should pull the score toward neutral. This is
    a leakage/lexicon-sensitivity check, not an alternative primary method.
    """
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    sentences = split_into_sentences(review_text)
    bucket: dict[str, list[float]] = {}
    for sent in sentences:
        s_low = sent.lower()
        for aspect, kws in lexicon.items():
            matched = matched_aspect_keywords(s_low, kws)
            if not matched:
                continue
            masked_sent = sent
            for kw in matched:
                masked_sent = mask_keyword_in_sentence(masked_sent, kw)
            bucket.setdefault(aspect, []).append(vader_compound(masked_sent))
    return {
        aspect: {
            "compound": float(np.mean(scores)),
            "label": vader_label(float(np.mean(scores))),
            "n_sentences": len(scores),
        }
        for aspect, scores in bucket.items()
    }


def split_into_clauses(sentence: str) -> list[str]:
    """
    Conservative clause split for the RQ2 sensitivity analysis.

    Splits on a semicolon or on one of a small set of contrastive/coordinating
    markers (which typically signal a shift to a different aspect or
    polarity within the same sentence: "good food, but slow delivery").
    Splitting is intentionally narrow (word-boundary, case-insensitive) so it
    does not fragment ordinary sentences; this is a sensitivity check, not a
    replacement for the sentence-level primary method.
    """
    markers = [";", "but", "however", "although", "though", "yet", "while", "whereas"]
    pattern = re.compile(
        r"\s*;\s*|\b(?:but|however|although|though|yet|while|whereas)\b",
        re.IGNORECASE,
    )
    parts = [p.strip() for p in pattern.split(sentence) if p and p.strip()]
    return parts if parts else [sentence]


CLAUSE_SPLIT_MARKERS = [";", "but", "however", "although", "though", "yet", "while", "whereas"]


def aspect_sentiment_vader_clause(
    review_text: str,
    lexicon: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """
    Clause-level counterpart to :func:`aspect_sentiment_vader` (RQ2 sensitivity
    analysis). Sentences are first split into clauses on
    :data:`CLAUSE_SPLIT_MARKERS`; an aspect is then attributed only to the
    clause(s) that actually contain its keyword, so a sentence such as
    "the content is great but billing is a nightmare" assigns *content* to the
    first clause and *billing* to the second, instead of scoring both aspects
    against the whole (mixed-sentiment) sentence.
    """
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    sentences = split_into_sentences(review_text)
    clauses: list[str] = []
    for sent in sentences:
        clauses.extend(split_into_clauses(sent))
    return _aspects_from_sentences(clauses, lexicon)


def sentence_vs_clause_review_table(
    texts,
    categories,
    lexicon: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Per-review, per-aspect sentence-level vs. clause-level VADER compound, for
    the RQ2 clause-level sensitivity analysis. One row per (review, aspect)
    that is mentioned under either method.
    """
    if lexicon is None:
        lexicon = DEFAULT_ASPECT_KEYWORDS
    rows = []
    for idx, (text, cat) in enumerate(zip(texts, categories)):
        if not isinstance(text, str):
            continue
        sent_scores = aspect_sentiment_vader(text, lexicon)
        clause_scores = aspect_sentiment_vader_clause(text, lexicon)
        aspects = set(sent_scores) | set(clause_scores)
        for aspect in aspects:
            s = sent_scores.get(aspect)
            c = clause_scores.get(aspect)
            rows.append({
                "review_idx": idx,
                "category": cat,
                "aspect": aspect,
                "sentence_compound": s["compound"] if s else np.nan,
                "clause_compound": c["compound"] if c else np.nan,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["diff"] = df["clause_compound"] - df["sentence_compound"]
        df["affected"] = df["diff"].abs() > 1e-9
    return df


def bootstrap_review_level_aspect_ci(
    long_df: pd.DataFrame,
    n_replications: int = 2000,
    random_state: int = 42,
    ci: float = 0.95,
    minuend_category: str | None = None,
) -> pd.DataFrame:
    """
    Review-level (cluster) bootstrap CI for mean VADER compound per
    (category, aspect), plus the CI of the cross-category difference.

    Resamples whole **reviews** (``review_idx``) with replacement rather than
    individual aspect-sentence rows, since several rows in ``long_df`` can come
    from the same review and are not independent observations. ``long_df``
    must have columns ``review_idx, category, aspect, compound`` (the output of
    :func:`build_aspect_sentiment_table`).

    The difference column is named after the actual category strings (e.g.
    ``diff_Magazine_Subscriptions_minus_Subscription_Boxes``) so the direction
    is unambiguous from the column name alone — no separate "a"/"b" convention
    to misread when copying values into a report. ``minuend_category`` picks
    which category is the minuend (first term); it defaults to the
    alphabetically *first* category for a deterministic default, but callers
    that want a specific direction (e.g. "Magazine minus Boxes") should pass it
    explicitly.
    """
    rng = np.random.RandomState(random_state)
    categories = sorted(long_df["category"].unique())
    if len(categories) != 2:
        raise ValueError("Cross-category difference CI requires exactly two categories")
    if minuend_category is None:
        cat_minuend, cat_subtrahend = categories
    elif minuend_category == categories[0]:
        cat_minuend, cat_subtrahend = categories[0], categories[1]
    elif minuend_category == categories[1]:
        cat_minuend, cat_subtrahend = categories[1], categories[0]
    else:
        raise ValueError(f"minuend_category {minuend_category!r} not in {categories}")
    diff_col = f"diff_{cat_minuend}_minus_{cat_subtrahend}"
    aspects = sorted(long_df["aspect"].unique())

    pivots = {}
    for cat in categories:
        piv = (
            long_df[long_df["category"] == cat]
            .pivot_table(index="review_idx", columns="aspect", values="compound", aggfunc="mean")
            .reindex(columns=aspects)
        )
        pivots[cat] = piv.to_numpy(dtype=float)

    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    samples = {}
    with np.errstate(invalid="ignore"):
        for cat in categories:
            mat = pivots[cat]
            n = mat.shape[0]
            boot_means = np.empty((n_replications, len(aspects)))
            for i in range(n_replications):
                idx = rng.randint(0, n, size=n)
                boot_means[i] = np.nanmean(mat[idx], axis=0)
            samples[cat] = boot_means

    point_means = long_df.groupby(["category", "aspect"])["compound"].mean()
    rows = []
    for j, asp in enumerate(aspects):
        row = {"aspect": asp}
        for cat in categories:
            vals = samples[cat][:, j]
            vals = vals[~np.isnan(vals)]
            lo, hi = (np.percentile(vals, [lo_pct, hi_pct]) if len(vals) else (np.nan, np.nan))
            row[f"mean_{cat}"] = round(float(point_means.get((cat, asp), np.nan)), 4)
            row[f"ci_low_{cat}"] = round(float(lo), 4)
            row[f"ci_high_{cat}"] = round(float(hi), 4)
        diffs = samples[cat_minuend][:, j] - samples[cat_subtrahend][:, j]
        diffs = diffs[~np.isnan(diffs)]
        dlo, dhi = (np.percentile(diffs, [lo_pct, hi_pct]) if len(diffs) else (np.nan, np.nan))
        row[diff_col] = round(float(point_means.get((cat_minuend, asp), np.nan)) - float(point_means.get((cat_subtrahend, asp), np.nan)), 4)
        row["diff_ci_low"] = round(float(dlo), 4)
        row["diff_ci_high"] = round(float(dhi), 4)
        row["diff_ci_excludes_zero"] = bool(dlo > 0 or dhi < 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["minuend_category"] = cat_minuend
    out.attrs["subtrahend_category"] = cat_subtrahend
    out.attrs["diff_column"] = diff_col
    return out


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
