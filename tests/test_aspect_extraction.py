from src import aspect_extraction as ae


def test_ad_keyword_has_word_boundaries():
    kws = ae.DEFAULT_ASPECT_KEYWORDS["advertising"]
    assert ae.sentence_mentions_aspect("there were too many ads in this issue", kws)
    assert ae.sentence_mentions_aspect("the ad was annoying", kws)
    # "ad" must NOT match as a substring inside unrelated words.
    assert not ae.sentence_mentions_aspect("i added some context and felt bad about it", kws)
    assert not ae.sentence_mentions_aspect("please give advice on this", kws)
    assert not ae.sentence_mentions_aspect("the radio show was great", kws)


def test_box_magazine_subscription_are_not_keywords():
    # These are broad product identifiers, deliberately excluded from the
    # lexicon so they don't falsely trigger any aspect.
    text = "I bought this box, it is a magazine subscription"
    assert ae.extract_aspects_keyword(text) == []


def test_matched_aspect_keywords_word_boundary():
    matched = ae.matched_aspect_keywords("the box arrived broken and the packaging was crushed", ae.DEFAULT_ASPECT_KEYWORDS["packaging"])
    assert matched == ["packaging"]


def test_clause_split_separates_contrastive_clauses():
    clauses = ae.split_into_clauses("the content is great but billing is a nightmare")
    assert len(clauses) == 2
    assert "content" in clauses[0]
    assert "billing" in clauses[1]


def test_clause_level_assigns_aspect_only_to_matching_clause():
    text = "The content is great but billing is a nightmare, they keep charging me."
    sentence_level = ae.aspect_sentiment_vader(text)
    clause_level = ae.aspect_sentiment_vader_clause(text)
    # Sentence-level mixes both aspects into one (whole-sentence) score.
    assert sentence_level["content"]["compound"] == sentence_level["billing"]["compound"]
    # Clause-level must separate them: content should score more positive
    # than billing once each aspect only sees its own clause.
    assert clause_level["content"]["compound"] > clause_level["billing"]["compound"]


def test_high_coverage_threshold():
    assert ae.is_high_coverage(0.05)
    assert ae.is_high_coverage(0.10)
    assert not ae.is_high_coverage(0.049)
