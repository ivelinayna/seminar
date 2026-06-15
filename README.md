# Classical NLP for Sentiment and Aspect Analysis on Amazon Reviews

Seminar project (Enterprise AI, JMU Würzburg). Classical NLP/ML only — no LLMs.
Two categories from the Amazon Reviews 2023 dataset: **Subscription Boxes**
(16,216 reviews) and **Magazine Subscriptions** (71,497 reviews), 87,713 total.

## Research questions

- **Main RQ.** Trade-off between predictive performance and interpretability of
  classical ML for sentiment polarity and product-aspect extraction.
- **RQ1.** Document-level sentiment classification: how well do classical models
  do, how does class imbalance affect them, and how do they compare? Naive
  Bayes vs Logistic Regression vs Gradient Boosting on TF-IDF features.
- **RQ2.** Aspect-based sentiment: which aspects are mentioned, and how does
  sentiment toward each aspect differ between the two categories?

## Repository structure

```
.
├── data/
│   ├── raw/                 # the four .jsonl files (reviews + metadata) — not in the bundle
│   └── processed/           # cached parquet written by notebook 02
├── notebooks/
│   ├── 01_eda.ipynb         # exploratory data analysis
│   ├── 02_preprocessing.ipynb   # cleaning, labelling, chronological split  (pre-run)
│   ├── 03_modeling_rq1.ipynb    # RQ1: classifiers, imbalance, interpretability
│   ├── 04_aspects_rq2.ipynb     # RQ2: aspect sentiment (VADER), robustness  (pre-run)
│   └── 05_paper_figures.ipynb   # word clouds + aspect radar + lexicon validation  (pre-run)
├── paper/                   # LaTeX seminar paper (Overleaf-ready)
│   ├── main.tex             # the full paper
│   ├── references.bib       # bibliography (natbib/bibtex)
│   ├── figures/             # all figures the paper embeds (self-contained)
│   ├── uni-siegel2.eps      # university seal (auto-converted by Overleaf)
│   └── Seminar_Paper_PREVIEW.pdf  # local compile (seal shown as placeholder)
├── src/
│   ├── nlp_engine.py        # cached spaCy pipelines + graceful fallback
│   ├── preprocessing.py     # clean / tokenize / lemmatize / sentence-split
│   ├── dataset.py           # load, label (binary+ternary), chronological split
│   ├── features.py          # TF-IDF vectorisation
│   ├── models.py            # baseline + NB / LR / GB + resampling
│   ├── evaluation.py        # metrics, confusion plots, LaTeX export
│   └── aspect_extraction.py # keyword + noun-phrase aspects, VADER sentiment
├── results/
│   ├── figures/             # all PNGs
│   └── tables/              # all CSVs + LaTeX tables for the paper
├── models/                  # saved binary LR + TF-IDF (for the RQ2 robustness check)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The code is built around `en_core_web_sm`. If the model is missing it falls back
to a blank English pipeline (sentencizer + lookup lemmatizer); results stay
valid, lemmas are slightly less accurate. (This fallback is what was used to
verify the pipeline in a sandbox without model-download access.)

## How to run

Run the notebooks in order:

1. **`01_eda.ipynb`** — dataset understanding (already complete).
2. **`02_preprocessing.ipynb`** — writes `data/processed/reviews_clean.parquet`
   (~15 s). Run this before 03 and 04.
3. **`03_modeling_rq1.ipynb`** — RQ1. ~5 min: the two gradient-boosting
   configurations per framing are the slow part; Naive Bayes and Logistic
   Regression fit in under a second.
4. **`04_aspects_rq2.ipynb`** — RQ2. Loads the cached aspect table if present,
   otherwise builds it (~40 s).

All figures and tables are (re)written into `results/` on each run.

5. **`05_paper_figures.ipynb`** — generates the two figures the task names
   explicitly (per-sentiment word clouds, aspect radar charts) plus a qualitative
   aspect-lexicon validation sample. Runs in seconds from the cached artefacts.

## Paper

`paper/main.tex` is a complete, Overleaf-ready LaTeX seminar paper (~19 pages)
built on the JMU Würzburg title-page style, with every figure and table embedded
and the real results filled in. To compile on Overleaf: upload the `paper/`
folder, set the main document to `main.tex`, and compile (pdfLaTeX + bibtex). The
university seal (`uni-siegel2.eps`) is converted automatically by Overleaf.
`Seminar_Paper_PREVIEW.pdf` is a local compile for quick reading (the seal shows
as a placeholder box there only; it renders correctly on Overleaf). Title-page
placeholders to fill: matriculation number, supervisor's full name, and the exact
chair name.

## Methodology

**RQ1 — document level, supervised.** Labels come from star ratings, which only
exist per review, so RQ1 stays at the review level. Two framings: **binary**
(1-2 = negative, 4-5 = positive, 3 dropped) and **ternary** (3 = neutral); both
are reported, binary leading. The split is **chronological within each category**
(oldest 80% train / newest 20% test) to mimic deployment while keeping both
categories on both sides. Because the classes are imbalanced, the primary
metrics are **macro-F1 and per-class recall**; accuracy is reference only. Each
model is run with and without imbalance handling (`class_weight='balanced'` for
LR/GB, random undersampling for NB, which has no class-weight parameter).

**RQ2 — sentence level, unsupervised.** A review-level label cannot express that
one review praises the content but criticises the price, so RQ2 splits reviews
into sentences, detects aspects per sentence (keyword lexicon + a noun-phrase
view), and scores each aspect-mentioning sentence with **VADER**.

> **Why VADER rather than the RQ1 classifier for aspect sentiment?** The RQ1
> model is trained on whole reviews; applying it to single sentences is
> out-of-distribution (sentences are much shorter, different feature
> distribution). VADER is lexicon-based, needs no training, and is designed for
> short text — so it is the primary sentence-level scorer. The RQ1 classifier is
> retained only as a **robustness check** (notebook 04, Section 6), and agreement
> between the two methods is reported.

The aspect lexicon extends the five conventional aspects (delivery, packaging,
quality, price, customer service) with two that the EDA motivated: **content**
(the substance of a subscription) and **billing** (cancel/refund/charged — the
negative-word cluster the log-odds analysis surfaced).

## Key results

**RQ1.**
- The majority-class baseline reaches ~69% accuracy but macro-F1 ≈ 0.41 with
  **zero recall on the negative class** — accuracy alone is misleading.
- Under the binary framing, **Logistic Regression (balanced)** is best:
  macro-F1 ≈ 0.87, negative recall lifted from ≈ 0.67 (no handling) to ≈ 0.88.
- Under the ternary framing the neutral (3★) class collapses without handling
  (Naive Bayes neutral recall ≈ 0.4%); balancing raises LR neutral recall from
  ≈ 9% to ≈ 43%, at an accuracy cost (≈ 0.81 → ≈ 0.77) — the genuine trade-off.
- Bigrams give a small but real gain over unigrams (macro-F1 ≈ 0.860 → 0.870).
- **Interpretability:** Logistic Regression matches gradient boosting here while
  remaining fully auditable; its strongest negative token is *cancel* — a
  billing word, anticipating RQ2.

**RQ2.**
- *What* is discussed is category-specific: **content** dominates magazine
  reviews (≈ 63% mention it), **packaging** dominates subscription-box reviews
  (≈ 47%).
- *What frustrates* customers is shared: **billing** has the lowest mean
  sentiment in both categories (≈ 0.17–0.18 compound, 21–28% negative);
  content and price are the most positive.
- Robustness: VADER and the document classifier agree on ≈ 76% of non-neutral
  aspect mentions, and VADER labels ≈ 22% of mentions neutral — a class the
  binary classifier structurally cannot express.

## Limitations

- **Sentence-level attribution** cannot separate multiple aspects that co-occur
  in one sentence with mixed sentiment ("great … but overpriced"): they all
  inherit the sentence's single VADER score. Clause-level splitting on
  coordinating conjunctions is the natural refinement.
- The aspect lexicon is hand-built and English-only; VADER is general-purpose,
  not domain-tuned.
- No aspect-level gold labels exist, so RQ2 is unsupervised and cannot be scored
  against ground truth — only cross-validated against a second method.
- Subscription Boxes contains only 641 distinct items, so its aspect frequencies
  can be influenced by a few popular boxes.
