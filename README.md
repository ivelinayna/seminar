# Classical NLP for Sentiment and Aspect Analysis on Amazon Reviews

Seminar project (Enterprise AI and Urban Analytics, JMU Würzburg). Classical NLP/ML only — no LLMs.
Two categories from the Amazon Reviews 2023 dataset: **Subscription Boxes**
(16,216 reviews) and **Magazine Subscriptions** (71,497 reviews), 87,713 total.

## Research questions

- **Main RQ.** To what extent can classical NLP and machine-learning methods
  extract review-level sentiment and product-related aspects from Amazon
  reviews, and what trade-offs between predictive performance, interpretability,
  and business usefulness emerge?
- **RQ1.** How do Naive Bayes, Logistic Regression, and Gradient Boosting compare
  in document-level sentiment classification, and what do their systematic
  errors, particularly for 3-star reviews, reveal about the limitations of
  bag-of-words representations?
- **RQ2.** Which product aspects are most frequently mentioned, and how do their
  sentence-level sentiment profiles differ across the two selected categories?

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
│   └── main.pdf               # final locally compiled PDF
├── src/
│   ├── nlp_engine.py        # cached spaCy pipelines
│   ├── preprocessing.py     # clean / tokenize / lemmatize / sentence-split
│   ├── dataset.py           # load, label, chronological 2-way/3-way split, matched sampling
│   ├── features.py          # TF-IDF vectorisation
│   ├── models.py            # baseline + NB / LR / GB + resampling + hyperparameter grids/selection
│   ├── evaluation.py        # metrics, bootstrap CIs, confusion plots, LaTeX export
│   └── aspect_extraction.py # keyword + noun-phrase + clause-level aspects, VADER sentiment, bootstrap
├── tests/                   # pytest suite for src/ (splits, leakage, negation, aspect matching, bootstrap)
├── results/
│   ├── figures/             # all PNGs
│   └── tables/              # all CSVs + LaTeX tables for the paper
├── models/                  # saved binary LR + TF-IDF (for the RQ2 robustness check)
├── Makefile                 # make test / make notebooks / make paper / make all
├── requirements.txt
├── requirements-lock.txt    # pinned exact versions (pip freeze) for exact reproduction
└── README.md
```

## Setup

### Data acquisition

This project uses the **Amazon Reviews 2023** release (McAuley Lab / UCSD):
https://amazon-reviews-2023.github.io/. Download the review and metadata files
for the **Subscription Boxes** and **Magazine Subscriptions** categories
(`.jsonl.gz`, decompress before use) from that page and place them in
`data/raw/`:

- `Subscription_Boxes.jsonl`
- `meta_Subscription_Boxes.jsonl`
- `Magazine_Subscriptions.jsonl`
- `meta_Magazine_Subscriptions.jsonl`

The raw files are not committed because of size/licensing.

### Environment

- **Python 3.13** (tested with 3.13.13). A `.venv` virtual environment is recommended.
- Exact, pinned dependency versions used for the final run are in
  `requirements-lock.txt` (`pip freeze` snapshot); `requirements.txt` lists the
  loose version ranges actually required.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact reproduction
# or: pip install -r requirements.txt  # loose ranges, latest compatible versions
python -m spacy download en_core_web_sm
```

The code is built around spaCy's **`en_core_web_sm` 3.8.x** pipeline. Install the
model before the final run so sentence splitting, lemmatisation and noun-phrase
extraction are reproducible. If the model is unavailable, `src/nlp_engine.py`
falls back to a blank pipeline (sentencizer + lookup lemmatizer); results stay
valid but lemmas are slightly less accurate.

### Random seeds

Every stochastic operation in this project uses a fixed seed (`random_state=42`
throughout: dataset sampling, model fitting, resampling, bootstrap replications,
and the word-cloud layout in notebook 05). All bootstrap procedures (`src/evaluation.py`,
`src/aspect_extraction.py`) additionally fix `n_replications=2000` for full
reproducibility of confidence intervals.

## How to run

Run the notebooks in order:

1. **`01_eda.ipynb`** — dataset understanding (~1 minute).
2. **`02_preprocessing.ipynb`** — writes `data/processed/reviews_clean.parquet`
   and demonstrates the dev-train/validation/final-test split (~2-5 minutes,
   dominated by spaCy preprocessing). Run this before 03 and 04.
3. **`03_modeling_rq1.ipynb`** — RQ1. Model, feature and hyperparameter
   selection use **only** a validation split (64% dev-train / 16% validation /
   20% final test, chronological, per category); the final test set is
   accessed only after model, feature and hyperparameter choices are fixed and
   is never used for selection or tuning (`src/dataset.py`,
   `chronological_split_three_way`). Includes a small deterministic
   hyperparameter grid search (the slowest part, ~10-15 minutes), bootstrap
   confidence intervals (2,000 replications), a matched-sample fair
   model-family comparison, and a direct analysis of 3-star test reviews.
   The final Gradient-Boosting comparison uses
   `sklearn.ensemble.GradientBoostingClassifier`, trained on a deterministic
   stratified sample of 12,000 reviews for tractable reproduction; this is
   reported transparently throughout (`results/tables/rq1_gradient_boosting_metadata.*`).
   **Total runtime: roughly 15-20 minutes.**
4. **`04_aspects_rq2.ipynb`** — RQ2. For final reproduction, rebuild the aspect
   table from the current lexicon (`REBUILD_ASPECT_TABLE = True`) after deleting
   old processed/cache files. Includes a clause-level sensitivity analysis,
   a billing-lexicon masking sensitivity check, and review-level bootstrap CIs
   for aspect sentiment (~3-5 minutes).
5. **`05_paper_figures.ipynb`** — generates the word clouds and the aspect
   radar charts (with a dynamic, non-clipping radial range), plus an
   illustrative qualitative spot-check of the aspect lexicon. Runs in seconds
   from the cached artefacts.

All figures and tables are (re)written into `results/` on each run.

Reproducible command-line run (or `make notebooks`, see `Makefile`):

```bash
find data/processed -maxdepth 1 -type f \( -name '*.parquet' -o -name '*.csv' \) -delete
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling_rq1.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_aspects_rq2.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_paper_figures.ipynb
```

Expected total runtime on the final local environment is roughly 25-35 minutes,
dominated by the RQ1 hyperparameter grid search and the spaCy preprocessing/aspect
extraction passes.

### Tests

```bash
python -m pytest tests/ -v   # or: make test
```

Covers: star-rating-to-sentiment mapping, the chronological dev-train/validation/
final-test split (proportions and no temporal overlap), TF-IDF fit only on
training data, negation surviving preprocessing, aspect keyword matching with
word boundaries (`ad` does not match inside other words; `box`/`magazine`/
`subscription` never trigger an aspect), clause-level aspect assignment, and
bootstrap reproducibility under a fixed seed.

## Paper

`paper/main.tex` is a complete, Overleaf-ready LaTeX seminar paper. The current
local PDF has 28 pages in total; the counted core text from Introduction through
Conclusion is 20 pages because the References begin on PDF page 25. The paper is
built on the JMU Würzburg title-page style, with every figure and table embedded
and the real results filled in. To compile on Overleaf: upload the `paper/`
folder, set the main document to `main.tex`, and compile (pdfLaTeX + bibtex). The
university seal (`uni-siegel2.eps`) is converted automatically by Overleaf.
`paper/main.pdf` is the current local compile; the title page already contains
the seminar title, matriculation number and supervisor.

Local compile:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Methodology

**RQ1 — document level, supervised.** Labels come from star ratings, which only
exist per review, so RQ1 stays at the review level. Two framings: **binary**
(1-2 = negative, 4-5 = positive, 3 dropped) and **ternary** (3 = neutral); both
are reported, binary leading. The split is **chronological within each category**
to mimic deployment while keeping both categories on both sides.

To avoid test-set leakage during model selection, the split is three-way:
**oldest 64% (dev-train) / next 16% (validation) / newest 20% (final test)**
(`src/dataset.py::chronological_split_three_way`). Feature representation
(unigram vs. uni+bigram), imbalance handling, and a small deterministic
hyperparameter grid (Naive Bayes `alpha`; Logistic Regression `C` ×
`class_weight`; Gradient Boosting `n_estimators` × `learning_rate` ×
`max_depth` × `class_weight`) are selected using **only the validation slice**
(macro-F1 primary, minority-class recall as tie-breaker,
`src/models.py::select_best_config`). The selected configuration is then
retrained on the full oldest 80% and evaluated on the final 20% test set, which
is accessed only after those choices are fixed and is never used for model
selection or tuning. Because the classes are imbalanced, the primary metrics are
**macro-F1 and per-class recall**; accuracy is reference only.

Two further result blocks isolate fair comparison from a leakage-free
selection: a **matched-sample comparison** (NB, LR and Gradient Boosting all
trained on an identical, deterministically sampled 12,000-row training set,
`src/dataset.py::deterministic_stratified_sample`) and a **full-data
comparison** (NB and LR on the full training pool; Gradient Boosting is
excluded there because its training is internally capped at 12,000 rows
regardless of input size, so a "full-data" run would not reflect more data than
the matched comparison). Final-test metrics are reported with **bootstrap 95%
confidence intervals** (2,000 replications, fixed seed,
`src/evaluation.py::bootstrap_classification_report_ci`), including a paired
bootstrap for the unigram-vs-bigram macro-F1 difference
(`paired_bootstrap_macro_f1_diff`).

**RQ2 — sentence level, unsupervised.** A review-level label cannot express that
one review praises the content but criticises the price, so RQ2 splits reviews
into sentences, detects aspects per sentence with the keyword lexicon, uses
noun-phrase chunking as exploratory candidate discovery/cross-check, and scores
each aspect-mentioning sentence with **VADER**.

> **Why VADER rather than the RQ1 classifier for aspect sentiment?** The RQ1
> model is trained on whole reviews; applying it to single sentences is
> out-of-distribution (sentences are much shorter, different feature
> distribution). VADER is lexicon-based, needs no training, and is designed for
> short text — so it is the primary sentence-level scorer. The RQ1 classifier is
> retained only as a **robustness check** (notebook 04, Section 6), and agreement
> between the two methods is reported as a limited consistency check.

The aspect lexicon extends the five conventional aspects (delivery, packaging,
quality, price, customer service) with three that the EDA motivated: **content**
(the substance of the product), **billing** (cancel/refund/charged), and
**advertising**. The final lexicon is intentionally conservative: broad product
identifiers such as `box`, `magazine`, and `subscription`, as well as sentiment
words such as `cheap`, `expensive`, `damaged`, `helpful`, and `rude`, are not
used as aspect-detection triggers.

**"High coverage" / "substantial" aspects are formally defined**: an aspect
counts as high-coverage within a category if at least **5%** of that
category's reviews mention it (`src/aspect_extraction.py::is_high_coverage`,
`HIGH_COVERAGE_MENTION_RATE_THRESHOLD = 0.05`). This single threshold is used
consistently in the notebook, the paper's methodology, results, discussion and
conclusion.

**Sentence-level attribution is the primary method** (per supervisor
feedback), but cannot separate two aspects that co-occur in one sentence with
mixed sentiment. A **clause-level sensitivity analysis**
(`split_into_clauses`, splitting conservatively on `;` or
`but/however/although/though/yet/while/whereas`) checks whether this changes
the headline findings; it is reported as a robustness check, not a
replacement. A **billing-lexicon sensitivity check** scores billing sentiment
with the matched keyword masked out before VADER scoring, to check whether the
billing-negative finding is driven by lexicon artefacts or by genuine
surrounding context. **Review-level (cluster) bootstrap CIs** (2,000
replications) are reported for mean aspect sentiment per category and for the
cross-category difference, since multiple aspect-sentence rows can come from
the same review and are not independent observations.

## Key results

**RQ1.** Model/feature/hyperparameter selection used **only the validation
slice**; numbers below are from the final test slice after selection
(`results/tables/rq1_binary_comparison.csv`, `rq1_ternary_comparison.csv`).
- The majority-class baseline reaches ~69% accuracy on the final test slice but
  macro-F1 ≈ 0.41 with
  **zero recall on the negative class** — accuracy alone is misleading.
- Under the binary framing, **Logistic Regression (balanced, C=2.0)** is the
  **strongest observed model under the evaluated setup**: macro-F1 = 0.888
  (95% class-stratified bootstrap CI [0.883, 0.893]), recall 0.910/0.884 (positive/negative).
- Under the ternary framing the neutral (3★) class collapses without handling
  (Naive Bayes neutral recall ≈ 0.9% on validation); the selected balanced LR
  (C=0.5) reaches macro-F1 = 0.644 (CI [0.636, 0.652]), neutral recall = 0.448
  — at a real accuracy cost (0.785 vs. ≈0.85 unbalanced) — the genuine trade-off.
  A **direct analysis of all 1,248 true 3-star test reviews**
  (`rq1_3star_error_analysis*.csv`) shows the model's confidence is similar
  whether right or wrong — this is a representational limitation, not a
  confidence problem.
- A **fair matched-sample comparison** (identical 12,000 rows for NB/LR/GB,
  each with its own validation-selected hyperparameters,
  `rq1_matched_sample_comparison.csv`) confirms LR wins even when every model
  sees the same data (macro-F1 0.876 vs. GB 0.815 vs. NB 0.814) — ruling out
  training-data volume as the explanation.
- Bigrams give a **small, stable gain** over unigrams (final-test macro-F1
  0.875 → 0.888; paired class-stratified bootstrap CI of the difference [0.009, 0.016],
  excludes zero), and a sanity check confirms 1,338 negation-bearing
  unigram/bigram features (e.g. `not worth`) survive preprocessing.
- **Interpretability:** Logistic Regression is the strongest observed model
  without sacrificing feature-level interpretability, including in the
  matched-sample comparison above. The strongest general negative cue is
  *not*; *cancel* is the strongest domain-specific negative feature.

**RQ2.** ("High coverage" / "substantial" = mention rate ≥ 5% within a category, see Methodology.)
- *What* is discussed is mostly **content** after lexicon cleaning: ≈ 43% of
  subscription-box reviews and ≈ 39% of magazine reviews mention product or
  editorial substance (both high-coverage). Packaging drops to ≈ 6% for boxes
  once `box/boxes` are removed as product-name artefacts (≈ 0.3% for magazines).
- *What frustrates* customers is shared: **billing is the weakest high-coverage
  aspect in both categories** (mean ≈ -0.09 for Boxes, ≈ 57% negative). Its
  negative polarity is **not robust for Subscription Boxes** once the
  masking check below removes the triggering terms themselves. For Magazine Subscriptions billing's
  mean (≈ 0.05) is only slightly positive — still the weakest high-coverage
  aspect there, but not described as an unconditional pain point. Customer
  service in Magazines is numerically more negative (≈ -0.03) but is a
  low-coverage signal (≈ 0.8% mention rate), not a primary finding.
- A **clause-level sensitivity analysis** (sentence vs. clause attribution)
  leaves the aspect ranking essentially unchanged (Spearman ρ = 0.976–1.000)
  and slightly *strengthens* the billing finding for Subscription Boxes
  (-0.080 → -0.098). A **billing-lexicon masking check** is an important
  honest caveat: masking the matched keyword shifts billing sentiment notably
  toward positive in both categories, **reversing the sign for Subscription
  Boxes** (-0.080 → +0.044) — part of the billing-negative finding reflects
  the lexicon valence of words like `cancel`/`charged` themselves, not solely
  surrounding context. **Review-level bootstrap CIs** confirm the billing
  cross-category difference is stable (CI excludes zero); see
  `results/tables/rq2_clause_sensitivity_summary.csv`,
  `rq2_billing_lexicon_masking_check.csv`, `rq2_aspect_sentiment_bootstrap_ci.csv`.
- **Advertising** is now a separate aspect, mainly visible in magazines
  (≈ 8% mention rate).
- Robustness: VADER and the document classifier agree on 77.1% of non-neutral
  aspect mentions (`n = 789`); VADER labels 26.1% of sampled aspect mentions
  neutral. An **illustrative, self-conducted qualitative spot-check** (not an
  independent or blinded manual annotation) found 77/80 aspect-assigned
  sentences plausible — reported as illustrative only, not a validated
  precision estimate.
- **Scope.** Both categories are subscription products, so this is a
  cross-category comparison within a fairly narrow domain, not a general claim
  about Amazon categories.

## Limitations

- **Sentence-level attribution** is the primary method and cannot, by itself,
  separate multiple aspects that co-occur in one sentence with mixed sentiment
  ("great … but overpriced"). A clause-level sensitivity analysis (notebook
  04, Section 2a) checks this directly; the main findings are only revised if
  that analysis actually reverses them.
- The aspect lexicon is hand-built and English-only; VADER is general-purpose,
  not domain-tuned. A billing-lexicon masking sensitivity check (notebook 04,
  Section 7) probes how much of the billing finding could be a lexicon
  artefact versus genuine review context.
- No aspect-level gold labels exist, so RQ2 is unsupervised and cannot be scored
  against ground truth. The VADER/classifier comparison and the "77/80"
  illustrative qualitative spot-check (notebook 05) are both **self-conducted
  by the author**, not independently/blindly annotated — they are reported as
  illustrative checks, not as a validated precision estimate.
- **Cross-category scope**: both categories studied are subscription products,
  so this is a comparison within a fairly narrow product domain, not a general
  claim about Amazon categories. Subscription Boxes has only 641 distinct
  items, so its aspect frequencies can be influenced by a few popular boxes.
  Review lengths differ between categories, which is why mention rates are also
  reported per 100 words (notebook 04, Section 4b).
- Results throughout this project are associative and descriptive, not causal.
- Hyperparameter and feature selection use a small, deterministic grid, not an
  exhaustive search, to keep notebook runtime realistic for a seminar project.
