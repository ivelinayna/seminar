# Classical NLP for Sentiment and Aspect Analysis on Amazon Reviews

Seminar project for Enterprise AI and Urban Analytics at JMU Würzburg.

The detailed methodology, results, discussion and limitations are documented in `paper/main.pdf`. This README only keeps the information needed to understand the repository and reproduce the project.

## Project Scope

Dataset: Amazon Reviews 2023 for `Subscription Boxes` and `Magazine Subscriptions`.

Research focus: classical NLP for document level sentiment classification and sentence level aspect sentiment analysis.

Final aspects: delivery, packaging, quality, price, content, customer service, billing and advertising.

Models: Naive Bayes, Logistic Regression and `sklearn.ensemble.GradientBoostingClassifier`.

## Repository Structure

`paper/`: LaTeX paper, bibliography, figures and final PDF.

`notebooks/`: numbered notebooks from EDA to final figures.

`src/`: reusable preprocessing, modelling, evaluation and aspect extraction code.

`results/`: generated result tables and figures.

`data/raw/`: expected raw Amazon files, not committed.

`data/processed/`: generated processed files, not committed.

`tests/`: reproducibility and helper tests.

## Data

Raw data is not committed. Download the Amazon Reviews 2023 review and metadata files for both categories and place them in `data/raw/`:

`Subscription_Boxes.jsonl`

`meta_Subscription_Boxes.jsonl`

`Magazine_Subscriptions.jsonl`

`meta_Magazine_Subscriptions.jsonl`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m spacy download en_core_web_sm
```

Use `requirements.txt` only if a less strict dependency install is needed.

## Reproduction

Run the notebooks in this order:

1. `01_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_modeling_rq1.ipynb`
4. `04_aspects_rq2.ipynb`
5. `05_paper_figures.ipynb`

Expected runtime on the final local environment: about 25 to 35 minutes.

## Tests

```bash
python -m pytest tests/ -v
```

## Paper

Compile from the `paper/` folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The final PDF is `paper/main.pdf`.

## Notes

All relevant random seeds are fixed at `random_state=42`.

Raw Amazon data, generated processed data and model artefacts should not be committed.
