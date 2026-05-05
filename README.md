# Sentiment Analysis on Amazon Product Reviews

Master's seminar project (EnterpriseAI, JMU Würzburg, Summer Semester 2026) investigating classical NLP and machine learning methods for sentiment classification and aspect-based sentiment analysis on the Amazon Reviews 2023 dataset.

## Research Questions

**Main RQ:** *To what extent can classical machine learning methods extract sentiment polarity and product-related aspects from Amazon reviews, and what trade-offs between predictive performance and interpretability emerge that are relevant for business decision-making?*

**RQ1 — Sentiment Classification & Error Analysis:** How do classical machine learning models compare in classifying document-level sentiment polarity in Amazon reviews, and what do their systematic errors reveal about the limitations of bag-of-words representations?

**RQ2 — Aspect-Based Sentiment Analysis:** Which product aspects are most frequently mentioned in Amazon reviews, and how do their sentence-level sentiment profiles differ across two selected product categories when using classical aspect extraction and sentiment scoring methods?

## Project Structure

```
.
├── data/
│   ├── raw/              #Original Amazon Reviews 2023 subsets (gitignored)
│   └── processed/        #Cleaned and tokenized data (gitignored)
├── notebooks/            #Jupyter notebooks for exploration and experiments
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classifiers.ipynb
│   ├── 04_aspect_analysis.ipynb
│   └── 05_visualizations.ipynb
├── src/                  #Reusable Python modules
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── aspect_extraction.py
├── results/
│   ├── figures/          #Plots and visualizations
│   └── tables/           #Metric tables, comparison results
├── paper/                #LaTeX source for the seminar paper
├── requirements.txt      #Pinned Python dependencies
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ivelinayna/seminar.git
cd seminar
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Download the dataset

The Amazon Reviews 2023 dataset is available at https://amazon-reviews-2023.github.io. Place the chosen category subset in `data/raw/` (the data folder is gitignored to keep the repo lightweight).

##Reproducibility

All experiments use `random_state=42` for reproducibility. Notebook outputs and result tables are versioned in `results/` to allow direct comparison across runs.

## Author

Ivelina Yaneva — Master's student, Information Systems, JMU Würzburg

## Supervisor

Viet Nguyen

## License

This repository is for academic purposes only. Dataset usage follows the terms specified by the Amazon Reviews 2023 release.
