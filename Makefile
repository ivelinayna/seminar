.PHONY: test notebooks paper clean-processed all

# Run the automated test suite (src/ correctness checks).
test:
	python -m pytest tests/ -v

# Re-run all notebooks top-to-bottom, in order, in place.
notebooks:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/02_preprocessing.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/03_modeling_rq1.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/04_aspects_rq2.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/05_paper_figures.ipynb

# Delete cached processed data so notebooks/01-02 rebuild it from raw data.
clean-processed:
	find data/processed -maxdepth 1 -type f \( -name '*.parquet' -o -name '*.csv' \) -delete

# Compile the LaTeX paper (pdflatex + bibtex via latexmk).
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

all: clean-processed notebooks test paper
