# Makefile for Project Professionalization

# Variables
PYTHON = python
PIP = pip
NOTEBOOK = exam_project/notebooks/exam_project.ipynb
OUTPUT_DIR = exam_project/notebooks

.PHONY: install analysis clean help lint format test check-all

help:
	@echo "Usage:"
	@echo "  make install   Install dependencies from requirements.txt"
	@echo "  make analysis  Execute the analysis notebook"
	@echo "  make clean     Remove cache files"
	@echo "  make lint      Run ruff for linting"
	@echo "  make format    Run ruff for formatting (replacing black)"
	@echo "  make test      Run pytest"
	@echo "  make audit     Run data audit and generate integrity reports"
	@echo "  make spatial   Run spatial diagnostics (Moran's I / LISA)"
	@echo "  make animate   Generate time-lapse fertility animation"
	@echo "  make memo      Generate a stylized supervisor memo (PDF)"
	@echo "  make check-all Run lint and test"

install:
	$(PIP) install -r requirements.txt

analysis:
	jupyter nbconvert --to html --execute $(NOTEBOOK) --output-dir=$(OUTPUT_DIR)

clean:
	rm -rf exam_project/data/processed/*.csv
	rm -rf dist/
	rm -rf build/

data:
	python exam_project/src/data/process_data.py

app:
	streamlit run exam_project/src/app.py

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

audit:
	python exam_project/src/data/audit.py

spatial:
	python scripts/spatial_diagnostics.py

spatial-reg:
	python scripts/spatial_regression.py

animate:
	python scripts/animate_transition.py

memo:
	quarto render scripts/generate_memo.qmd --to pdf --output research_brief.pdf
	mv scripts/research_brief.pdf research_brief.pdf

full-audit: spatial spatial-reg memo

check-all: lint test full-audit
