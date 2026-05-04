# Makefile for Project Professionalization

# Variables
PYTHON = python
PIP = pip
NOTEBOOK = exam_project2/notebooks/exam_project.ipynb
OUTPUT_DIR = exam_project2/notebooks

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
	cd exam_project2 && dvc repro analyze
	jupyter nbconvert --to html --execute $(NOTEBOOK) --output-dir=$(OUTPUT_DIR)

clean:
	rm -rf exam_project2/data/processed/*.parquet
	rm -rf exam_project2/data/processed/*.csv
	rm -rf dist/
	rm -rf build/

data:
	cd exam_project2 && dvc repro build

app:
	@echo "App has been moved to Astro Digital Monograph (kulturkampf_site/)"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

lock:
	pip install pip-tools
	pip-compile requirements.txt -o requirements.lock
	@if [ -f exam_project2/requirements.txt ]; then cd exam_project2 && pip-compile requirements.txt -o requirements.lock; fi

audit:
	cd exam_project2 && python -m src.data.audit_schema

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
