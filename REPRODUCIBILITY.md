# Reproducibility Guide

This document provides step-by-step instructions for reproducing the results of the Kulturkampf fertility study.

## 1. Prerequisites

- Python 3.9 or higher.
- Git.
- (Optional) [Binder](https://mybinder.org/v2/gh/AntonEbsen/Demography/main?urlpath=lab) for one-click cloud reproduction.

## 2. Environment Setup

Clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/AntonEbsen/Demography.git
cd Demography
pip install -e "./exam_project2[dev]"
```

## 3. Data Verification

Run the automated data integrity suite to ensure the archival files are correctly loaded and mapped:

```bash
pytest tests/test_exam_project2.py
```

## 4. Running the Analysis

Open the analysis notebooks to regenerate all regressions and figures:

```bash
jupyter-lab exam_project2/notebooks/02_main_analysis.ipynb
```

Alternatively, run the raw source scripts:

```bash
python -m exam_project2.src.data.build_dataset
python -m exam_project2.src.analysis.regressions
```

## 5. Audit Trail

- **Provenance**: See `exam_project2/data/README.md` for raw data sources.
- **Lineage**: See the Data Hub on the [Digital Monograph](https://AntonEbsen.github.io/Demography/kulturkampf/data) for the visual lineage graph.
- **Computation**: All fixed-effects models utilize the `linearmodels` package with entity-clustered standard errors to match standard Stata `reghdfe` implementations.
