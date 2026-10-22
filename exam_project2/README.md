# Bismarck’s Lost Battle: code and data

The submitted course paper covers **1867–1890**. The rolling analysis has since expanded, including a default **1862–1910** panel. Current outputs are not an exact reproduction of the submitted manuscript.

Start with the [paper and project page](https://www.antonebsen.com/en/projects/demography-project), [Figure 4.1 validation report](reports/figure_4_1_validation.md), or [evolving data appendix](DATA_APPENDIX.md).

## Reproduce the narrow validation

From the repository root, using a separate environment with Python 3.12 or newer:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r exam_project2/requirements-validation.txt
python -m unittest discover -s tests -p test_figure_4_1_validation.py -v
cd exam_project2
python -m src.validation.figure_4_1
```

The check reads source workbooks without editing them and writes JSON results, an input-hash manifest, runtime versions, and a reproduced PNG under `outputs/validation/figure_4_1/`. It reports a known prose discrepancy: **92/392 = 23.47%**, rather than approximately 22%, of counties have Catholic shares between 20% and 80%. The submitted PDF is unchanged. Optional `--paper /path/to/submitted.pdf` verifies the recorded PDF hash.

## Broader analysis

From the repository root:

```bash
python -m pip install -e "./exam_project2[dev]"
```

The broader package uses minimum-version constraints, not a fully locked environment. It has additional data and optional-dependency requirements; installation is not a claim that every notebook or historical output has been reproduced. See [notebook roles](notebooks/README.md), [test scope](../tests/README.md), and the appendix before running the evolving pipeline.

## Structure

- `data/raw/`: third-party sources; preserve originals and source conditions.
- `data/processed/`: generated panels, not a frozen manuscript dataset.
- `src/data/`, `src/analysis/`, `src/visualization/`: loaders, analysis, and plots.
- `src/validation/`: narrow source-level manuscript checks.
- `notebooks/`: exploratory and analytical notebooks.
- `outputs/`: results, including separate validation artifacts.
- `reports/`: explanatory notes and validation reports.
