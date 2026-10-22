# Reproducibility: verified scope and broader analysis

The repository is an evolving analysis, not a frozen replication package for the submitted May 2026 paper. Passing software tests does not certify archival data, every result, or causal identification.

## Verified source-level check: submitted Figure 4.1

See the [validation report](exam_project2/reports/figure_4_1_validation.md) for the reference PDF hash, sample rules, results, and known prose discrepancy. From the repository root in an activated Python 3.12+ environment:

```bash
python -m pip install -r exam_project2/requirements-validation.txt
python -m unittest discover -s tests -p test_figure_4_1_validation.py -v
cd exam_project2
python -m src.validation.figure_4_1
```

This reads raw sources without editing them, writes separate validation artifacts, and reports **92/392 = 23.47%**, not approximately 22%, in the 20–80% Catholic-share interval. Seven matching figure/sample claims pass; regression estimates are not checked. The submitted PDF remains unchanged.

## Broader analysis: not certified by that check

Use a separate checkout and environment to avoid overwriting ongoing work. From the repository root:

```bash
python -m pip install -e "./exam_project2[dev]"
python -m pytest tests/
```

The full package has minimum-version constraints rather than a complete lockfile. Some modules require additional dependencies or generated data. Skipped tests are not evidence that those checks passed.

See the [current notebook list](exam_project2/notebooks/README.md) and [data appendix](exam_project2/DATA_APPENDIX.md) for evolving analysis entry points. Their longer windows and specifications can differ from the submitted 1867–1890 paper. Neither a successful full notebook run nor a working Binder launch is established here.

## Provenance and audit trail

- [Primary data providers and reuse conditions](exam_project2/data/README.md).
- [Raw-file preservation and workbook conventions](exam_project2/data/raw/README.md).
- [Validation code](exam_project2/src/validation/figure_4_1.py).
- [Recorded statistics, hashes, and environment](exam_project2/outputs/validation/figure_4_1/).

Inspect the workflow results as well as local tests. No existing lint, type-check, data-pipeline, or empirical checks have been weakened to make the narrow validation pass.
