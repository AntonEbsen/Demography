# Test scope

Software tests do not validate every empirical result or causal identification.

## Submitted Figure 4.1

From the repository root, after installing `exam_project2/requirements-validation.txt`:

```bash
python -m unittest discover -s tests -p test_figure_4_1_validation.py -v
```

These cover column normalization, equivalent/conflicting workbook variants, county weighting, thresholds, invalid/duplicate inputs, actual-year filtering, and manuscript comparison precision. The separate `Submitted figure validation` workflow runs the real-data reproduction and uploads artifacts. See the [validation report](../exam_project2/reports/figure_4_1_validation.md) for the known prose discrepancy.

## Broader tests

`test_exam_project2.py` and other modules exercise parts of the evolving analysis, including synthetic-data behavior. After installing relevant package/development dependencies, run `pytest tests/` to discover them. Test counts and workflow coverage change; inspect `.github/workflows/` rather than assuming every test runs on every push. Passing synthetic tests alone does not establish reproduction of submitted tables.
