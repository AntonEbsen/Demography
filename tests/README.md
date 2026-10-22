# Test scope

Software tests do not validate every empirical result or causal identification.

## Submitted Figure 4.1

From the repository root, after installing `exam_project2/requirements-validation.txt`:

```bash
python -m unittest discover -s tests -p test_figure_4_1_validation.py -v
```

These cover column normalization, equivalent/conflicting workbook variants, county weighting, thresholds, invalid/duplicate inputs, actual-year filtering, and manuscript comparison precision. The separate `Submitted figure validation` workflow runs the real-data reproduction and uploads artifacts. See the [validation report](../exam_project2/reports/figure_4_1_validation.md) for the known prose discrepancy.

## Broader tests

The [one-command Figure 4.1 package](../reproduction/figure-4-1/README.md) also runs
`test_figure_reproduction_package.py`: five wrapper checks for changed/missing
inputs, changed report fields/counts/bins, numerical tolerance and safe output
destinations. Its CLI is the real-data integration check; synthetic fixtures
are explicitly labelled and are not historical evidence.

`test_exam_project2.py` and other modules exercise parts of the evolving analysis, including synthetic-data behavior. After installing relevant package/development dependencies, run `pytest tests/` to discover them. Test counts and workflow coverage change; inspect `.github/workflows/` rather than assuming every test runs on every push. Passing synthetic tests alone does not establish reproduction of submitted tables.
