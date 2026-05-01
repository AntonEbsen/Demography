# Automated Test Suite

This directory contains the verification suite for the research projects.

## Test Coverage

- `test_exam_project2.py`: 13 specialized tests for the Kulturkampf analysis, including:
  - Data integrity checks (no negative birth rates, year ranges).
  - Econometric validation (correct DiD coefficients on dummy data).
  - Merge consistency (ensuring no county-code loss during processing).

## Execution

Tests are run automatically on every GitHub push via **GitHub Actions**. To run locally:
```bash
pytest tests/
```

## Significance

Automated testing in a research repository signals a "production-grade" commitment to accuracy and structural integrity, crucial for peer review and replication.
