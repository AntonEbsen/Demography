# Contributing to the Kulturkampf Research Project

Thank you for your interest in this research. As an Open Science project, I welcome contributions that improve the accuracy, transparency, or scope of the analysis.

## How to Contribute

### 1. Data Contributions
If you have access to digitized archival records for Prussian provinces currently missing or incomplete in the Galloway database, please reach out or open an Issue.

### 2. Methodological Suggestions
If you have suggestions for alternative identification strategies (e.g., Synthetic Control Methods for specific provinces) or improvements to standard error clustering, please open a Pull Request.

### 3. Bug Reports
If you find inconsistencies in the data cleaning pipeline or errors in the digital monograph, please use the GitHub Issue tracker.

## Code Standards

- All Python code should follow PEP 8 styling (enforced by **Ruff**).
- All functions must include **Type Hints** and Google-style docstrings.
- Any change to the analysis pipeline must pass the existing `pytest` suite in `tests/`.

## License

By contributing, you agree that your contributions will be licensed under the project's **MIT License**.
