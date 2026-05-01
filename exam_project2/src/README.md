# Source Code

This directory contains the modular Python implementation of the research pipeline.

## Package Structure

- `data/`: Data loading, harmonization, and panel building logic.
- `analysis/`: Econometric models (DiD, Event Study, Robustness) and statistical utilities.
- `visualization/`: Plotting logic for trajectories, event studies, and maps.

## Design Philosophy

The codebase is designed as a professional Python package (`exam_project2`). It prioritizes:
1. **Reproducibility**: Explicit type hints and automated CI/CD validation.
2. **Modularity**: Separation of data engineering and statistical analysis.
3. **Transparency**: Detailed logging of all data transformations and regression steps.
