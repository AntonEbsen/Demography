# Econometric Analysis Modules

This directory contains the core statistical logic for the Kulturkampf fertility study.

## Modules

- `regressions.py`: Implements the main Difference-in-Differences (DiD) and Event Study specifications using `PanelOLS`.
- `utils.py`: Shared utilities for model fitting, standard error clustering (entity-level), and specification testing.

## Methodology Implemented

1. **Two-Way Fixed Effects (TWFE)**: Controlling for unobserved county-level heterogeneity and time-specific shocks.
2. **Event Study Design**: Testing for pre-trends and identifying the exact temporal "break" following the 1873 May Laws and 1875 Civil Marriage Act.
3. **Specification Curves**: Automated robustness checks across alternative post-treatment cutoffs and Catholic share thresholds.
