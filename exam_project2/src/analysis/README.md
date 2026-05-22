# Econometric Analysis Modules

This directory contains the core statistical logic for the Kulturkampf fertility study.

## Modules

- `regressions.py`: Implements the main Difference-in-Differences (DiD) and Event Study specifications using `PanelOLS`.
- `synthetic_did.py`: Synthetic Difference-in-Differences (Arkhangelsky et al. 2021) with hand-rolled simplex-constrained unit/time weights and placebo-permutation standard errors. Robust to non-parallel pre-trends.
- `utils.py`: Shared utilities for model fitting, standard error clustering (entity-level), and specification testing.

## Methodology Implemented

1. **Two-Way Fixed Effects (TWFE)**: Controlling for unobserved county-level heterogeneity and time-specific shocks.
2. **Event Study Design**: Testing for pre-trends and identifying the exact temporal "break" following the 1873 May Laws and 1875 Civil Marriage Act.
3. **Specification Curves**: Automated robustness checks across alternative post-treatment cutoffs and Catholic share thresholds.
4. **Synthetic Difference-in-Differences (SDID)**: Reweights control counties (unit weights $\hat\omega$) and pre-period years (time weights $\hat\lambda$) so the synthetic control's pre-1873 trajectory matches the treated trajectory by construction. Closes the parallel-trends attack surface. See `synthetic_did.py`.

## Quick start: SDID

```python
from exam_project2.src.analysis.synthetic_did import (
    run_sdid, run_sdid_threshold_sweep, plot_synthetic_vs_treated,
)

# Single specification: CMR, >50% Catholic counties, 1862–1885 window
res = run_sdid(
    df, outcome="marriage_rate", treat_col="high_cath",
    treatment_year=1873, year_start=1862, year_end=1885,
    n_placebo=500,
)
print(res.summary_line())
plot_synthetic_vs_treated(res)         # visual proof of pre-trend match

# Threshold sweep (Catholic share > 40 / 50 / 60) for CMR and CBR
table = run_sdid_threshold_sweep(
    df, thresholds=(40, 50, 60),
    outcomes=("marriage_rate", "cbr"),
)
```

`run_sdid` requires a balanced county-year rectangle for the chosen
outcome and window. Counties with any missing year are dropped (logged
as a warning). Placebo SEs reassign treatment among the surviving
controls and rerun the full estimator `n_placebo` times.
