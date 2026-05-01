# Raw Data Archives

This directory contains the immutable source files for the Prussian Kulturkampf research.

## Data Sources

1. **Galloway Prussia Database (1862–1914)**:
   - `VIT*.xlsx`: Annual vital statistics (Births, Deaths, Marriages) for 450+ counties.
   - `POP*.xlsx`: Population census files used for denominator interpolation.
   - `REL1871.xlsx`: The 1871 Religion Census used to define the cross-sectional treatment intensity.

2. **iPEHD (Becker-Woessmann)**:
   - `ipehd_1849_indu_trans.csv`: Industrialization and occupational controls.
   - `ipehd_qje2009_master.dta`: Reference data for the 1871 baseline.

## Preservation Note

Files in this directory should **never be modified manually**. All cleaning and harmonization must be performed programmatically via the `src/data/` package to ensure a 100% reproducible audit trail.
