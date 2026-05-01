# Data Directory

This directory contains the archival and processed datasets for the Kulturkampf research project.

## Structure

- `raw/`: Contains the original Excel and Stata files from the Galloway Prussia Database and iPEHD.
  - `galloway_data/`: Vital statistics (VIT), Census (POP), and Religion (REL) files.
  - `ipehd_data/`: Becker-Woessmann (2009) replication data for industrial and religious controls.
- `processed/`: Contains the final harmonized panel dataset (`analysis_panel.parquet`) used for regressions.

## Data Provenance

All raw data is sourced from the [Galloway Prussia Database](https://www.populationspast.org/) and the [iPEHD](https://www.cesifo.org/en/ipehd). The processing pipeline in `src/data/` handles the normalization of column names, population interpolation, and variable construction.
