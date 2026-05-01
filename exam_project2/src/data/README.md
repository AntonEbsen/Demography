# Data Engineering Modules

This directory manages the transformation of raw archival spreadsheets into a clean, analysis-ready panel.

## Modules

- `load_data.py`: Specialized loaders for the Galloway Excel format (handling 50+ individual years) and iPEHD Stata files.
- `build_dataset.py`: The "Master Pipeline" that merges denomination data with vital statistics, interpolates missing population denominators, and constructs the final DiD interaction terms.

## Key Transformations

- **Harmonization**: Mapping inconsistent column names across 50 years of Prussian vital records to a unified schema.
- **Interpolation**: Using 5-year census intervals to linearly interpolate the total population (`Poptot`) for years where annual registration was incomplete.
- **Normalization**: Converting raw counts into rates (CBR, IMR, etc.) per 1,000 citizens.
