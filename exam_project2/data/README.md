# Data provenance and generated panels

`raw/galloway_data/` contains vital-statistics (VIT), population (POP), and religion (REL) workbooks. `raw/ipehd_data/` contains iPEHD-related files, including Becker–Woessmann (2009) replication data. Generated panels in `processed/` may reflect later extensions rather than the submitted paper’s 1867–1890 sample.

## Primary sources

- **Galloway, Patrick R. (2007), _Galloway Prussia Database 1861 to 1914_.** [Provider’s page and codebook](https://www.patrickrgalloway.com/prussia.htm). Population Past/CAMPOP is not its provider. Commercial use requires Patrick R. Galloway’s permission, according to the provider.
- **Becker, Sascha O., Francesco Cinnirella, Erik Hornung, and Ludger Woessmann (2014), “iPEHD – The ifo Prussian Economic History Database,” _Historical Methods_ 47(2), 57–66.** [Official database page](https://www.ifo.de/en/welcome-ipehd-ifo-prussian-economic-history-database). The Becker–Woessmann (2009) QJE replication is a particular application, not the citation for the entire database.

The software’s MIT license does not override third-party data conditions. Consult providers and codebooks before reuse or redistribution; do not infer unrestricted rights from files being present here.

## What has been checked

The [Figure 4.1 validation](../reports/figure_4_1_validation.md) reconstructs county membership and Catholic-share statistics from REL1871 and VIT1867–VIT1890. Its manifest records selected files and SHA-256 hashes. It does not validate the iPEHD crosswalk, population interpolation, outcome denominators, or regressions.
