# Raw data archives

Preserve source files unchanged. Perform cleaning in code and record exclusions rather than silently correcting raw cells.

- `galloway_data/`: Galloway Prussia Database 1861 to 1914; VIT annual vital statistics, POP census files, and REL1871 religion counts.
- `ipehd_data/`: iPEHD-related county data and Becker–Woessmann replication files.

See [primary-source citations and reuse conditions](../README.md). The software license does not relicense these data.

## Workbook conventions

Extensions and column casing vary. VIT1875–VIT1878 each have both `.XLS` and `.xlsx` versions. The Figure 4.1 check prefers `.xlsx`, requires every cell to agree with alternatives (ignoring storage dtype), and records hashes of both. Missing files or conflicting conversions fail the check.

Select the window using the actual `Year` field, not only the filename. One matched record in VIT1867 has `Year = 1866` (Code 545, ISERLOHN). The narrow validation excludes it from 1867–1890 without relabeling it. This does not correct the raw file or certify all other data fields.
