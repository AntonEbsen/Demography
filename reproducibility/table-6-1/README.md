# Submitted Table 6.1: a narrow numerical audit

Checked 24 October 2026, with AI coding assistance. This extends the separate
Figure 4.1 check; it does **not** make the whole evolving repository a frozen
replication package for the submitted paper.

## Target and result

Target: Table 6.1, printed page 15 of the course paper submitted 29 May 2026,
*Bismarck's Lost Battle: Marriage, Fertility, and the Kulturkampf in Prussia,
1867–1890*. The unchanged PDF is linked from the
[paper page](https://www.antonebsen.com/en/projects/demography-project).
PDF SHA-256: `086a7d333c7a9b85f6476e90de30f4f3cfd76d1720996ca0aebfb532803d5e41`.

All **40 displayed fields** match: coefficient, county-clustered standard error,
observation count, within-R², and pre-trends p-value in each of eight models.
N matches exactly; other fields match the table's **three decimal places**, not
unavailable unrounded original estimates. `p = 0.000` denotes rounding, not zero.
Full recomputed precision is retained in `expected-results.json`.

| Outcome | A: County + year FE, coefficient (SE) | B: County + year × region FE, coefficient (SE) | N |
| --- | --- | --- | --- |
| Crude birth rate | −0.011 (0.003) | 0.001 (0.005) | 9,255 |
| GMFR, fixed 1871 prevalence | −0.023 (0.026) | 0.043 (0.041) | 9,254 |
| Non-marital fertility, time-varying denominator | −0.036 (0.006) | −0.021 (0.008) | 9,255 |
| General marriage rate | −0.011 (0.003) | −0.003 (0.002) | 9,261 |

## Re-run (Python 3.12)

From this directory's repository revision, not an arbitrary future main branch:

```sh
python -m venv .venv-table-6-1
# Windows:
.venv-table-6-1/Scripts/python -m pip install -r requirements-table-6-1.txt
.venv-table-6-1/Scripts/python scripts/audit_table_6_1.py --verify
.venv-table-6-1/Scripts/python -m unittest discover -s tests -p test_table_6_1_audit.py
# macOS/Linux: replace .venv-table-6-1/Scripts/python with .venv-table-6-1/bin/python
```

Use `py -3.12 -m venv .venv-table-6-1` on Windows if needed. First setup requires
internet to install pinned packages. The tested local platform is Windows,
Python 3.12.14. Dedicated CI tests Windows and Linux separately from the legacy
repository-wide jobs; check its actual result before claiming cross-platform success.

Default output: ignored `scratch/table-6-1-results.json`. `--output PATH` selects
another report. `--verify` fails if any displayed field differs. Missing, changed
or additional source inputs fail before estimation. Do **not** use the maintainer
`--record-inputs` option to make a changed dataset pass.

The source spreadsheets/CSVs already in `exam_project2/data/raw/galloway_data`
and `ipehd_data` are required. No new raw data are distributed by this audit.
If missing, obtain the original providers' versions under their applicable terms;
this script does not download substitutes. See `exam_project2/DATA_APPENDIX.md`.

## Explicit specification and source-version differences

- Unbalanced **1867–1890** sample, 9,261 county-years, 392 counties. The legacy
  loader selects filenames by year but includes Code 545, **Year 1866**, from
  VIT1867. This audit filters the actual Year field; it never relabels that row.
- Continuous 1871 Catholic share measured in **percentage points**, interacted
  with `Year >= 1873`; no weights or additional controls, including no `ln_pop`.
- County + year FE (A); county + year-by-Regierungsbezirk FE (B). Standard errors
  use `PanelOLS.fit(cov_type="clustered", cluster_entity=True)` defaults.
- Rates follow the paper's equations B.2, B.4, B.5 and B.7. **Non-marital fertility
  uses contemporaneous unmarried women** (`women_15_49 − married_women_15_49`),
  as specified in **B.5**. The evolving code also provides a fixed-1871 unmarried
  share; that is a different outcome, despite conflicting shorthand in the paper's
  earlier variable descriptions. This distinction is documented, not silently tuned.
- Apply the builder's existing six flagged implausible-birth rows to B.5 as well
  as CBR/GMFR. The current builder does not propagate that mask to its static
  non-marital alternative. Flagged county-years are listed in the report. One
  additional static-GMFR extreme produces its N of 9,254. No thresholds are fit
  to the displayed estimates; the existing builder supplies them unchanged.
- Separate event studies under **each** FE design, omitted year **1872**. Joint
  Wald χ² test (5 df) of the five Catholic-share × year interactions **1867–1871**.
  Do not copy the current table generator's TWFE p-values into Panel B.
- Coefficients are independently cross-checked by alternating-projection
  residualization of outcome and treatment on the two FE sets (absolute tolerance
  `1e-9`). This is **not** a second independent standard-error implementation.

The script rebuilds the panel in memory. It never loads the cached parquet panel,
changes the submitted PDF, overwrites existing tables, or edits the research
modules. The legacy builder's ancillary crosswalk write is redirected to a
temporary directory and removed by that temporary-directory context.

## Integrity and limits

`inputs.json` pins 146 data/code files conservatively, including some raw files
not used by these four outcomes. Manifest hashes normalize CRLF→LF only for text
`.py`/`.csv` files to allow Git checkouts on both operating systems; spreadsheet
bytes are exact. Every run also compares **all original bytes before/after** so
newline changes during analysis do not go unnoticed. Filenames are repo-relative.

Successful numerical reproduction does not validate the historical database,
age-structure interpolation, substantive denominator choices, other tables,
figures or the causal claim. The submitted paper's parallel-trends failures
remain. This later AI-assisted audit is not part of the original assessed work
and is not independent human validation of every source.
