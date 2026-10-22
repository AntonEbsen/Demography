# Reproduce Figure 4.1

This narrow package reconstructs the Catholic-share distribution in the submitted
May 2026 demography paper. It does **not** reproduce regressions or identify a
causal effect. The submitted PDF is unchanged.

## Run from a fresh checkout

Install Python **3.12**, clone this repository and open its root. Use the fixed
revision linked from the [paper page](https://www.antonebsen.com/en/projects/demography-project)
when reproducing that version rather than following later `main` changes.

```sh
git clone https://github.com/AntonEbsen/Demography.git
cd Demography
python scripts/reproduce_figure_4_1.py
```

On Windows, use `py -3.12 scripts/reproduce_figure_4_1.py` if `python` selects a
different version. Stata, GIS packages, a DVC account, a processed panel and
private credentials are not needed.

The command creates `.venv-figure-4-1/` without system site packages, installs
the 17 exact runtime versions in `requirements.lock.txt`, checks dependencies,
runs 12 software safeguard tests, and executes the actual-source reconstruction.
First setup requires access to the Python package index. Subsequent calls reuse
that environment and check the lock again. Do not use it for unrelated packages.

Each run creates a **new** `reproduction-runs/figure-4-1-*` directory. Optional
`--output-dir /path/to/new/results` selects a new/empty destination. Previous
results and raw workbooks are not overwritten. A failed run may leave partial
diagnostic files, but never a new successful `verification.json`.

## Sources and access

Inputs are REL1871 and VIT1867–VIT1890 in
`exam_project2/data/raw/galloway_data/`. Existing repository files, including
four equivalent conversion pairs, are identified by the 29 SHA-256 values in
the [reference manifest](../../exam_project2/outputs/validation/figure_4_1/input_manifest.json).
The wrapper requires those exact bytes before running and checks them again at
the end. Missing files or different versions fail explicitly.

Use `--data-dir /path/to/galloway_data` for separately stored inputs. Different
downloads/conversions must not be silently substituted for this frozen version.
The lower-level validation module can investigate alternatives, but its success
alone is not a pass of this exact-source check.

Source: Patrick R. Galloway (2007), *Galloway Prussia Database 1861 to 1914*.
[Provider, documentation and conditions](https://www.patrickrgalloway.com/prussia.htm).
The provider requires permission for commercial use. This package adds no raw
data copies and the code's MIT license does not override data conditions.

## Expected scientific output

| Quantity | Expected |
|---|---:|
| Counties (one equal-weight observation each) | 392 |
| Catholic share at or below 50% / above 50% | 262 / 130 |
| Unweighted mean / median (%) | 34.009050 / 13.332378 |
| Matched county-years, actual years 1867–1890 | 9,261 |
| Counties with incomplete 24-year coverage | 33 |
| Catholic share between 20% and 80% | 92/392 = 23.469388% |

The last quantity **disagrees** with the paper's “approximately 22%” sentence.
The wrapper preserves that discrepancy rather than marking the sentence correct.
It also preserves the excluded Code 545, Year 1866 record from VIT1867.XLS.

Outputs are a PNG figure, `report.json`, `input_manifest.json`, `environment.json`
and `verification.json`. The command prints `VERIFIED` and exits zero only after
every report field matches the reference and all 29 input hashes still match.
Counts, categories and 30 histogram bin counts match exactly; floating statistics
and edges use an absolute tolerance of `1e-10`. Rendering can differ by OS/font,
so **no pixel-identical PNG hash is promised**.

The reference PDF SHA-256 is recorded in the report; this command does not
download or re-read it. The original module's optional `--paper` flag verifies
that local PDF separately. See the [original audit](../../exam_project2/reports/figure_4_1_validation.md)
for source selection, exclusions and interpretation.

## Verification boundary

This is a later, AI-assisted reproducibility aid, not code represented as the
original submitted analysis. Python 3.12 on Windows was tested locally in an
isolated environment; the dedicated workflow tests the command on Windows and
Linux. Check its actual outcome rather than treating a configured job as a
successful test. Raw inputs, reference results, original analysis functions and
the submitted paper are unchanged.

Not covered: outcome denominators, population interpolation, iPEHD linkage,
regression coefficients/standard errors, parallel trends, causal identification
or later exploratory extensions.
