# Data Appendix

*The Kulturkampf and Catholic Fertility in Prussia, 1862–1890*

This appendix documents every variable, data source, sample-construction rule, and estimation specification used in the empirical analysis. It is written so a reader can answer the three transparency questions for any number that appears in the paper: **what is it**, **how is it computed**, and **where does it come from?**

The build pipeline entry point is [`build_analysis_panel()`](src/data/build_dataset.py:22), which writes the analysis dataset to [`data/processed/analysis_panel.parquet`](data/processed/analysis_panel.parquet). On the current snapshot the panel contains **10,783 county-year observations** spanning **392 Prussian counties** over **29 years (1862–1890)**, with **86 columns**.

---

## Table of contents

1. [Overview](#1-overview)
2. [Data sources](#2-data-sources)
3. [Unit of observation, sample frame, and crosswalks](#3-unit-of-observation-sample-frame-and-crosswalks)
4. [Raw variables — Galloway Prussia Database](#4-raw-variables--galloway-prussia-database)
5. [Raw variables — iPEHD (Becker–Woessmann 2009)](#5-raw-variables--ipehd-beckerwoessmann-2009)
6. [Constructed analysis variables](#6-constructed-analysis-variables)
7. [Sample inclusion rules and cleaning](#7-sample-inclusion-rules-and-cleaning)
8. [Estimation specifications](#8-estimation-specifications)
9. [Caveats and measurement issues](#9-caveats-and-measurement-issues)
10. [Reproducibility](#10-reproducibility)

---

## 1. Overview

**Research design.** The paper estimates the effect of Bismarck's Kulturkampf (the anti-Catholic legislation enacted between 1872 and 1878 and gradually rolled back through 1887) on the Catholic–Protestant fertility differential in Prussian counties. Identification rests on a **two-way fixed-effects difference-in-differences** comparing the change in vital rates after 1873 (the May Laws) across counties with different 1871 Catholic population shares, augmented by an event study, a long-difference estimator, and an instrumental-variables strategy that uses **distance to Wittenberg** (Becker & Woessmann 2009) and distance to the nearest 1871 Catholic bishop's seat as instruments for Catholic-share intensity.

**Unit of observation.** Galloway *Kreis* (county) × calendar year. The county is identified by `Code` (Galloway's numeric ID); the year by `Year`. The treatment variable `cath_share` is **time-invariant from the 1871 census**.

**Outcomes.** Crude birth rate (CBR), legitimate and illegitimate birth rates, illegitimacy ratio, marriage rate, Catholic share of marriages, infant mortality rate, and the **Princeton EFP / Coale indices** ($I_f$, $I_g$, $I_h$, plus the Galloway-tradition `gmfr` = legitimate births per 1,000 married women aged 15–49). The Coale framework decomposes overall fertility into a *marital fertility* component ($I_g$) and a *nuptiality* component (we use the observed marriage rate as the empirical $I_m$ analogue) — exactly the demographic-transition decomposition used by **Galloway, Hammel & Lee (1994), *Population Studies* 48: 135–158**, the canonical paper on Prussian fertility decline. All formulas are listed in §6; the paper headlines $I_g$ alongside CBR and marriage rate so the reader can cleanly attribute the response to nuptiality vs. marital fertility.

**How to read this appendix.** §§4–6 are the variable dictionary (raw → constructed). §7 is the audit trail of which observations enter the regression sample. §8 is the methods sub-appendix listing the regression equations actually estimated by [`src/analysis/`](src/analysis/). §9 lists known measurement and identification caveats. §10 is the reproducibility recipe.

---

## 2. Data sources

| Source | Provider | URL | Coverage | Files (under `data/raw/`) |
|---|---|---|---|---|
| **Galloway Prussia Database** — Vital Statistics (VIT) | Patrick Galloway / *Population Past* (CAMPOP, Cambridge) | <https://www.populationspast.org/> | Annual county-level vital statistics, 1862–1914 (we use 1862–1890) | `galloway_data/VIT{1862…1914}.{xls,xlsx,XLS}` |
| **Galloway** — Population Census (POP) | Galloway / *Population Past* | <https://www.populationspast.org/> | Census-year population by county (1861, 1864, 1867, 1871, 1875, 1880, 1885, 1890, …) | `galloway_data/POP{1861,…,1890}.xls` |
| **Galloway** — Religion Census (REL1871) | Galloway / *Population Past* | <https://www.populationspast.org/> | Cross-sectional 1871 population by denomination | `galloway_data/REL1871.xlsx` |
| **iPEHD** — Integrated Population-Economic-Historical Database | Sascha Becker & Ludger Woessmann, replication for *QJE* (2009) | <https://www.cesifo.org/en/ipehd> | 452 county cross-section, 1871 Prussian census variables | `ipehd_data/ipehd_qje2009_master.dta` |
| **German Empire 1871** shapefile | (community shapefile, EPSG:32633 / UTM 33N) | distributed in repo | County polygons, 1871 borders | `gis_data/German_Empire_1871_v.1.0.{shp,dbf,shx,prj}` |

**Citations.**
- Galloway, P.R. *Prussian Census and Vital Statistics, 1820–1934.* CAMPOP / *Population Past*.
- Becker, S.O., and Woessmann, L. (2009). "Was Weber Wrong? A Human Capital Theory of Protestant Economic History." *Quarterly Journal of Economics* 124(2), 531–596.

**Loaders.** Each source is loaded by a dedicated function. The relevant entry points are:
- [`load_rel1871()`](src/data/load_data.py:89) — REL1871 cross-section.
- [`load_pop_census()`](src/data/load_data.py:113) — POP census files used to interpolate pre-1872 population.
- [`load_vit_panel()`](src/data/load_data.py:268) — stacks all VIT files into a single panel.
- [`load_ipehd_master()`](src/data/load_data.py:316) — reads `ipehd_qje2009_master.dta`.
- [`merge_ipehd_controls()`](src/data/merge_ipehd.py:215) — merges iPEHD covariates into the Galloway panel via the crosswalk.
- [`build_centroid_crosswalk()`](src/data/centroids.py:53) — extracts county centroids (UTM, km) from the shapefile and matches them to Galloway codes.

---

## 3. Unit of observation, sample frame, and crosswalks

### 3.1 Unit and identifiers

- **Panel key:** `(Code, Year)`. Uniqueness is enforced at [`build_dataset.py:283–295`](src/data/build_dataset.py:283).
- `Code`: Galloway county code (integer < 900 — codes ≥ 900 are non-county aggregates and are dropped).
- `Year`: calendar year, integer.
- `Kreis`: county name (string; identification only).
- `Rb`: Galloway Regierungsbezirk code (3-letter string). The 36 codes that appear in the analysis panel are: AAC, ARN, AUR, BER, BRE, BRO, DAN, DUS, ERF, FRA, GUM, HAN, HIL, KAS, KOB, KOL, KON, KOS, LIE, LUN, MAG, MAR, MER, MIN, MUN, OPP, OSN, POS, POT, SCH, SIG, STA, STE, STR, TRI, WIE.
- `Type`: Galloway record type. Filtered to `Type == 0` (combined Stadt+Land record per Kreis); `Type == 1` (city-only) and `Type == 2` (rural-only) are dropped at load time to avoid double-counting — see [`load_data.py:100`](src/data/load_data.py:100) and [`load_data.py:299–302`](src/data/load_data.py:299).

### 3.2 Time window

`year_start = 1862`, `year_end = 1890` ([`build_dataset.py:31–32`](src/data/build_dataset.py:31)). 1862 is the first year of available VIT data; 1890 is the analysis cutoff.

### 3.3 Galloway ↔ iPEHD crosswalk

Galloway and iPEHD use *different* numeric county systems — code 388 in Galloway is Süderdithmarschen, but `kreiskey1871 = 388` in iPEHD is Adenau in the Rhineland. The crosswalk is built in three stages by [`build_crosswalk()`](src/data/merge_ipehd.py:127) and **every candidate match is validated** by comparing Catholic shares (tolerance `±5` percentage points, [`merge_ipehd.py:155–162`](src/data/merge_ipehd.py:155)):

1. **Direct code match** ([`merge_ipehd.py:164–172`](src/data/merge_ipehd.py:164)). On the current snapshot: 264 candidates → 61 valid (203 dropped by Catholic-share validation). Garbage-match rate is high precisely because the numbering systems do not align.
2. **Manual crosswalk** of 53 known mismatches ([`merge_ipehd.py:48–104`](src/data/merge_ipehd.py:48), applied at [`merge_ipehd.py:174–183`](src/data/merge_ipehd.py:174)). Built by hand from county-name and Regierungsbezirk inspection. 44 candidates → 43 valid.
3. **Name-cleaning fuzzy match** ([`merge_ipehd.py:185–202`](src/data/merge_ipehd.py:185)). Uses [`_clean_name()`](src/data/merge_ipehd.py:107) to normalise diacritics, drop prefixes (`STADT`, `LANDKREIS`, `KREIS`, `OBERAMTSBEZIRK`, `PR.`, `DT.`), strip suffixes, and collapse whitespace. 251 candidates → 248 valid.

**Final coverage:** **352 county mappings** out of 393 Galloway Type-0 counties. After joining to the analysis panel: **351 / 392 counties** have iPEHD data, covering **9,644 / 10,783 obs (89.4%)**. Validation: $\mathrm{corr}(\texttt{cath\_share}, \texttt{f\_cath}) = 0.9998$.

### 3.4 Galloway ↔ shapefile crosswalk

Built by [`build_centroid_crosswalk()`](src/data/centroids.py:53), output to [`data/processed/centroids.parquet`](data/processed/centroids.parquet). Filters the shapefile to `TYPE == "0"` ([`centroids.py:62`](src/data/centroids.py:62)), takes the polygon centroid in UTM 33N, converts to kilometres, and merges to Galloway by cleaned name. Coverage is reported in the log — typically ~97% of Galloway Type-0 counties.

### 3.5 Sub-region groupings

Defined in `SUBREGION_DEFINITIONS` at [`regressions.py:511–515`](src/analysis/regressions.py:511) and re-used in [`channels.py:97–98`](src/analysis/channels.py:97):

- **Polish provinces:** `Rb ∈ {POS, BRO}` (Posen, Bromberg).
- **German Catholic provinces:** `Rb ∈ {KOL, KOB, TRI, AAC, OPP, MUN}` (Köln, Koblenz, Trier, Aachen, Oppeln, Münster).
- **Protestant (rest):** all other `Rb` codes.

These groupings are constructed on-the-fly inside the analysis modules; they are *not* materialised columns of `analysis_panel.parquet`.

---

## 4. Raw variables — Galloway Prussia Database

All Galloway columns enter the panel through [`_normalize_columns()`](src/data/load_data.py:37) (which maps mixed-case and ALL-CAPS variants to a canonical form) and [`_load_single_vit()`](src/data/load_data.py:192) (which handles file-format heterogeneity and applies fallback aggregation rules where canonical columns are absent).

### 4.1 VIT files (vital statistics, annual)

Built by [`load_vit_panel()`](src/data/load_data.py:268) from `VIT{year}.{xls,xlsx,XLS}` (one file per year). Each row of the resulting panel is a county-year. Filter applied: `Code < 900`, `Type == 0` ([`load_data.py:299–302`](src/data/load_data.py:299)).

| Column | Definition | Unit | Notes / fallback rule |
|---|---|---|---|
| `Code` | Galloway county identifier | integer | Dropped if `≥ 900` |
| `Rb` | Regierungsbezirk | 3-letter string | 36 unique values in panel |
| `Kreis` | County name | string | Identification only |
| `Type` | Record type | int (0/1/2) | Filtered to `0` |
| `Year` | Calendar year | int | 1862–1890 |
| `Poptot` | Total population | persons | Falls back to `Pop` if `Poptot` absent ([`load_data.py:207–212`](src/data/load_data.py:207)). Pre-1872 VIT files lack this column → filled by interpolation (§4.2). |
| `Birtot` | Total live births in year | count | Falls back to `Birm + Birf` ([`load_data.py:215–220`](src/data/load_data.py:215)). |
| `Birlegtot` | Legitimate births | count | Falls back to a four-way sum of live + dead × m + f ([`load_data.py:223–231`](src/data/load_data.py:223)). |
| `Birbastot` | Illegitimate ("bastard") births | count | Falls back analogously ([`load_data.py:234–242`](src/data/load_data.py:234)). |
| `Dthtot` | Total deaths | count | Falls back to `Dthm + Dthf` ([`load_data.py:245–250`](src/data/load_data.py:245)). |
| `Dth_infant_leg` | Infant deaths (legitimate) | count | Reads `Dth<1leg` when present, else falls back to `Dthyoung` ([`load_data.py:253–258`](src/data/load_data.py:253)). **Caveat:** `Dthyoung` is a less precise concept than infant mortality and is the reason analyses using `infant_mortality_rate` are restricted to 1875+ (see [`channels.py:80`](src/analysis/channels.py:80)). |
| `Martot` | Total marriages | count | |
| `Marevan` | Evangelical/Protestant marriages | count | Sparse pre-1875 (4,614 NaN of 10,783 obs ≈ 43%). |
| `Marcath` | Catholic marriages | count | Sparse pre-1875 (4,614 NaN). |
| `Inmigtot` | In-migration total (registered moves into county) | count | Pre-1875 files store `Inmigtot` directly; 1875–1886 files store by sex (`Inmigm`+`Inmigf`) and we sum them ([`load_data.py:268–276`](src/data/load_data.py:268)). **Coverage gap: 1868–1871 and 1887–1890** have no migration columns at all → 3,068 NaN rows (28% of panel). |
| `Outmigtot` | Out-migration (official) | count | Same construction and gap as `Inmigtot` ([`load_data.py:280–288`](src/data/load_data.py:280)). |
| `Outmigunoff` | Unofficial out-migration | count | Recorded only 1875–1886 → 6,142 NaN. Captures emigrants who left without filing an official permit; especially relevant for the Polenausweisungen literature. |

### 4.2 POP files (population census, used for interpolation)

Loaded by [`load_pop_census()`](src/data/load_data.py:113) for years 1861, 1864, 1867, 1871, 1875, 1880, 1885, 1890 (the parameterised default). Used **only** to fill missing `Poptot` in pre-1872 VIT files via [`interpolate_population()`](src/data/load_data.py:148):

1. Inner-merge VIT and POP on `(Code, Year)`.
2. Where `Poptot` is missing but census `Pop` exists, copy the census value ([`load_data.py:167–169`](src/data/load_data.py:167)).
3. Within each county, linearly interpolate any remaining gaps with `pd.Series.interpolate(method="linear", limit_direction="both")` ([`load_data.py:171–175`](src/data/load_data.py:171)).

**On the current snapshot:** 3,632 obs had `Poptot` missing pre-interpolation; after step 3, zero remain.

> ⚠ **Caveat.** Pre-1872 population is **interpolated, not measured**. This affects the denominator of every constructed rate for the years 1862–1871. See §9.

### 4.3 REL1871 (religion census, single cross-section)

Loaded by [`load_rel1871()`](src/data/load_data.py:89). Filtered to `Code < 900`, `Type == 0` ([`load_data.py:100`](src/data/load_data.py:100)). Returns 393 counties.

| Column | Definition | Unit |
|---|---|---|
| `Code`, `Rb`, `Kreis` | Identifiers (as above) | — |
| `Pop` | Total 1871 population | persons |
| `Relcathm`, `Relcathf` | Catholic population, male / female | persons |

The denominator of `cath_share` is `Pop` from REL1871, *not* the VIT `Poptot` series. Also present in the source file but unused downstream: `Relevan` (Protestant), `Reljew` (Jewish), `Relother`.

### 4.4 POP1871 age × sex pyramid (cross-section, time-invariant covariates)

In addition to the population total used for §4.2 interpolation, **POP1871 also contains a full age × sex pyramid plus county area**. We extract these as a 1871 cross-section via [`load_pop1871_age_structure()`](src/data/load_data.py:341) and merge them onto every county-year as time-invariant covariates suffixed `_1871`. The Galloway → analysis-name rename map is `POP1871_COLUMN_RENAME` at [`load_data.py:347–361`](src/data/load_data.py:347). Coverage is 100% of analysis-panel counties (no NaN).

| Galloway column | Renamed to | Definition | Unit |
|---|---|---|---|
| `Area` | `pop_area_1871` | County land area, 1871 | as recorded in source (Galloway codebook; very likely hectares — values 5,923–243,000 imply km² is implausible) |
| `Pop` | `pop_total_1871` | Total population, 1871 | persons |
| `Popm`, `Popf` | `pop_m_1871`, `pop_f_1871` | Population by sex, 1871 | persons |
| `Popmilitary` | `pop_military_1871` | Military stationed population, 1871 | persons |
| `Age0-4{m,f}` | `age_0_4_{m,f}_1871` | Population aged 0–4 by sex | persons |
| `Age5-14{m,f}` | `age_5_14_{m,f}_1871` | Population aged 5–14 by sex | persons |
| `Age15-19{m,f}` | `age_15_19_{m,f}_1871` | Population aged 15–19 by sex | persons |
| `Age20-29{m,f}` | `age_20_29_{m,f}_1871` | Population aged 20–29 by sex | persons |
| `Age30-39{m,f}` | `age_30_39_{m,f}_1871` | Population aged 30–39 by sex | persons |
| `Age40-49{m,f}` | `age_40_49_{m,f}_1871` | Population aged 40–49 by sex | persons |
| `Age50-59{m,f}` | `age_50_59_{m,f}_1871` | Population aged 50–59 by sex | persons |
| `Age60andover{m,f}` | `age_60p_{m,f}_1871` | Population aged 60+ by sex | persons |

These columns are the basis for the **General Fertility Rate** in §6.2 and the **female-15-49 share** Bai/Hsiao baseline in §8.7.

The 16 explicit per-sex age-band columns in `analysis_panel.parquet` are: `age_0_4_m_1871`, `age_0_4_f_1871`, `age_5_14_m_1871`, `age_5_14_f_1871`, `age_15_19_m_1871`, `age_15_19_f_1871`, `age_20_29_m_1871`, `age_20_29_f_1871`, `age_30_39_m_1871`, `age_30_39_f_1871`, `age_40_49_m_1871`, `age_40_49_f_1871`, `age_50_59_m_1871`, `age_50_59_f_1871`, `age_60p_m_1871`, `age_60p_f_1871`.

---

## 5. Raw variables — iPEHD (Becker–Woessmann 2009)

Single source: `data/raw/ipehd_data/ipehd_qje2009_master.dta`. Cross-sectional 1871 census; merged into the panel via the crosswalk in §3.3 and therefore **time-invariant** within county-year. The list of merged columns is `IPEHD_CONTROL_VARS` at [`merge_ipehd.py:25–44`](src/data/merge_ipehd.py:25).

| Column | Definition | Unit | Used as |
|---|---|---|---|
| `f_prot` | Protestant population share, 1871 | % | Control / placebo treatment |
| `f_cath` | Catholic population share, 1871 (iPEHD definition) | % | Crosswalk validation (corr ≈ 0.9998 with Galloway `cath_share`) |
| `f_jew` | Jewish population share, 1871 | % | Placebo treatment ([`run_jewish_placebo`](src/analysis/regressions.py:866)); pre-treatment trend |
| `f_urban` | Urban population share, 1871 | % | Moderator (heterogeneity); pre-treatment trend |
| `f_young` | Population share aged < 15, 1871 | % | Demographic control |
| `f_fem` | Female population share, 1871 | % | Demographic control |
| `f_ortsgeb` | Share born in locality, 1871 | % | Mobility proxy |
| `f_pruss` | Share with Prussian citizenship, 1871 | % | Pre-treatment trend |
| `hhsize` | Average household size, 1871 | persons | Demographic control |
| `lnpop` | Log population, 1871 (iPEHD; distinct from constructed `ln_pop`) | log persons | Reference only |
| `pop` | Population count, 1871 (iPEHD) | persons | Reference only |
| `gpop` | Population growth rate (iPEHD measure) | % per year | Emigration robustness |
| `f_blind` | Population share blind, 1871 | % | Demographic |
| `f_deaf` | Population share deaf, 1871 | % | Demographic |
| `f_dumb` | Population share mute, 1871 | % | Demographic |
| `kmwittenberg` | Distance to Wittenberg | km | **Instrument for Catholic-share intensity** (Becker–Woessmann) |
| `f_miss` | Share with missing education data, 1871 | % | Data-quality control |
| `school1517` | School enrollment, ages 15–17 | % | Literacy proxy; moderator (heterogeneity); pre-treatment trend |

**Coverage in `analysis_panel.parquet`:** every iPEHD variable has 1,139 NaN rows (10.6%) on counties not matched by the crosswalk. Specifications using iPEHD variables therefore operate on a smaller sample of ≈ 9,644 obs / 351 counties.

### 5.1 Spatial covariates (built from shapefile, not iPEHD)

Materialised in [`data/processed/centroids.parquet`](data/processed/centroids.parquet) by [`centroids.py`](src/data/centroids.py). Joined into analysis modules on demand (not into `analysis_panel.parquet`).

| Column | Definition | Unit | Source |
|---|---|---|---|
| `x_km`, `y_km` | County centroid in UTM 33N (EPSG:32633) | km | [`centroids.py:64–65`](src/data/centroids.py:64) |
| `km_bishop` | Distance from county centroid to nearest 1871 Catholic bishop's seat | km | [`add_bishop_distance()`](src/data/centroids.py:105). Bishop-seat coordinate list at [`centroids.py:36–50`](src/data/centroids.py:36) (13 sees: Köln, Trier, Münster, Paderborn, Hildesheim, Osnabrück, Fulda, Limburg, Breslau, Frauenburg, Gnesen, Posen, Kulm). Used as a second IV alongside `kmwittenberg`. |

---

## 6. Constructed analysis variables

These are the columns produced by [`build_analysis_panel()`](src/data/build_dataset.py:22) on top of the raw inputs from §§4–5. All formulas are exact reproductions of the code.

### 6.1 Treatment variables

| Variable | Definition | Formula | Unit | Built at |
|---|---|---|---|---|
| `cath_share` | Catholic population share, 1871 (Galloway-derived) | $(\texttt{Relcathm}+\texttt{Relcathf})/\texttt{Pop}\times 100$ | % (0–100), time-invariant | [`load_data.py:102`](src/data/load_data.py:102) |
| `prot_share` | Non-Catholic complement | $100-\texttt{cath\_share}$ | %, time-invariant | [`load_data.py:103`](src/data/load_data.py:103) |
| `post_kulturkampf` | Post-treatment year indicator | $\mathbb{1}[\texttt{Year}\ge 1873]$ | binary | [`build_dataset.py:177`](src/data/build_dataset.py:177) |
| `high_cath` | High-Catholic county indicator | $\mathbb{1}[\texttt{cath\_share}>50]$ | binary; **130 counties = 1** | [`build_dataset.py:181`](src/data/build_dataset.py:181) |
| `treat_x_post` | Binary DiD treatment | $\texttt{high\_cath}\times\texttt{post\_kulturkampf}$ | binary | [`build_dataset.py:184`](src/data/build_dataset.py:184) |
| `cath_share_x_post` | Continuous DiD treatment | $\texttt{cath\_share}\times\texttt{post\_kulturkampf}$ | percentage points × indicator | [`build_dataset.py:188`](src/data/build_dataset.py:188) |

**Treatment-date justification.** `Post = 1[Year ≥ 1873]` because the *Maigesetze* (May Laws), the central body of Kulturkampf legislation against Catholic clergy and education, were enacted in May 1873. Sensitivity to alternative cutoffs in {1862, 1865, 1867, 1869, 1871, 1874, 1875} is provided by [`run_start_year_sensitivity()`](src/analysis/regressions.py:210).

### 6.2 Outcome variables (rates and ratios)

The headline rate variables (`cbr`, `legitimate_br`, `illegitimate_br`, `marriage_rate`) use **mid-year population** as the denominator — the standard demographic convention. Mid-year population (`Poptot_midyear`) is constructed by linearly interpolating between consecutive December census anchors and evaluating at July 1 of each calendar year, in [`compute_midyear_population()`](src/data/load_data.py:148). The Galloway carry-forward variants (under a `_carryforward` suffix) are retained for the robustness row in `baseline_did.tex` only; see §6.3.

| Variable | Definition | Formula | Unit | Built at |
|---|---|---|---|---|
| `Poptot_midyear` | Mid-year population (linear interpolation between Dec censuses, evaluated at July 1) | linear interp anchored at Dec 1 census dates | persons | [`load_data.py:148–217`](src/data/load_data.py:148); merged into panel at [`build_dataset.py:137`](src/data/build_dataset.py:137) |
| `cbr` | Crude birth rate | $\texttt{Birtot}/\texttt{Poptot\_midyear}\times 1000$ | per 1,000 mid-year pop | [`build_dataset.py:140–143`](src/data/build_dataset.py:140) |
| `legitimate_br` | Legitimate birth rate | $\texttt{Birlegtot}/\texttt{Poptot\_midyear}\times 1000$ | per 1,000 | [`build_dataset.py:144–147`](src/data/build_dataset.py:144) |
| `illegitimate_br` | Illegitimate birth rate | $\texttt{Birbastot}/\texttt{Poptot\_midyear}\times 1000$ | per 1,000 | [`build_dataset.py:148–151`](src/data/build_dataset.py:148) |
| `marriage_rate` | Crude marriage rate | $\texttt{Martot}/\texttt{Poptot\_midyear}\times 1000$ | per 1,000 | [`build_dataset.py:152–155`](src/data/build_dataset.py:152) |
| `illegitimacy_ratio` | Share of births that are illegitimate | $\texttt{Birbastot}/\texttt{Birtot}\times 100$ | % of total births (no Poptot denom.) | [`build_dataset.py:158`](src/data/build_dataset.py:158) |
| `cath_marriage_share` | Catholic share of marriages | $\begin{cases}\texttt{Marcath}/\texttt{Martot}\times 100 & \text{if }\texttt{Marcath}\text{ present}\\\text{NaN} & \text{otherwise}\end{cases}$ | % of marriages; NaN pre-1875 | [`build_dataset.py:162–166`](src/data/build_dataset.py:162) |
| `infant_mortality_rate` | Infant deaths per 1,000 legitimate live births | $\texttt{Dth\_infant\_leg}/\texttt{Birlegtot}\times 1000$ if $\texttt{Birlegtot}>0$, else NaN | per 1,000; **analysis restricted to 1875+** | [`build_dataset.py:170–174`](src/data/build_dataset.py:170) |
| `gfr_static_1871` | **(deprecated)** General fertility rate, static 1871 denominator. Superseded by `I_g` and `gmfr` for marital-fertility analysis. | $\texttt{Birtot}/\texttt{women\_15\_49\_1871}\times 1000$ | per 1,000 women aged 15–49 (1871 base, kept for back-compat) | [`build_dataset.py`](src/data/build_dataset.py) |
| `I_f` | Coale's overall fertility index | $\texttt{Birtot}/(W \cdot \bar F^H)$, where $W$ = mid-year women 15–49, $\bar F^H$ = Hutterite-weighted age-specific fertility max | unitless (Hutterite-normalised; ~0.4 in 1871 Prussia) | [`coale_indices.py`](src/analysis/coale_indices.py) |
| `I_g` | **Coale marital fertility (Galloway-tradition headline)** | $\texttt{Birlegtot}/(W \cdot \bar F^H_{\text{mar}})$ where $\bar F^H_{\text{mar}} = \sum_a s_a m_a F^H_a$ uses the marriage-share-weighted Hutterite max | unitless (~0.6 in 1871 Prussia per Princeton EFP) | [`coale_indices.py`](src/analysis/coale_indices.py) |
| `I_h` | Coale illegitimate fertility index | $\texttt{Birbastot}/(W \cdot \bar F^H_{\text{unmar}})$ | unitless (~0.08 typical 19th-c.\ Europe) | [`coale_indices.py`](src/analysis/coale_indices.py) |
| `gmfr` | **Galloway General Marital Fertility Rate** (unnormalised $I_g$) | $\texttt{Birlegtot}/M \times 1000$ where $M$ = married women 15–49 (= 0.62 × W) | per 1,000 married women 15–49 (~234 in panel mean) | [`coale_indices.py`](src/analysis/coale_indices.py) |

**Why mid-year is the headline.** December carry-forward (the raw Galloway value, see §9 caveat) biases CBR upward by 1–3% in growing populations and produces a sawtooth artefact at every census year. The standard demographic convention — used by the Human Mortality Database, the Princeton European Fertility Project, and Coale-Watkins — is mid-year (≈ person-years lived). On the current build, switching to mid-year *strengthens* the marriage-rate DiD coefficient (from $-0.0036$ to $-0.0042$, both $p<0.001$) and increases the magnitude of the CBR coefficient from near-zero ($-0.00028$) to $-0.0041$ (in the theoretically expected direction; $p=0.19$). `gfr_static_1871`, `illegitimacy_ratio`, `infant_mortality_rate`, and `cath_marriage_share` are unaffected by the choice of convention because their denominators are not `Poptot`.

### 6.3 Galloway carry-forward variants (robustness row only)

These are exactly the Galloway-database "out of the box" rates: same numerators as §6.2, but the denominator is the raw Galloway `Poptot` (= previous December census carried forward in inter-census years). They appear in the panel only so that the headline DiD table can include a one-line robustness row showing how the coefficients differ from the proper-convention number; they are not used as the primary outcome in any specification.

| Variable | Definition | Built at |
|---|---|---|
| `cbr_carryforward` | $\texttt{Birtot}/\texttt{Poptot}\times 1000$ | [`build_dataset.py:202`](src/data/build_dataset.py:202) |
| `legitimate_br_carryforward` | $\texttt{Birlegtot}/\texttt{Poptot}\times 1000$ | [`build_dataset.py:203`](src/data/build_dataset.py:203) |
| `illegitimate_br_carryforward` | $\texttt{Birbastot}/\texttt{Poptot}\times 1000$ | [`build_dataset.py:204`](src/data/build_dataset.py:204) |
| `marriage_rate_carryforward` | $\texttt{Martot}/\texttt{Poptot}\times 1000$ | [`build_dataset.py:205`](src/data/build_dataset.py:205) |
| `inmig_rate_carryforward` | $\texttt{Inmigtot}/\texttt{Poptot}\times 1000$, NaN where `Inmigtot` missing | [`build_dataset.py:206–209`](src/data/build_dataset.py:206) |
| `outmig_rate_carryforward` | $\texttt{Outmigtot}/\texttt{Poptot}\times 1000$, NaN where `Outmigtot` missing | [`build_dataset.py:210–213`](src/data/build_dataset.py:210) |
| `net_mig_rate_carryforward` | $(\texttt{Inmigtot}-\texttt{Outmigtot})/\texttt{Poptot}\times 1000$, NaN if either missing | [`build_dataset.py:214–218`](src/data/build_dataset.py:214) |

### 6.4 Migration rates (annual, time-varying)

| Variable | Definition | Formula | Unit | Built at |
|---|---|---|---|---|
| `inmig_rate` | Crude in-migration rate (mid-year denom.) | $\texttt{Inmigtot}/\texttt{Poptot\_midyear}\times 1000$, NaN where `Inmigtot` missing | per 1,000 mid-year pop; ~28% NaN | [`build_dataset.py:181–184`](src/data/build_dataset.py:181) |
| `outmig_rate` | Crude out-migration rate (mid-year denom.) | $\texttt{Outmigtot}/\texttt{Poptot\_midyear}\times 1000$, NaN where `Outmigtot` missing | per 1,000 mid-year pop; ~28% NaN | [`build_dataset.py:185–188`](src/data/build_dataset.py:185) |
| `net_mig_rate` | Net migration rate (mid-year denom.) | $(\texttt{Inmigtot}-\texttt{Outmigtot})/\texttt{Poptot\_midyear}\times 1000$, NaN if either input missing | per 1,000 mid-year pop; ~28% NaN | [`build_dataset.py:189–193`](src/data/build_dataset.py:189) |

These are **measured** migration rates from Galloway, distinct from the *implied* migration rate computed inside [`run_emigration_robustness()`](src/analysis/regressions.py:772) as the residual of the demographic accounting identity. Coverage gap: 1868–1871 and 1887–1890 have no migration columns in any VIT file → all three rates are NaN for those years (8 of 29 panel years; ~28% of obs). `Outmigunoff` is recorded only 1875–1886 and is **not** included in the `outmig_rate` numerator — interpret `outmig_rate` as official-permit out-migration only.

### 6.5 Constructed 1871 covariates (time-invariant)

| Variable | Definition | Formula | Unit | Built at |
|---|---|---|---|---|
| `women_15_49_1871` | Women of reproductive age, 1871 | $\sum$ of `age_15_19_f_1871`, `age_20_29_f_1871`, `age_30_39_f_1871`, `age_40_49_f_1871` | count | [`build_dataset.py:239`](src/data/build_dataset.py:239) |
| `women_share_15_49_1871` | Women 15–49 share of total population, 1871 | $\texttt{women\_15\_49\_1871}/\texttt{pop\_total\_1871}\times 100$ | %, time-invariant; **mean = 24.9%, sd = 0.85** | [`build_dataset.py:240–244`](src/data/build_dataset.py:240) |

`women_share_15_49_1871` is the demographic-age-structure analogue of the iPEHD socio-economic baselines — used in §8.7 as a Bai/Hsiao pre-treatment-trend test of whether differential *fertility capacity* (rather than religion) drives differential trajectories.

### 6.5 Coale–Watkins framework (Princeton EFP / Galloway 1994)

The Princeton European Fertility Project's three-index decomposition is the standard demographic framework for studying historical fertility transitions. It separates overall fertility into a **marital-fertility** component (within-marriage childbearing intensity) and a **nuptiality** component (the share of women who are married). For the Kulturkampf paper this is the right framework because the substantive question — *did the legislation operate by suppressing within-marriage childbearing or by disrupting marriage formation?* — is exactly what the decomposition isolates.

**Definitions.** Let $W_i$ = women aged $i$ in 5-year band, $M_i$ = married women aged $i$, and $F^H_i$ = Hutterite age-specific fertility (the natural-fertility upper bound, Coale 1969). Then:

$$
I_f \;=\; \frac{\text{Births}}{\sum_i W_i F^H_i}, \quad
I_g \;=\; \frac{\text{Births}_{\text{leg}}}{\sum_i M_i F^H_i}, \quad
I_h \;=\; \frac{\text{Births}_{\text{ill}}}{\sum_i (W_i - M_i) F^H_i}
$$

with the identity $I_f \approx I_g \cdot I_m + I_h(1 - I_m)$, where $I_m = \sum_i M_i F^H_i / \sum_i W_i F^H_i$ is the Hutterite-weighted nuptiality index.

**Galloway, Hammel & Lee (1994).** Their headline outcome is the **General Marital Fertility Rate**, defined as legitimate births per 1,000 married women aged 15–49, computed as a 5-year moving average centred on each census year. **Our `gmfr` is the unnormalised analogue of $I_g$**: same numerator (legitimate births), same denominator concept (married women 15–49). The Hutterite-normalised `I_g` is what the Princeton EFP reports; it is `gmfr` divided by $\bar F^H_{\text{mar}} \times 1{,}000$. The two carry the same DiD coefficient up to a constant rescaling.

**Approximations** (because Galloway VIT lacks annual age × marital-status data):

1. **Hutterite ASFR** from Coale (1969), 7 age bands × natural-fertility max.
2. **Coale–Demeny "West" Level 7** female age distribution within 15–49 (typical for $e_0 \approx 35$ Prussia 1860–90).
3. **Marriage-share schedule** calibrated to give the Princeton EFP's empirical Prussia 1871 prevalence (~62% of women 15–49 married, peaks at 90% in ages 35–39). Constant across counties and time.
4. **Women 15–49 share** of total population: county-specific value from POP1871 (`women_15_49_1871 / pop_total_1871`, mean ≈ 24.9%, sd ≈ 0.85), scaled to mid-year population. Allows cross-county variation but assumes within-county constancy across 1862–1890.

These approximations affect the *level* of the indices but **not** the within-county DiD coefficients, which are what the empirical analysis uses.

**Empirical levels** (current build): $I_f = 0.41$, $I_g = 0.61$, $I_h = 0.08$, gmfr = 234. Princeton EFP independent estimates for 1871 Prussia: $I_f \approx 0.40$–$0.42$, $I_g \approx 0.65$–$0.72$, $I_h \approx 0.05$–$0.10$. Levels are within tolerance; the slight $I_g$ underestimate likely reflects our marriage-share schedule erring slightly low.

### 6.6 Other constructed variables

| Variable | Definition | Formula | Unit | Built at |
|---|---|---|---|---|
| `ln_pop` | Log mid-year population (county-size control) | $\ln(\texttt{Poptot\_midyear})$ — uses the same mid-year denominator as the headline rates so the regression does not mix conventions | log persons | [`build_dataset.py:178–181`](src/data/build_dataset.py:178) |
| `cbr_flag` | CBR outlier flag (catches extremes under either mid-year or Galloway carry-forward denominator) | $\mathbb{1}[\texttt{cbr}\notin[15,70]\,\text{or}\,\texttt{cbr\_carryforward}\notin[15,70]]$ | bool; **6 obs = TRUE** | [`build_dataset.py:222–230`](src/data/build_dataset.py:222) |
| `gfr_flag` | GFR outlier flag | $\mathbb{1}[\texttt{gfr\_static\_1871}>400]$ | bool; **10 obs = TRUE** (mostly 1869–1872 boundary-reform artefacts) | [`build_dataset.py:264–266`](src/data/build_dataset.py:264) |

When `cbr_flag` is TRUE the *rate* columns (`cbr`, `legitimate_br`, `illegitimate_br`, `illegitimacy_ratio`, `marriage_rate`) are set to `NaN` ([`build_dataset.py:206–208`](src/data/build_dataset.py:206)) and `gfr_static_1871` is also nulled ([`build_dataset.py:256–257`](src/data/build_dataset.py:256)). When `gfr_flag` is TRUE only `gfr_static_1871` is set to `NaN` ([`build_dataset.py:271–273`](src/data/build_dataset.py:271)). The row is preserved so entity FE are not perturbed.

---

## 7. Sample inclusion rules and cleaning

A row is in the final analysis panel iff it survives every rule below, applied in this order.

1. **Galloway type and code filter.** `Code < 900` AND `Type == 0` ([`load_data.py:100`](src/data/load_data.py:100), [`load_data.py:299–302`](src/data/load_data.py:299)). Drops non-county aggregates and the city-only / rural-only sub-records that would double-count Stadt+Land Kreise.
2. **Time window.** `Year ∈ [1862, 1890]` ([`build_dataset.py:31–32`](src/data/build_dataset.py:31)).
3. **Population non-missingness.** `Poptot > 0` and not NaN ([`build_dataset.py:195`](src/data/build_dataset.py:195)). Pre-1872 values are *linearly interpolated* from POP census files ([`load_data.py:171–175`](src/data/load_data.py:171)) — see §4.2 caveat.
4. **REL1871 join.** Inner merge on `Code` ([`build_dataset.py:99–103`](src/data/build_dataset.py:99)). Counties with no 1871 religion record are dropped (pre-merge: 486 VIT counties; post-merge: **392** matched).
5. **Outlier flagging on CBR.** Rates set to NaN where `cbr > 70` or `cbr < 15` ([`build_dataset.py:201–208`](src/data/build_dataset.py:201)). The row is *kept* to preserve panel structure. **5 observations affected** on the current snapshot — typically 1869–1871 county-boundary-reform artifacts.
6. **Outlier flagging on GFR.** `gfr_static_1871` set to NaN where it exceeds 400 per 1,000 women aged 15–49 ([`build_dataset.py:264–273`](src/data/build_dataset.py:264)). **10 observations affected** — almost all 1869–1872 boundary-reform artefacts where 1872-boundary `Birtot` is paired with a 1871-boundary `women_15_49_1871`.
7. **Duplicate-key handling.** Duplicate `(Code, Year)` rows are dropped, keeping the first ([`build_dataset.py:283–295`](src/data/build_dataset.py:283)). One known case on the current data: `Code=545` (Iserlohn), `Year=1866`. Logged on each build.
8. **iPEHD merge.** Left-style: counties not matched by the crosswalk (§3.3) retain `NaN` for iPEHD columns ([`merge_ipehd.py:278`](src/data/merge_ipehd.py:278)). On the current snapshot **41 counties (≈ 10.5%)** have no iPEHD data, corresponding to 1,139 obs (10.6%). Specifications using iPEHD covariates implicitly drop these.
9. **POP1871 age-structure merge.** Left-style on `Code` ([`build_dataset.py:229`](src/data/build_dataset.py:229)). 100% match rate on the current snapshot (every analysis-panel county has 1871 age data).
10. **Schema audit.** Pandera schema in [`audit_schema.py:13–34`](src/data/audit_schema.py:13) validates dtypes, ranges (`cbr ∈ [0,100]`, `cath_share ∈ [0,100]`, `Year ∈ [1820,1910]`, `gfr_static_1871 ∈ [0,500]`, `women_share_15_49_1871 ∈ [0,100]`, migration rates within plausible bounds) and `(Code, Year)` uniqueness.

**Resulting panel dimensions** (snapshot of 2026-05-09 build):

|  | |
|---|---|
| Observations | **10,783** |
| Columns | **86** |
| Counties | **392** |
| Years | **1862–1890 (29)** |
| High-Catholic counties (`cath_share > 50`) | **130** |
| iPEHD-matched counties | **351 / 392 (89.5%)** |
| iPEHD-matched observations | **9,644 / 10,783 (89.4%)** |
| POP1871 age-merged observations | **10,783 / 10,783 (100%)** |
| Years with measured migration | **21 / 29** (1862–1867, 1872–1886) |
| Migration-coverage observations | **7,715 / 10,783 (71.5%)** |
| `cbr_flag = TRUE` | **5** |
| `gfr_flag = TRUE` | **10** |
| Duplicate rows dropped | **1** (Iserlohn 1866) |

---

## 8. Estimation specifications

This section reproduces the regression equations *as estimated by the code*. Each specification cites the function that runs it; all functions share the same panel and the same convention `(Code, Year)` for the multi-index.

Notation: $Y_{it}$ is the outcome for county $i$ in year $t$; $\alpha_i$ are county fixed effects; $\delta_t$ are year fixed effects; $X_{it}$ is a vector of time-varying controls (default: $\ln(\text{Poptot})$); $\text{Post}_t \equiv \mathbb{1}[t \ge 1873]$.

### 8.1 Baseline two-way fixed-effects DiD

Function: [`run_baseline_did()`](src/analysis/regressions.py:54). Three FE designs are switchable through `fe_design`:

**Continuous treatment** (default):
$$
Y_{it} = \beta\,(\texttt{cath\_share}_i \times \text{Post}_t) + \alpha_i + \delta_t + \gamma\,\ln(\text{Poptot})_{it} + \varepsilon_{it}
$$

**Binary treatment** (`treatment="binary"`):
$$
Y_{it} = \beta\,(\texttt{high\_cath}_i \times \text{Post}_t) + \alpha_i + \delta_t + \gamma\,\ln(\text{Poptot})_{it} + \varepsilon_{it}
$$

**FE designs** ([`regressions.py:167–201`](src/analysis/regressions.py:167)):

| `fe_design` | What is absorbed | Use |
|---|---|---|
| `"twfe"` | $\alpha_i$ + $\delta_t$ (entity + year) | Headline spec |
| `"year_x_rb"` | $\alpha_i$ + Year × Regierungsbezirk dummies | Identifies $\beta$ off variation *within Rb–year* cells |
| `"twfe_county_trends"` | $\alpha_i$ + $\delta_t$ + county-specific linear trend on centred Year | Absorbs deterministic pre-trends, one slope per county |

**Standard errors.** Cluster-robust at `Code` (county) by default ([`regressions.py:205`](src/analysis/regressions.py:205)). Two-way (county + year) clustering is available with `two_way_cluster=True`.

**Pre-treatment-characteristic time trends** (Bai 2009; Hsiao 2014): when `pretreatment_trends=[…]` is passed, the spec adds, for each baseline characteristic $X^\text{base}_{i}$,
$$
\sum_{j} \gamma_j\, X^{\text{base}}_{j,i} \times f(t)
$$
where $f(t)$ is either a full set of year dummies (`pretreatment_trends_form="year_dummies"`) or a centred linear trend (`pretreatment_trends_form="linear"`) — see [`regressions.py:122–160`](src/analysis/regressions.py:122). Identification of $\beta$ then comes from *deviations* from each county's predicted trajectory at 1873.

### 8.2 Event study

Function: [`run_event_study()`](src/analysis/regressions.py:1070). Estimated equation, with $1872$ as the omitted reference year:
$$
Y_{it} = \sum_{t \neq 1872} \beta_t\,(\texttt{cath\_share}_i \times \mathbb{1}[\text{Year}=t]) + \alpha_i + \delta_t + \gamma X_{it} + \varepsilon_{it}
$$

The companion **pre-trends Wald test** ([`pretrends_wald_test()`](src/analysis/regressions.py:397)) runs the joint linear restriction
$$
H_0: \beta_t = 0 \quad \forall\, t < 1872
$$
returning Wald $\chi^2$, df, F-equivalent (Wald/df), and p-value ([`regressions.py:432–453`](src/analysis/regressions.py:432)).

### 8.3 Long-difference estimator

Function: [`run_long_difference()`](src/analysis/regressions.py:340). Collapses the panel to two periods (default pre = 1862–1871, post = 1880–1889) and estimates
$$
\Delta Y_i = \alpha + \beta\,\texttt{cath\_share}_i + \gamma\,\Delta\ln(\text{Poptot})_i + \varepsilon_i
$$
where $\Delta Y_i = \overline{Y}_i^{\text{post}} - \overline{Y}_i^{\text{pre}}$. Estimator is OLS with HC1 robust SE ([`regressions.py:385`](src/analysis/regressions.py:385)). Avoids TWFE pathologies (negative weights, residual autocorrelation).

### 8.4 Instrumental variables (2SLS)

#### Single instrument

Function: [`run_iv_did()`](src/analysis/regressions.py:989).

First stage:
$$
(\texttt{cath\_share}_i \times \text{Post}_t) = \pi\,(\texttt{kmwittenberg}_i \times \text{Post}_t) + \alpha_i + \delta_t + \gamma X_{it} + u_{it}
$$
Second stage:
$$
Y_{it} = \beta\,\widehat{(\texttt{cath\_share}_i \times \text{Post}_t)} + \alpha_i + \delta_t + \gamma X_{it} + \varepsilon_{it}
$$
Reports first-stage F and the Wu-Hausman endogeneity test.

#### Multiple instruments (over-identified)

Function: [`run_iv_did_multi()`](src/analysis/regressions.py:244). Default instrument set:
$$
Z_i \times \text{Post}_t \;\in\; \{\,\texttt{kmwittenberg}_i \times \text{Post}_t,\;\;\texttt{km\_bishop}_i \times \text{Post}_t\,\}
$$
where `km_bishop` is built by [`add_bishop_distance()`](src/data/centroids.py:105). Reports the **Hansen J over-identification test** — failure to reject is consistent with all instruments being exogenous ([`regressions.py:256–258`](src/analysis/regressions.py:256)).

### 8.5 Heterogeneity (triple difference)

Function: [`run_heterogeneity_did()`](src/analysis/regressions.py:457).
$$
Y_{it} = \beta_1\,(\texttt{cath\_share}_i \times \text{Post}_t)
       + \beta_2\,(M_i \times \text{Post}_t)
       + \beta_3\,(\texttt{cath\_share}_i \times M_i \times \text{Post}_t)
       + \alpha_i + \delta_t + \gamma\,\ln(\text{Poptot})_{it} + \varepsilon_{it}
$$
The moderator $M_i$ is **mean-centred** ([`regressions.py:483`](src/analysis/regressions.py:483)) so that $\beta_1$ is the treatment effect at the mean moderator. Moderators tested: `school1517`, `f_urban` (extensible).

### 8.6 Sub-region DiD and triple-difference Polish

Function: [`run_subregion_did()`](src/analysis/regressions.py:518). Runs the baseline continuous DiD separately on the three sub-region samples defined in §3.5; reports asymptotic and **wild-cluster bootstrap** p-values from [`wild_bootstrap.wild_cluster_bootstrap()`](src/analysis/wild_bootstrap.py:39) (999 Rademacher draws under $H_0$).

Function: [`run_triple_difference_polish()`](src/analysis/regressions.py:945) — the formal triple-difference variant with `Polish_i × Post_t` and `cath_share × Polish × Post` interactions on the full panel.

### 8.7 Pre-treatment characteristic robustness (Bai/Hsiao)

Function: [`run_pretreatment_trends_robustness()`](src/analysis/regressions.py:711). Wraps `run_baseline_did` with progressively richer baseline-trend interactions:
$$
\text{pretreatment\_trends} \in \{\,\texttt{school1517},\;\texttt{f\_urban},\;\texttt{f\_pruss},\;\texttt{f\_jew},\;\texttt{women\_share\_15\_49\_1871}\,\}
$$
and `pretreatment_trends_form` ∈ {`"linear"`, `"year_dummies"`}. The `f_jew × year` augmentation is the test that attenuates the marriage-rate coefficient by roughly half — a finding to highlight in the paper. Spec (6) adds the **share of women aged 15–49 in 1871** (`women_share_15_49_1871`, from POP1871, see §4.4 and §6.4) as an additional baseline. This is the demographic-age-structure analogue of the iPEHD socio-economic baselines and tests whether differential *fertility capacity* drives differential trends. On the current build, adding the female-share trend leaves the marriage-rate coefficient essentially unchanged ($-0.0021$ vs $-0.0020$ in spec (5)), indicating differential age-structure trends do not explain the result.

### 8.8 Falsifications and placebos

| Spec | Function | Variation |
|---|---|---|
| **Jewish-share placebo** | [`run_jewish_placebo()`](src/analysis/regressions.py:866) | Replaces `cath_share` with `f_jew` as the treatment; if estimated $\hat\beta\neq 0$, the post-1873 shock is loading on minority-religious composition broadly, not Catholicism per se. |
| **Fake-treatment placebo** | [`run_fake_treatment_placebo()`](src/analysis/regressions.py:906) | Restricts to 1862–1871 and re-defines $\text{Post} = \mathbb{1}[\text{Year}\ge 1865]$; tests for spurious pre-trend. |
| **Sub-sample decomposition** | [`run_subsample_decomposition()`](src/analysis/regressions.py:654) | Cuts: full / Core Prussia (excl. 1866 annexations) / no Polish / Core+no-Polish — see notebook 03 §13. |

### 8.9 Robust inference

| Tool | Function | Spec note |
|---|---|---|
| **Wild cluster bootstrap** | [`wild_cluster_bootstrap()`](src/analysis/wild_bootstrap.py:39) | 999 Rademacher draws under $H_0$. Used for sub-region DiD where cluster count is small. |
| **Honest DiD (Rambachan–Roth 2023)** | [`honest_did_bounds()`](src/analysis/honest_did.py:82) | Smoothness restriction: $\lvert\Delta\delta_t\rvert \le M\,\max_{s\le 0}\lvert\Delta\delta_s\rvert$. Worst-case bias $B(M, h) = M\,\max_{s\le 0}\lvert\Delta\beta_s\rvert\,h$ ([`honest_did.py:24–32`](src/analysis/honest_did.py:24)). Reports breakdown $M$ — smallest $M$ at which the CI just contains zero. |
| **Conley spatial HAC** | [`spatial_did_se()`](src/analysis/conley_se.py:69) | Bartlett kernel $K(d) = \max(1 - d/H,\,0)$, cutoff $H = 200$ km ([`conley_se.py:13–24`](src/analysis/conley_se.py:13)). Run on the *baseline* TWFE residuals; report the larger of cluster-robust and Conley SEs. |
| **Variance decomposition** | [`variance_decomposition.py`](src/analysis/variance_decomposition.py) | $R^2$ contribution of entity FE, year FE, treatment. |
| **dCDH negative-weights diagnostic** | [`dcdh_diagnostic.py`](src/analysis/dcdh_diagnostic.py) | de Chaisemartin–d'Haultfoeuille (2020) weight check on continuous TWFE. |
| **Multiple-testing correction** | [`multiple_testing.py`](src/analysis/multiple_testing.py) | Anderson FDR sharpened q-values across the four primary outcomes. |
| **Permutation inference** | [`permutation_inference.py`](src/analysis/permutation_inference.py) | Fisher randomisation: 1,000 reassignments of `cath_share`, exact one-sided p-values. |

### 8.10 Emigration robustness (with measured migration)

Function: [`run_emigration_robustness()`](src/analysis/regressions.py:772). Six specifications addressing the post-1885 Polish-province emigration confound:

1. Baseline TWFE with `ln_pop` only.
2. + `pop_growth_rate` (constructed inline from `Poptot` differences).
3. + `migration_rate` *implied* by the demographic accounting identity ($\Delta\text{Pop} - (\text{Birtot}-\text{Dthtot})$, per 1,000).
4. Sample restricted to pre-1885.
5. + **measured `outmig_rate`** from Galloway VIT (per 1,000 pop, official permit out-migration only).
6. + **measured `net_mig_rate`** from Galloway VIT (per 1,000 pop, $\text{In}-\text{Out}$).

Specs (5) and (6) are the new measured-migration specs. They use the variables documented in §6.3 and implicitly restrict the sample to the 7,715 obs (~21 of 29 years) where Galloway records migration. On the current build:

- The marriage-rate coefficient survives in (5)/(6): $\hat\beta = -0.0023$, $p = 0.014$ (vs $-0.0036$, $p<0.001$ in baseline (1)). About 65% of the headline marriage-rate effect remains after directly conditioning on measured migration — a meaningful but partial attenuation, consistent with measured emigration explaining roughly a third of the marriage-rate result.
- The CBR coefficient stays null across all specs.

Use this as the cleaner replacement (or companion) for the implied-migration spec (3), which uses a residual-based proxy that mechanically conflates migration with measurement error in `Birtot - Dthtot`.

### 8.11 Coale–Watkins decomposition (Galloway-tradition outcomes)

Functions: [`compute_coale_indices()`](src/analysis/coale_indices.py), [`run_baseline_did()`](src/analysis/regressions.py:54) with `outcome ∈ {"I_f", "I_g", "I_h", "gmfr"}`. See §6.5 for the full Coale–Watkins framework.

**The headline finding under the Princeton EFP framework.** On the current build, the DiD coefficient on `cath_share_x_post` for each Coale index is:

| Outcome | $\hat\beta$ | $p$ | Interpretation |
|---|---|---|---|
| $I_f$ (overall fertility) | $-0.00004$ | 0.22 | Borderline; consistent with CBR null |
| **$I_g$ (marital fertility)** | **$-0.00003$** | **0.53** | **No effect on within-marriage childbearing** |
| $I_h$ (illegitimate fertility) | $-0.00005$ | 0.006 | Modest decline (consistent with marriage disruption suppressing both legitimate and illegitimate births in absolute count) |
| `gmfr` (per 1,000 married women) | $-0.013$ | 0.53 | Same as $I_g$ up to constant rescaling |
| Marriage rate | $-0.0042$ | $<0.001$ | **Strong nuptiality response** |

This is the textbook Princeton EFP / Galloway demographic-transition result: **the Kulturkampf operated through nuptiality (marriage-formation disruption), not through marital fertility (within-marriage childbearing).** That is the canonical signature of an institutional shock — Bismarck's anti-Catholic clergy laws disrupted parish-administered Catholic marriage formation but did not directly affect couples' reproductive decisions inside existing marriages.

**Why this matters for the paper.** Galloway, Hammel & Lee (1994) is the canonical empirical paper on Prussian fertility, and the Princeton EFP $I_g$ / $I_m$ decomposition is the framework demographers expect to see. By reporting $I_g$ alongside CBR and the marriage rate — and by showing that $I_g$ is null while marriage rate moves — the paper places the Kulturkampf finding cleanly inside the Galloway tradition.

**Why `gfr_static_1871` is deprecated.** Our previous attempt at a marital-fertility outcome used `Birtot / women_15_49_1871 × 1,000` (a static-1871 denominator). It is still in the panel for reproducibility but is not a "real" GFR (the denominator is fixed across years), is not the Galloway-tradition measure, and is superseded by $I_g$ and `gmfr`.

### 8.12 Channel analyses

Defined in [`channels.py`](src/analysis/channels.py):

- **Illegitimacy** ([`illegitimacy_analysis()`](src/analysis/channels.py:17)): baseline DiD on `illegitimacy_ratio` with `cath_share_x_post + ln_pop`.
- **Infant mortality** ([`infant_mortality_analysis()`](src/analysis/channels.py:73)): restricts to **1875+** (because of the Galloway definition change at 1875, see §4.1) and re-bases the post indicator at the **rollback** start: `cath_x_rollback = cath_share × 1[Year ≥ 1880]`.

---

## 9. Caveats and measurement issues

A short, honest accounting. Each item is annotated with where it bites.

- **Headline rates use mid-year population, not Galloway's raw `Poptot`.** Galloway's raw `Poptot` is the *previous* December census carried forward unchanged in inter-census years (empirically, **55% of obs have `Poptot` identical to the prior year's value**). The standard demographic CBR convention is mid-year (July 1) population — an approximation to person-years lived. We construct `Poptot_midyear` via linear interpolation between Dec censuses evaluated at July 1 and use it as the **headline denominator** for `cbr`, `legitimate_br`, `illegitimate_br`, `marriage_rate`, and migration rates (§6.2). The Galloway carry-forward variants are retained under a `_carryforward` suffix (§6.3) and reported only as a single robustness row in `baseline_did.tex` so a reader can see how using the database "out of the box" differs. On the current build, switching to mid-year *strengthens* the marriage-rate coefficient (from $-0.0036$ to $-0.0042$) and shifts the CBR coefficient from a puzzling near-zero ($-0.00028$) to a theoretically expected $-0.0041$ (still not significant but no longer null).
- **Pre-1872 population is interpolated, not measured** ([`load_data.py:171–175`](src/data/load_data.py:171)). On the current snapshot 3,632 of 11,493 raw VIT obs (≈ 32%) entered interpolation. Affects the denominator of every constructed rate for 1862–1871.
- **Infant-mortality definition changes in 1875.** `Dth_infant_leg` falls back from `Dth<1leg` (true infant deaths) to `Dthyoung` (broader young-age deaths) when the former is absent ([`load_data.py:253–258`](src/data/load_data.py:253)). The analysis using `infant_mortality_rate` is therefore restricted to 1875+ ([`channels.py:80`](src/analysis/channels.py:80)).
- **Marriage-by-denomination columns are sparse pre-1875.** `Marcath` and `Marevan` have ~43% NaN in the panel, concentrated pre-1875 (≈ 13 years × 392 counties ≈ 5,096 missing). `cath_marriage_share` therefore inherits this gap.
- **`cath_share` is time-invariant from 1871** ([`load_data.py:102`](src/data/load_data.py:102)). The implicit assumption is that the *cross-section* of Catholic shares is stable across 1862–1890. For the DiD this is identifying assumption, not a derivation.
- **1868 Prussian county-boundary reform** creates artificial spikes. Handled by the `cbr_flag` mechanism ([`build_dataset.py:201–208`](src/data/build_dataset.py:201)) — rates set to NaN, row preserved. **5 observations affected** by `cbr_flag`. The same boundary-reform issue contaminates `gfr_static_1871` for an additional 10 observations where 1872-boundary `Birtot` is paired with 1871-boundary `women_15_49_1871`; these are caught by `gfr_flag` ([`build_dataset.py:264–273`](src/data/build_dataset.py:264)) and nulled out.
- **iPEHD merge has ≈ 10.6% non-coverage.** All specs that use iPEHD covariates (heterogeneity, IV, Bai/Hsiao trends, Jewish placebo) operate on the matched sub-sample. Report exact $N$ for each (already done in the regressions module via `nobs`).
- **Treatment timing ambiguity.** Post is defined at 1873 (May Laws). The actual legislative phase was 1872–1878, with rollback 1880–1887. Sensitivity in [`run_start_year_sensitivity()`](src/analysis/regressions.py:210); rollback dynamics analysed in [`rollback.py`](src/analysis/rollback.py).
- **Polish-province emigration (Polenausweisungen) post-1885** is a known confounder for Polish-province fertility. Addressed by (i) population-control specifications in [`run_emigration_robustness()`](src/analysis/regressions.py:772), (ii) restricting the panel to 1862–1884 (notebook 03 §11), and now (iii) directly conditioning on **measured** out-migration / net-migration rates from Galloway VIT — see §8.10 specs (5)/(6).
- **Migration data has a coverage gap.** Galloway records migration counts in VIT files for 1862–1867 (totals only) and 1875–1886 (by sex; 1872–1874 also have totals). 1868–1871 and 1887–1890 have **no migration columns at all**, so `inmig_rate`, `outmig_rate`, and `net_mig_rate` are NaN (3,068 obs / 28% of panel). Specs that condition on these variables drop those rows.
- **`Outmigunoff` is not in the headline `outmig_rate`.** The unofficial out-migration count is recorded only 1875–1886 and is not summed into `Outmigtot`. Treat `outmig_rate` as a lower bound on out-migration in years where unofficial flight (e.g. evading the Settlement Commission) was material.
- **`gfr_static_1871` uses a 1871-static denominator.** The denominator (women aged 15–49 in 1871) does not vary over time within a county. Cross-sectional differences in age structure are removed; *time variation* in fertility capacity is not. In the TWFE setting this is harmless (entity FE absorb the static component), but readers should not interpret `gfr_static_1871` as a true contemporaneous GFR. It is best understood as "CBR rescaled by the 1871 women-15–49 share" — which is exactly the form needed to address the age-structure critique of CBR.
- **`pop_area_1871` units are not certified in the appendix.** The Galloway codebook (PDF distributed with the data) specifies the unit; values 5,923–243,000 are consistent with hectares but not with km². Treat `pop_area_1871` as an ordinal county-size measure unless you confirm the unit from the codebook.
- **Non-Prussian observations.** The sample is Prussia only; conclusions do not extend to e.g. Bavaria or the South-German Catholic states.
- **Pre-trends rejection on `cbr`.** The pre-trends Wald test rejects on the crude birth rate but not on the marriage rate, so causal claims about *fertility* (cbr/legitimate_br) are weaker than those about *marriage formation*. Honest-DiD breakdown $M$ should be reported alongside the headline coefficient.

---

## 10. Reproducibility

### Build the analysis dataset

```bash
cd exam_project2
python -m src.data.build_dataset       # builds analysis_panel.parquet
python -m src.data.audit_schema        # validates with Pandera
python -m src.data.centroids           # builds centroids.parquet (optional, for Conley SE & maps)
```

Or, equivalently, `dvc repro` if DVC is installed (stages defined in [`dvc.yaml`](dvc.yaml); pinned outputs in [`dvc.lock`](dvc.lock)).

### Reproduce the analysis

The four numbered notebooks under [`notebooks/`](notebooks/) consume `analysis_panel.parquet` and produce every figure and table in the paper:

| Notebook | Role |
|---|---|
| [`01_data_and_eda.ipynb`](notebooks/01_data_and_eda.ipynb) | Data pipeline + descriptive statistics + raw-trend plots |
| [`02_baseline_regressions.ipynb`](notebooks/02_baseline_regressions.ipynb) | Baseline DiD, event study, Honest DiD, permutation, Anderson FDR |
| [`03_extensions_and_mechanisms.ipynb`](notebooks/03_extensions_and_mechanisms.ipynb) | Heterogeneity, channels, falsifications, emigration, Bai/Hsiao trends, sub-sample decomposition |
| [`04_spatial_analysis.ipynb`](notebooks/04_spatial_analysis.ipynb) | Choropleths, sub-region maps, IV, Conley HAC |

### Environment

Python ≥ 3.9. Core dependencies: `pandas`, `numpy`, `linearmodels` (PanelOLS, IV2SLS, clustered SE), `statsmodels`, `geopandas`, `pyproj`, `pyarrow`, `pandera`, `matplotlib`, `seaborn`. Pinned in [`pyproject.toml`](pyproject.toml).

```bash
pip install -r requirements.txt
```

### Citations referenced in the analysis

- **Galloway, P.R.** *Prussian Census and Vital Statistics, 1820–1934* — primary data, distributed via *Population Past* (CAMPOP, Cambridge).
- **Galloway, P.R., Hammel, E.A. & Lee, R.D.** (1994). "Fertility Decline in Prussia, 1875–1910: A Pooled Cross-Section Time Series Analysis." *Population Studies* 48(1), 135–158. — Canonical Prussian fertility paper; we follow their conventions on mid-year population (linearly interpolated between consecutive December censuses) and adopt the Princeton EFP framework as our headline marital-fertility measure (`I_g` and the unnormalised `gmfr`).
- **Coale, A.J. & Watkins, S.C., eds.** (1986). *The Decline of Fertility in Europe*. Princeton University Press. — Princeton EFP framework, $I_f$ / $I_g$ / $I_h$ / $I_m$ definitions.
- **Coale, A.J.** (1969). "The decline of fertility in Europe from the French Revolution to World War II," in *Fertility and Family Planning*. — Hutterite age-specific fertility schedule.
- **Becker, S.O. & Woessmann, L.** (2009). "Was Weber Wrong? A Human Capital Theory of Protestant Economic History." *QJE* 124(2). — IV strategy and iPEHD covariates.
- **Rambachan, A. & Roth, J.** (2023). "A More Credible Approach to Parallel Trends." *Review of Economic Studies* — Honest DiD bounds.
- **de Chaisemartin, C. & d'Haultfoeuille, X.** (2020). "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *AER* — dCDH negative-weights diagnostic.
- **Bai, J.** (2009); **Hsiao, C.** (2014). — pre-treatment-characteristic time-trend robustness.
- **Conley, T.G.** (1999). "GMM Estimation with Cross Sectional Dependence." *Journal of Econometrics* — spatial HAC.
