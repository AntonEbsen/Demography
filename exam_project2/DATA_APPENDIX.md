# Data Appendix

*The Kulturkampf and Catholic Fertility in Prussia, 1862–1890*

This appendix documents every variable, data source, sample-construction rule, and estimation specification used in the empirical analysis. It is written so a reader can answer the three transparency questions for any number that appears in the paper: **what is it**, **how is it computed**, and **where does it come from?**

The build pipeline entry point is [`build_analysis_panel()`](src/data/build_dataset.py:22), which writes the analysis dataset to [`data/processed/analysis_panel.parquet`](data/processed/analysis_panel.parquet). On the current snapshot the panel contains **10,783 county-year observations** spanning **392 Prussian counties** over **29 years (1862–1890)**, with **103 columns**.

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
| `Dth_infant_leg` | Infant deaths (legitimate, under 1 year) | count | Reads `Dth<1leg` when present, else falls back to `Dthyoung` ([`load_data.py:325–334`](src/data/load_data.py:325)). **Caveat:** `Dthyoung` is a broader / less complete young-deaths category than `Dth<1leg`; analyses using `infant_mortality_rate` are restricted to 1875+ to avoid the ~3–4× level discontinuity at 1875 documented in [`fig_imr_break.png`](outputs/figures/fig_imr_break.png) (see [`channels.py:80`](src/analysis/channels.py:80)). |
| `Dth_infant_bas` | Infant deaths (illegitimate, under 1 year) | count | Reads `Dth<1bas` when present. Pre-1875 the column does not exist in any VIT file, so this is **NaN for 1862–1874** (~43% of obs) ([`load_data.py:336–345`](src/data/load_data.py:336)). Combined with `Dth_infant_leg` to construct the headline total-IMR outcome `infant_mortality_rate` (defined for 1875+ by construction). |
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

### 4.5 Additional Galloway cross-sections (BIR, STA, TAX, AGR, GEL, EDU)

Six additional Galloway tables were merged in to widen the covariate base around the Kulturkampf. Each is a single cross-section that enters the panel as a time-invariant row (one value per county, broadcast over all panel years).

| File | Year | Loader | New panel columns | Use |
|---|---|---|---|---|
| `BIR1871.XLS` | 1871 | [`load_bir1871()`](src/data/load_data.py:917) | `born_in_locality_share_1871`, `born_in_kreis_share_1871`, `born_in_prussia_share_1871`, `born_outside_prussia_share_1871` | Migration baseline. Galloway's BIR1871 birthplace categories are *nested* (locality ⊂ Kreis ⊂ Provinz ⊂ Prussia); each share is reported separately rather than cumulated. Adds a Bai/Hsiao baseline in §8.7 spec (7). |
| `STA1871.XLS` | 1871 | [`load_sta1871()`](src/data/load_data.py:1099) | `pct_never_married_m_1871`, `pct_never_married_f_1871`, `pct_widowed_f_1871`, `married_share_over15_f_1871`, `hh_avg_size_1871` | Marital-status baseline. `married_share_over15_f_1871` (= Marriedover15f / Popover15f, mean 0.516, sd 0.033) drives the proper Coale $I_g$ recalibration (§6.5b): each county's actual marriage prevalence shifts the marital-fertility denominator instead of the previous Prussia-wide constant. |
| `TAX1876.XLS` | 1876 | [`load_tax1876()`](src/data/load_data.py:958) | `income_tax_pc_1876`, `ln_income_tax_pc_1876` | County income proxy. **Mid-treatment** (1876 is year 3 of the May Laws) — enters only as a heterogeneity moderator or as a robustness control, never as a pre-period regressor. Added to §8.7 spec (9). |
| `AGR1882.XLS` | 1882 | [`load_agr1882()`](src/data/load_data.py:982) | `farms_total_1882`, `farms_share_under_2ha_1882`, `farms_share_over_50ha_1882`, `land_gini_1882` | Land-inequality moderator. The Gini is computed over the six farm-size bins (<1, 1–2, 2–10, 10–50, 50–100, >100 ha) using mid-point land area as the welfare measure. AGR1882 is post-treatment but farm-size distribution is approximately structural (slowly evolving). Added to §8.7 spec (9). |
| `GEL1882.XLS` | 1882 | [`load_gel1882()`](src/data/load_data.py:1015) | `rel_edu_emp_1882`, `transport_emp_1882`, `health_emp_1882`, `finance_emp_1882`, `pop_1880_gel`, plus per-1k versions | Service-sector employment by branch. `rel_edu_emp_1882` (religion + education + instruction occupations) is the **Kulturkampf-channel outcome** in §6.7 (paired with `rel1849_cat_priest`). |
| `EDU1886.XLS` | 1886 | [`load_edu1886()`](src/data/load_data.py:1058) | `school_age_pop_1886`, `attend_public_1886`, `attend_private_1886`, `attend_rate_1886`, `teachers_1886`, `teacher_income_1886`, `pupils_per_teacher_1886` | Post-Kulturkampf schooling cross-section. **Schooling-channel endpoint** (paired with EDU1849 / iPEHD `school1517`). |

These columns are populated for every Galloway Type-0 county in the source file; merge coverage is 393/393 (BIR1871, STA1871), 404/393 (TAX1876, AGR1882, GEL1882; 11 city-only `Type=1` counties not present in our analysis), 453/393 (EDU1886).

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

### 5.0 iPEHD 1849 cross-section (merged via separate `kreiskey1849` crosswalk)

In addition to the 1871 master file, the project ingests seven iPEHD CSVs covering 1849 (and one 1816–21 mortality file) located under `data/raw/ipehd_data/`. These files key on **`kreiskey1849`** — a different numbering system from `kreiskey1871` — and require a separate crosswalk because of the 1815/1818 and 1866 Prussian boundary reorganisations. The crosswalk is built by [`build_crosswalk_1849()`](src/data/merge_ipehd.py:316) using a four-stage name match within Regierungsbezirk (exact → across Rb → substring → fuzzy / edit-distance), persisted to [`data/processed/crosswalk_1849.csv`](data/processed/crosswalk_1849.csv), and validated by correlating 1849 Catholic-priest density with 1871 `cath_share` (corr ≈ 0.84, N = 280). Coverage is **280 of 393** Galloway Type-0 counties (71.2%); unmatched counties retain `NaN` for the 1849 variables.

The merge is driven by [`merge_ipehd_1849()`](src/data/merge_ipehd.py:402) and brings in the columns below:

| Source file | Columns merged | Use |
|---|---|---|
| `ipehd_1849_rel_church.csv` | `rel1849_cat_priest`, `rel1849_cat_chaplain_vicar`, `rel1849_cat_main_church`, `rel1849_pro_priest`, `rel1849_pro_main_church`, `rel1849_jew_meetplace` | **Religious-infrastructure channel** (§6.7): paired with `rel_edu_emp_1882` from GEL1882 in a long-difference DiD on log religious-sector density. |
| `ipehd_1849_edu_stud.csv` | `edu1849_pub_ele_stud_m`, `edu1849_pub_ele_stud_f`, `edu1849_pub_mim_stud_m`, `edu1849_pub_mif_stud_f`, `edu1849_pub_high_stud_m`, `edu1849_pub_gym_stud_m` | **Schooling channel** (§6.7): paired with `school1517` (1871) and `attend_rate_1886` in a 3-period schooling DiD. The gender split provides a placebo-style sharpness check (Kulturkampf bit boys' education harder than girls'). |
| `ipehd_1849_pop_demo.csv` | `pop1849_tot`, `pop1849_m_tot`, `pop1849_f_tot`, `pop1849_f_17to45` | 1849 baseline population; denominator for the religious / schooling intensity measures. Includes women aged 17–45 (a 1849 analogue of `women_15_49_1871`). |
| `ipehd_1849_pop_mari.csv` | `pop1849_families`, `pop1849_m_wedlock`, `pop1849_f_wedlock` | 1849 marital-status baseline. Source for `avg_household_size_1849` in the §6.7 balance table. |
| `ipehd_1849_indu_fac.csv` | `ipehd_1849_indu_fac_total` (sum over 100+ industry-specific factory counts) | 1849 industrial-structure baseline. Source for `factories_per_10k_1849` in the §6.7 balance table. |

The 1849 iPEHD CSVs ship in cp1252-encoded format with a small number of replacement characters in place of umlauts; [`_clean_name()`](src/data/merge_ipehd.py:107) strips the orphan character so name matching still succeeds.

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
| `infant_mortality_rate` | **Total** infant mortality rate (headline; standard demographic definition, Galloway, Hammel & Lee 1994 / Princeton EFP convention) | $(\texttt{Dth\_infant\_leg}+\texttt{Dth\_infant\_bas})/(\texttt{Birlegtot}+\texttt{Birbastot})\times 1000$ if both numerator terms present and denominator $>0$, else NaN | per 1,000 total live births; **1875+ only** (pre-1875 the illegitimate-infant-death column `Dth<1bas` does not exist in any Galloway VIT file, so the rate is undefined by construction) | [`build_dataset.py:178–193`](src/data/build_dataset.py:178) |
| `infant_mortality_rate_leg` | **(diagnostic)** Legitimate-only IMR -- retained to document the 1875 data break in [`fig_imr_break.png`](outputs/figures/fig_imr_break.png), not as an analytical outcome | $\texttt{Dth\_infant\_leg}/\texttt{Birlegtot}\times 1000$ if $\texttt{Birlegtot}>0$, else NaN | per 1,000 legitimate live births. Pre-1875 the numerator falls back to `Dthyoung`, producing the ~3–4× level discontinuity at 1875. Do *not* use as a regression outcome. | [`build_dataset.py:195–202`](src/data/build_dataset.py:195) |
| `gfr_static_1871` | **(deprecated)** General fertility rate, static 1871 denominator. Superseded by `I_g` and `gmfr` for marital-fertility analysis. | $\texttt{Birtot}/\texttt{women\_15\_49\_1871}\times 1000$ | per 1,000 women aged 15–49 (1871 base, kept for back-compat) | [`build_dataset.py`](src/data/build_dataset.py) |
| `I_f` | Coale's overall fertility index | $\texttt{Birtot}/(W \cdot \bar F^H)$, where $W$ = mid-year women 15–49, $\bar F^H$ = Hutterite-weighted age-specific fertility max | unitless (Hutterite-normalised; ~0.4 in 1871 Prussia) | [`coale_indices.py`](src/analysis/coale_indices.py) |
| `I_g` | **Coale marital fertility (Galloway-tradition headline)** | $\texttt{Birlegtot}/(W \cdot k_i \cdot \bar F^H_{\text{mar}})$ where $k_i = \mu_i / \bar\mu^{\text{Prussia}}$ is a county-specific marriage shifter from STA1871 (= `married_share_over15_f_1871` / 0.516, clipped to $[0.5, 1.5]$). Recalibrated in May 2026: previously the marriage-share schedule was Prussia-wide constant; with STA1871 each county's actual marriage prevalence scales the marital-fertility denominator. | unitless (~0.55 panel mean, 0.66 for high-Catholic counties; Princeton EFP target 0.65–0.72) | [`coale_indices.py`](src/analysis/coale_indices.py) |
| `I_h` | Coale illegitimate fertility index. Uses $\bar F^H_{\text{unmar},i} = \bar F^H - k_i \bar F^H_{\text{mar}}$ so the partition identity $\bar F^H_{\text{mar},i} + \bar F^H_{\text{unmar},i} = \bar F^H$ holds row-by-row. | $\texttt{Birbastot}/(W \cdot \bar F^H_{\text{unmar},i})$ | unitless (~0.08 typical 19th-c.\ Europe) | [`coale_indices.py`](src/analysis/coale_indices.py) |
| `gmfr` | **Galloway General Marital Fertility Rate** (unnormalised $I_g$). Denominator is `married_women_15_49`. | $\texttt{Birlegtot}/M \times 1000$ where $M$ = `married_women_15_49` (now county-specific via STA1871) | per 1,000 married women 15–49 (~234 in panel mean) | [`coale_indices.py`](src/analysis/coale_indices.py) |
| `married_women_15_49` | Estimated count of married women aged 15–49. Under the STA1871 recalibration, $M_i = W_i \cdot k_i \cdot \bar\rho^{\text{ref}}$ where $\bar\rho^{\text{ref}}\approx 0.619$ is the Prussia-wide implied marriage prevalence under the constant schedule. | $W_i \cdot k_i \cdot \bar\rho^{\text{ref}}$ | count | [`coale_indices.py`](src/analysis/coale_indices.py) |

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

### 6.5a Constructed 1871 covariates (time-invariant)

| Variable | Definition | Formula | Unit | Built at |
|---|---|---|---|---|
| `women_15_49_1871` | Women of reproductive age, 1871 | $\sum$ of `age_15_19_f_1871`, `age_20_29_f_1871`, `age_30_39_f_1871`, `age_40_49_f_1871` | count | [`build_dataset.py:239`](src/data/build_dataset.py:239) |
| `women_share_15_49_1871` | Women 15–49 share of total population, 1871 | $\texttt{women\_15\_49\_1871}/\texttt{pop\_total\_1871}\times 100$ | %, time-invariant; **mean = 24.9%, sd = 0.85** | [`build_dataset.py:240–244`](src/data/build_dataset.py:240) |

`women_share_15_49_1871` is the demographic-age-structure analogue of the iPEHD socio-economic baselines — used in §8.7 as a Bai/Hsiao pre-treatment-trend test of whether differential *fertility capacity* (rather than religion) drives differential trajectories.

### 6.5b Political-economy covariates from the 1871 Reichstag election (ELE1871)

Galloway's ELE1871 reports Reichstag vote totals at the *Wahlkreis* (electoral district) level. Each Wahlkreis is an aggregation of one or more Type-0 Kreise; the constituent Kreis names are encoded in the Wahlkreis label (e.g. "6 BRAUNSBERG–HEILSBERG" pools Kreise 13 and 14). [`load_ele1871()`](src/data/load_data.py:484) parses these labels, matches constituent names to panel Kreise via [`_clean_name()`](src/data/merge_ipehd.py) and a four-step fallback (exact-within-Rb → exact-across-Rbs → contains-within-Rb → contains-across-Rbs), then assigns each Kreis its Wahlkreis vote shares. Coverage: **338 of 392 panel Kreise (86%)**; unmatched are mostly city–rural splits (e.g. "DANZIG STADT" vs "DANZIG LAND") that do not exist as separate Type-0 Kreise.

| Variable | Definition | Built at |
|---|---|---|
| `zentrum_share_1871` | Vote share of the Catholic Centre Party (Zentrum), per cent of valid votes | [`load_data.py:484+`](src/data/load_data.py:484) |
| `polen_share_1871` | Vote share of the Polish-nationalist Catholic party (Polen) | as above |
| `catholic_party_share_1871` | `zentrum_share_1871 + polen_share_1871` (combined Catholic political mobilisation) | as above |
| `conservative_share_1871` | Konservativ + Deutsche Reichspartei | as above |
| `liberal_share_1871` | National-liberal + Liberal Reichspartei + Fortschritt + Volkspartei | as above |
| `nat_liberal_share_1871` | National-liberal (the dominant Protestant-aligned liberal party in 1871) | as above |
| `sozialdemokrat_share_1871` | Sozialdemokrat (founded 1875–78; near-zero in 1871) | as above |

**Validation.** All variables are time-invariant 1871 cross-sections and have empirical correlations with `cath_share` consistent with the established political geography of unified Germany:

- $\mathrm{corr}(\texttt{cath\_share}, \texttt{zentrum\_share\_1871}) = +0.66$
- $\mathrm{corr}(\texttt{cath\_share}, \texttt{catholic\_party\_share\_1871}) = +0.77$
- $\mathrm{corr}(\texttt{cath\_share}, \texttt{conservative\_share\_1871}) = -0.35$
- $\mathrm{corr}(\texttt{cath\_share}, \texttt{nat\_liberal\_share\_1871}) = -0.27$

Zentrum was founded in 1870 specifically in response to anti-Catholic legislation, and these correlations confirm it functioned as the political vehicle for Catholic identity in 1871. The Polen vote share is positively but weakly correlated with `cath_share` (r = +0.26): Polish counties were Catholic, but most Catholic counties (especially Rhineland) voted Zentrum rather than Polen.

**Usage in the analysis.** Zentrum and Polen vote shares enter the heterogeneity DiD ([`heterogeneity_table`](src/analysis/latex_tables.py)) as time-invariant moderators of `cath_share × Post`. The interaction with Zentrum is positive (effect *diminishes* with Zentrum mobilisation, $p<0.01$ for CBR, marriage rate, and $I_g$); the interaction with Polen is negative (effect *strengthens* with Polish-nationalist mobilisation, $p<0.05$). The combined pattern indicates the Kulturkampf marriage-rate disruption operated through Polish-Catholic counties rather than through politically-organised German Catholic counties — adding a political-economy mechanism to the established Polish-vs-German Catholic heterogeneity story.

#### 6.5c Time-varying Reichstag vote shares (1871–1890)

Galloway publishes seven Reichstag election cross-sections during the analysis window: 1871 (pre-Kulturkampf), 1874 and 1878 (enforcement), 1881, 1884, 1887 (rollback), and 1890 (post-rollback). [`load_election_panel()`](src/data/load_data.py) loads all seven, applies the same Wahlkreis-to-Kreis crosswalk as `load_ele1871`, and returns a 7 × 338-row long-format panel of `zentrum_share`, `polen_share`, and `catholic_party_share` (Zentrum + Polen) by Kreis × election year.

The annual analysis panel includes three **time-varying** columns built from this election panel via carry-forward of the most-recent-election value:

| Variable | Definition | Built at |
|---|---|---|
| `zentrum_share_current` | Most-recent-Reichstag-election Zentrum vote share at panel year $t$ (carry-forward from each election to the next) | [`build_dataset.py`](src/data/build_dataset.py) merge step |
| `polen_share_current` | Same, Polen vote share | as above |
| `catholic_party_share_current` | Same, Zentrum + Polen combined | as above |

Panel years 1862–1870 inherit the 1871 election share (backfill); 1871–1873 use 1871; 1874–1877 use 1874; 1878–1880 use 1878; 1881–1883 use 1881; 1884–1886 use 1884; 1887–1889 use 1887; 1890 uses 1890.

**Headline analytical use (`political_mobilization.py`).** The seven elections constitute a stacked-cross-section DiD panel with `zentrum_share` as the outcome and `cath_share × Post` as the treatment. The estimated coefficient is $\hat\beta = +0.277$ ($p<0.001$) — for each percentage point of 1871 Catholic share, Zentrum vote share rose by 0.28 pp in post-Kulturkampf elections. A 100% Catholic vs 0% Catholic comparison implies a 27.7 pp *additional* Zentrum mobilisation attributable to the Kulturkampf legislation. Phase-specific estimates show the effect peaks during rollback (enforcement $\hat\beta=+0.236$; rollback $\hat\beta=+0.302$; post-rollback $\hat\beta=+0.284$, all $p<0.001$) — the Kulturkampf *permanently* politicised Catholic identity. See [`fig_zentrum_mobilization.png`](outputs/figures/fig_zentrum_mobilization.png) and notebook 03 §14.

#### 6.5d Time-varying urban share (URB1875–1890)

Galloway publishes Kreis-level urbanisation cross-sections at **1875, 1880, 1885, and 1890** in `URB{year}.XLS`. [`load_urb_panel()`](src/data/load_data.py) loads all four (Type-0 Kreise, Code<900), harmonises a Galloway formatting quirk in URB1885 (urban-population column is `Poptot-1` rather than `Popurban`), and returns a long-format panel of (Code, Year, percenturban, popurban, poptot). The panel is merged into the analysis frame via **linear interpolation between anchors** to produce an annual `urban_share_current` variable for panel years 1875–1890.

| Variable | Definition | Built at |
|---|---|---|
| `urban_share_current` | Linearly-interpolated urban population share at panel year $t$, anchored at URB1875/1880/1885/1890 measurements | [`build_dataset.py`](src/data/build_dataset.py) merge step |

**Pre-1875: `urban_share_current` is NaN by construction.** No Galloway URB cross-section exists before 1875; iPEHD's `f_urban` (1871 cross-section, time-invariant) remains the appropriate urbanisation control for the pre-treatment period. The two measures are **not directly comparable in levels**: iPEHD's 1871 `f_urban` averages 22.5% across panel Kreise, while URB1875's `Percenturban` averages 28.6% in 1875. The 6 pp gap reflects different urban-place thresholds (URB uses Galloway's stricter definition; iPEHD harmonises to Becker–Woessmann's Reichstag-1871 base) rather than four years of urbanisation. Keep them separate: `f_urban` for the 1871-static iPEHD heterogeneity slot, `urban_share_current` for the time-varying Bai/Hsiao spec.

**Empirical pattern.** Mean `urban_share_current` rises gently from 22.9% in 1875 to 24.6% in 1890 — a ~1.7 pp increase over the post-Kulturkampf window. By Catholic-share group, high-Catholic counties remain substantially less urban than low-Catholic counties throughout (20.5% vs 26.7% in 1890), and the two trajectories are parallel — urbanisation did not differentially accelerate in either group. Useful as a control variable: lets the Bai/Hsiao specification allow each county to follow its own urbanisation-trajectory-implied trend rather than relying on the 1871-static iPEHD value.

### 6.6 Coale–Watkins framework (Princeton EFP / Galloway 1994)

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
3. **Marriage-share schedule.** Originally a Prussia-wide constant Hajnal-derived schedule (overall prevalence ~62% of women 15–49 married, peaking at 90% in ages 35–39). **As of May 2026 the schedule is *recalibrated per county*** using STA1871's `Marriedover15f / Popover15f`: each county receives a scalar shifter $k_i = \mu_i / \bar\mu^{\text{Prussia}}$ (clipped to $[0.5, 1.5]$) that scales the marital-fertility denominator. This preserves the Prussia-wide age pattern of marriage but lets the *level* of marriage prevalence vary across counties — closing a known weakness in the constant-schedule approximation. Within-county across time, the shifter is still constant (STA1871 is a 1871 cross-section).
4. **Women 15–49 share** of total population: county-specific value from POP1871 (`women_15_49_1871 / pop_total_1871`, mean ≈ 24.9%, sd ≈ 0.85), scaled to mid-year population. Allows cross-county variation but assumes within-county constancy across 1862–1890.

These approximations affect the *level* of the indices but **not** the within-county DiD coefficients, which are what the empirical analysis uses. The STA1871 recalibration also leaves the within-county DiD invariant (the shifter is time-invariant within a county), but it reduces *cross-county* dispersion in $I_g$ (sd drops from 0.079 → 0.068 in 1871), correctly partitioning cross-county marriage-prevalence variation onto the nuptiality side instead of the marital-fertility component.

**Empirical levels** (current build, post-STA1871-recalibration): $I_f = 0.37$, $I_g = 0.55$ panel mean (0.66 in high-Catholic counties — within the Princeton EFP target range 0.65–0.72), $I_h = 0.07$, gmfr ≈ 230. Cross-county dispersion of $I_g$ is now within the Princeton EFP norm.

### 6.7 Other constructed variables

| Variable | Definition | Formula | Unit | Built at |
|---|---|---|---|---|
| `ln_pop` | Log mid-year population. **No longer a default control** (see §8 caveat below): every rate outcome already has mid-year population in its denominator, so adding $\ln(\text{Pop})$ on the right-hand side is mechanically correlated with the LHS and risks "bad-control" bias. Kept in the panel and as one of the tested specifications in the population/migration robustness table. | $\ln(\texttt{Poptot\_midyear})$ | log persons | [`build_dataset.py:178–181`](src/data/build_dataset.py:178) |
| `cbr_flag` | CBR outlier flag (catches extremes under either mid-year or Galloway carry-forward denominator) | $\mathbb{1}[\texttt{cbr}\notin[15,70]\,\text{or}\,\texttt{cbr\_carryforward}\notin[15,70]]$ | bool; **6 obs = TRUE** | [`build_dataset.py:222–230`](src/data/build_dataset.py:222) |
| `gfr_flag` | GFR outlier flag | $\mathbb{1}[\texttt{gfr\_static\_1871}>400]$ | bool; **10 obs = TRUE** (mostly 1869–1872 boundary-reform artefacts) | [`build_dataset.py:264–266`](src/data/build_dataset.py:264) |

### 6.8 Religious-infrastructure and schooling channels (1849 → 1882/1886)

Two long-difference DiD specifications use the 1849 iPEHD baseline alongside the 1882 Galloway endpoint to measure the Kulturkampf's effect on Catholic religious-sector employment and on schooling participation. Both are implemented in [`channels.py`](src/analysis/channels.py).

| Function | Outcome | Period structure | Specification |
|---|---|---|---|
| [`religious_infrastructure_channel()`](src/analysis/channels.py:147) | $\log$(religious workers per 1,000 Catholics): 1849 measure = `rel1849_cat_priest`; 1882 measure = `rel_edu_emp_1882` | 2-period stacked panel | $\log(y_{it}) = \alpha + \beta_1 \texttt{cath\_share}_i + \beta_2 \texttt{post1882}_t + \beta_3 (\texttt{cath\_share}_i \times \texttt{post1882}_t) + \delta_{Rb_i} + \varepsilon_{it}$. SE clustered at county level. Predicted sign on $\beta_3$ is **negative** (Anzeigegesetz, clerical exile). On the current snapshot $\hat\beta_3 \approx -0.036$ (SE 0.002, $p < 0.001$, $N=577$). |
| [`schooling_channel()`](src/analysis/channels.py:265) | Elementary attendance rate: 1849 = $\frac{\texttt{edu1849\_pub\_ele\_stud\_m+f}}{\texttt{pop1849\_tot}}$; 1886 = $\frac{\texttt{attend\_public\_1886+private}}{\texttt{school\_age\_pop\_1886}}$. Reports `school1517` (1871) as a midpoint diagnostic. | 2-period stacked panel (1849, 1886) | Same DiD structure as above with year/Rb FE. On the current snapshot $\hat\beta_3 \approx -0.00022$ (SE $5 \times 10^{-5}$, $p < 0.001$). Catholic counties saw a slightly smaller expansion of compulsory schooling between 1849 and 1886. |

**Caveats.** The two measures in the religious-infrastructure long-difference are not identical concepts: 1849 measures clerical positions narrowly, 1882 measures religion + education + instruction occupations broadly (the GEL1882 occupational category does not separate clergy from teachers). The two-period DiD identifies the *change* in the Catholic-share gradient between baseline and endpoint, which is the parameter of interest; the levels are not directly comparable.

### 6.9 1849 pretreatment balance table

[`pretreatment_balance_1849_table()`](src/analysis/latex_tables.py:399) groups counties by 1871 Catholic-share quartile and reports means of:
- `attend_rate_1849` (elementary attendance rate, students / total pop)
- `cat_priest_per_1k_1849` (mechanical validation row)
- `factories_per_10k_1849` (industrial-structure baseline)
- `avg_household_size_1849`
- `born_in_kreis_share_1871` (mobility baseline)

with Welch t-tests of the Q4 − Q1 difference. The table goes into [`outputs/tables/pretreatment_balance_1849.tex`](outputs/tables/pretreatment_balance_1849.tex). It is the strongest balance evidence in the paper: in 1849, 23 years before the May Laws, the would-be high- and low-Catholic groups did not differ on industrial structure (∼50 factories per 10k in both), and schooling was if anything *higher* in Catholic counties (+0.9 pp). The differences that do exist (∼0.14-person household-size gap, ∼5-pp mobility gap) move in directions that are not confounders for the fertility result.

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
| Columns | **103** |
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
\text{pretreatment\_trends} \in \{\,\texttt{school1517},\;\texttt{f\_urban},\;\texttt{f\_pruss},\;\texttt{f\_jew},\;\texttt{women\_share\_15\_49\_1871},\;\texttt{born\_in\_kreis\_share\_1871},\;\texttt{attend\_rate\_1849\_baseline},\;\texttt{land\_gini\_1882},\;\texttt{ln\_income\_tax\_pc\_1876}\,\}
$$
and `pretreatment_trends_form` ∈ {`"linear"`, `"year_dummies"`}. The `f_jew × year` augmentation is the test that attenuates the marriage-rate coefficient by roughly half — a finding to highlight in the paper. Spec (6) adds the **share of women aged 15–49 in 1871** (`women_share_15_49_1871`, from POP1871, see §4.4 and §6.4) as an additional baseline. This is the demographic-age-structure analogue of the iPEHD socio-economic baselines and tests whether differential *fertility capacity* drives differential trends. On the current build, adding the female-share trend leaves the marriage-rate coefficient essentially unchanged ($-0.0021$ vs $-0.0020$ in spec (5)), indicating differential age-structure trends do not explain the result.

Specs (7)–(9) layer on moderators from the new Galloway / iPEHD merges:
- **(7) `born_in_kreis_share_1871` × trend.** Addresses the concern that high-Catholic Polish provinces had a distinctively immobile pre-treatment population whose demography evolved on its own trajectory. The marriage-rate coefficient is *strengthened* by this control ($-0.00267$, $p=0.004$).
- **(8) `attend_rate_1849_baseline` × trend.** A literally pre-treatment human-capital baseline (23 years before the May Laws), constructed from `edu1849_pub_ele_stud_{m,f}` and `pop1849_tot`. Available for only ~280 counties (the 1849 crosswalk coverage), shrinking $N$ to 7,143. Marriage-rate coefficient $-0.00220$ ($p=0.030$) — still significant.
- **(9) `land_gini_1882 × trend` + `ln_income_tax_pc_1876 × trend`.** Land-inequality and income gradients added on top of (8). Both controls are post-treatment in date (1882, 1876) but are approximately structural; their inclusion is a stress test, not a clean identification claim. Marriage-rate coefficient $-0.00259$ ($p=0.017$).

The point of the (7)–(9) ladder is to test that the marriage-rate effect survives nine progressively more demanding pretreatment-trajectory controls, including ones built on entirely pre-Kulturkampf 1849 data. It does.

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

- **Illegitimacy** ([`illegitimacy_analysis()`](src/analysis/channels.py:17)): baseline DiD on `illegitimacy_ratio` with `cath_share_x_post` (no $\ln(\text{Pop})$ control — see §6.7/§8 caveat).
- **Infant mortality** ([`infant_mortality_analysis()`](src/analysis/channels.py:73)): restricts to **1875+** (because of the Galloway definition change at 1875, see §4.1) and re-bases the post indicator at the **rollback** start: `cath_x_rollback = cath_share × 1[Year ≥ 1880]`.

---

## 9. Caveats and measurement issues

A short, honest accounting. Each item is annotated with where it bites.

- **Headline rates use mid-year population, not Galloway's raw `Poptot`.** Galloway's raw `Poptot` is the *previous* December census carried forward unchanged in inter-census years (empirically, **55% of obs have `Poptot` identical to the prior year's value**). The standard demographic CBR convention is mid-year (July 1) population — an approximation to person-years lived. We construct `Poptot_midyear` via linear interpolation between Dec censuses evaluated at July 1 and use it as the **headline denominator** for `cbr`, `legitimate_br`, `illegitimate_br`, `marriage_rate`, and migration rates (§6.2). The Galloway carry-forward variants are retained under a `_carryforward` suffix (§6.3) and reported only as a single robustness row in `baseline_did.tex` so a reader can see how using the database "out of the box" differs. On the current build, switching to mid-year *strengthens* the marriage-rate coefficient (from $-0.0036$ to $-0.0042$) and shifts the CBR coefficient from a puzzling near-zero ($-0.00028$) to a theoretically expected $-0.0041$ (still not significant but no longer null).
- **Pre-1872 population is interpolated, not measured** ([`load_data.py:171–175`](src/data/load_data.py:171)). On the current snapshot 3,632 of 11,493 raw VIT obs (≈ 32%) entered interpolation. Affects the denominator of every constructed rate for 1862–1871.
- **Infant-mortality data break in 1875.** The headline `infant_mortality_rate` (total IMR: total infant deaths per 1,000 total live births) is **defined only from 1875 onwards** because Galloway's illegitimate-infant-death column `Dth<1bas` does not appear in any pre-1875 VIT file. The legitimate-only diagnostic series `infant_mortality_rate_leg` (= `Dth_infant_leg / Birlegtot × 1000`) is available for the full panel via a `Dthyoung` fallback pre-1875, but the fallback captures a different (broader / less complete) age window than the post-1875 `Dth<1leg`, producing a sharp ~3–4× level discontinuity at 1875 — visible in [`fig_imr_break.png`](outputs/figures/fig_imr_break.png) and uniform across Catholic-share groups in [`fig_imr_by_group.png`](outputs/figures/fig_imr_by_group.png). The discontinuity is a measurement-definition artefact, not a real change in infant survival. [`channels.infant_mortality_analysis`](src/analysis/channels.py) restricts to 1875+ to use the proper total IMR.
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
