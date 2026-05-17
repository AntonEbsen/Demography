# Transcribing married-women-15-49 from *Preußische Statistik*

This folder holds the CSV templates and ingest infrastructure for the
Galloway 1994 GMFR replication. The denominator (married women aged
15–49) must be transcribed from *Preußische Statistik* directly because
the populationspast.org electronic Galloway database includes the
marital-status table only for 1871 (`STA1871.XLS`).

## Source-volume mapping (confirmed from Galloway 2007 *Description Tables*, Table 4, p. 5)

| Census | *Preußische Statistik* volume | Per-Kreis cell available |
|---|---|---|
| 1871 | Heft 30  | Married Female **≥15** only (no 15–49 cap; we use the Princeton schedule, see §6.6b of DATA_APPENDIX) |
| 1880 | Heft 66 | **No marital-status × age data at Kreis level.** Skip or use Rb proxy. |
| 1885 | Heft 96 | **Totals only, no age breakdown.** Skip or use Rb proxy. |
| 1890 | Heft **121:1** | **Married Female 15–49 direct cell** |
| 1895 | Heft **148:1** | **Married Female 15–49 direct cell** |
| 1900 | Heft **177** | **Married Female 15–49 direct cell** |
| 1905 | Heft **206:1** | **Married Female 15–49 direct cell** |
| 1910 | Heft **234:1** | Married Female **15–44** direct cell (1-yr truncation) |

For the four golden years (1890, 1895, 1900, 1905) the source publishes
**one number per Kreis**: total married women aged 15–49. That's the
only cell you need. For 1910 the cap is 15–44, slightly narrower —
acceptable for replication but flag it in the methods section.

## What to fill in

Each template has one row per Type-0 Kreis, pre-populated with `Code`,
`Rb`, `Kreis`, and the published total population from `POP{year}` for
cross-checking. The cells:

| Column                  | What to put in                                                     |
|-------------------------|--------------------------------------------------------------------|
| **`mwf_15_49_direct`** ← primary | The single published "Married Female 15–49" cell from the source. Use this for 1890–1905. For 1910 use the published 15–44 cell. |
| `mwf_15_19`, `mwf_20_24`, …, `mwf_45_49` | Optional 5-year-band breakdown (use only if the source publishes bands rather than a single 15–49 cell). The loader sums them as a fallback. |
| `mwf_50p`               | Optional: married women 50+ (sanity check, not used in GMFR).      |
| `twf_15_19`, …, `twf_50p` | Optional: total women in each band (cross-validation).            |
| `source_page`           | Page number(s) in the source volume.                                |
| `notes`                 | Free-text comments.                                                |

If the source page gives a single 15–49 number, fill **only**
`mwf_15_49_direct` and leave the band columns blank — that's the
intended workflow for 1890–1910.

## Workflow

1. **Open** `MAR{year}_transcription_template.csv` in your spreadsheet
   editor (Excel / LibreOffice / Sheets). Do NOT save as `.xlsx`; the
   loader expects CSV.
2. **Fetch the source volume** — see [Where to find scans](#where-to-find-scans).
3. **Locate the per-Kreis Familienstand × Alter table.** In *Preußische
   Statistik* it's usually a multi-page Tabelle ordered by Provinz →
   Regierungsbezirk → Kreis. The column "Verheiratet weiblich, 15–49"
   (or similar German wording) is the one you transcribe.
4. **Match by Kreis name** — the Galloway `Code` order in the template
   is not the same as the source-volume order, so go row-by-row by name.
5. **Save** the filled CSV (overwriting the template) and run
   `python -m src.data.build_dataset`. The pipeline auto-merges the
   transcribed data and produces `gmfr_galloway_{year}` columns in the
   analysis panel.

## Where to find scans

- **HathiTrust** holds *Preußische Statistik* Heft 121–177, which covers
  the **1890 (121:1), 1895 (148:1), and 1900 (177)** censuses. Catalog:
  https://catalog.hathitrust.org/Record/000680750. Full-text scans
  available directly through HathiTrust's BookReader.
- HathiTrust gaps include Heft 64–69 and Heft 95–97, so Heft 66 (1880)
  and Heft 96 (1885) are not there — but those years don't have the
  Kreis-level cross-tab anyway, so the gap is moot for GMFR purposes.
- Heft 206:1 (1905) and Heft 234:1 (1910) are in HathiTrust's 179–222
  and >222 ranges respectively; check the catalog.
- **Internet Archive** carries several Google Books scans of
  *Preußische Statistik* — search `"Preussische Statistik"` and look at
  the title-page year to identify volumes.
- **Bayerische Staatsbibliothek MDZ** (`digitale-sammlungen.de`) has
  alternative scans of many German statistical volumes — useful when
  HathiTrust is gated.

## What if a Kreis is missing from the source?

Some Kreise underwent boundary reform between censuses. If a Kreis
in the template has no corresponding row in the source volume,
leave its cells blank — the loader treats blank as NaN and the
panel-merge will assign NaN for that Code. The Galloway crosswalk
already handles a few of these.

## Validation hints

- **Sanity check vs. population**. A Kreis with population ~50,000
  typically has ~10,000 women aged 15–49, of whom 50–70% are married
  (so `mwf_15_49_direct` should land in the **5,000–7,000** range for
  a typical mid-sized rural Kreis). Cities have higher % unmarried, so
  expect lower values.
- **Cross-time sanity**. The same Kreis across 1890, 1895, 1900, 1905
  should show slow, monotonic growth in married women (typically 1–3%
  per 5 years). A jump of more than 25% between adjacent censuses is
  almost certainly a transcription error or a boundary change.
- **Compute the implied GMFR after transcribing 10 Kreise**:
  `5-yr-avg(Birlegtot) / mwf_15_49_direct × 1000` should fall in
  [180, 450] for late-19th-century Prussia.

## Citation

When the new GMFR appears in the paper, cite the source volume(s)
explicitly:

> Married women aged 15–49 by Kreis are transcribed from *Preußische
> Statistik*, Heft 121:1 (1890), Heft 148:1 (1895), Heft 177 (1900),
> Heft 206:1 (1905), and Heft 234:1 (1910). Volume mapping confirmed
> against Galloway, Patrick R. 2007, *Prussia Vital Registration and
> Census Data Description Tables 1849 to 1914*, Table 4, p. 5
> (www.patrickrgalloway.com).

## AI-OCR assistance

If you supply a PDF page URL or screenshot of one or two Kreise of the
table, the assistant can OCR the "Married Female 15–49" cell and write
it into the template directly. Realistic per-session throughput: a
single page (typically 4–8 Kreise) at a time, with you spot-checking
~10% of transcribed values against the source.
