# Transcribing married-women-by-age from *Preußische Statistik*

This folder holds CSV templates for the Galloway 1994 GMFR replication, one
per Prussian census year (1880, 1885, 1890, 1895, 1900). Each row is a
Type-0 Kreis (the analysis unit), pre-populated with `Code`, `Rb`,
`Kreis`, and the published `pop_total_{year}_galloway` for cross-validation.

The cells to fill in are the seven 5-year age bands of **married women**
aged 15–49 (plus 50+), transcribed from the corresponding source volume.

## What to transcribe

For every Kreis, fill in:

| Column        | What                                                       |
|---------------|------------------------------------------------------------|
| `mwf_15_19`   | Married women (verheiratet, weiblich) aged 15–19 (or 15–20 in some volumes) |
| `mwf_20_24`   | Married women aged 20–24                                   |
| `mwf_25_29`   | Married women aged 25–29                                   |
| `mwf_30_34`   | Married women aged 30–34                                   |
| `mwf_35_39`   | Married women aged 35–39                                   |
| `mwf_40_44`   | Married women aged 40–44                                   |
| `mwf_45_49`   | Married women aged 45–49                                   |
| `mwf_50p`     | Married women aged 50+ (sanity check, not used in GMFR)    |
| `source_page` | Page number(s) in the source volume                        |
| `notes`       | Free text (e.g., "table uses 10-year bins, split 50:50")   |

If the source volume uses 10-year bins (e.g., 15–25, 25–35), put the
whole 10-year count in the lower band and zero in the upper band, then
flag it in `notes` — the loader sums across bands anyway, so this is
harmless for the 15–49 total.

The optional `twf_*` columns hold **total women** in each band — useful
for validation (you can check the implied marriage rate is plausible)
but not required.

## Source volumes (start here)

The marital-status × age cross-tab Galloway used lives in *Preußische
Statistik* (the official Royal Prussian Statistical Office series). The
specific Band numbers are:

- **1880 census**: *Preußische Statistik* **Band LXVI** (66), "Die
  definitiven Ergebnisse der Volkszählung vom 1. December 1880 im
  preussischen Staate." HathiTrust does **not** carry Band 66 (its
  gap covers 64–69). Internet Archive's Google Books scans may have
  it — search archive.org for *"Preussische Statistik 1880"* or
  *"Volkszählung 1880"*. Alternatively, the Bayerische Staatsbibliothek
  (BSB) MDZ portal at `digitale-sammlungen.de`.
- **1885 census**: *Preußische Statistik* Band 96 (probable; verify in
  the Galloway 1988/2007 inventory document). HathiTrust gap.
- **1890 census**: *Preußische Statistik* Band 121 and/or 122
  (verify). HathiTrust covers 121–177, so this one is on
  babel.hathitrust.org.
- **1895 census**: Band 145 (probable). HathiTrust.
- **1900 census**: Band 177 (probable). HathiTrust.

**Confirm volume numbers** by downloading Patrick Galloway's
*"Prussia vital registration and census data description tables 1849
to 1914"* inventory PDF from `patrickrgalloway.com` — it lists every
table he used by Band/Heft number.

## Workflow

1. **Open** `MAR1880_transcription_template.csv` in your spreadsheet
   editor of choice (Excel, LibreOffice, Google Sheets). Save to a new
   name so the template stays clean: e.g., `MAR1880.csv`.
2. **Fetch the source volume** PDF and find the per-Kreis marital-
   status table (usually titled "Bevölkerung nach Familienstand und
   Alter" or similar, organised by Provinz → Regierungsbezirk → Kreis).
3. **Transcribe row-by-row**. The Galloway Code in the template matches
   the order in which Kreise appear in his database, *not* the order in
   *Preußische Statistik*. Use the Kreis name as the matching key.
4. **Save** the filled CSV as `MAR{year}_transcription_template.csv`
   (overwriting the empty template). The loader picks it up
   automatically on the next `python -m src.data.build_dataset` run.
5. **Run** `python -m src.data.build_dataset` — the pipeline merges the
   transcribed data and produces `gmfr_galloway_{year}` columns.

## Validation hints

- **Population cross-check**: For each row, `(mwf + total non-married
  women all ages) + (total men all ages) ≈ pop_total_{year}_galloway`.
  If you transcribe `twf_*` too, you can check `sum(twf) ≈ total
  women`. The exact equality won't hold (military, institutional,
  unknown-age subtractions), but should be within ~5%.
- **Married-50+ sanity**: In late-19th-century Prussia, married women
  50+ are typically 25–35% of all married women. If your `mwf_50p` is
  outside that range relative to the 15–49 sum, double-check.
- **Provincial means** of `married_women_15_49` should land in the
  hundreds to low thousands for a typical Kreis (population ~50k,
  women 15–49 ~10k, ~60% married → ~6k).

## Citation

When the GMFR appears in the paper, cite the source volume(s) explicitly,
e.g.:

> Married women aged 15–49 by Kreis are transcribed from *Preußische
> Statistik*, Band LXVI (1880), Tabelle X, pp. yyy–zzz. Legitimate
> birth counts are from Galloway, Patrick R., 2007, "Galloway Prussia
> Database 1861 to 1914," www.patrickrgalloway.com.

## Asking the AI assistant for OCR help

If you have a PDF page URL or screenshot of a single table page, the
assistant can read the numbers off the scan and fill in the
corresponding rows. Practical limits:

- One page at a time works best (typically 4–8 Kreise per page).
- The assistant cannot bulk-process a 200-page PDF in one shot, so the
  page-by-page workflow needs to be driven by you.
- Always spot-check at least 5–10 transcribed rows against the source
  before trusting the full file.
