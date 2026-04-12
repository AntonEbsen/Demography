"""
build_1831_treatment_merge.py
=============================
Merges 1831 Census occupational data into the 1851-1881 PopulationsPast panel
to construct a pre-treatment "industrial intensity" variable for DiD analysis.

Problem solved
--------------
The 1831 Census file identifies registration districts by numeric code (RGNUM),
while the 1851-1881 panel identifies them by name (REGDIST). The two files also
operate at different granularities: 1831 is at parish level (~15,600 rows),
the panel is at registration sub-district level (~10,000 rows per decade).

This script:
  1. Derives a crosswalk  REGDIST name <-> RGNUM  from the CEN_1851 code
     in the 1851 source file  (CEN_1851 // 10000 == RGNUM).
  2. Aggregates 1831 parishes up to registration-district level (RGNUM).
  3. Computes treatment intensity  (MANUFAC / TOT1831)  and a binary
     above-median treatment indicator.
  4. Merges into the panel via the crosswalk.

Inputs  (adjust paths below)
------
  - census_1831_baseline.xlsx        : 1831 Census, parish-level
  - PopulationsPast_census_data_1851.xlsx : 1851 Census, sub-district-level
  - master_panel_data.csv            : your 1851-1881 panel

Output
------
  - master_panel_with_1831.csv       : panel + treatment variables
"""

import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
# Adjust these to match your project structure.
PATH_1831   = "../data/processed/census_1831_baseline.xlsx"
PATH_1851   = "../data/raw/PopulationsPast_census_data_1851.xlsx"
PATH_PANEL  = "../data/processed/master_panel_data.csv"
PATH_OUTPUT = "../data/processed/master_panel_with_1831.csv"


# ── 1. Load data ─────────────────────────────────────────────────────────
df_1831 = pd.read_excel(PATH_1831)
df_1851 = pd.read_excel(PATH_1851)
df_panel = pd.read_csv(PATH_PANEL)

print(f"Loaded 1831: {df_1831.shape}  |  1851: {df_1851.shape}  |  Panel: {df_panel.shape}")


# ── 2. Build crosswalk: REGDIST name  →  RGNUM ──────────────────────────
# The 1851 CEN_1851 code encodes the registration district number:
#   CEN_1851 = RGNUM * 10_000  +  sub-district sequence
# So integer-dividing by 10 000 recovers RGNUM.
#
# We restrict to England & Wales because the 1831 file has no Scottish data.

SCOTTISH_COUNTIES = [
    "SHETLAND", "ORKNEY", "CAITHNESS", "SUTHERLAND",
    "ROSS AND CROMARTY", "INVERNESS", "MORAY", "ARGYLL", "NAIRN",
    "BANFF", "ABERDEEN", "KINCARDINE", "FORFAR", "PERTH", "FIFE",
    "KINROSS", "CLACKMANNAN", "STIRLING", "DUNBARTON", "BUTE",
    "RENFREW", "LANARK", "LINLITHGOW", "EDINBURGH", "HADDINGTON",
    "BERWICK", "ROXBURGH", "SELKIRK", "PEEBLES", "DUMFRIES",
    "KIRKCUDBRIGHT", "WIGTOWN", "AYR", "ELGIN",
]

df_1851_ew = df_1851[~df_1851["REGCNTY"].isin(SCOTTISH_COUNTIES)].copy()
df_1851_ew["RGNUM"] = (df_1851_ew["CEN_1851"] // 10_000).astype(float)

# One REGDIST name per RGNUM (there is one known split: RGNUM 504 ->
# Pontefract / Hemsworth; both get the same 1831 values, which is correct).
crosswalk = df_1851_ew[["REGDIST", "RGNUM"]].drop_duplicates("REGDIST")

print(f"Crosswalk: {len(crosswalk)} district-name entries, "
      f"{crosswalk['RGNUM'].nunique()} unique RGNUM values")


# ── 3. Handle punctuation variants ──────────────────────────────────────
# The 1851 source file uses "ST," (e.g. "ST, MARTIN IN THE FIELDS")
# while the panel may use "ST." or "ST ".  We build a lookup dict that
# maps both variants to the correct RGNUM.

crosswalk_dict: dict[str, float] = {}
for _, row in crosswalk.iterrows():
    name    = row["REGDIST"]
    rgnum   = row["RGNUM"]
    # Original form  (as in the 1851 file, typically "ST,")
    crosswalk_dict[name] = rgnum
    # Common variant: "ST," -> "ST."
    crosswalk_dict[name.replace("ST,", "ST.")] = rgnum


# ── 4. Aggregate 1831 parishes to registration-district level ───────────
# Clean negative values (flagged in the 1831 README as undocumented).
COLS_TO_CLEAN = ["MANUFAC", "FAMTRADE", "FAMAGRI", "TOT1831"]
for col in COLS_TO_CLEAN:
    df_1831.loc[df_1831[col] < 0, col] = 0

agg_1831 = (
    df_1831
    .groupby("RGNUM")
    .agg(
        MANUFAC_1831  = ("MANUFAC",  "sum"),
        FAMTRADE_1831 = ("FAMTRADE", "sum"),
        FAMAGRI_1831  = ("FAMAGRI",  "sum"),
        TOT1831       = ("TOT1831",  "sum"),
    )
    .reset_index()
)


# ── 5. Compute treatment variables ──────────────────────────────────────
agg_1831["Industrial_Ratio_1831"] = (
    agg_1831["MANUFAC_1831"] / agg_1831["TOT1831"]
).replace([np.inf, -np.inf], 0).fillna(0)

median_intensity = agg_1831["Industrial_Ratio_1831"].median()

agg_1831["is_treated_baseline"] = (
    (agg_1831["Industrial_Ratio_1831"] > median_intensity).astype(int)
)

print(f"\n1831 aggregated to {len(agg_1831)} registration districts")
print(f"Median industrial ratio: {median_intensity:.4f}")
print(f"Treated / Control: "
      f"{(agg_1831['is_treated_baseline']==1).sum()} / "
      f"{(agg_1831['is_treated_baseline']==0).sum()}")


# ── 6. Map panel districts to RGNUM via the crosswalk ────────────────────
df_panel["RGNUM"] = df_panel["REGDIST"].map(crosswalk_dict)

matched   = df_panel["RGNUM"].notna().sum()
unmatched = df_panel["RGNUM"].isna().sum()
print(f"\nPanel rows matched to RGNUM: {matched}/{len(df_panel)} "
      f"({matched/len(df_panel)*100:.1f}%)")
print(f"Unmatched: {unmatched}  (Scottish districts + post-1851 boundary changes)")


# ── 7. Merge 1831 treatment data into the panel ─────────────────────────
merge_cols = [
    "RGNUM",
    "Industrial_Ratio_1831",
    "is_treated_baseline",
    "MANUFAC_1831",
    "FAMTRADE_1831",
    "FAMAGRI_1831",
    "TOT1831",
]

df_merged = df_panel.merge(agg_1831[merge_cols], on="RGNUM", how="left")

has_treatment = df_merged["Industrial_Ratio_1831"].notna().sum()
print(f"\nFinal panel rows with 1831 data: {has_treatment}/{len(df_merged)} "
      f"({has_treatment/len(df_merged)*100:.1f}%)")


# ── 8. Save ──────────────────────────────────────────────────────────────
df_merged.to_csv(PATH_OUTPUT, index=False)
print(f"\nSaved to {PATH_OUTPUT}")
print(f"Shape: {df_merged.shape}")
print(f"New columns: RGNUM, Industrial_Ratio_1831, is_treated_baseline, "
      f"MANUFAC_1831, FAMTRADE_1831, FAMAGRI_1831, TOT1831")


# ── 9. Diagnostics ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

# Treatment balance at district level (should be ~50/50)
yr1 = df_merged[df_merged["Year"] == 1851].dropna(subset=["is_treated_baseline"])
dist_treat = yr1.groupby("REGDIST")["is_treated_baseline"].first()
print(f"\nDistrict-level balance (1851):")
print(f"  Treated:  {(dist_treat == 1).sum()}")
print(f"  Control:  {(dist_treat == 0).sum()}")

# Treatment intensity distribution
print(f"\nIndustrial_Ratio_1831 summary (district-level):")
print(agg_1831["Industrial_Ratio_1831"].describe().to_string())

# Reminder about clustering
print("""
NOTE FOR REGRESSIONS
--------------------
All sub-districts within a registration district share the same 1831
treatment value. Cluster standard errors at the RGNUM (district) level
in your DiD regressions to account for this, e.g.:

  import statsmodels.api as sm

  model = sm.OLS(y, X).fit(
      cov_type='cluster',
      cov_kwds={'groups': df['RGNUM']}
  )
""")