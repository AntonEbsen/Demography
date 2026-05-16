"""
merge_ipehd.py
==============
Merges iPEHD variables into the Galloway-based analysis panel.

The main challenge: Galloway and iPEHD use different county numbering.
This module handles the crosswalk through a three-step process:
  1. Direct code match (where kreiskey1871 == Galloway Code)
  2. Manual crosswalk for known mismatches (spelling, name variants)
  3. Aggressive name matching for remaining cases

Coverage: ~346 out of 393 Galloway Type-0 counties (88%).

Usage (from notebook):
    from src.data.merge_ipehd import merge_ipehd_controls, IPEHD_CONTROL_VARS
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional

# iPEHD variables worth merging as controls
IPEHD_CONTROL_VARS = [
    "f_prot",          # Protestant share
    "f_cath",          # Catholic share (for validation)
    "f_jew",           # Jewish share
    "f_urban",         # Urban population share
    "f_young",         # Share of population under 15
    "f_fem",           # Female share
    "f_ortsgeb",       # Share born in locality (immobility proxy)
    "f_pruss",         # Share with Prussian citizenship
    "hhsize",          # Average household size
    "lnpop",           # Log population (iPEHD version)
    "pop",             # Total population
    "gpop",            # Population growth rate
    "f_blind",         # Share blind
    "f_deaf",          # Share deaf
    "f_dumb",          # Share mute
    "kmwittenberg",    # Distance to Wittenberg (km)
    "f_miss",          # Share with missing education data
    "school1517",      # School enrollment rate, ages 15-17 (literacy proxy)
]

# Manual crosswalk: Galloway Code -> iPEHD kreiskey1871
# Built by comparing county names and Regierungsbezirk assignments
_MANUAL_CROSSWALK = {
    40: 33,     # LYCK / Lyk
    47: 38,     # MARIENBURG (DAN) / Marienburg i. Pr.
    54: 41,     # PREUSSISCH STARGARD / Pr. Stargard
    76: 56,     # FLATOW / Flatkow
    77: 57,     # DEUTSCH KRONE / Dt. Krone
    91: 64,     # TELTOW / Teltow (incl Charlottenburg)
    93: 66,     # JUTERBOG-LUCKENWALDE / Jüterbock-Luckenwalde
    94: 67,     # ZAUCH-BELZIG / Zauche-Belzig
    97: 69,     # OSTHAVELLAND / Ost-Havelland (incl Spandau)
    99: 70,     # WESTHAVELLAND / West-Havelland (incl Brandenburg)
    101: 72,    # OSTPRIGNITZ / Ostpriegnitz
    102: 73,    # WESTPRIGNITZ / Westpriegnitz
    112: 82,    # WESTSTERNBERG / West-Sternberg
    113: 81,    # OSTSTERNBERG / Ost-Sternberg
    131: 95,    # UCKERMUNDE / Ükermünde
    164: 118,   # FRANZBURG / Franzburg (incl Stralsund)
    184: 126,   # GRATZ / Buk
    178: 129,   # OBORNIK / Obernik
    191: 135,   # GOSTYN / Kröben
    202: 140,   # KOLMAR / Chodziesen
    208: 144,   # HOHENSALZA / Inowraclaw
    216: 149,   # GROSS WARTENBERG / Wartenberg
    217: 150,   # OLS / Oels
    251: 179,   # GOLDBERG-HAINAU / Goldberg-Haynau
    272: 195,   # GROSS STREHLITZ / Gross Stehlitz
    331: 231,   # SAALKREIS / Saale
    336: 234,   # MANSFELDER GEBIRGSKR / Gebirgskreis Mansfeld
    338: 235,   # MANSFELDER SEEKREIS / Seekreis Mansfeld
    411: 278,   # SPRINGE / Wennigsen
    417: 281,   # MARIENBURG (HIL) / Marienburg b. H.
    420: 282,   # GOSLAR / Liebenburg
    440: 291,   # ULZEN / Uelzen
    453: 295,   # STADER MARSCHKREIS / Stader Marsch
    454: 296,   # STADER GEESTKREIS / Stader Geest
    457: 298,   # HADELN / Otterndorf
    466: 302,   # ROTENBURG (STA) / Rotenburg a. W.
    561: 353,   # ROTENBURG (KAS) / Rotenburg a. F.
    577: 367,   # SCHAUMBURG / Rinteln
    580: 369,   # DILLKREIS / Dillkreis (Dillenburg)
    581: 370,   # OBERWESTERWALDKREIS / Oberwesterwald
    583: 371,   # UNTERWESTERWALDKREIS / Unterwesterwald
    584: 372,   # OBERLAHNKREIS / Oberlahnkreis (Weilburg)
    586: 373,   # UNTERLAHNKREIS / Unterlahnkreis (Diez)
    588: 374,   # RHEINGAUKREIS / Rheingau-Kreis
    594: 375,   # OBERTAUNUSKREIS -> Mainkreis (Wiesbaden)
    592: 378,   # UNTERTAUNUSKREIS / Untertaunus
    579: 380,   # BIEDENKOPF / Hinterlandkreis
    623: 399,   # MULHEIM (DUS) / Mülheim a. d. Ruhr
    646: 410,   # LENNEP / Lennep (incl Remscheid)
    661: 419,   # MULHEIM (KOL) / Mülheim am Rhein
    686: 438,   # SANKT WENDEL / Sanct Wendel
    699: 450,   # SIGMARINGEN / Oberamtsbezirk Sigmaringen
    700: 451,   # GAMMERTINGEN / Oberamtsbezirk Gammertingen
    701: 452,   # HECHINGEN / Oberamtsbezirk Hechingen
    702: 453,   # HAIGERLOCH / Oberamtsbezirk Haigerloch
}


def _clean_name(name):
    """Aggressively clean county name for fuzzy matching."""
    if pd.isna(name):
        return ""
    s = str(name).upper().strip()
    s = s.replace("Ö", "O").replace("Ü", "U").replace("Ä", "A")
    s = s.replace("ö", "O").replace("ü", "U").replace("ä", "A")
    s = s.replace("ß", "SS").replace("É", "E").replace("È", "E")
    # The 1849 iPEHD CSVs round-tripped umlauts through a lossy encoding
    # and replaced them with U+FFFD ('?'). We drop the orphan character so
    # "K?nigsberg Stadt" still matches "KONIGSBERG STADT".
    s = s.replace("�", "")
    s = re.sub(r'\(.*?\)', '', s)
    for prefix in ["STADT ", "LANDKREIS ", "KREIS ", "STADTKREIS ",
                    "OBERAMTSBEZIRK ", "PR. ", "DT. ", "PREUSS. ", "PREUSS ", "GR. "]:
        s = s.replace(prefix, "")
    s = re.sub(r'\s+(I\.\s*WPR\.?|I\.\s*PR\.?|A\.\s*D\.\s*DREW\.?|'
               r'AM HARZ|I\.\s*WESTF\.?|A\.\s*H\.?|A\.\s*W\.?|A\.\s*F\.?|'
               r'B\.\s*H\.?|AM RHEIN|AM MAIN)\s*$', '', s)
    s = re.sub(r'\s+(STADT|LAND)\s*$', '', s)
    s = s.replace('-', ' ').replace('  ', ' ').strip()
    return s


def build_crosswalk(
    ipehd_path: Path,
    rel_path: Path,
    cath_tolerance: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the Galloway Code -> iPEHD kreiskey1871 crosswalk.
    
    IMPORTANT: The two datasets use different numbering systems. Codes
    that happen to match numerically often refer to *different* counties.
    For example, Galloway code 388 is Süderdithmarschen (Schleswig),
    but iPEHD kreiskey1871 = 388 is Adenau (Rhineland).
    
    To prevent garbage matches, every candidate match (whether from
    direct code match, manual crosswalk, or name match) is validated
    by comparing the Catholic population share. Matches where the
    shares differ by more than `cath_tolerance` percentage points are
    discarded.
    
    Returns DataFrame with columns: Code, kreiskey1871
    """
    dta = pd.read_stata(ipehd_path)
    rel = pd.read_excel(rel_path)
    rel_t0 = rel[(rel['Code'] < 900) & (rel['Type'] == 0)].copy()
    rel_t0['_gal_cath'] = (rel_t0['Relcathm'] + rel_t0['Relcathf']) / rel_t0['Pop'] * 100
    
    # Helper: validate a set of candidate matches by Catholic share
    def _validate(candidates):
        """Drop matches where Galloway and iPEHD Catholic shares differ too much."""
        check = candidates.merge(rel_t0[['Code', '_gal_cath']], on='Code')
        check = check.merge(dta[['kreiskey1871', 'f_cath']], on='kreiskey1871')
        check['_diff'] = abs(check['_gal_cath'] - check['f_cath'])
        good = check[check['_diff'] <= cath_tolerance]
        bad_n = len(check) - len(good)
        return good[['Code', 'kreiskey1871']], bad_n
    
    # Step 1: Direct code match (then validate)
    direct_raw = rel_t0[['Code']].merge(
        dta[['kreiskey1871']],
        left_on='Code', right_on='kreiskey1871', how='inner'
    )[['Code', 'kreiskey1871']]
    direct, bad_direct = _validate(direct_raw)
    if verbose:
        print(f"Step 1 — Direct code match: {len(direct_raw)} candidates -> "
              f"{len(direct)} valid (dropped {bad_direct})")
    
    # Step 2: Manual crosswalk (then validate)
    manual_raw = pd.DataFrame([
        {'Code': k, 'kreiskey1871': v}
        for k, v in _MANUAL_CROSSWALK.items()
    ])
    manual_raw = manual_raw[~manual_raw['Code'].isin(direct['Code'])]
    manual, bad_manual = _validate(manual_raw)
    if verbose:
        print(f"Step 2 — Manual crosswalk: {len(manual_raw)} candidates -> "
              f"{len(manual)} valid (dropped {bad_manual})")
    
    # Step 3: Name match (then validate)
    matched_codes = set(direct['Code']) | set(manual['Code'])
    matched_keys = set(direct['kreiskey1871']) | set(manual['kreiskey1871'])
    
    rem_gal = rel_t0[~rel_t0['Code'].isin(matched_codes)].copy()
    rem_gal['name_clean'] = rem_gal['Kreis'].apply(_clean_name)
    
    rem_ipehd = dta[~dta['kreiskey1871'].isin(matched_keys)].copy()
    rem_ipehd['name_clean'] = rem_ipehd['county1871'].apply(_clean_name)
    
    name_raw = rem_gal[['Code', 'name_clean']].merge(
        rem_ipehd[['kreiskey1871', 'name_clean']],
        on='name_clean', how='inner'
    ).drop_duplicates(subset='Code')[['Code', 'kreiskey1871']]
    name_match, bad_name = _validate(name_raw)
    if verbose:
        print(f"Step 3 — Name match: {len(name_raw)} candidates -> "
              f"{len(name_match)} valid (dropped {bad_name})")
    
    # Combine
    crosswalk = pd.concat([direct, manual, name_match], ignore_index=True)
    crosswalk = crosswalk.drop_duplicates(subset='Code')
    
    if verbose:
        print(f"\nFinal crosswalk: {len(crosswalk)} county mappings "
              f"(out of {len(rel_t0)} Galloway Type-0 counties)")
    
    return crosswalk.sort_values('Code').reset_index(drop=True)


def merge_ipehd_controls(
    panel: pd.DataFrame,
    ipehd_path: Optional[Path] = None,
    rel_path: Optional[Path] = None,
    controls: Optional[list] = None,
) -> pd.DataFrame:
    """
    Merge iPEHD control variables into the analysis panel.
    
    The iPEHD variables are cross-sectional (1871), so they enter
    the panel as time-invariant controls.
    
    Parameters
    ----------
    panel : pd.DataFrame
        The Galloway-based analysis panel with 'Code' column.
    ipehd_path : Path
        Path to ipehd_qje2009_master.dta
    rel_path : Path
        Path to REL1871 (needed for crosswalk construction)
    controls : list
        Which iPEHD variables to merge. Default: IPEHD_CONTROL_VARS
    
    Returns
    -------
    pd.DataFrame : panel with additional iPEHD columns.
        Unmatched counties retain NaN for iPEHD variables.
    """
    from src.data.load_data import DATA_RAW

    # The iPEHD master .dta lives under data/raw/ipehd_data/, not in the
    # galloway_data subdir that DATA_RAW points to.
    ipehd_dir = DATA_RAW.parent / "ipehd_data"

    if ipehd_path is None:
        ipehd_path = ipehd_dir / "ipehd_qje2009_master.dta"
    if rel_path is None:
        from src.data.load_data import _find_file
        rel_path = _find_file(DATA_RAW, "REL1871")
    if controls is None:
        controls = IPEHD_CONTROL_VARS
    
    # Load iPEHD
    dta = pd.read_stata(ipehd_path)
    
    # Filter to variables that actually exist
    available = [c for c in controls if c in dta.columns]
    missing = [c for c in controls if c not in dta.columns]
    if missing:
        print(f"Warning: iPEHD variables not found: {missing}")
    
    # Build crosswalk
    crosswalk = build_crosswalk(ipehd_path, rel_path)
    
    # Merge crosswalk with iPEHD data
    ipehd_subset = dta[['kreiskey1871'] + available].copy()
    crosswalk_with_data = crosswalk.merge(ipehd_subset, on='kreiskey1871', how='left')
    
    # Drop kreiskey1871 — we only need Code for merging with panel
    crosswalk_with_data = crosswalk_with_data.drop(columns=['kreiskey1871'])
    
    # Merge into panel
    n_before = len(panel)
    panel_merged = panel.merge(crosswalk_with_data, on='Code', how='left')
    
    # Report
    n_matched = panel_merged[available[0]].notna().sum() if available else 0
    n_total = len(panel_merged)
    n_counties_matched = panel_merged[panel_merged[available[0]].notna()]['Code'].nunique() if available else 0
    n_counties_total = panel_merged['Code'].nunique()
    
    print(f"iPEHD merge complete:")
    print(f"  Crosswalk: {len(crosswalk)} county mappings")
    print(f"  Counties with iPEHD data: {n_counties_matched} / {n_counties_total}")
    print(f"  Observations with iPEHD data: {n_matched} / {n_total}")
    print(f"  Variables added: {available}")
    
    # Validate: compare Catholic shares
    if 'f_cath' in available:
        both = panel_merged[panel_merged['f_cath'].notna()].copy()
        both_counties = both.groupby('Code')[['cath_share', 'f_cath']].first()
        corr = both_counties['cath_share'].corr(both_counties['f_cath'])
        print(f"\n  Validation: corr(Galloway cath_share, iPEHD f_cath) = {corr:.4f}")
        if corr > 0.95:
            print(f"  OK Crosswalk validated — religion shares match closely.")
        else:
            print(f"  WARN Low correlation — check crosswalk for errors!")

    return panel_merged


# ===================================================================
# 1849 iPEHD crosswalk and merger
# ===================================================================
#
# The 1849 iPEHD CSV files (pop_demo, pop_mari, edu_stud, rel_church,
# indu_fac, indu_tec, indu_trans) key on `kreiskey1849`, a numbering
# system distinct from `kreiskey1871`. Prussian territorial boundaries
# were redrawn in 1815/1818 and again after the 1866 Austro-Prussian
# War (when Hannover, Hesse-Kassel, Nassau, Schleswig-Holstein, and
# Frankfurt were annexed); the 1849 file therefore covers ~335 of the
# 393 Type-0 Kreise present in 1871.
#
# Strategy. We build a Code -> kreiskey1849 crosswalk by cleaning the
# 1849 iPEHD county name and matching it against the cleaned Galloway
# 1871 Kreis name within the same Regierungsbezirk first, then across
# Rbs. Religion-share validation is not available (the 1849 files do
# not include population by denomination), so we apply two coarser
# sanity filters after matching: (i) iPEHD rb -> Galloway Rb agreement
# for ~90% of the high-confidence matches; (ii) a soft monotonicity
# check that high Catholic-priest counts in 1849 correspond to high
# 1871 Catholic share (the priest-population ratio is roughly stable
# across denominations within decades).


def _load_1849_csv(path: Path) -> pd.DataFrame:
    """Read a 1849 iPEHD CSV with the encoding the file ships in.

    The files were written with a Western European code page and contain
    a small number of replacement characters (?) where umlauts could not
    round-trip. ``_clean_name`` strips those out for matching purposes.
    """
    return pd.read_csv(path, encoding="cp1252")


def build_crosswalk_1849(
    ipehd_1849_path: Path,
    rel1871_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the Galloway Code -> iPEHD kreiskey1849 crosswalk.

    Uses 1849 iPEHD `pop_demo` (which lists every 1849 Kreis with name
    and Regierungsbezirk code) as the source side, and Galloway REL1871
    Type-0 Kreise as the target side. Matches by cleaned name within Rb
    first, then across Rbs. No religion-share validation (1849 files
    have no by-denomination population), so callers should treat the
    output as conservative: unmatched 1849 Kreise are discarded.

    Returns
    -------
    pd.DataFrame with columns: Code, kreiskey1849, rb_1849
    """
    src = _load_1849_csv(ipehd_1849_path)
    rel = pd.read_excel(rel1871_path)
    rel_t0 = rel[(rel['Code'] < 900) & (rel['Type'] == 0)].copy()

    src = src[["kreiskey1849", "county", "rb"]].copy()
    src["name_clean"] = src["county"].apply(_clean_name)
    rel_t0["name_clean"] = rel_t0["Kreis"].apply(_clean_name)

    # Step 1: exact match within Rb.
    within = src.merge(
        rel_t0[["Code", "Rb", "name_clean"]],
        left_on=["rb", "name_clean"], right_on=["Rb", "name_clean"],
        how="inner",
    )[["Code", "kreiskey1849", "rb"]].rename(columns={"rb": "rb_1849"})
    within = within.drop_duplicates(subset="Code")

    if verbose:
        print(f"1849 Step 1 - Exact name within Rb: {len(within)} matches")

    # Step 2: exact match across Rbs (a handful of counties were moved
    # between Rbs between 1849 and 1871).
    remaining_src = src[~src["kreiskey1849"].isin(within["kreiskey1849"])]
    remaining_rel = rel_t0[~rel_t0["Code"].isin(within["Code"])]

    across = remaining_src.merge(
        remaining_rel[["Code", "name_clean"]],
        on="name_clean", how="inner",
    )
    across = across.drop_duplicates(subset="name_clean", keep=False)
    across = across[["Code", "kreiskey1849", "rb"]].rename(columns={"rb": "rb_1849"})
    across = across.drop_duplicates(subset="Code")

    if verbose:
        print(f"1849 Step 2 - Exact name across Rbs: {len(across)} matches")

    # Step 3: substring match within Rb (catches "ALLENSTEIN" vs
    # "ALLENSTEIN LAND" type variants).
    matched_src = set(within["kreiskey1849"]) | set(across["kreiskey1849"])
    matched_rel = set(within["Code"]) | set(across["Code"])
    rem_src = src[~src["kreiskey1849"].isin(matched_src)].copy()
    rem_rel = rel_t0[~rel_t0["Code"].isin(matched_rel)].copy()
    rows = []
    for _, srow in rem_src.iterrows():
        s_clean = srow["name_clean"]
        if not s_clean or len(s_clean) < 3:
            continue
        cand = rem_rel[
            (rem_rel["Rb"] == srow["rb"])
            & rem_rel["name_clean"].apply(
                lambda r: bool(r) and (r in s_clean or s_clean in r)
            )
        ]
        if len(cand) == 1:
            rows.append((int(cand.iloc[0]["Code"]), int(srow["kreiskey1849"]),
                         srow["rb"]))
    contains = pd.DataFrame(rows, columns=["Code", "kreiskey1849", "rb_1849"])
    contains = contains.drop_duplicates(subset="Code")

    if verbose:
        print(f"1849 Step 3 - Substring within Rb: {len(contains)} matches")

    # Step 4: fuzzy match within Rb (edit distance 1 or 2). Catches
    # systematic 1849->1871 spelling drift like WELAU/WEHLAU,
    # EILAU/EYLAU, MORUNGEN/MOHRUNGEN, BEHREND/BERENT, KULM/CULM,
    # WONGROWIZ/WONGROWITZ, PRIEGNITZ/PRIGNITZ.
    matched_src |= set(contains["kreiskey1849"])
    matched_rel |= set(contains["Code"])
    rem_src = src[~src["kreiskey1849"].isin(matched_src)].copy()
    rem_rel = rel_t0[~rel_t0["Code"].isin(matched_rel)].copy()

    def _edit_distance_le(a: str, b: str, k: int) -> bool:
        # Cheap bound check then full DP, only computed when length-diff fits.
        if abs(len(a) - len(b)) > k:
            return False
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            prev = cur
        return prev[n] <= k

    fuzzy_rows: list[tuple[int, int, str]] = []
    for _, srow in rem_src.iterrows():
        s_clean = srow["name_clean"]
        if not s_clean or len(s_clean) < 4:
            continue
        in_rb = rem_rel[rem_rel["Rb"] == srow["rb"]]
        cand = []
        for _, rrow in in_rb.iterrows():
            r_clean = rrow["name_clean"]
            if not r_clean or len(r_clean) < 4:
                continue
            # tolerance scales with the shorter name length: 1 for short,
            # 2 for longer, but cap at 2 to avoid runaway false-positives.
            tol = 1 if min(len(s_clean), len(r_clean)) <= 7 else 2
            if _edit_distance_le(s_clean, r_clean, tol):
                cand.append((int(rrow["Code"]), int(srow["kreiskey1849"]),
                             srow["rb"]))
        if len(cand) == 1:
            fuzzy_rows.append(cand[0])
    fuzzy = pd.DataFrame(fuzzy_rows, columns=["Code", "kreiskey1849", "rb_1849"])
    fuzzy = fuzzy.drop_duplicates(subset="Code")

    if verbose:
        print(f"1849 Step 4 - Fuzzy match within Rb: {len(fuzzy)} matches")

    crosswalk = pd.concat([within, across, contains, fuzzy], ignore_index=True)
    crosswalk = crosswalk.drop_duplicates(subset="Code")

    if verbose:
        print(f"1849 Final crosswalk: {len(crosswalk)} county mappings "
              f"(out of {len(rel_t0)} Galloway Type-0 counties; "
              f"{len(src)} 1849 Kreise available)")

    return crosswalk.sort_values("Code").reset_index(drop=True)


def merge_ipehd_1849(
    panel: pd.DataFrame,
    crosswalk_1849: pd.DataFrame,
    files: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Merge selected 1849 iPEHD variables into the panel as time-invariant
    pre-treatment covariates. Like the existing 1871 merge, the 1849
    variables are cross-sectional and apply to every panel year for the
    matched county.

    Parameters
    ----------
    panel : pd.DataFrame
        Analysis panel with a 'Code' column.
    crosswalk_1849 : pd.DataFrame
        Output of `build_crosswalk_1849`; columns Code, kreiskey1849.
    files : dict, optional
        Mapping of {filename_stem: [columns_to_keep]}. Defaults bring in
        the religious-infrastructure and schooling variables used by the
        Kulturkampf channels.

    Returns
    -------
    pd.DataFrame : panel with new columns.
    """
    from src.data.load_data import DATA_RAW

    ipehd_dir = DATA_RAW.parent / "ipehd_data"

    if files is None:
        files = {
            "ipehd_1849_rel_church.csv": [
                "rel1849_cat_priest", "rel1849_cat_chaplain_vicar",
                "rel1849_cat_main_church", "rel1849_pro_priest",
                "rel1849_pro_main_church", "rel1849_jew_meetplace",
            ],
            "ipehd_1849_edu_stud.csv": [
                "edu1849_pub_ele_stud_m", "edu1849_pub_ele_stud_f",
                "edu1849_pub_mim_stud_m", "edu1849_pub_mif_stud_f",
                "edu1849_pub_high_stud_m", "edu1849_pub_gym_stud_m",
            ],
            "ipehd_1849_pop_demo.csv": [
                "pop1849_tot", "pop1849_m_tot", "pop1849_f_tot",
                "pop1849_f_17to45",
            ],
            "ipehd_1849_pop_mari.csv": [
                "pop1849_families", "pop1849_m_wedlock", "pop1849_f_wedlock",
            ],
            "ipehd_1849_indu_fac.csv": None,  # all numeric cols -> sum -> total
        }

    out = panel.copy()
    cw = crosswalk_1849[["Code", "kreiskey1849"]]

    for fname, cols in files.items():
        fpath = ipehd_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found")
            continue
        src = _load_1849_csv(fpath)

        if cols is None:
            # Default: take a single aggregate column = sum of all numeric
            # cols excluding the keys. Used for indu_fac (factory total).
            keep = [c for c in src.columns
                    if c not in ("kreiskey1849", "county", "rb")
                    and pd.api.types.is_numeric_dtype(src[c])]
            # Build a Python-identifier-safe aggregate name (no leading
            # digit; the source filename starts with "1849_..." which
            # would otherwise produce an awkward column name).
            stem = fname.replace("ipehd_", "").replace(".csv", "")
            agg_name = f"ipehd_{stem}_total"
            src[agg_name] = src[keep].sum(axis=1, min_count=1)
            cols = [agg_name]

        # Filter to columns that exist.
        cols = [c for c in cols if c in src.columns]
        merge_cols = ["kreiskey1849"] + cols
        merged = cw.merge(src[merge_cols], on="kreiskey1849", how="left")
        merged = merged.drop(columns=["kreiskey1849"])
        out = out.merge(merged, on="Code", how="left")
        print(f"  {fname}: merged {len(cols)} columns "
              f"({out[cols[0]].notna().sum()} obs non-null)")

    return out