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
    "f_litrate",       # Literacy rate
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
    594: 375,   # OBERTAUNUSKREIS → Mainkreis (Wiesbaden)
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
    s = re.sub(r'\(.*?\)', '', s)
    for prefix in ["STADT ", "LANDKREIS ", "KREIS ", "STADTKREIS ",
                    "OBERAMTSBEZIRK ", "PR. ", "DT. "]:
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
        print(f"Step 1 — Direct code match: {len(direct_raw)} candidates → "
              f"{len(direct)} valid (dropped {bad_direct})")
    
    # Step 2: Manual crosswalk (then validate)
    manual_raw = pd.DataFrame([
        {'Code': k, 'kreiskey1871': v}
        for k, v in _MANUAL_CROSSWALK.items()
    ])
    manual_raw = manual_raw[~manual_raw['Code'].isin(direct['Code'])]
    manual, bad_manual = _validate(manual_raw)
    if verbose:
        print(f"Step 2 — Manual crosswalk: {len(manual_raw)} candidates → "
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
        print(f"Step 3 — Name match: {len(name_raw)} candidates → "
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
            print(f"  ✓ Crosswalk validated — religion shares match closely.")
        else:
            print(f"  ⚠ Low correlation — check crosswalk for errors!")
    
    return panel_merged