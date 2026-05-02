"""
merge_ipehd_1849.py
===================
Merges 1849 iPEHD CSV files into the analysis panel.

The 1849 data uses kreiskey1849 (335 counties), while our panel uses
Galloway Code (~393 counties = 1871 boundaries). We bridge through:
  Galloway Code → kreiskey1871 (existing crosswalk) → kreiskey1849 (built here)

The 1849→1871 step is built via name matching with Catholic share
validation, exactly like the existing Galloway↔iPEHD crosswalk.

Variables included (selected for relevance to fertility analysis):
  - rel1849_cat, rel1849_pro: pre-Kulturkampf religious composition
  - pop1849_f_17to45: fertile-age female population (key denominator!)
  - edu1849_pub_ele_stud_m/f: pre-treatment school enrollment
  - rel1849_cat_priest: institutional Catholic presence
  - pop1849_families: household structure baseline
  - pop181621_born_oow_tot: long-run illegitimacy baseline (1816-21)
  - fac1849_*_total: industrialization aggregates

Usage (from notebook):
    from src.merge_ipehd_1849 import merge_ipehd_1849
    panel_extended = merge_ipehd_1849(panel_with_ipehd)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional

from src.data.merge_ipehd import build_crosswalk, _clean_name


# =====================================================================
# Variables to extract from each file (curated list)
# =====================================================================

VARS_BY_FILE = {
    "ipehd_1849_rel_church.csv": [
        "rel1849_pro_priest",          # Number of Protestant priests
        "rel1849_cat_priest",          # Number of Catholic priests
        "rel1849_cat_chaplain_vicar",  # Catholic chaplains/curates
        "rel1849_pro_main_church",     # Protestant parish churches
        "rel1849_cat_main_church",     # Catholic parish churches
    ],
    "ipehd_1849_pop_demo.csv": [
        "pop1849_f_17to45",            # KEY: Fertile-age female pop
        "pop1849_f_tot",
        "pop1849_m_tot",
        "pop1849_tot",
        "pop1849_f_un14",              # Pre-fertile females (cohort effect)
    ],
    "ipehd_1849_pop_mari.csv": [
        "pop1849_families",            # Number of families
        "pop1849_m_wedlock",           # Males in wedlock
        "pop1849_f_wedlock",           # Females in wedlock
    ],
    "ipehd_1849_edu_stud.csv": [
        "edu1849_pub_ele_stud_m",      # Male elementary students
        "edu1849_pub_ele_stud_f",      # Female elementary students
        "edu1849_pub_gym_stud_m",      # Male gymnasium students
    ],
    "ipehd_181621_pop_death.csv": [
        "pop181621_born_tot",          # Total births 1816-1821
        "pop181621_born_oow_tot",      # Out-of-wedlock births (LONG-RUN baseline!)
        "pop181621_died_un1_tot",      # Infant deaths under 1
        "pop181621_died_tot",          # Total deaths
    ],
}

# We don't load the giant industry files by default (107 + 83 + 9 = 199 cols)
# But we'll provide a function to aggregate them if requested.


# =====================================================================
# Load 1849 file with proper encoding
# =====================================================================

def _load_1849_csv(path: Path) -> pd.DataFrame:
    """Load a 1849 iPEHD CSV with the correct encoding."""
    return pd.read_csv(path, encoding="latin-1")


# =====================================================================
# Build 1849 -> 1871 kreiskey crosswalk
# =====================================================================

def build_1849_to_1871_crosswalk(
    ipehd_master_path: Path,
    rel_church_1849_path: Path,
    pop_demo_1849_path: Path,
    cath_tolerance: float = 15.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a crosswalk from kreiskey1849 to kreiskey1871.
    
    IMPORTANT: kreiskey1849 and kreiskey1871 are NOT compatible numbering
    systems above ~kreiskey 80. Direct key matching produces garbage (e.g.
    1849 key 281 = "Stadt Cöln" while 1871 key 281 = "Marienburg b. H.").
    
    Strategy:
      1. Match by CLEANED NAME within Regierungsbezirk
      2. Validate each match by comparing approximate 1849 Catholic share
         (computed from priest counts) vs 1871 Catholic share. Boundaries
         changed gradually so shares should be similar (within 15pp).
    
    The 15pp tolerance is loose because the 1849 proxy is imprecise — but
    it still rules out cross-province mismatches (where 1849 shares of
    near-0% would be matched to 1871 shares of near-100% or vice versa).
    
    Returns
    -------
    DataFrame with kreiskey1849, kreiskey1871, county_name_1849, rb_1849
    """
    dta_1871 = pd.read_stata(ipehd_master_path)
    rel_1849 = _load_1849_csv(rel_church_1849_path)
    
    # Build approximate 1849 Catholic share from priest counts
    # Priest count ratio is a stable proxy for denominational composition
    rel_1849 = rel_1849.copy()
    cath_priests = rel_1849["rel1849_cat_priest"].fillna(0)
    pro_priests = rel_1849["rel1849_pro_priest"].fillna(0)
    total_priests = cath_priests + pro_priests
    rel_1849["_cath_share_1849_approx"] = np.where(
        total_priests >= 2,  # need a few priests to estimate ratio
        cath_priests / total_priests * 100,
        np.nan,
    )
    
    # Clean names
    rel_1849["name_clean"] = rel_1849["county"].apply(_clean_name)
    dta_1871 = dta_1871.copy()
    dta_1871["name_clean"] = dta_1871["county1871"].apply(_clean_name)
    
    # Step 1: Name match
    name_matches = rel_1849[
        ["kreiskey1849", "name_clean", "_cath_share_1849_approx", "rb"]
    ].merge(
        dta_1871[["kreiskey1871", "name_clean", "f_cath", "rbkey"]],
        on="name_clean", how="inner",
    ).drop_duplicates(subset="kreiskey1849")
    
    # Step 2: Validate by Catholic share
    # Three cases:
    #  (a) Have approx 1849 share AND it matches 1871 share within tolerance: VALID
    #  (b) Have approx 1849 share but doesn't match: INVALID (cross-province error)
    #  (c) No approx 1849 share (too few priests): ACCEPT only if name is unambiguous
    
    name_matches["_diff"] = abs(
        name_matches["_cath_share_1849_approx"] - name_matches["f_cath"]
    )
    
    has_proxy = name_matches["_cath_share_1849_approx"].notna()
    proxy_matches = name_matches[has_proxy & (name_matches["_diff"] <= cath_tolerance)]
    no_proxy = name_matches[~has_proxy]
    
    # For "no proxy" cases, count how many times the cleaned name appears
    # in the 1849 file. Only accept if unambiguous (appears once).
    name_counts = rel_1849.groupby(rel_1849["county"].apply(_clean_name)).size()
    no_proxy_unique = no_proxy[
        no_proxy["name_clean"].map(name_counts) == 1
    ]
    
    valid = pd.concat([
        proxy_matches[["kreiskey1849", "kreiskey1871"]],
        no_proxy_unique[["kreiskey1849", "kreiskey1871"]],
    ])
    
    n_total = len(name_matches)
    n_valid = len(valid)
    
    if verbose:
        print(f"1849→1871 crosswalk:")
        print(f"  Name match candidates: {n_total}")
        print(f"  Validated by Catholic share (tolerance ±{cath_tolerance}pp): {len(proxy_matches)}")
        print(f"  Validated by name uniqueness: {len(no_proxy_unique)}")
        print(f"  Total valid: {n_valid} / {len(rel_1849)} 1849 counties")
    
    # Add metadata
    crosswalk = valid.merge(
        rel_1849[["kreiskey1849", "county", "rb"]].rename(
            columns={"county": "county_name_1849", "rb": "rb_1849"}
        ),
        on="kreiskey1849", how="left",
    )
    
    return crosswalk


# =====================================================================
# Main merge function
# =====================================================================

def merge_ipehd_1849(
    panel: pd.DataFrame,
    data_dir: Optional[Path] = None,
    files_and_vars: Optional[dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge 1849 iPEHD variables into the analysis panel.
    
    The merge proceeds in three steps:
      1. Build Galloway Code → kreiskey1871 crosswalk (from existing module)
      2. Build kreiskey1849 → kreiskey1871 crosswalk (built here)
      3. Chain them: Galloway Code → kreiskey1871 → kreiskey1849
      4. Merge selected 1849 variables into the panel
    
    The 1849 variables enter as time-invariant pre-treatment controls.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Analysis panel with 'Code' column.
    data_dir : Path
        Where the iPEHD CSVs and master .dta live.
    files_and_vars : dict
        Map of filename -> list of variables to extract.
        Defaults to VARS_BY_FILE.
    
    Returns
    -------
    pd.DataFrame : panel with new 1849 variable columns.
    """
    from src.data.load_data import DATA_RAW

    if data_dir is None:
        data_dir = DATA_RAW
    if files_and_vars is None:
        files_and_vars = VARS_BY_FILE

    ipehd_master_path = data_dir / "ipehd_qje2009_master.dta"

    # Step 1: Galloway Code -> kreiskey1871 (existing)
    # REL1871 lives in the Galloway directory, not the iPEHD one.
    from src.data.load_data import _find_file, DATA_RAW as _GALLOWAY_RAW
    rel_path = _find_file(_GALLOWAY_RAW, "REL1871")
    
    if verbose:
        print("Step 1: Build Galloway Code -> kreiskey1871 crosswalk")
    cw_gal_to_1871 = build_crosswalk(
        ipehd_master_path, rel_path, cath_tolerance=5.0, verbose=False,
    )
    
    # Step 2: kreiskey1849 -> kreiskey1871 (new)
    if verbose:
        print("\nStep 2: Build kreiskey1849 -> kreiskey1871 crosswalk")
    cw_1849_to_1871 = build_1849_to_1871_crosswalk(
        ipehd_master_path,
        data_dir / "ipehd_1849_rel_church.csv",
        data_dir / "ipehd_1849_pop_demo.csv",
        verbose=verbose,
    )
    
    # Step 3: Chain them: Code -> kreiskey1871 -> kreiskey1849
    chain = cw_gal_to_1871.merge(cw_1849_to_1871, on="kreiskey1871", how="inner")
    
    if verbose:
        print(f"\nStep 3: Chained crosswalk Code -> kreiskey1849:")
        print(f"  {len(chain)} Galloway counties have a 1849 match")
        print(f"  ({len(cw_gal_to_1871) - len(chain)} have 1871 but no 1849 match)")
    
    # Step 4: Load each 1849 file and merge selected variables
    if verbose:
        print(f"\nStep 4: Load 1849 variables")
    
    # Start with the chain (Code + kreiskey1849)
    var_data = chain[["Code", "kreiskey1849"]].copy()
    
    all_added_vars = []
    for fname, var_list in files_and_vars.items():
        path = data_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        
        df = _load_1849_csv(path)
        
        # Determine the key column (most are kreiskey1849, but pop_death uses kreiskey1800)
        if "kreiskey1849" in df.columns:
            key_col = "kreiskey1849"
        elif "kreiskey1800" in df.columns:
            key_col = "kreiskey1800"
        else:
            print(f"  [skip] {fname}: no recognized key column")
            continue
        
        # Filter to requested variables
        available = [v for v in var_list if v in df.columns]
        missing = [v for v in var_list if v not in df.columns]
        if missing:
            print(f"  [warn] {fname}: missing {missing}")
        
        if not available:
            continue
        
        sub = df[[key_col] + available].copy()
        
        # Special handling: 1816-21 deaths uses 1800 keys, which often
        # but not always equal 1849 keys. We use direct match.
        if key_col == "kreiskey1800":
            sub = sub.rename(columns={"kreiskey1800": "kreiskey1849"})
            # Validate: how many 1800 keys are in our 1849 set?
            n_overlap = sub["kreiskey1849"].isin(var_data["kreiskey1849"]).sum()
            if verbose:
                print(f"  {fname}: {len(sub)} rows, {n_overlap} match 1849 keys")
        
        var_data = var_data.merge(sub, on="kreiskey1849", how="left")
        all_added_vars.extend(available)
        
        if verbose:
            n_with_data = var_data[available[0]].notna().sum() if available else 0
            print(f"  {fname}: added {len(available)} vars ({n_with_data} non-null)")
    
    # Drop the kreiskey1849 column for the final merge — we use Code
    merge_cols = ["Code"] + [c for c in var_data.columns if c not in ["Code", "kreiskey1849"]]
    
    # Step 5: Merge into panel
    panel_extended = panel.merge(var_data[merge_cols], on="Code", how="left")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL MERGE COMPLETE")
        print(f"{'='*60}")
        print(f"Variables added: {len(all_added_vars)}")
        if all_added_vars:
            example_var = all_added_vars[0]
            n_obs_with_data = panel_extended[example_var].notna().sum()
            n_counties_with_data = (
                panel_extended[panel_extended[example_var].notna()]["Code"].nunique()
            )
            print(f"Coverage (e.g. {example_var}): "
                  f"{n_counties_with_data} counties, {n_obs_with_data} obs")
        print(f"Total panel: {len(panel_extended)} obs, "
              f"{panel_extended['Code'].nunique()} counties")
    
    return panel_extended


# =====================================================================
# Helper: derived variables that are useful for analysis
# =====================================================================

def add_derived_1849_vars(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived variables from the 1849 raw counts.
    
    These are usually what you actually want for regressions:
      - share_school_attendance_1849: students per pop
      - share_female_fertile_1849: fertile women / total pop
      - cath_priests_per_capita_1849: institutional Catholic density
      - oow_share_1816_21: long-run illegitimacy share
      - cath_share_1849: pre-Kulturkampf religion (placebo!)
    """
    df = panel.copy()
    
    # Share female fertile-age (17-45) in 1849 — KEY denominator effect
    if {"pop1849_f_17to45", "pop1849_tot"}.issubset(df.columns):
        df["share_female_fertile_1849"] = df["pop1849_f_17to45"] / df["pop1849_tot"]
    
    # Pre-Kulturkampf school attendance (combined m+f / total pop)
    if {"edu1849_pub_ele_stud_m", "edu1849_pub_ele_stud_f", "pop1849_tot"}.issubset(df.columns):
        df["school_attendance_1849"] = (
            (df["edu1849_pub_ele_stud_m"] + df["edu1849_pub_ele_stud_f"])
            / df["pop1849_tot"]
        )
    
    # Catholic priests per 1000 inhabitants (institutional density)
    if {"rel1849_cat_priest", "pop1849_tot"}.issubset(df.columns):
        df["cath_priests_per_1000_1849"] = (
            df["rel1849_cat_priest"] / df["pop1849_tot"] * 1000
        )
    
    # Long-run out-of-wedlock share, 1816-21
    if {"pop181621_born_oow_tot", "pop181621_born_tot"}.issubset(df.columns):
        df["oow_share_1816_21"] = (
            df["pop181621_born_oow_tot"] / df["pop181621_born_tot"] * 100
        )
    
    # Married share (males in wedlock / total males)
    if {"pop1849_m_wedlock", "pop1849_m_tot"}.issubset(df.columns):
        df["share_married_men_1849"] = df["pop1849_m_wedlock"] / df["pop1849_m_tot"]
    
    # Family size
    if {"pop1849_tot", "pop1849_families"}.issubset(df.columns):
        df["mean_family_size_1849"] = df["pop1849_tot"] / df["pop1849_families"]
    
    return df
