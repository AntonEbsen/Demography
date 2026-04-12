"""
build_dataset.py
================
Merges religion and vital registration data into the analysis panel.

Usage (from notebook):
    from src.build_dataset import build_analysis_panel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.load_data import load_rel1871, load_vit_panel, DATA_RAW, DATA_PROCESSED


def build_analysis_panel(
    data_dir: Optional[Path] = None,
    year_start: int = 1862,
    year_end: int = 1890,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the main analysis dataset by merging REL1871 with the VIT panel.
    
    Steps
    -----
    1. Load REL1871 → county-level Catholic share (time-invariant treatment).
    2. Load VIT panel → county×year vital statistics.
    3. Merge on Code.
    4. Construct outcome variables (birth rates, marriage rates, etc.).
    5. Add Kulturkampf treatment indicators.
    
    Parameters
    ----------
    data_dir : Path
        Directory with raw data files.
    year_start, year_end : int
        Year range for the panel.
    save : bool
        If True, saves to data/processed/analysis_panel.parquet.
    
    Returns
    -------
    pd.DataFrame  with columns:
        Identifiers:  Code, Rb, Kreis, Year
        Treatment:    cath_share, high_cath, post_kulturkampf, treat_x_post
        Outcomes:     cbr, legitimate_br, illegitimate_br, marriage_rate,
                      cath_marriage_share, infant_mortality_rate
        Controls:     Poptot, ln_pop
    """
    if data_dir is None:
        data_dir = DATA_RAW

    # ------------------------------------------------------------------
    # 1. Load religion data (cross-section)
    # ------------------------------------------------------------------
    rel = load_rel1871(data_dir / "REL1871.XLS")
    print(f"REL1871: {len(rel)} counties loaded")
    
    # ------------------------------------------------------------------
    # 2. Load vital registration panel
    # ------------------------------------------------------------------
    vit = load_vit_panel(
        data_dir=data_dir,
        year_start=year_start,
        year_end=year_end,
        type_filter=0,
    )
    
    # ------------------------------------------------------------------
    # 3. Merge on Code
    # ------------------------------------------------------------------
    panel = vit.merge(
        rel[["Code", "cath_share", "prot_share"]],
        on="Code",
        how="inner",
    )
    
    n_before = vit["Code"].nunique()
    n_after = panel["Code"].nunique()
    print(f"Merge: {n_before} VIT counties → {n_after} matched with REL1871 "
          f"({n_before - n_after} dropped)")
    
    # ------------------------------------------------------------------
    # 4. Construct outcome variables
    # ------------------------------------------------------------------
    
    # Crude birth rate (per 1,000)
    panel["cbr"] = panel["Birtot"] / panel["Poptot"] * 1000
    
    # Legitimate birth rate (per 1,000)
    panel["legitimate_br"] = panel["Birlegtot"] / panel["Poptot"] * 1000
    
    # Illegitimate birth rate (per 1,000)
    panel["illegitimate_br"] = panel["Birbastot"] / panel["Poptot"] * 1000
    
    # Illegitimacy ratio (illegitimate / total births, %)
    panel["illegitimacy_ratio"] = panel["Birbastot"] / panel["Birtot"] * 100
    
    # Marriage rate (per 1,000)
    panel["marriage_rate"] = panel["Martot"] / panel["Poptot"] * 1000
    
    # Catholic marriage share (% of marriages that are Catholic)
    # Only available from 1875 onwards
    panel["cath_marriage_share"] = np.where(
        panel["Marcath"].notna() & (panel["Martot"] > 0),
        panel["Marcath"] / panel["Martot"] * 100,
        np.nan,
    )
    
    # Infant mortality rate (infant deaths / live births, per 1,000)
    panel["infant_mortality_rate"] = np.where(
        panel["Dth_infant_leg"].notna() & (panel["Birlegtot"] > 0),
        panel["Dth_infant_leg"] / panel["Birlegtot"] * 1000,
        np.nan,
    )
    
    # Log population
    panel["ln_pop"] = np.log(panel["Poptot"])
    
    # ------------------------------------------------------------------
    # 5. Treatment variables for the Kulturkampf DiD
    # ------------------------------------------------------------------
    
    # Post-Kulturkampf indicator
    # Main Kulturkampf legislation: 1872-1875
    # May Laws: May 1873 (key moment)
    # We define post = 1 for years >= 1873
    # (You can test sensitivity to this cutoff: 1872, 1874, 1875)
    panel["post_kulturkampf"] = (panel["Year"] >= 1873).astype(int)
    
    # High-Catholic indicator (binary treatment)
    # Counties with > 50% Catholic population
    panel["high_cath"] = (panel["cath_share"] > 50).astype(int)
    
    # Interaction term (standard DiD)
    panel["treat_x_post"] = panel["high_cath"] * panel["post_kulturkampf"]
    
    # Continuous treatment intensity version
    # Uses cath_share directly (more flexible)
    panel["cath_share_x_post"] = panel["cath_share"] * panel["post_kulturkampf"]
    
    # ------------------------------------------------------------------
    # 6. Clean up and handle problematic observations
    # ------------------------------------------------------------------
    
    # Drop observations where population is missing or zero
    panel = panel[panel["Poptot"].notna() & (panel["Poptot"] > 0)].copy()
    
    # Flag extreme birth rates (likely data errors)
    panel["cbr_flag"] = (panel["cbr"] > 70) | (panel["cbr"] < 15)
    n_flagged = panel["cbr_flag"].sum()
    if n_flagged > 0:
        print(f"Warning: {n_flagged} observations with extreme CBR (>70 or <15 per 1000)")
    
    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    panel = panel.sort_values(["Code", "Year"]).reset_index(drop=True)
    
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out_path = DATA_PROCESSED / "analysis_panel.parquet"
        panel.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Analysis panel: {len(panel)} obs, "
          f"{panel['Code'].nunique()} counties, "
          f"{panel['Year'].min()}-{panel['Year'].max()}")
    print(f"High-Catholic counties (>50%): {panel.groupby('Code')['high_cath'].first().sum()}")
    print(f"Low-Catholic counties (≤50%):  {panel.groupby('Code')['high_cath'].first().apply(lambda x: 1-x).sum()}")
    print(f"Mean CBR: {panel['cbr'].mean():.1f} per 1,000")
    print(f"{'='*50}")
    
    return panel
