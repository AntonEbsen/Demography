"""
build_dataset.py
================
Merges religion and vital registration data into the analysis panel.

Usage (from notebook):
    from src.data.build_dataset import build_analysis_panel
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from src.data.load_data import (
    load_rel1871,
    load_vit_panel,
    interpolate_population,
    load_pop1871_age_structure,
    DATA_RAW,
    DATA_PROCESSED,
)
from src.data.merge_ipehd import merge_ipehd_controls


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
                      cath_marriage_share, infant_mortality_rate,
                      gfr_static_1871
        Migration:    Inmigtot, Outmigtot, Outmigunoff,
                      inmig_rate, outmig_rate, net_mig_rate
        1871 census:  pop_*_1871, age_*_1871, women_15_49_1871,
                      women_share_15_49_1871
        Controls:     Poptot, ln_pop, plus iPEHD covariates
    """
    if data_dir is None:
        data_dir = DATA_RAW

    # ------------------------------------------------------------------
    # 1. Load religion data (cross-section)
    # ------------------------------------------------------------------
    rel = load_rel1871()
    logger.info("REL1871: %d counties loaded", len(rel))
    
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
    # 2b. Interpolate missing population (pre-1872 VIT files lack Poptot)
    # ------------------------------------------------------------------
    n_missing_before = vit["Poptot"].isna().sum()
    if n_missing_before > 0:
        logger.info("Poptot missing for %d obs — interpolating from POP census files...", n_missing_before)
        vit = interpolate_population(vit, data_dir=data_dir)
    
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
    logger.info("Merge: %d VIT counties -> %d matched with REL1871 (%d dropped)",
                n_before, n_after, n_before - n_after)
    
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

    # Migration rates (per 1,000), where Galloway records migration that year.
    # Coverage: 1862-1867 (totals only) and 1872-1886 (sex-summed). Years
    # 1868-1871 and 1887+ have no migration columns -> NaN propagates.
    panel["inmig_rate"] = np.where(
        panel["Inmigtot"].notna() & (panel["Poptot"] > 0),
        panel["Inmigtot"] / panel["Poptot"] * 1000.0,
        np.nan,
    )
    panel["outmig_rate"] = np.where(
        panel["Outmigtot"].notna() & (panel["Poptot"] > 0),
        panel["Outmigtot"] / panel["Poptot"] * 1000.0,
        np.nan,
    )
    panel["net_mig_rate"] = np.where(
        panel["Inmigtot"].notna()
        & panel["Outmigtot"].notna()
        & (panel["Poptot"] > 0),
        (panel["Inmigtot"] - panel["Outmigtot"]) / panel["Poptot"] * 1000.0,
        np.nan,
    )

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
    
    # Flag extreme birth rates (likely boundary-reform artifacts or data
    # errors) and nullify the derived rate columns so downstream regressions
    # — and the Pandera audit — drop them via NaN handling rather than
    # treating them as real values.
    panel["cbr_flag"] = (panel["cbr"] > 70) | (panel["cbr"] < 15)
    n_flagged = panel["cbr_flag"].sum()
    if n_flagged > 0:
        logger.warning("%d observations with extreme CBR (>70 or <15 per 1000); "
                       "rate columns set to NaN", n_flagged)
        rate_cols = ["cbr", "legitimate_br", "illegitimate_br",
                     "illegitimacy_ratio", "marriage_rate"]
        panel.loc[panel["cbr_flag"], rate_cols] = np.nan

    # ------------------------------------------------------------------
    # 6b. Merge iPEHD controls (cross-sectional, time-invariant).
    # The crosswalk currently covers ~88% of Type-0 counties; unmatched
    # counties keep NaN for these columns. Required for the IV strategy
    # using distance to Wittenberg and for additional iPEHD covariates.
    # ------------------------------------------------------------------
    try:
        panel = merge_ipehd_controls(panel)
    except FileNotFoundError as exc:
        logger.warning("iPEHD merge skipped (file not found): %s", exc)

    # ------------------------------------------------------------------
    # 6c. Merge POP1871 age x sex pyramid (time-invariant 1871 cross-section)
    # and construct the General Fertility Rate using women aged 15-49 in 1871
    # as a static denominator. This addresses the standard demographic
    # critique that CBR is mechanically affected by age structure.
    # ------------------------------------------------------------------
    try:
        age1871 = load_pop1871_age_structure()
        panel = panel.merge(age1871, on="Code", how="left")

        # Women of reproductive age, 1871 census (count and share of total pop)
        repro_age_cols = [
            "age_15_19_f_1871",
            "age_20_29_f_1871",
            "age_30_39_f_1871",
            "age_40_49_f_1871",
        ]
        if all(c in panel.columns for c in repro_age_cols):
            panel["women_15_49_1871"] = panel[repro_age_cols].sum(axis=1, min_count=1)
            panel["women_share_15_49_1871"] = np.where(
                panel["pop_total_1871"] > 0,
                panel["women_15_49_1871"] / panel["pop_total_1871"] * 100.0,
                np.nan,
            )

            # General Fertility Rate (per 1,000 women aged 15-49 in 1871).
            # Static denominator -> interpretable as "births per 1,000 women
            # of reproductive age, holding age structure at its 1871 level".
            panel["gfr_static_1871"] = np.where(
                panel["women_15_49_1871"].fillna(0) > 0,
                panel["Birtot"] / panel["women_15_49_1871"] * 1000.0,
                np.nan,
            )
            # Mirror the cbr_flag treatment: an extreme CBR signals a bad
            # Birtot or denominator that year, which equally contaminates GFR.
            if "cbr_flag" in panel.columns:
                panel.loc[panel["cbr_flag"].fillna(False), "gfr_static_1871"] = np.nan

            # GFR-specific flag: a static 1871 denominator paired with a
            # post-1873 boundary-reform Birtot can produce demographically
            # impossible rates (>1 birth per woman per year). Cap at a tight
            # historical-plausibility ceiling (400 per 1,000 women 15-49 ~
            # cohort TFR of 12, well above any pre-modern observation).
            panel["gfr_flag"] = (
                panel["gfr_static_1871"].notna() & (panel["gfr_static_1871"] > 400)
            )
            n_gfr_flagged = int(panel["gfr_flag"].sum())
            if n_gfr_flagged > 0:
                logger.warning(
                    "%d observations with extreme gfr_static_1871 (>400); set to NaN",
                    n_gfr_flagged,
                )
                panel.loc[panel["gfr_flag"], "gfr_static_1871"] = np.nan
            n_matched = panel["women_15_49_1871"].notna().sum()
            logger.info("POP1871 age structure merged: %d of %d obs matched", n_matched, len(panel))
        else:
            logger.warning("POP1871 age structure merge: expected columns missing -> skipping GFR")
    except FileNotFoundError as exc:
        logger.warning("POP1871 age-structure merge skipped (file not found): %s", exc)

    # Enforce panel-key uniqueness. Source files occasionally contain a
    # duplicate (Code, Year) — most often a mislabeled year in a city/county
    # split — which would silently double-weight that observation.
    n_before = len(panel)
    dup_mask = panel.duplicated(["Code", "Year"], keep=False)
    if dup_mask.any():
        dup_keys = (
            panel.loc[dup_mask, ["Code", "Kreis", "Year"]]
            .drop_duplicates()
            .to_dict("records")
        )
        logger.warning("Dropping duplicate (Code, Year) rows (keeping first): %s",
                       dup_keys)
        panel = panel.drop_duplicates(["Code", "Year"], keep="first")
        logger.warning("Dropped %d duplicate row(s)", n_before - len(panel))

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    panel = panel.sort_values(["Code", "Year"]).reset_index(drop=True)
    
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out_path = DATA_PROCESSED / "analysis_panel.parquet"
        panel.to_parquet(out_path, index=False)
        logger.info("Saved to %s", out_path)
    
    logger.info("Analysis panel: %d obs, %d counties, %d-%d",
                len(panel), panel['Code'].nunique(), panel['Year'].min(), panel['Year'].max())
    logger.info("High-Catholic counties (>50%%): %d", panel.groupby('Code')['high_cath'].first().sum())
    logger.info("Mean CBR: %.1f per 1,000", panel['cbr'].mean())

    return panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    build_analysis_panel()