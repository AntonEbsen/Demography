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
    compute_midyear_population,
    load_pop1871_age_structure,
    load_ele1871,
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
    pd.DataFrame with columns. Headline rate variables (`cbr`,
    `legitimate_br`, `illegitimate_br`, `marriage_rate`, `inmig_rate`,
    `outmig_rate`, `net_mig_rate`) use mid-year population
    (`Poptot_midyear`) as the denominator -- the standard demographic
    convention, matching Galloway, Hammel & Lee (1994).
    Carry-forward variants under a `_carryforward` suffix use the raw
    Galloway `Poptot` and are reported as a robustness row only.

        Identifiers:  Code, Rb, Kreis, Year
        Treatment:    cath_share, high_cath, post_kulturkampf, treat_x_post
        Outcomes:     cbr, legitimate_br, illegitimate_br, marriage_rate,
                      cath_marriage_share, illegitimacy_ratio,
                      infant_mortality_rate (TOTAL IMR, headline -- 1875+
                      only by construction; Galloway 1994 convention),
                      infant_mortality_rate_leg (legitimate-only,
                      diagnostic variable used by fig_imr_break.png to
                      show the 1875 measurement break -- not analytical)
        Coale/Galloway: I_f (general fertility), I_g (marital fertility,
                       Hutterite-normalised; analogue of Galloway's GMFR),
                       I_h (illegitimate fertility), gmfr (legitimate
                       births per 1,000 married women 15-49, the
                       unnormalised Galloway-tradition GMFR)
        Deprecated:   gfr_static_1871 (kept for back-compat; superseded
                      by I_g for marital-fertility analysis)
        Migration:    Inmigtot, Outmigtot, Outmigunoff,
                      inmig_rate, outmig_rate, net_mig_rate
        Mid-year pop: Poptot_midyear (used as the denominator above)
        Carry-fwd:    cbr_carryforward, legitimate_br_carryforward,
                      illegitimate_br_carryforward,
                      marriage_rate_carryforward,
                      inmig_rate_carryforward, outmig_rate_carryforward,
                      net_mig_rate_carryforward
        1871 census:  pop_*_1871, age_*_1871, women_15_49_1871,
                      women_share_15_49_1871
        1871 election: zentrum_share_1871, polen_share_1871,
                       catholic_party_share_1871, conservative_share_1871,
                       liberal_share_1871, nat_liberal_share_1871,
                       sozialdemokrat_share_1871
        Controls:     ln_pop (= log Poptot_midyear), plus iPEHD covariates;
                      raw Galloway `Poptot` is also retained as a column
                      for users who want to recompute carry-forward rates
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
    # 4. Mid-year population (proper demographic CBR convention).
    # Galloway's raw `Poptot` is the *previous* December census carried
    # forward in inter-census years; mid-year (July 1) population is the
    # standard CBR denominator (an approximation to person-years lived).
    # We construct `Poptot_midyear` first and use it as the *primary*
    # denominator for cbr / legitimate_br / illegitimate_br /
    # marriage_rate / migration rates. The original carry-forward
    # versions are retained under a `_carryforward` suffix for the
    # robustness row in the headline DiD table. See
    # compute_midyear_population() docstring.
    # ------------------------------------------------------------------
    panel = compute_midyear_population(panel, data_dir=data_dir)
    pop_my = panel["Poptot_midyear"]

    # Headline rates: mid-year denominator (standard demographic convention).
    panel["cbr"] = np.where(
        pop_my.notna() & (pop_my > 0),
        panel["Birtot"] / pop_my * 1000.0, np.nan,
    )
    panel["legitimate_br"] = np.where(
        pop_my.notna() & (pop_my > 0),
        panel["Birlegtot"] / pop_my * 1000.0, np.nan,
    )
    panel["illegitimate_br"] = np.where(
        pop_my.notna() & (pop_my > 0),
        panel["Birbastot"] / pop_my * 1000.0, np.nan,
    )
    panel["marriage_rate"] = np.where(
        pop_my.notna() & (pop_my > 0),
        panel["Martot"] / pop_my * 1000.0, np.nan,
    )

    # Illegitimacy ratio: birth-only denominator -- unaffected by Poptot.
    panel["illegitimacy_ratio"] = panel["Birbastot"] / panel["Birtot"] * 100

    # Catholic marriage share (% of marriages that are Catholic).
    # Only available from 1875 onwards.
    panel["cath_marriage_share"] = np.where(
        panel["Marcath"].notna() & (panel["Martot"] > 0),
        panel["Marcath"] / panel["Martot"] * 100,
        np.nan,
    )

    # Headline infant mortality rate: TOTAL infant deaths per 1,000
    # TOTAL live births -- the standard demographic definition (HMD,
    # WHO, Princeton EFP, Galloway, Hammel & Lee 1994 convention).
    # Well-defined only from 1875 onwards because Galloway's
    # illegitimate-infant-death column Dth<1bas does not appear earlier;
    # pre-1875 this is NaN by construction. Channels.py
    # infant_mortality_analysis already restricts to 1875+ so this is
    # consistent with the existing analysis.
    total_imr_denom = (panel["Birlegtot"].fillna(0) + panel["Birbastot"].fillna(0))
    total_imr_num = (
        panel["Dth_infant_leg"].fillna(0) + panel["Dth_infant_bas"].fillna(0)
    )
    panel["infant_mortality_rate"] = np.where(
        panel["Dth_infant_bas"].notna()
        & panel["Dth_infant_leg"].notna()
        & (total_imr_denom > 0),
        total_imr_num / total_imr_denom * 1000.0,
        np.nan,
    )

    # Legitimate-only infant mortality rate, retained as a diagnostic
    # variable used by fig_imr_break.png to document the Galloway 1875
    # data-definition change (Dthyoung -> Dth<1leg). Pre-1875 falls
    # back to Dthyoung, which produces the ~3-4x level discontinuity at
    # 1875 visible in the figure. NOT recommended as an analytical
    # outcome -- the headline `infant_mortality_rate` above is the
    # standard demographic measure.
    panel["infant_mortality_rate_leg"] = np.where(
        panel["Dth_infant_leg"].notna() & (panel["Birlegtot"] > 0),
        panel["Dth_infant_leg"] / panel["Birlegtot"] * 1000,
        np.nan,
    )

    # Log mid-year population: standard control for county-size effects,
    # using the same mid-year-interpolated denominator as the headline
    # rate variables (avoids mixing carry-forward and mid-year
    # conventions inside the same regression).
    panel["ln_pop"] = np.log(panel["Poptot_midyear"])

    # Headline migration rates: mid-year denominator (per 1,000 pop).
    # Coverage: 1862-1867 (totals only) and 1872-1886 (sex-summed); years
    # 1868-1871 and 1887+ have no Galloway migration columns -> NaN.
    panel["inmig_rate"] = np.where(
        panel["Inmigtot"].notna() & pop_my.notna() & (pop_my > 0),
        panel["Inmigtot"] / pop_my * 1000.0, np.nan,
    )
    panel["outmig_rate"] = np.where(
        panel["Outmigtot"].notna() & pop_my.notna() & (pop_my > 0),
        panel["Outmigtot"] / pop_my * 1000.0, np.nan,
    )
    panel["net_mig_rate"] = np.where(
        panel["Inmigtot"].notna() & panel["Outmigtot"].notna()
        & pop_my.notna() & (pop_my > 0),
        (panel["Inmigtot"] - panel["Outmigtot"]) / pop_my * 1000.0, np.nan,
    )

    # ------------------------------------------------------------------
    # 4b. Carry-forward (Galloway raw) variants for the robustness row in
    # the headline DiD table. These use the unmodified Galloway `Poptot`
    # (= previous December census carried forward) as denominator and are
    # what one gets by using the Galloway database "out of the box".
    # ------------------------------------------------------------------
    panel["cbr_carryforward"] = panel["Birtot"] / panel["Poptot"] * 1000
    panel["legitimate_br_carryforward"] = panel["Birlegtot"] / panel["Poptot"] * 1000
    panel["illegitimate_br_carryforward"] = panel["Birbastot"] / panel["Poptot"] * 1000
    panel["marriage_rate_carryforward"] = panel["Martot"] / panel["Poptot"] * 1000
    panel["inmig_rate_carryforward"] = np.where(
        panel["Inmigtot"].notna() & (panel["Poptot"] > 0),
        panel["Inmigtot"] / panel["Poptot"] * 1000.0, np.nan,
    )
    panel["outmig_rate_carryforward"] = np.where(
        panel["Outmigtot"].notna() & (panel["Poptot"] > 0),
        panel["Outmigtot"] / panel["Poptot"] * 1000.0, np.nan,
    )
    panel["net_mig_rate_carryforward"] = np.where(
        panel["Inmigtot"].notna() & panel["Outmigtot"].notna()
        & (panel["Poptot"] > 0),
        (panel["Inmigtot"] - panel["Outmigtot"]) / panel["Poptot"] * 1000.0, np.nan,
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
    # Flag extreme CBR under *either* the headline mid-year convention
    # or the Galloway carry-forward convention. The mid-year convention
    # can produce demographically impossible values when the 1871-anchored
    # interpolated denominator does not match the post-1872 boundary-reform
    # Birtot (e.g. Tarnowitz 1872, where 1873 county splits mean the 1872
    # numerator covers a different geography than the 1871/1875 census
    # anchors). One unified flag keeps both rate series consistent with
    # the same set of "good" rows.
    panel["cbr_flag"] = (
        (panel["cbr"] > 70) | (panel["cbr"] < 15)
        | (panel["cbr_carryforward"].fillna(panel["cbr"]) > 70)
        | (panel["cbr_carryforward"].fillna(panel["cbr"]) < 15)
    )
    n_flagged = panel["cbr_flag"].sum()
    if n_flagged > 0:
        logger.warning("%d observations with extreme CBR (>70 or <15 per 1000); "
                       "rate columns set to NaN", n_flagged)
        rate_cols = ["cbr", "legitimate_br", "illegitimate_br",
                     "illegitimacy_ratio", "marriage_rate"]
        panel.loc[panel["cbr_flag"], rate_cols] = np.nan
        # Mirror the null-out for the Galloway carry-forward variants
        # (used in the robustness row): if Birtot is implausible the
        # carry-forward ratio is equally contaminated.
        carryforward_rate_cols = [
            "cbr_carryforward", "legitimate_br_carryforward",
            "illegitimate_br_carryforward", "marriage_rate_carryforward",
        ]
        for col in carryforward_rate_cols:
            if col in panel.columns:
                panel.loc[panel["cbr_flag"], col] = np.nan

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

    # ------------------------------------------------------------------
    # 6c-bis. Merge 1871 Reichstag election vote shares from ELE1871.
    # Galloway publishes vote totals at the Wahlkreis (electoral
    # district) level; load_ele1871 parses Wahlkreis names to recover
    # the constituent Kreise and assigns the Wahlkreis vote shares to
    # each. Coverage ~85% of Type-0 panel Kreise; unmatched rows
    # retain NaN. Zentrum (Catholic Centre Party) and Polen (Polish
    # nationalist Catholic party) shares are the new variables of
    # interest -- direct political-economy measures of Catholic
    # affiliation, used as (a) heterogeneity moderators, (b) an
    # alternative instrument for cath_share, and (c) cross-validation
    # of the religious-census measure.
    # ------------------------------------------------------------------
    try:
        ele1871 = load_ele1871(
            panel_kreise=panel[["Code", "Kreis", "Rb"]].drop_duplicates(["Code"]),
        )
        panel = panel.merge(ele1871, on="Code", how="left")
        logger.info(
            "ELE1871 vote shares merged: %d of %d Kreise have Zentrum-share data",
            int(panel["zentrum_share_1871"].notna().sum() // (panel["Year"].nunique())),
            panel["Code"].nunique(),
        )
    except FileNotFoundError as exc:
        logger.warning("ELE1871 merge skipped (file not found): %s", exc)

    # ------------------------------------------------------------------
    # 6d. Princeton EFP Coale indices (I_f, I_g, I_h) and the Galloway-
    # tradition GMFR (legitimate births per 1,000 married women aged
    # 15-49). I_g is the central marital-fertility outcome in Galloway,
    # Hammel & Lee (1994); we report it alongside CBR throughout the
    # paper. See coale_indices.py docstring for the approximation
    # assumptions (Hutterite ASFR, Coale-Demeny "West" age distribution,
    # county-specific 1871 women-15-49 share scaled to mid-year pop).
    # ------------------------------------------------------------------
    try:
        from src.analysis.coale_indices import compute_coale_indices
        panel = compute_coale_indices(
            panel,
            pop_col="Poptot_midyear",
            use_county_specific_share=True,
        )
        # Mirror the cbr_flag null-out: the Coale indices share Birtot /
        # Birlegtot / Birbastot in their numerators, so an extreme CBR
        # signals contaminated index values for the same row.
        if "cbr_flag" in panel.columns:
            for col in ("I_f", "I_g", "I_h", "gmfr"):
                if col in panel.columns:
                    panel.loc[panel["cbr_flag"].fillna(False), col] = np.nan
        # Light flag for demographically implausible I_g (>1.0 means
        # marital fertility above the Hutterite maximum -- a measurement
        # artefact). Also flag I_g > 1 from boundary-reform residuals.
        n_ig_extreme = int((panel["I_g"] > 1.2).sum())
        if n_ig_extreme > 0:
            logger.warning("%d obs with I_g > 1.2 (Hutterite max); set to NaN", n_ig_extreme)
            panel.loc[panel["I_g"] > 1.2, ["I_g", "gmfr"]] = np.nan
        logger.info("Coale indices computed: I_f, I_g, I_h, gmfr")
    except Exception as exc:
        logger.warning("Coale-indices computation skipped: %s", exc)

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