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
    load_election_panel,
    ELECTION_YEARS,
    load_urb_panel,
    URB_YEARS,
    load_bir1871,
    load_tax1876,
    load_agr1882,
    load_gel1882,
    load_edu1886,
    load_sta1871,
    load_pop1885_marital_status,
    load_age1882,
    load_age1890,
    _find_file,
    DATA_RAW,
    DATA_PROCESSED,
)
from src.data.merge_ipehd import (
    merge_ipehd_controls,
    build_crosswalk_1849,
    merge_ipehd_1849,
)


def build_analysis_panel(
    data_dir: Optional[Path] = None,
    year_start: int = 1862,
    year_end: int = 1910,
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
        Election TV:   zentrum_share_current, polen_share_current,
                       catholic_party_share_current (carry-forward from
                       Reichstag elections 1871-1890)
        Urban TV:      urban_share_current (linearly interpolated from
                       URB1875/80/85/90; NaN pre-1875)
        1886 schooling: school_age_pop_1886, attend_public_1886,
                        attend_private_1886, attend_rate_1886,
                        teachers_1886, teacher_income_1886,
                        pupils_per_teacher_1886 (EDU1886 cross-section,
                        used by channels.schooling_channel() for the
                        1849->1886 long-difference DiD on attendance
                        rates; time-invariant after merge)
        Controls:     iPEHD baseline covariates (time-invariant);
                      `ln_pop = log Poptot_midyear` is built and retained
                      for descriptive use but is NOT in the default
                      regression controls -- entity FE absorb time-
                      invariant size differences, and population is
                      itself responsive to the Kulturkampf;
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
        panel_kreise_ref = panel[["Code", "Kreis", "Rb"]].drop_duplicates(["Code"])
        ele1871 = load_ele1871(panel_kreise=panel_kreise_ref)
        panel = panel.merge(ele1871, on="Code", how="left")
        logger.info(
            "ELE1871 vote shares merged: %d of %d Kreise have Zentrum-share data",
            int(panel["zentrum_share_1871"].notna().sum() // (panel["Year"].nunique())),
            panel["Code"].nunique(),
        )

        # Time-varying Zentrum / Polen / Catholic-party vote shares.
        # Galloway publishes Reichstag results in 1871, 1874, 1878,
        # 1881, 1884, 1887, 1890 -- one pre-treatment election (1871)
        # and six post-treatment elections covering enforcement
        # (1874, 1878) and rollback (1881, 1884, 1887, 1890).
        #
        # We construct three time-varying columns in the panel:
        #   zentrum_share_current, polen_share_current,
        #   catholic_party_share_current
        # Each takes the most-recent-election vote share at panel year
        # t (i.e. carry-forward from each election until the next one).
        # Panel years before 1871 inherit the 1871 result; panel years
        # 1890 inherit the 1890 result.
        ele_long = load_election_panel(panel_kreise=panel_kreise_ref)
        if len(ele_long) > 0:
            # Pivot to wide: one row per Kreis, columns per election year.
            ele_wide = ele_long.pivot(
                index="Code", columns="election_year",
                values=["zentrum_share", "polen_share", "catholic_party_share"],
            )
            ele_wide.columns = [f"{a}__{int(b)}" for a, b in ele_wide.columns]
            ele_wide = ele_wide.reset_index()
            panel = panel.merge(ele_wide, on="Code", how="left")

            # Build carry-forward time-varying columns.
            election_yrs = sorted(ele_long["election_year"].unique())
            for share_col in ("zentrum_share", "polen_share", "catholic_party_share"):
                cur = pd.Series(np.nan, index=panel.index, dtype=float)
                for ey in election_yrs:
                    src_col = f"{share_col}__{int(ey)}"
                    if src_col not in panel.columns:
                        continue
                    # Apply this election's share to all panel rows whose
                    # Year >= this election_year (overwrites earlier ones).
                    mask = panel["Year"] >= ey
                    cur.loc[mask] = panel.loc[mask, src_col].values
                # For panel years before the earliest election, use the
                # earliest election's share as a backfill (1862-1870 -> 1871).
                first_ey = election_yrs[0]
                first_src = f"{share_col}__{int(first_ey)}"
                if first_src in panel.columns:
                    pre_mask = panel["Year"] < first_ey
                    cur.loc[pre_mask] = panel.loc[pre_mask, first_src].values
                panel[f"{share_col}_current"] = cur.values

            # Drop the per-election wide columns -- they were intermediate.
            panel = panel.drop(
                columns=[c for c in panel.columns if "__" in c and any(
                    c.startswith(s) for s in ("zentrum_share", "polen_share",
                                              "catholic_party_share"))]
            )
            logger.info(
                "Time-varying election shares merged: %d election years (%s)",
                len(election_yrs), election_yrs,
            )
    except FileNotFoundError as exc:
        logger.warning("ELE merge skipped (file not found): %s", exc)

    # ------------------------------------------------------------------
    # 6c-ter. Merge time-varying urban share from URB1875/80/85/90.
    # Galloway publishes Kreis-level Percenturban at four cross-
    # sections during the analysis window. We interpolate linearly
    # between anchors to produce annual urban_share_current values for
    # panel years 1875-1890. Pre-1875 the variable is NaN (no Galloway
    # urban measurement is available before 1875; iPEHD's f_urban
    # remains as a separate static 1871 cross-section). This enables
    # the Bai/Hsiao time-varying-urbanisation trend spec.
    # ------------------------------------------------------------------
    try:
        urb_long = load_urb_panel()
        if len(urb_long) > 0:
            urb_anchors = (
                urb_long.set_index(["Code", "Year"])["percenturban"]
                .unstack("Year")
            )

            def _interp_row(row: pd.Series) -> pd.Series:
                # Linear interpolation between URB anchors (1875, 1880,
                # 1885, 1890); endpoints carried forward outside the
                # observed range, but we only emit for years within the
                # closed [min, max] anchor span.
                anchor_years = [y for y in row.index if pd.notna(row[y])]
                if not anchor_years:
                    return pd.Series(dtype=float)
                xs = np.asarray(anchor_years, dtype=float)
                ys = np.asarray([row[y] for y in anchor_years], dtype=float)
                target_years = list(range(int(min(anchor_years)), int(max(anchor_years)) + 1))
                return pd.Series(
                    np.interp(np.asarray(target_years, dtype=float), xs, ys),
                    index=target_years,
                )

            urb_interp = urb_anchors.apply(_interp_row, axis=1)
            urb_interp.columns.name = "Year"
            urb_interp_long = (
                urb_interp.stack().rename("urban_share_current").reset_index()
            )
            panel = panel.merge(urb_interp_long, on=["Code", "Year"], how="left")
            n_matched = panel["urban_share_current"].notna().sum()
            logger.info(
                "Time-varying urban share merged: %d obs have a "
                "URB-interpolated value (1875-1890 only); mean = %.2f%%",
                n_matched, float(panel["urban_share_current"].mean()),
            )
    except FileNotFoundError as exc:
        logger.warning("URB merge skipped (file not found): %s", exc)

    # ------------------------------------------------------------------
    # 6c-quater. Additional Galloway cross-sections.
    # BIR1871: birthplace shares -> migration controls.
    # TAX1876: county income tax -> income proxy (mid-treatment).
    # AGR1882: farm-size Gini -> land-inequality moderator.
    # GEL1882: religion-education employment -> Kulturkampf-channel
    #          outcome / endpoint (paired with rel1849_cat_priest).
    # EDU1886: post-Kulturkampf school attendance -> schooling channel
    #          endpoint (paired with EDU1849).
    # STA1871: marital status -> available for future nuptiality work.
    # Each is a single cross-section that enters the panel as a time-
    # invariant row (one value per county, repeated across panel years).
    # ------------------------------------------------------------------
    for loader, label in [
        (load_bir1871, "BIR1871"),
        (load_tax1876, "TAX1876"),
        (load_agr1882, "AGR1882"),
        (load_gel1882, "GEL1882"),
        (load_edu1886, "EDU1886"),
        (load_sta1871, "STA1871"),
        # POP1885: married_men_1885, married_women_1885,
        # married_sex_ratio_1885. Second anchor for the time-varying
        # married_sex_ratio series built post-merge below.
        (load_pop1885_marital_status, "POP1885 marital"),
        # AGE1890: actual count of women 15-49 and married women 15-49
        # per Kreis from Galloway's own age-by-marital tabulation. Acts
        # as the 1890 anchor for the time-varying marriage-prevalence
        # interpolation in compute_coale_indices. Also returns
        # `r_w_15_49_in_popf_1890` and `r_m_15_49_in_marriedf_1890` --
        # within-Kreis ratios used to extract 15-49 counts from
        # AGE1882's coarser bins below.
        (load_age1890, "AGE1890"),
        # AGE1882: only resolves coarse bins (0-19, 20-69, 70+); use
        # the AGE1890 per-Kreis ratios to extract approximate 15-49
        # counts (built in the post-merge step below).
        (load_age1882, "AGE1882"),
    ]:
        try:
            cs = loader()
            n_overlap = panel["Code"].isin(cs["Code"]).any()
            panel = panel.merge(cs, on="Code", how="left")
            logger.info(
                "%s merged: %d new columns, %d Kreise matched",
                label, cs.shape[1] - 1, cs["Code"].nunique(),
            )
        except FileNotFoundError as exc:
            logger.warning("%s merge skipped (file not found): %s", label, exc)
        except Exception as exc:
            logger.warning("%s merge failed: %s", label, exc)

    # Note: the previous ``compute_galloway_gmfr()`` loop over census
    # transcription templates has been superseded by the AGE1890 +
    # AGE1882 loaders above. Time-varying women_15_49 and
    # married_women_15_49 series are now built inside
    # compute_coale_indices via the piecewise-linear interpolation
    # documented in src/analysis/coale_indices.py. The transcription
    # templates and CLI helper in
    # ``data/raw/transcribed_marital_status/`` remain in the repo as
    # an opt-in path for any 1895/1900/1905/1910 census years where
    # the user wants to add ground-truth cells later, but are not
    # auto-merged into the headline panel.

    # ------------------------------------------------------------------
    # AGE1882 has only coarse 0-19 / 20-69 / 70+ marital bins, so we
    # derive approximate 15-49 counts by applying the AGE1890 within-
    # Kreis ratios. This buys us a mid-sample (1882) anchor for the
    # piecewise-linear interpolation of women_15_49 and
    # married_women_15_49 in compute_coale_indices, splitting the
    # 1871-1890 stretch into 1871-1882 and 1882-1890 segments.
    # ------------------------------------------------------------------
    if {"pop_1882f", "marriedf_1882", "r_w_15_49_in_popf_1890",
            "r_m_15_49_in_marriedf_1890"}.issubset(panel.columns):
        panel["women_15_49_1882"] = (
            panel["pop_1882f"] * panel["r_w_15_49_in_popf_1890"]
        )
        panel["married_women_15_49_1882"] = (
            panel["marriedf_1882"] * panel["r_m_15_49_in_marriedf_1890"]
        )
        panel["married_share_15_49_f_1882"] = (
            panel["married_women_15_49_1882"]
            / panel["women_15_49_1882"].replace(0, np.nan)
        )
        # Sanity filter: a handful of Kreise have AGE1882 marital
        # totals that look like transcription errors (e.g. KLEVE Code
        # 614 reports Married20-69f = 2,742 against Pop1882f = 25,046,
        # implying an 11% marriage rate -- vs ~33% panel-wide and 50%+
        # in nearby Kreise). When the implied 15-49 marriage share is
        # outside [0.30, 0.70] (panel sd ~0.05 around mean 0.52),
        # null the 1882 anchor for that Kreis. compute_coale_indices
        # then falls back to the 1871-1890 linear interpolation for
        # those rows (i.e. the two-anchor case).
        bad = (
            (panel["married_share_15_49_f_1882"] < 0.30)
            | (panel["married_share_15_49_f_1882"] > 0.70)
        )
        n_dropped = panel.loc[bad, "Code"].nunique()
        for col in (
            "women_15_49_1882", "married_women_15_49_1882",
            "married_share_15_49_f_1882",
        ):
            panel.loc[bad, col] = np.nan
        n_kept = panel.dropna(subset=["women_15_49_1882"])["Code"].nunique()
        logger.info(
            "AGE1882 approximate 15-49 anchors materialised: %d Kreise "
            "kept, %d Kreise nulled as implausible (marriage share <0.30 "
            "or >0.70); AGE1890 calibration ratios R_W=%.4f, R_M=%.4f",
            n_kept, n_dropped,
            float(panel["r_w_15_49_in_popf_1890"].mean()),
            float(panel["r_m_15_49_in_marriedf_1890"].mean()),
        )

    # ------------------------------------------------------------------
    # General marriage rate (Newell 1988 / standard demographic
    # textbook): marriages per 1,000 mid-year population aged 15+. The
    # 15+ denominator strips out the (very large) under-15 population
    # which cannot legally marry, giving a "marriageable-age" rate
    # rather than the crude rate (per total population). Built as a
    # time-varying quantity from three age-structured anchors:
    #
    #   1871  STA1871, exact (Popover15m + Popover15f).
    #   1882  AGE1882 coarse bins (0-19, 20-69, 70+) with the 15-19
    #         portion of the 0-19 bin recovered via the AGE1890
    #         within-bin ratio r_15to19_in_0to19_1890 -- the same
    #         calibration trick used for the Coale women_15_49 series.
    #   1890  AGE1890, 5/6 of Age14-19 to extract the 15-19 portion
    #         plus Age20+ bins.
    #
    # Piecewise-linear interpolation in year across the three anchors,
    # with the share clamped outside [1871, 1890] and pop-scaled by
    # Poptot_midyear so the resulting count tracks population growth.
    # Compatible with the `marriage_rate` (crude per Poptot_midyear)
    # which we keep alongside.
    # ------------------------------------------------------------------
    if {"pop_15plus_1871", "pop_15plus_1890",
            "pop_total_1871"}.issubset(panel.columns):
        # 1871 share: exact from STA1871.
        share_15plus_1871 = (
            panel["pop_15plus_1871"]
            / panel["pop_total_1871"].replace(0, np.nan)
        ).clip(lower=0.40, upper=0.80)

        # 1890 share: use AGE1890 pop_15plus_1890 over the Poptot_midyear
        # at year 1890 (cheapest reliable total-population denominator).
        pop_1890_by_code = (
            panel.loc[panel["Year"] == 1890, ["Code", "Poptot_midyear"]]
            .dropna()
            .drop_duplicates(subset="Code")
            .set_index("Code")["Poptot_midyear"]
        )
        panel["_pop_1890_total"] = panel["Code"].map(pop_1890_by_code)
        share_15plus_1890 = (
            panel["pop_15plus_1890"]
            / panel["_pop_1890_total"].replace(0, np.nan)
        ).clip(lower=0.40, upper=0.80)

        # 1882 share: third anchor from AGE1882 coarse bins, using the
        # AGE1890 within-bin (15-19)/(0-19) ratio to extract the 15-19
        # portion of the 0-19 bin. The resulting pop_15plus_1882 is then
        # divided by AGE1882's Pop1882 total to get the share. Computed
        # only for Kreise that have all three AGE1882 bin totals plus
        # the AGE1890 ratio (typically the full panel).
        share_15plus_1882: pd.Series | None = None
        if {"pop_0to19_1882", "pop_20to69_1882", "pop_70plus_1882",
                "pop_1882", "r_15to19_in_0to19_1890"}.issubset(panel.columns):
            pop_15plus_1882 = (
                panel["r_15to19_in_0to19_1890"] * panel["pop_0to19_1882"]
                + panel["pop_20to69_1882"]
                + panel["pop_70plus_1882"]
            )
            share_15plus_1882 = (
                pop_15plus_1882 / panel["pop_1882"].replace(0, np.nan)
            ).clip(lower=0.40, upper=0.80)
            # Store the 1882 anchor for the audit schema and the data
            # appendix to reference (parallels pop_15plus_1871 /
            # pop_15plus_1890).
            panel["pop_15plus_1882"] = pop_15plus_1882

        # Piecewise-linear interpolation of share_15+ across the three
        # anchors. Pre-1871 and post-1890 clamp to the nearest anchor;
        # 1871 <= t <= 1882 interpolates 1871 -> 1882; 1882 < t <= 1890
        # interpolates 1882 -> 1890. If the 1882 anchor is unavailable,
        # fall back to two-anchor linear (1871 -> 1890) for that Kreis.
        years = panel["Year"].astype(float)

        if share_15plus_1882 is not None:
            w_a = ((years - 1871.0) / (1882.0 - 1871.0)).clip(0.0, 1.0)
            w_b = ((years - 1882.0) / (1890.0 - 1882.0)).clip(0.0, 1.0)
            seg_a = share_15plus_1871 * (1 - w_a) + share_15plus_1882 * w_a
            seg_b = share_15plus_1882 * (1 - w_b) + share_15plus_1890 * w_b
            share_t = np.where(years <= 1882.0, seg_a, seg_b)
            share_t = pd.Series(share_t, index=panel.index)
            # Two-anchor fallback where the 1882 anchor is missing.
            missing_1882 = share_15plus_1882.isna()
            if missing_1882.any():
                weight = ((years - 1871) / 19.0).clip(0.0, 1.0)
                fallback = (
                    share_15plus_1871 * (1 - weight)
                    + share_15plus_1890 * weight
                )
                share_t = share_t.where(~missing_1882, fallback)
        else:
            weight = ((years - 1871) / 19.0).clip(0.0, 1.0)
            share_t = (
                share_15plus_1871 * (1 - weight) + share_15plus_1890 * weight
            )

        # Final fallback: if AGE1890 is missing entirely for a Kreis,
        # carry the 1871 share forward (matches previous behaviour).
        share_t = share_t.where(share_15plus_1890.notna(), share_15plus_1871)

        panel["pop_15plus"] = share_t * panel["Poptot_midyear"]
        panel["general_marriage_rate"] = np.where(
            panel["pop_15plus"].notna() & (panel["pop_15plus"] > 0),
            panel["Martot"] / panel["pop_15plus"] * 1000.0, np.nan,
        )
        panel = panel.drop(columns=["_pop_1890_total"])

        n_have = panel.dropna(subset=["general_marriage_rate"])["Code"].nunique()
        mean_1882 = (
            float(share_15plus_1882.dropna().mean())
            if share_15plus_1882 is not None and share_15plus_1882.notna().any()
            else float("nan")
        )
        logger.info(
            "General marriage rate (Newell 1988) materialised: %d Kreise. "
            "1871 share-15+ mean = %.3f, 1882 share-15+ mean = %.3f, "
            "1890 share-15+ mean = %.3f.",
            n_have,
            float(share_15plus_1871.dropna().mean()),
            mean_1882,
            float(share_15plus_1890.dropna().mean()),
        )

    # ------------------------------------------------------------------
    # 6c-sexies. Time-varying married sex ratio (Galloway, Hammel & Lee
    # 1994 control). Constructed as
    #
    #   married_sex_ratio_t = 100 * MarriedM_t / MarriedF_t
    #
    # from two anchor cross-sections: 1871 STA1871 (over-15) and 1885
    # POP1885 (all ages; under-15 marriage is negligible so the
    # definitional mismatch is empirically tiny -- see
    # load_pop1885_marital_status docstring). Piecewise-linear
    # interpolation in Year between 1871 and 1885; clamps to the
    # nearest anchor outside that window (i.e. constant at the 1871
    # value for 1862-1870, constant at the 1885 value for 1886-1890).
    # If the 1885 anchor is missing for a Kreis, carry the 1871 value
    # forward (and vice versa). Galloway, Hammel & Lee (1994) include
    # this variable as a control "to measure the separation of spouses
    # due to temporary or permanent relocation of the husband or wife"
    # -- the mechanical channel by which migration / military service
    # depresses period marital fertility without a behavioural
    # response. In the Kulturkampf panel this is the natural
    # bad-control test for whether the Polish-province CBR /
    # marriage-rate effects are driven by men leaving Posen / Bromberg
    # for Pittsburgh / the Ruhr, rather than by an institutional shock
    # to Catholic family formation.
    # ------------------------------------------------------------------
    if {"married_sex_ratio_1871", "married_sex_ratio_1885"}.issubset(
        panel.columns
    ):
        years = panel["Year"].astype(float)
        a71 = panel["married_sex_ratio_1871"].astype(float)
        a85 = panel["married_sex_ratio_1885"].astype(float)
        # Linear interpolation 1871 <= t <= 1885; clamp outside.
        w = ((years - 1871.0) / (1885.0 - 1871.0)).clip(0.0, 1.0)
        msr = a71 * (1 - w) + a85 * w
        # If either anchor is missing for a Kreis, fall back to the
        # available one; if both missing, leave NaN.
        msr = msr.where(a71.notna() & a85.notna(), a71.fillna(a85))
        # Clip to a plausible historical range: 19th-century county
        # populations had married_sex_ratio in [70, 105] with the
        # great majority in [85, 100]. Values <70 or >110 are almost
        # certainly mis-transcribed or boundary-reform artefacts.
        panel["married_sex_ratio"] = msr.clip(lower=70, upper=110)

        n_have = panel["married_sex_ratio"].notna().sum()
        mean_71 = float(a71.dropna().mean()) if a71.notna().any() else float("nan")
        mean_85 = float(a85.dropna().mean()) if a85.notna().any() else float("nan")
        logger.info(
            "married_sex_ratio (Galloway, Hammel & Lee 1994) materialised: "
            "%d / %d obs non-null. 1871 mean = %.2f, 1885 mean = %.2f.",
            n_have, len(panel), mean_71, mean_85,
        )

    # ------------------------------------------------------------------
    # 6c-quinquies. iPEHD 1849 covariates via kreiskey1849 crosswalk.
    # Brings in pre-Kulturkampf religious infrastructure (Catholic /
    # Protestant priests, churches), schooling (students by gender and
    # school type), 1849 population age/sex/marital structure, and a
    # 1849 factory-total aggregate. Crosswalk is name-based within Rb
    # with edit-distance backup; expect ~70% coverage of the analysis
    # Type-0 Kreise (boundary churn 1849 -> 1871 caps the upper bound).
    # ------------------------------------------------------------------
    try:
        ipehd_dir = DATA_RAW.parent / "ipehd_data"
        rel1871_path = _find_file(DATA_RAW, "REL1871")
        pop1849_path = ipehd_dir / "ipehd_1849_pop_demo.csv"
        if rel1871_path is not None and pop1849_path.exists():
            cw_1849 = build_crosswalk_1849(
                ipehd_1849_path=pop1849_path,
                rel1871_path=rel1871_path,
                verbose=False,
            )
            # Persist the crosswalk for downstream reuse (channels and
            # balance tables read it via DATA_PROCESSED / 'crosswalk_1849.csv').
            DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            cw_1849.to_csv(DATA_PROCESSED / "crosswalk_1849.csv", index=False)

            panel = merge_ipehd_1849(panel, cw_1849)
            n_matched_priest = panel["rel1849_cat_priest"].notna().sum() if "rel1849_cat_priest" in panel.columns else 0
            logger.info(
                "1849 iPEHD merged: crosswalk has %d county mappings; "
                "%d obs got non-null Catholic-priest count",
                len(cw_1849), int(n_matched_priest),
            )

            # 1849 elementary-school attendance rate (students / total
            # population, both sexes), built from EDU1849 + pop1849_tot.
            # Used as a continuous, truly pre-treatment literacy /
            # human-capital baseline (23 years before the May Laws) by
            # `run_pretreatment_trends` and the heterogeneity table.
            # Mirrors the construction in `channels.schooling_channel`.
            if {"edu1849_pub_ele_stud_m", "edu1849_pub_ele_stud_f",
                    "pop1849_tot"}.issubset(panel.columns):
                students_1849 = (
                    panel["edu1849_pub_ele_stud_m"].fillna(0)
                    + panel["edu1849_pub_ele_stud_f"].fillna(0)
                )
                panel["attend_rate_1849_baseline"] = (
                    students_1849 / panel["pop1849_tot"].replace(0, np.nan)
                )
                n_kreis = panel.dropna(subset=["attend_rate_1849_baseline"])["Code"].nunique()
                logger.info(
                    "attend_rate_1849_baseline materialised: %d Kreise with non-null value",
                    n_kreis,
                )
        else:
            logger.warning("1849 iPEHD merge skipped (REL1871 or pop_demo not found)")
    except Exception as exc:
        logger.warning("1849 iPEHD merge failed: %s", exc)

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
            for col in (
                "I_f", "I_g", "I_h", "gmfr", "lgfr", "gfr",
                "Ig_static_1871", "gmfr_static_1871",
            ):
                if col in panel.columns:
                    panel.loc[panel["cbr_flag"].fillna(False), col] = np.nan
        # Light flag for demographically implausible I_g (>1.0 means
        # marital fertility above the Hutterite maximum -- a measurement
        # artefact). Also flag I_g > 1 from boundary-reform residuals.
        n_ig_extreme = int((panel["I_g"] > 1.2).sum())
        if n_ig_extreme > 0:
            logger.warning("%d obs with I_g > 1.2 (Hutterite max); set to NaN", n_ig_extreme)
            panel.loc[panel["I_g"] > 1.2, ["I_g", "gmfr"]] = np.nan
        if "Ig_static_1871" in panel.columns:
            n_ig_static_extreme = int((panel["Ig_static_1871"] > 1.2).sum())
            if n_ig_static_extreme > 0:
                logger.warning(
                    "%d obs with Ig_static_1871 > 1.2 (Hutterite max); set to NaN",
                    n_ig_static_extreme,
                )
                panel.loc[
                    panel["Ig_static_1871"] > 1.2,
                    ["Ig_static_1871", "gmfr_static_1871"],
                ] = np.nan
        logger.info("Coale indices computed: I_f, I_g, I_h, gmfr, Ig_static_1871, gmfr_static_1871")
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