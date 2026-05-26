"""
coale_indices.py
================
Princeton European Fertility Project (Coale & Watkins 1986) fertility indices
for the Kulturkampf panel.

Background. The Princeton EFP's central methodological contribution was a
trio of Hutterite-benchmarked indices that decompose total fertility into
marital-fertility and nuptiality components:

    I_f  = B / sum_i (W_i * F_i^H)            (overall fertility)
    I_g  = B_leg / sum_i (M_i * F_i^H)        (marital fertility)
    I_h  = B_ill / sum_i ((W_i - M_i) * F_i^H)  (non-marital fertility)
    I_m  = sum_i (M_i * F_i^H) / sum_i (W_i * F_i^H)  (proportion married,
                                            weighted by Hutterite ASFR)

Identity (when illegitimate fertility is small):

    I_f  ≈  I_g * I_m + I_h * (1 - I_m)

So a fall in I_f can be decomposed into either a fall in marital fertility
(I_g, contraception, abstinence) or a fall in nuptiality (I_m, marriage delay
or non-marriage). For the Kulturkampf the demographic question is exactly:
did Catholic counties experience a quantum (marital-fertility) shock or a
nuptiality (marriage-formation) shock?

Approximations. Galloway does not include age structure of women or married
women, so we approximate using:

    1. Coale-Demeny "West" Level 7 (e0 ≈ 35) female age distribution within
       the 15-49 reproductive interval. Standard for 19th-century Europe.
    2. A constant share rho_W = 0.23 of total population that is female
       aged 15-49 (typical for high-fertility / high-mortality regimes).
    3. A nuptiality schedule for 19th-century Prussia derived from Hajnal
       (1965) and Knodel's village data, used to estimate married women by
       age. Calibrated to a 55% overall marriage prevalence among women
       15-49 by default (override via the ``marriage_prevalence`` arg).

These approximations affect the *level* of the indices but not the
cross-county or pre/post *ratios*, which are what the empirical analysis
relies on.

References:
- Coale, A. J. (1969). The decline of fertility in Europe.
- Coale, A. J. and Watkins, S. C., eds. (1986). The Decline of Fertility in
  Europe. Princeton University Press.
- Hajnal, J. (1965). European marriage patterns in perspective.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (standard Princeton EFP / Coale-Demeny references)
# ---------------------------------------------------------------------------

# Hutterite age-specific fertility rates, 1921--30 (Coale 1969). Used as the
# upper benchmark of natural fertility.
HUTTERITE_ASFR: dict[tuple[int, int], float] = {
    (15, 19): 0.300,
    (20, 24): 0.550,
    (25, 29): 0.502,
    (30, 34): 0.447,
    (35, 39): 0.406,
    (40, 44): 0.222,
    (45, 49): 0.061,
}

# Coale-Demeny "West" Level 7 (e0 ≈ 35y) female age distribution, share
# of women aged 15-49 in each 5-year bracket. Approximates 19th-century
# European female age structure within the reproductive interval.
CD_WEST_FEMALE_15_49: dict[tuple[int, int], float] = {
    (15, 19): 0.197,
    (20, 24): 0.179,
    (25, 29): 0.159,
    (30, 34): 0.140,
    (35, 39): 0.122,
    (40, 44): 0.105,
    (45, 49): 0.098,
}

# Nuptiality schedule for 19th-century Prussia: share of women in each
# 5-year age group who are currently married. Calibrated against the Princeton
# EFP's empirical Prussia I_g ≈ 0.70 in 1871 to give an overall marriage
# prevalence near 64% among women aged 15-49, consistent with the eastern
# side of the Hajnal line.
DEFAULT_MARRIED_SHARE: dict[tuple[int, int], float] = {
    (15, 19): 0.08,
    (20, 24): 0.45,
    (25, 29): 0.78,
    (30, 34): 0.88,
    (35, 39): 0.90,
    (40, 44): 0.85,
    (45, 49): 0.78,
}

# Share of total population that is female aged 15-49. Calibrated against
# Coale-Demeny "West" tables for life expectancy at birth e0 ≈ 35-40 with
# crude birth rate ≈ 38 per 1{,}000 (typical for 1860-90 Prussia).
WOMEN_15_49_SHARE = 0.25


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _hutterite_weighted(age_dist: dict, asfr: dict = HUTTERITE_ASFR) -> float:
    """Sum_i (s_i * F_i^H) for a given age distribution s_i."""
    return float(sum(age_dist[ages] * asfr[ages] for ages in asfr))


def _hutterite_marital_weighted(
    age_dist: dict,
    married_share: dict,
    asfr: dict = HUTTERITE_ASFR,
) -> float:
    """Sum_i (s_i * m_i * F_i^H) — Hutterite-weighted average over married women."""
    return float(sum(age_dist[a] * married_share[a] * asfr[a] for a in asfr))


def _hutterite_unmarried_weighted(
    age_dist: dict,
    married_share: dict,
    asfr: dict = HUTTERITE_ASFR,
) -> float:
    return float(sum(age_dist[a] * (1 - married_share[a]) * asfr[a] for a in asfr))


def compute_coale_indices(
    panel: pd.DataFrame,
    women_share_of_pop: float = WOMEN_15_49_SHARE,
    age_dist: dict | None = None,
    married_share: dict | None = None,
    pop_col: str = "Poptot_midyear",
    use_county_specific_share: bool = True,
    use_sta1871: bool = True,
    marriage_col: str = "married_share_over15_f_1871",
    use_age1890: bool = True,
) -> pd.DataFrame:
    """
    Compute Coale's $I_f$, $I_g$, $I_h$ for each county-year, plus the
    Galloway-style General Marital Fertility Rate (GMFR).

    Following Galloway, Hammel & Lee (1994, *Population Studies*), the
    primary fertility measure for Prussia is the GMFR (legitimate births
    per 1{,}000 married women aged 15--49). Coale's $I_g$ is the
    Hutterite-normalised version of the same quantity (= GMFR / Hutterite
    age-weighted natural-fertility maximum). Both are computed and
    returned: ``I_g`` (unitless, normalised) and ``gmfr`` (per 1{,}000
    married women, the Galloway-tradition headline level).

    Returns a copy of ``panel`` with new columns ``I_f``, ``I_g``, ``I_h``,
    ``gmfr``, and (when STA1871 is available) ``married_women_15_49``.

    Implementation. Galloway VIT files lack age structure of married
    women, so we approximate the contemporaneous count of women aged
    15-49 by either:

      - **County-specific 1871 share** (default, ``use_county_specific_share=True``):
        $W_t = (\\mathrm{women\\_15\\_49\\_1871} / \\mathrm{pop\\_total\\_1871})
        \\times \\mathrm{Poptot}_t$. Allows the share of reproductive-age
        women to vary across counties (e.g.\\ Polish vs Rhineland) but
        assumes within-county constancy across 1862--1890.
      - **Constant share** (``use_county_specific_share=False``):
        $W_t = 0.25 \\times \\mathrm{Poptot}_t$. The Coale-Demeny "West"
        Level 7 reference value used in the Princeton EFP.

    Marital-fertility denominator (the proper Princeton EFP one
    requires $\\sum_a m_a F^H_a$ -- the Hutterite weighted *count of
    married women* in each age group). Two recalibration paths:

      - **STA1871 recalibration** (default, ``use_sta1871=True``):
        when ``married_share_over15_f_1871`` (= Marriedover15f /
        Popover15f from Galloway's STA1871 cross-section) is on the
        panel, scale the constant marital-fertility weight by the
        county-specific marriage prevalence shifter
        $k_i = \\mu_i / \\bar\\mu^{\\mathrm{Prussia}}$,
        where $\\bar\\mu^{\\mathrm{Prussia}}$ is the 1871 cross-section
        mean of $\\mu_i$ (~ 0.516). This preserves the Prussia-wide
        age-pattern of marriage but allows the *level* of marriage
        prevalence to vary across counties, which is the dominant
        source of cross-county variation in actual marital-fertility
        denominators. Assumes within-county marriage prevalence is
        constant across 1862-1890.
      - **Constant schedule** (``use_sta1871=False``): the original
        approximation -- ``DEFAULT_MARRIED_SHARE`` schedule applied
        identically to every county.

    The ``$I_m$`` (nuptiality) component is captured separately by the
    observed marriage rate (``Martot / Poptot_midyear``) rather than
    by a constructed Coale $I_m$ index.

    Default population denominator is ``Poptot_midyear`` (the
    linearly-interpolated mid-year column built by
    ``compute_midyear_population``); pass ``pop_col="Poptot"`` to fall
    back on the raw Galloway carry-forward population.
    """
    age_dist = age_dist or CD_WEST_FEMALE_15_49
    married_share = married_share or DEFAULT_MARRIED_SHARE

    Fbar_all = _hutterite_weighted(age_dist)
    Fbar_mar = _hutterite_marital_weighted(age_dist, married_share)
    Fbar_unmar = _hutterite_unmarried_weighted(age_dist, married_share)
    married_share_overall_const = sum(
        age_dist[a] * married_share[a] for a in age_dist
    )

    df = panel.copy()
    year = df["Year"]
    pop = df[pop_col] if pop_col in df.columns else df["Poptot"]

    # -----------------------------------------------------------------
    # Time-varying W (women 15-49 count) and M (married women 15-49
    # count) anchored at the two Galloway cross-sections we have:
    #
    #   1871 anchor : POP1871 (women_15_49_1871) and STA1871
    #                 (married_share_over15_f_1871, scaled to 15-49 via
    #                 the implied schedule ratio).
    #   1890 anchor : AGE1890 (women_15_49_1890,
    #                 married_women_15_49_1890) -- Galloway's own
    #                 age-by-marital-status tabulation, giving us actual
    #                 counts of women 15-49 and married women 15-49 per
    #                 Kreis at the 1890 census.
    #
    # For 1871 <= t <= 1890: linear interpolation in year.
    # For t < 1871: scale 1871 count by contemporaneous population
    # ratio (preserves the prior pre-anchor behaviour).
    # -----------------------------------------------------------------
    have_age1890 = (
        use_age1890
        and "women_15_49_1890" in df.columns
        and "married_women_15_49_1890" in df.columns
        and df["women_15_49_1890"].notna().any()
    )
    have_age1882 = (
        use_age1890
        and "women_15_49_1882" in df.columns
        and "married_women_15_49_1882" in df.columns
        and df["women_15_49_1882"].notna().any()
    )

    def _piecewise_linear_anchored(
        year_col: pd.Series,
        anchors: list[tuple[int, pd.Series]],
    ) -> pd.Series:
        """
        Piecewise linear interpolation in `year_col` across an ordered
        list of (year, value_series) anchors. **Per row**, anchors with
        missing values are skipped -- so a Kreis where the 1882 anchor
        is NaN (e.g. nulled as implausible) still gets clean
        1871-1890 interpolation. Inside the anchor range, each row is
        linearly interpolated between the two nearest *non-null*
        surrounding anchors. Outside the range, the nearest-anchor
        value is returned (caller handles pre-anchor pop scaling).
        """
        anchors = sorted(anchors, key=lambda x: x[0])
        years_a = np.array([y for y, _ in anchors], dtype=float)
        # Stack the anchor values into a (n_rows, n_anchors) array.
        values_mat = np.column_stack(
            [v.astype(float).to_numpy() for _, v in anchors]
        )
        year_arr = year_col.to_numpy(dtype=float)
        out = np.full(len(year_arr), np.nan, dtype=float)
        for i in range(len(year_arr)):
            t = year_arr[i]
            valid_mask = ~np.isnan(values_mat[i])
            if not valid_mask.any():
                continue
            y_valid = years_a[valid_mask]
            v_valid = values_mat[i, valid_mask]
            if t <= y_valid[0]:
                out[i] = v_valid[0]
            elif t >= y_valid[-1]:
                out[i] = v_valid[-1]
            else:
                # Find bracketing anchors and linearly interpolate.
                k = np.searchsorted(y_valid, t)
                y_lo, y_hi = y_valid[k - 1], y_valid[k]
                v_lo, v_hi = v_valid[k - 1], v_valid[k]
                w = (t - y_lo) / (y_hi - y_lo)
                out[i] = v_lo * (1 - w) + v_hi * w
        return pd.Series(out, index=year_col.index, dtype=float)

    # --- W (women 15-49 count, county-year) ---------------------------
    if (
        use_county_specific_share
        and "women_15_49_1871" in df.columns
        and "pop_total_1871" in df.columns
    ):
        share_1871 = (
            df["women_15_49_1871"] / df["pop_total_1871"]
        ).clip(lower=0.10, upper=0.40)
        W_pop_scaled = share_1871 * pop  # pre-1871 / fallback

        if have_age1890:
            anchors_W = [(1871, df["women_15_49_1871"].astype(float))]
            if have_age1882:
                anchors_W.append((1882, df["women_15_49_1882"].astype(float)))
            anchors_W.append((1890, df["women_15_49_1890"].astype(float)))
            W_interp = _piecewise_linear_anchored(year, anchors_W)
            # Use the interpolation for 1871 onward when the latest
            # anchor is present; pre-1871 falls back to pop-scaled.
            w_anchor_90 = df["women_15_49_1890"]
            use_interp_W = w_anchor_90.notna() & (year >= 1871)
            W = pd.Series(
                np.where(use_interp_W, W_interp, W_pop_scaled),
                index=df.index, dtype=float,
            )
        else:
            W = W_pop_scaled
    else:
        W = women_share_of_pop * pop

    # --- M (married women 15-49 count, county-year) -------------------
    # 1871 anchor: STA1871 over-15 prevalence shifter k_71 = mu_i / mean(mu).
    # k_71 measures relative marriage prevalence vs Prussia average.
    # M_71 = W_71 * k_71 * married_share_overall_const  (current logic).
    # 1890 anchor: AGE1890 married_women_15_49_1890 directly.
    # For 1871 <= t <= 1890: interpolate M linearly in year.

    used_sta1871 = False
    used_age1890 = False
    mu_ref = float("nan")
    if (
        use_sta1871
        and marriage_col in df.columns
        and df[marriage_col].notna().any()
    ):
        # STA1871 reports `married_share_over15_f_1871` = Marriedover15f /
        # Popover15f. We use this empirical prevalence *directly* as the
        # 1871 marriage prevalence among 15-49 women. This is a small
        # approximation (over-15 includes some 50+ women, who in 1871
        # Prussia are mostly still married until widowhood) but it's the
        # closest measure to AGE1890's 15-49 prevalence -- the Prussia
        # means are 0.516 vs 0.524, within 2 percentage points.
        mu_1871 = df[marriage_col]
        mu_ref = float(df.drop_duplicates("Code")[marriage_col].dropna().mean())
        used_sta1871 = True

        if (
            have_age1890
            and "women_15_49_1871" in df.columns
        ):
            # 1871 anchor: empirical M_1871 = W_1871 * mu_1871.
            M_71_anchor = (
                df["women_15_49_1871"].astype(float) * mu_1871
            )
            # 1890 anchor: directly observed in AGE1890.
            M_90_anchor = df["married_women_15_49_1890"].astype(float)
            anchors_M = [(1871, M_71_anchor)]
            if have_age1882:
                # 1882 anchor: AGE1882 marital totals adjusted to 15-49
                # by AGE1890 within-Kreis ratios.
                M_82_anchor = df["married_women_15_49_1882"].astype(float)
                anchors_M.append((1882, M_82_anchor))
            anchors_M.append((1890, M_90_anchor))
            M_interp = _piecewise_linear_anchored(year, anchors_M)
            # Pre-1871 / fallback: scale 1871 anchor by pop ratio.
            M_pre = M_71_anchor * (pop / df["pop_total_1871"]).clip(lower=0.5, upper=2.0)
            use_interp_M = M_90_anchor.notna() & (year >= 1871)
            M = pd.Series(
                np.where(use_interp_M, M_interp, M_pre),
                index=df.index, dtype=float,
            )
            used_age1890 = True
        else:
            # STA1871 only: M_t = W_1871 * mu_1871 * (pop / pop_1871).
            if "women_15_49_1871" in df.columns:
                M_71_anchor = (
                    df["women_15_49_1871"].astype(float) * mu_1871
                )
                M = M_71_anchor * (
                    pop / df["pop_total_1871"]
                ).clip(lower=0.5, upper=2.0)
            else:
                M = W * mu_1871
    else:
        # No STA1871 -- fall back to the constant Prussia-wide schedule.
        M = W * married_share_overall_const

    # --- Post-1890 pop-scaled extrapolation --------------------------
    # The 1871 -> 1882 -> 1890 piecewise interpolation clamps post-1890
    # values to the 1890 anchor (constant per county). For the long
    # window 1891-1910 this understates W and M in growing counties
    # and overstates them in shrinking ones, biasing the marital-
    # fertility rate denominator. Mirror the pre-1871 fallback by
    # scaling the 1890 anchor proportionally to mid-year population:
    #
    #     W_t   = W_1890   * (Poptot_t / Poptot_1890)    for t > 1890
    #     M_t   = M_1890   * (Poptot_t / Poptot_1890)    for t > 1890
    #
    # Implicitly assumes (a) the share of women 15-49 in total pop, and
    # (b) the marriage prevalence among women 15-49, are constant from
    # 1890 forward. Both assumptions are reasonable in the short run
    # but become strained by 1910 as the secular fertility transition
    # accelerates (Knodel 1974; Galloway, Hammel & Lee 1994). Treat
    # post-1890 I_g and GMFR as approximate.
    if have_age1890:
        pop_1890_by_code = (
            df.loc[df["Year"] == 1890, ["Code", pop_col]]
            .drop_duplicates("Code").set_index("Code")[pop_col]
        )
        pop_1890_series = df["Code"].map(pop_1890_by_code)
        post_mask = (year > 1890) & pop_1890_series.notna() & (pop_1890_series > 0)
        if post_mask.any():
            pop_ratio = (pop / pop_1890_series).clip(lower=0.5, upper=2.0)
            W_post = df["women_15_49_1890"].astype(float) * pop_ratio
            if used_age1890:
                M_post = df["married_women_15_49_1890"].astype(float) * pop_ratio
            else:
                M_post = M  # leave M unchanged for fallback branches
            W = pd.Series(
                np.where(post_mask & W_post.notna(), W_post, W),
                index=df.index, dtype=float,
            )
            M = pd.Series(
                np.where(post_mask & M_post.notna(), M_post, M),
                index=df.index, dtype=float,
            )
            logger.info(
                "Post-1890 pop-scaled extrapolation applied to %d obs "
                "(W and M scaled by Poptot_t / Poptot_1890)",
                int(post_mask.sum()),
            )

    # --- Effective Hutterite-weighted marital fertility maximum -------
    # Re-express the time-varying M as a county-year-specific k_t and
    # apply it to the Hutterite weights so the existing I_g formula
    # (B_leg / (W * Fbar_mar_eff)) generalises cleanly. k_t = (M/W) /
    # married_share_overall_const so that, when M / W equals
    # married_share_overall_const, k_t = 1 (the reference Prussia
    # schedule).
    k_t = (M / W.replace(0, np.nan) / married_share_overall_const).clip(
        lower=0.5, upper=1.5
    ).fillna(1.0)
    Fbar_mar_eff = k_t * Fbar_mar
    Fbar_unmar_eff = (Fbar_all - Fbar_mar_eff).clip(lower=1e-6)

    # --- Compute the headline indices ---------------------------------
    df["I_f"] = df["Birtot"] / (W * Fbar_all)
    df["I_g"] = df["Birlegtot"] / (W * Fbar_mar_eff)
    df["I_h"] = np.where(
        (W > 0) & df["Birbastot"].notna(),
        df["Birbastot"] / (W * Fbar_unmar_eff),
        np.nan,
    )
    df["gmfr"] = np.where(M > 0, df["Birlegtot"] / M * 1000.0, np.nan)

    # --- Static-1871 prevalence variants ------------------------------
    # The contemporaneous M_t and Fbar_mar_eff are mechanically affected
    # by the Kulturkampf, which directly regulates marriage formation
    # (Notzivilehe 1874, Personenstandsgesetz 1875). A treatment-induced
    # drop in M_t inflates the GMFR and I_g denominators are "bad
    # controls" in the Pearl/Angrist sense. The static-1871 variants
    # freeze the marriage *prevalence* mu_1871 = M_1871 / W_1871 to its
    # pre-Kulturkampf county-specific baseline while keeping the
    # fertile-age female population W_t time-varying (so the rate still
    # scales with population growth):
    #
    #   M_static_t   = mu_1871 * W_t
    #   GMFR_static  = B_leg / M_static_t * 1000
    #   I_g_static   = B_leg / (W_t * k_1871 * Fbar_mar)
    #
    # where k_1871 = mu_1871 / married_share_overall_const is the
    # county-specific 1871 shifter. The construction inherits the
    # Coale-Watkins (1986) decomposition logic: nuptiality (mu) is
    # held analytically fixed while marital fertility (births per
    # fixed-prevalence married woman) is the identified outcome.
    if used_sta1871:
        mu_1871_series = df[marriage_col].astype(float)
        # Counties missing STA1871 fall back to the constant schedule,
        # which by construction has zero static treatment-channel bias.
        mu_1871_series = mu_1871_series.fillna(married_share_overall_const)
        M_static_1871 = mu_1871_series * W
        k_1871 = (mu_1871_series / married_share_overall_const).clip(
            lower=0.5, upper=1.5
        )
        Fbar_mar_static = k_1871 * Fbar_mar
        df["gmfr_static_1871"] = np.where(
            M_static_1871 > 0,
            df["Birlegtot"] / M_static_1871 * 1000.0,
            np.nan,
        )
        df["Ig_static_1871"] = np.where(
            (W > 0) & (Fbar_mar_static > 0),
            df["Birlegtot"] / (W * Fbar_mar_static),
            np.nan,
        )
        # Symmetric static-1871 illegitimate-fertility rate. Illegitimate
        # births per 1{,}000 *unmarried* women 15-49 with the unmarried
        # share fixed at its 1871 county-specific level. This is the
        # mechanical analogue of gmfr_static_1871 on the non-marital
        # margin: it purges (i) the total-population denominator that
        # contaminates `illegitimate_br` and (ii) the marriage-prevalence
        # composition shock that contaminates the `illegitimacy_ratio`.
        # If the Kulturkampf moved marriage formation, only this static
        # rate isolates the behavioural non-marital fertility response.
        U_static_1871 = (1.0 - mu_1871_series).clip(lower=0.0) * W
        df["illegitimate_br_static_1871"] = np.where(
            U_static_1871 > 0,
            df["Birbastot"] / U_static_1871 * 1000.0,
            np.nan,
        )
    # General fertility rate (GFR): total births per 1,000 women aged
    # 15-49 mid-year. The standard demographic textbook fertility
    # measure (Newell 1988): strips out the under-15 / over-49 / male
    # portion of the population that CBR's denominator includes. Uses
    # the *time-varying* W_t from the AGE1890+AGE1882+STA1871 anchored
    # interpolation, not the static 1871 cross-section (which is the
    # deprecated `gfr_static_1871` column).
    df["gfr"] = np.where(W > 0, df["Birtot"] / W * 1000.0, np.nan)
    # Legitimate general fertility rate: legitimate births per 1{,}000
    # *women* aged 15-49 (NOT per married woman). The natural
    # marital-style counterpart to CBR -- uses the proper age-restricted
    # denominator from POP1871/AGE1890 rather than total population.
    df["lgfr"] = np.where(W > 0, df["Birlegtot"] / W * 1000.0, np.nan)

    # Coale's nuptiality index I_m: the Hutterite-weighted proportion of
    # women 15-49 who are married. Formally
    #
    #   I_m = sum_a (m_a * F_a^H) / sum_a (w_a * F_a^H)
    #
    # where m_a / w_a is the age-specific married share. We approximate
    # the age structure of m_a and w_a within 15-49 by holding the
    # Princeton DEFAULT_MARRIED_SHARE schedule's age profile fixed and
    # rescaling its level by the time-varying county-year shifter k_t =
    # (M/W) / married_share_overall_const (the same shifter used above
    # to make I_g county-year-specific). This yields
    #
    #   I_m_it = k_t_it * (Fbar_mar / Fbar_all),
    #
    # so I_m inherits the AGE1890 + AGE1882 + STA1871 anchored time
    # variation in M/W. The Coale identity
    #
    #   I_f = I_g * I_m + I_h * (1 - I_m)
    #
    # is approximately satisfied by these series (exactly satisfied if
    # the marital-fertility/non-marital-fertility partition uses the
    # same k_t shifter, which it does here -- Fbar_unmar_eff = Fbar_all
    # - Fbar_mar_eff). For descriptive use the index makes the nuptiality
    # channel directly observable as a DiD outcome.
    Im_ref = Fbar_mar / Fbar_all
    df["I_m"] = (k_t * Im_ref).clip(lower=0.05, upper=0.95)

    # Expose the time-varying counts as panel columns so downstream
    # code (general_marriage_rate, schema audit, sensitivity checks)
    # can read them. These are the AGE1890+AGE1882+STA1871-anchored
    # piecewise-interpolated series.
    df["women_15_49"] = W
    df["married_women_15_49"] = M

    if used_age1890:
        anchors_log = "1871 + 1890"
        if have_age1882:
            anchors_log = "1871 + 1882 (AGE1882 via AGE1890 ratios) + 1890"
        logger.info(
            "Coale indices computed with AGE anchors (%s): mu_ref=%.4f at "
            "1871; piecewise-linear interpolation of W and M; pop-scaled "
            "extrapolation pre-1871",
            anchors_log, mu_ref,
        )
    elif used_sta1871:
        logger.info(
            "Coale I_g computed with STA1871 recalibration only "
            "(AGE1890 column not found or all-null): mu_ref=%.4f",
            mu_ref,
        )
    else:
        logger.info(
            "Coale I_g computed with constant Prussia-wide marriage "
            "schedule (STA1871 column not found or all-null)"
        )

    return df


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------

def aggregate_by_group_period(
    panel_with_indices: pd.DataFrame,
    pre_years: tuple[int, int] = (1862, 1872),
    post_years: tuple[int, int] = (1873, 1890),
) -> pd.DataFrame:
    """
    Group means of Coale indices and observed nuptiality by (high_cath ×
    period). Useful for the decomposition narrative: "I_f fell in
    high-Catholic counties post-1873 — was this driven by I_g (marital
    fertility) or by nuptiality (the marriage rate)?"
    """
    df = panel_with_indices.copy()
    df["period"] = np.where(
        df["Year"].between(*post_years), "Post",
        np.where(df["Year"].between(*pre_years), "Pre", "Other"),
    )
    df = df[df["period"] != "Other"]
    df["group"] = np.where(df["high_cath"] == 1, "High Cath", "Low Cath")
    cols = [
        c for c in
        ["I_f", "I_g", "I_h", "I_m",
         "marriage_rate", "general_marriage_rate"]
        if c in df.columns
    ]
    return (
        df.groupby(["group", "period"], observed=True)[cols]
        .mean()
        .reset_index()
    )


def did_on_indices(
    panel_with_indices: pd.DataFrame,
    indices: Sequence[str] = (
        "I_f", "I_g", "I_h", "I_m",
        "marriage_rate", "general_marriage_rate",
    ),
) -> pd.DataFrame:
    """
    DiD coefficient on cath_share x post for each Coale index. Reports the
    decomposition: the same shock affects I_f, I_g, I_m differently,
    revealing whether the channel is marital-fertility (I_g) or nuptiality
    (I_m). The Coale identity I_f = I_g * I_m + I_h * (1 - I_m) means the
    I_f coefficient should approximately equal a weighted average of the
    I_g and I_m coefficients.
    """
    from src.analysis.regressions import run_baseline_did

    rows = []
    for idx in indices:
        try:
            res = run_baseline_did(
                panel_with_indices, outcome=idx, treatment="continuous"
            )["result"]
            rows.append({
                "index": idx,
                "coef": float(res.params["cath_share_x_post"]),
                "se": float(res.std_errors["cath_share_x_post"]),
                "p": float(res.pvalues["cath_share_x_post"]),
                "n": int(res.nobs),
            })
        except Exception as exc:
            logger.warning("DiD on %s failed: %s", idx, exc)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    panel = pd.read_parquet(
        Path(__file__).resolve().parent.parent.parent
        / "data" / "processed" / "analysis_panel.parquet"
    )
    df = compute_coale_indices(panel)
    print("Sample county-year I_f, I_g, I_h values:")
    print(df[["Code", "Year", "high_cath", "I_f", "I_g", "I_h"]].head(10).to_string(index=False))
    print()
    print("Mean values by group × period:")
    print(aggregate_by_group_period(df).to_string(index=False))
    print()
    print("DiD on each index:")
    print(did_on_indices(df).to_string(index=False))
