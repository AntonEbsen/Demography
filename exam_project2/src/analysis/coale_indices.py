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

    df = panel.copy()
    pop = df[pop_col] if pop_col in df.columns else df["Poptot"]

    if (
        use_county_specific_share
        and "women_15_49_1871" in df.columns
        and "pop_total_1871" in df.columns
    ):
        # County-specific share of women aged 15-49 from POP1871, applied
        # to contemporaneous mid-year population.
        share = (df["women_15_49_1871"] / df["pop_total_1871"]).clip(lower=0.10, upper=0.40)
        W = share * pop
    else:
        W = women_share_of_pop * pop

    # County-specific marriage shifter. k_i applies *proportionally* to
    # both Fbar_mar and Fbar_unmar so they remain a partition of Fbar_all:
    # if a county has 10% higher marriage prevalence then 10% more of
    # the Hutterite-weighted fertility "potential" is in the married
    # category and 10% less in the unmarried category (subject to
    # clipping to keep Fbar_unmar non-negative).
    if (
        use_sta1871
        and marriage_col in df.columns
        and df[marriage_col].notna().any()
    ):
        mu = df[marriage_col]
        mu_ref = float(
            df.drop_duplicates("Code")[marriage_col].dropna().mean()
        )
        k = (mu / mu_ref).clip(lower=0.5, upper=1.5)
        Fbar_mar_eff = k * Fbar_mar
        # Keep the marital + non-marital partition consistent: rescale
        # the unmarried weight so that the implied weighted-average
        # married share equals the rescaled value, preserving
        # Fbar_mar + Fbar_unmar' = Fbar_all (the Princeton identity).
        Fbar_unmar_eff = Fbar_all - Fbar_mar_eff
        Fbar_unmar_eff = Fbar_unmar_eff.clip(lower=1e-6)
        used_sta1871 = True
    else:
        Fbar_mar_eff = Fbar_mar
        Fbar_unmar_eff = Fbar_unmar
        used_sta1871 = False

    # M = implied count of married women 15-49 (used as GMFR denominator
    # and exposed as a panel column). Under the STA1871 recalibration
    # this varies across counties; under the constant-schedule fallback
    # it equals W * mean(rho_a^const).
    married_share_overall_const = sum(
        age_dist[a] * married_share[a] for a in age_dist
    )
    if used_sta1871:
        # Implied married-women count among 15-49 is W * mu_i * (ratio
        # of "implied married rate among 15-49 under reference schedule"
        # to "empirical married rate over 15"). The ratio is just the
        # constant married_share_overall_const / mu_ref, baked into k.
        M = W * k * married_share_overall_const
    else:
        M = W * married_share_overall_const

    df["I_f"] = df["Birtot"] / (W * Fbar_all)
    df["I_g"] = df["Birlegtot"] / (W * Fbar_mar_eff)
    df["I_h"] = np.where(
        (W > 0) & df["Birbastot"].notna(),
        df["Birbastot"] / (W * Fbar_unmar_eff),
        np.nan,
    )

    # Galloway-tradition GMFR: legitimate births per 1{,}000 married women
    # 15--49. Direct unnormalised analogue of the Princeton I_g.
    df["gmfr"] = np.where(M > 0, df["Birlegtot"] / M * 1000.0, np.nan)
    # Expose the implied married-women count as a panel column so
    # downstream code can recompute its own rates.
    df["married_women_15_49"] = M

    if used_sta1871:
        logger.info(
            "Coale I_g computed with STA1871 recalibration: mu_ref=%.4f, "
            "county-specific shifter k_i applied (clipped to [0.5, 1.5])",
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
    cols = [c for c in ["I_f", "I_g", "I_h", "marriage_rate"] if c in df.columns]
    return (
        df.groupby(["group", "period"], observed=True)[cols]
        .mean()
        .reset_index()
    )


def did_on_indices(
    panel_with_indices: pd.DataFrame,
    indices: Sequence[str] = ("I_f", "I_g", "I_h", "marriage_rate"),
) -> pd.DataFrame:
    """
    DiD coefficient on cath_share x post for each Coale index. Reports the
    decomposition: the same shock affects I_f, I_g, I_m differently,
    revealing whether the channel is marital-fertility or nuptiality.
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
