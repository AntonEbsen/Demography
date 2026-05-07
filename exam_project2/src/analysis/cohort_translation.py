"""
cohort_translation.py
=====================
Translate the period CBR coefficient into demographically-interpretable
quantities: cumulative "missing" births during the Kulturkampf and an
implied translation to total fertility rate (TFR) and completed cohort
fertility (CCF).

We do NOT have women-15--45 counts in the Galloway panel, so the
translation goes via a constant-share approximation. With the share of
women aged 15-45 fixed at f_w (default 0.22 of total population), the
general fertility rate is GFR ~= CBR / f_w. A flat age-specific
fertility schedule across a 30-year reproductive span gives the rough
identity TFR ~= GFR x 30 / 1000. Both are approximations and should be
labelled as such in the manuscript.

We report:
  - cumulative birth deficit per 1{,}000 population (post 1873-1890)
  - implied reduction in TFR
  - implied reduction in CCF for the cohorts of women whose reproductive
    careers intersect the Kulturkampf window
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def cohort_translation(
    panel: pd.DataFrame,
    iv_coef: float,
    outcomes: Sequence[str] = ("cbr",),
    pre_years: tuple[int, int] = (1862, 1871),
    post_years: tuple[int, int] = (1873, 1890),
    high_threshold: float = 75.0,
    low_threshold: float = 25.0,
    f_w: float = 0.22,            # share of pop that is women 15-45
    repro_span: int = 30,         # years of reproductive lifespan
) -> pd.DataFrame:
    """Translate the IV CBR coefficient into TFR/CCF and missing-births terms."""
    pre_a, pre_b = pre_years
    post_a, post_b = post_years
    n_post_years = post_b - post_a + 1

    # Mean cath_share by group (high vs low)
    sub = panel.dropna(subset=["cath_share"]).copy()
    high_mean = sub.loc[sub["cath_share"] > high_threshold, "cath_share"].mean()
    low_mean = sub.loc[sub["cath_share"] < low_threshold, "cath_share"].mean()
    delta_cath = high_mean - low_mean

    # Annual implied effect for a high-vs-low contrast: beta x delta_cath (CBR units)
    annual_cbr_diff = iv_coef * delta_cath  # per 1{,}000 population per year

    # Cumulative missing births per 1{,}000 over the post period
    cumulative_per_1000 = annual_cbr_diff * n_post_years

    # GFR equivalent: CBR / f_w
    annual_gfr_diff = annual_cbr_diff / f_w

    # TFR equivalent: GFR x repro_span / 1000
    tfr_diff = annual_gfr_diff * repro_span / 1000.0

    # CCF: same approximation if cohort spans entire reproductive window
    # within the Kulturkampf period. Otherwise, scale by overlap fraction.
    # A cohort of women aged 15 in 1873 finishes their reproductive career
    # in 1903; the Kulturkampf-rollback period covers 1873-1890 = 18 of
    # 30 years -> overlap ~0.6.
    overlap_frac = min(n_post_years, repro_span) / repro_span
    ccf_diff = tfr_diff * overlap_frac

    rows = [{
        "high_cath_mean": high_mean,
        "low_cath_mean": low_mean,
        "delta_cath": delta_cath,
        "iv_coef_cbr": iv_coef,
        "annual_cbr_diff": annual_cbr_diff,
        "n_post_years": n_post_years,
        "cumulative_per_1000": cumulative_per_1000,
        "annual_gfr_diff": annual_gfr_diff,
        "tfr_diff": tfr_diff,
        "ccf_diff": ccf_diff,
        "f_w": f_w,
        "repro_span": repro_span,
        "overlap_frac": overlap_frac,
    }]
    return pd.DataFrame(rows)
