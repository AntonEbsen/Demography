"""
honest_did.py
=============
Honest DiD inference under non-parallel pre-trends (Rambachan & Roth 2023).

This is a *simplified* implementation of the smoothness-bound (M-bound)
restriction. The full reference is the ``HonestDiD`` R package; the
Python port ``honestdid`` is unmaintained, so we hand-roll the core
construction.

Setup. Let beta_t denote the event-study coefficient on
$\\mathrm{CathShare} \\times \\mathbb{1}[t]$ relative to a pre-treatment
reference year. We observe beta_t = tau_t + delta_t, where tau_t is the
treatment effect (zero for t < 0 by assumption) and delta_t is the
unobserved differential trend. The smoothness restriction is

    |delta_t - delta_{t-1}|  <=  M * max_{s<=0} |delta_s - delta_{s-1}|

i.e. the post-period trend's first-difference is bounded by M times the
largest pre-period first-difference. M = 0 is the parallel-trends
assumption; M = 1 says "the post-period trend can vary at most as much
between adjacent years as the worst pre-period change".

Worst-case bias accumulates over the post-period horizon:

    B(M, h)  =  M * max_{s<=0} |Delta beta_s| * h

where h is the number of periods past treatment for the coefficient of
interest. The honest 1 - alpha confidence interval for tau is

    CI  =  [tau_hat - B(M, h) - z_alpha * SE(tau_hat),
            tau_hat + B(M, h) + z_alpha * SE(tau_hat)]

The *breakdown M* is the smallest M for which 0 is contained in the CI.
A large breakdown M means the result is robust to substantial pre-trend
extrapolation; a small one means the result is fragile.

We summarise both for (i) the average post-period treatment effect and
(ii) the first post-period coefficient (most directly identified).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.analysis.regressions import run_event_study

logger = logging.getLogger(__name__)


@dataclass
class HonestDiDResult:
    target: str
    tau_hat: float
    se: float
    max_pre_diff: float
    breakdown_m: float
    table: pd.DataFrame  # one row per M, with CI bounds


def _ci_for_m(tau_hat: float, se: float, max_pre_diff: float, M: float, h: int,
              z: float = 1.96) -> tuple[float, float, float]:
    bias = M * max_pre_diff * h
    return tau_hat - bias - z * se, tau_hat + bias + z * se, bias


def _breakdown_m(tau_hat: float, se: float, max_pre_diff: float, h: int,
                 z: float = 1.96) -> float:
    """Smallest M at which the honest CI just contains zero."""
    if max_pre_diff <= 0:
        return float("inf")
    if abs(tau_hat) <= z * se:
        return 0.0  # already non-significant under parallel trends
    return (abs(tau_hat) - z * se) / (max_pre_diff * h)


def honest_did_bounds(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment_var: str = "cath_share",
    ref_year: int = 1872,
    pre_cutoff: int = 1872,
    M_values: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    target: str = "average",
) -> HonestDiDResult:
    """
    Compute Honest DiD bounds for the post-period treatment effect.

    Parameters
    ----------
    target : "average" or "first"
        "average": tau_hat is the average of post-period event-study coefs.
        "first":   tau_hat is the first post-period coef (e.g. 1873).
    """
    es = run_event_study(df, outcome=outcome, treatment_var=treatment_var, ref_year=ref_year)
    coefs = es["coefs"].sort_values("Year").reset_index(drop=True)

    pre = coefs[coefs["Year"] < pre_cutoff]
    post = coefs[coefs["Year"] >= pre_cutoff]
    if pre.empty or post.empty:
        raise ValueError("Need both pre-period and post-period coefficients.")

    # Largest |delta_t - delta_{t-1}| in the pre period (using betas as
    # estimates of the differential trend).
    pre_diffs = np.abs(np.diff(pre["beta"].values))
    max_pre_diff = float(pre_diffs.max())

    # Target estimand
    if target == "average":
        tau_hat = float(post["beta"].mean())
        # SE of the mean of the post coefficients
        se = float(post["se"].mean() / np.sqrt(len(post)))
        # Use the median post-period horizon for bias accumulation
        h = int((post["Year"] - pre_cutoff + 1).median())
    elif target == "first":
        first_post = post.iloc[0]
        tau_hat = float(first_post["beta"])
        se = float(first_post["se"])
        h = 1
    else:
        raise ValueError(f"target must be 'average' or 'first', got {target!r}")

    rows = []
    for M in M_values:
        lo, hi, bias = _ci_for_m(tau_hat, se, max_pre_diff, M, h)
        rows.append({
            "M": M,
            "bias": bias,
            "ci_lo": lo,
            "ci_hi": hi,
            "contains_zero": lo <= 0 <= hi,
        })

    bd = _breakdown_m(tau_hat, se, max_pre_diff, h)

    return HonestDiDResult(
        target=target,
        tau_hat=tau_hat,
        se=se,
        max_pre_diff=max_pre_diff,
        breakdown_m=bd,
        table=pd.DataFrame(rows),
    )
