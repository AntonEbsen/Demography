"""
permutation_inference.py
========================
Fisher-style randomization inference for the baseline DiD.

Permutes the cross-sectional cath_share assignment across counties B times,
re-estimates the TWFE coefficient on each draw, and computes the exact
two-sided p-value as the share of permutation coefficients at least as
extreme as the observed coefficient. Avoids reliance on asymptotic
cluster-robust SEs (which are wobbly in finite Catholic-county samples
and questionable when pre-trends are non-zero).

For speed, we work in the within-demeaned space: the OLS coefficient on
a single regressor X after entity + year FE absorption is

    beta = sum(eps_X * eps_Y) / sum(eps_X^2),

where eps denotes the within-transformed series. Each permutation only
needs to reconstruct eps_X for the shuffled treatment, which is O(N).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fast_two_way_demean(
    x: np.ndarray, code_inv: np.ndarray, year_inv: np.ndarray,
    n_codes: int, n_years: int, n_iter: int = 5,
) -> np.ndarray:
    """Iterative within-transformation by entity and time (numpy-only)."""
    code_count = np.bincount(code_inv, minlength=n_codes).astype(float)
    year_count = np.bincount(year_inv, minlength=n_years).astype(float)
    out = x.astype(float).copy()
    for _ in range(n_iter):
        code_sum = np.bincount(code_inv, weights=out, minlength=n_codes)
        out = out - (code_sum / code_count)[code_inv]
        year_sum = np.bincount(year_inv, weights=out, minlength=n_years)
        out = out - (year_sum / year_count)[year_inv]
    return out


def permutation_p_value(
    panel: pd.DataFrame,
    outcome: str = "cbr",
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Two-sided permutation p-value for cath_share x post on ``outcome``."""
    rng = np.random.default_rng(seed)

    df = (
        panel[["Code", "Year", outcome, "cath_share", "post_kulturkampf"]]
        .dropna()
        .sort_values(["Code", "Year"])
        .reset_index(drop=True)
    )
    n = len(df)
    codes = df["Code"].values
    years = df["Year"].values
    post = df["post_kulturkampf"].values.astype(float)
    y = df[outcome].values

    unique_codes, code_inv = np.unique(codes, return_inverse=True)
    unique_years, year_inv = np.unique(years, return_inverse=True)

    eps_Y = _fast_two_way_demean(y, code_inv, year_inv,
                                 len(unique_codes), len(unique_years))

    # Build the cross-sectional cath_share vector keyed by county order
    cath_by_code = (
        df.drop_duplicates("Code")[["Code", "cath_share"]]
        .set_index("Code")["cath_share"]
        .reindex(unique_codes)
        .values
    )

    # cath_share for each obs (broadcast via code_inv)
    cath_obs = cath_by_code[code_inv]

    D_obs = cath_obs * post
    eps_D_obs = _fast_two_way_demean(D_obs, code_inv, year_inv,
                                     len(unique_codes), len(unique_years))
    denom_obs = float((eps_D_obs ** 2).sum())
    if denom_obs <= 0:
        raise ValueError("Treatment has no within variation.")
    beta_obs = float((eps_D_obs * eps_Y).sum() / denom_obs)

    perm_betas = np.zeros(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(cath_by_code)
        D_perm = shuffled[code_inv] * post
        eps_Dp = _fast_two_way_demean(D_perm, code_inv, year_inv,
                                      len(unique_codes), len(unique_years))
        denom_p = float((eps_Dp ** 2).sum())
        perm_betas[k] = (
            float((eps_Dp * eps_Y).sum() / denom_p) if denom_p > 0 else 0.0
        )

    p_two_sided = float(np.mean(np.abs(perm_betas) >= abs(beta_obs)))

    return {
        "beta_obs": beta_obs,
        "n_permutations": n_permutations,
        "p_value": p_two_sided,
        "perm_distribution_mean": float(perm_betas.mean()),
        "perm_distribution_std": float(perm_betas.std()),
    }
