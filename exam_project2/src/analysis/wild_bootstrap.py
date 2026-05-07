"""
wild_bootstrap.py
=================
Wild cluster bootstrap for the baseline DiD (Cameron, Gelbach & Miller 2008).

Conventional cluster-robust SEs become unreliable when the number of
clusters is small (a common rule of thumb is < 50). The Polish-province
sub-sample has only 24 counties, so the wild bootstrap is the canonical
fix. We implement the "wild bootstrap" with restricted residuals
(imposing the null beta = 0):

  1. Demean Y, D by entity + year FE (FWL); compute beta_hat and the
     restricted residuals u_i = eps_Y_i (the model's prediction under
     the null is zero in the demeaned space).
  2. For B bootstrap draws:
       - draw Rademacher weights w_g in {-1, +1}, one per cluster.
       - construct pseudo-outcome y_i^star = w_g(i) * u_i.
       - re-estimate beta on (eps_D, y_star).
  3. Two-sided p-value = share of |beta_star| >= |beta_hat|.

This is the basic Rademacher wild bootstrap. Mammen weights are an
alternative; we stick with Rademacher because it has better small-sample
properties for hypothesis testing under the null.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.permutation_inference import _fast_two_way_demean

logger = logging.getLogger(__name__)


def wild_cluster_bootstrap(
    panel: pd.DataFrame,
    outcome: str = "cbr",
    sample_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    n_boot: int = 999,
    seed: int = 42,
) -> dict:
    """
    Wild cluster bootstrap p-value for the cath_share x post coefficient.

    Parameters
    ----------
    sample_filter : callable, optional
        Boolean filter applied to ``panel`` before running. Use e.g.
        ``lambda d: d['Rb'].isin(['POS', 'BRO'])`` for the Polish-province
        sub-sample.
    """
    rng = np.random.default_rng(seed)

    sub = panel if sample_filter is None else panel.loc[sample_filter(panel)].copy()
    sub = (
        sub[["Code", "Year", outcome, "cath_share", "post_kulturkampf"]]
        .dropna()
        .sort_values(["Code", "Year"])
        .reset_index(drop=True)
    )

    sub["D"] = sub["cath_share"] * sub["post_kulturkampf"]
    codes = sub["Code"].values
    years = sub["Year"].values
    unique_codes, code_inv = np.unique(codes, return_inverse=True)
    unique_years, year_inv = np.unique(years, return_inverse=True)
    n_clusters = len(unique_codes)

    eps_Y = _fast_two_way_demean(
        sub[outcome].values, code_inv, year_inv, n_clusters, len(unique_years)
    )
    eps_D = _fast_two_way_demean(
        sub["D"].values, code_inv, year_inv, n_clusters, len(unique_years)
    )

    denom = float((eps_D ** 2).sum())
    if denom <= 0:
        raise ValueError("Treatment has no within variation in this sub-sample.")
    beta_obs = float((eps_D * eps_Y).sum() / denom)

    # Under H0 (beta = 0), the restricted residuals are just eps_Y.
    u_restricted = eps_Y

    boot_betas = np.zeros(n_boot)
    for b in range(n_boot):
        weights = rng.choice([-1.0, 1.0], size=n_clusters)
        w_obs = weights[code_inv]
        y_star = w_obs * u_restricted
        boot_betas[b] = float((eps_D * y_star).sum() / denom)

    p = float(np.mean(np.abs(boot_betas) >= abs(beta_obs)))
    return {
        "beta_obs": beta_obs,
        "n_clusters": n_clusters,
        "n_obs": int(len(sub)),
        "n_boot": n_boot,
        "p_value": p,
        "boot_distribution_std": float(boot_betas.std()),
        "boot_ci_2_5": float(np.quantile(boot_betas, 0.025)),
        "boot_ci_97_5": float(np.quantile(boot_betas, 0.975)),
    }
