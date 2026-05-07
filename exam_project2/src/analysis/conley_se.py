"""
conley_se.py
============
Conley (1999) spatial HAC standard errors for the baseline DiD.

Conventional cluster-robust SEs at the county level allow arbitrary
correlation *within* a county over time but assume independence *across*
counties. In a setting where counties near each other share unobserved
demographic shocks (e.g. epidemics, regional grain shocks), this is too
optimistic. Conley HAC corrects for spatial correlation up to a cutoff
distance, with weights from a Bartlett kernel.

Specification implemented (panel version, e.g. Hsiang 2010):

    V_Conley = (X'X)^{-1} [Σ_t Σ_{i,j} K(d_ij) X_it u_it u_jt X_jt'] (X'X)^{-1}

where
    K(d) = max(1 - d / H, 0)        # Bartlett kernel
    H    = cutoff distance (km), default 200

We do *not* additionally adjust for serial correlation within county; that
is what entity FE + year FE already handle here. The resulting SEs are
therefore directly comparable to the entity-clustered SEs reported as the
default — pick the larger of the two for inference.

Public API:
    spatial_did_se(panel, outcome="cbr", cutoff_km=200, controls=("ln_pop",))
        -> dict with coef, conley_se, cluster_se, n
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.centroids import load_centroids

logger = logging.getLogger(__name__)


def _two_way_demean(
    df: pd.DataFrame,
    cols: Sequence[str],
    entity: str = "Code",
    time: str = "Year",
    max_iter: int = 50,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Iteratively subtract entity and time means until convergence."""
    out = df[list(cols) + [entity, time]].copy()
    for _ in range(max_iter):
        before = out[cols].values.copy()
        out[list(cols)] = (
            out.groupby(entity, group_keys=False)[list(cols)]
            .transform(lambda s: s - s.mean())
        )
        out[list(cols)] = (
            out.groupby(time, group_keys=False)[list(cols)]
            .transform(lambda s: s - s.mean())
        )
        if np.max(np.abs(out[list(cols)].values - before)) < tol:
            break
    return out[list(cols) + [entity, time]]


def spatial_did_se(
    panel: pd.DataFrame,
    outcome: str = "cbr",
    cutoff_km: float = 200.0,
    controls: Sequence[str] = ("ln_pop",),
    treatment: str = "cath_share_x_post",
) -> dict:
    """Run the baseline DiD and return cluster + Conley spatial HAC SEs."""
    centroids = load_centroids()

    needed = ["Code", "Year", outcome, treatment] + list(controls)
    sub = panel[needed].dropna().merge(centroids, on="Code", how="inner")
    if len(sub) == 0:
        raise ValueError("No rows left after centroid merge — check coverage.")

    # Two-way within-transformation for entity + year FE
    rhs = [treatment] + list(controls)
    demean_cols = [outcome] + rhs
    dem = _two_way_demean(sub, demean_cols)

    y = dem[outcome].values
    X = dem[rhs].values  # k = 1 + len(controls)

    # OLS on demeaned data — coefficients identical to PanelOLS with TWFE
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta

    # Reattach (Code, Year, x_km, y_km) for spatial weighting
    work = pd.DataFrame({
        "Code": sub["Code"].values,
        "Year": sub["Year"].values,
        "x_km": sub["x_km"].values,
        "y_km": sub["y_km"].values,
        "u": u,
    })
    work[rhs] = X

    # Build the Conley meat matrix year-by-year. Spatial weights are
    # time-invariant (centroids don't move) so we precompute once.
    coords = centroids[["x_km", "y_km"]].values
    code_index = {c: i for i, c in enumerate(centroids["Code"].values)}
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    K_full = np.maximum(1.0 - dist / cutoff_km, 0.0)

    k = X.shape[1]
    meat = np.zeros((k, k))
    for t, yr_grp in work.groupby("Year"):
        idx = np.array([code_index[c] for c in yr_grp["Code"].values])
        K_t = K_full[np.ix_(idx, idx)]
        ut = yr_grp["u"].values
        Xt = yr_grp[rhs].values
        weighted = K_t * np.outer(ut, ut)
        meat += Xt.T @ weighted @ Xt

    V_conley = XtX_inv @ meat @ XtX_inv
    conley_se = np.sqrt(np.diag(V_conley))

    # Cluster-robust SE (county-clustered, no spatial correction) for
    # side-by-side comparison.
    meat_cluster = np.zeros((k, k))
    for c, grp in work.groupby("Code"):
        u_c = grp["u"].values
        X_c = grp[rhs].values
        s_c = X_c.T @ u_c
        meat_cluster += np.outer(s_c, s_c)
    V_cluster = XtX_inv @ meat_cluster @ XtX_inv
    cluster_se = np.sqrt(np.diag(V_cluster))

    return {
        "coef": dict(zip(rhs, beta.tolist())),
        "conley_se": dict(zip(rhs, conley_se.tolist())),
        "cluster_se": dict(zip(rhs, cluster_se.tolist())),
        "n": int(len(sub)),
        "cutoff_km": cutoff_km,
    }
