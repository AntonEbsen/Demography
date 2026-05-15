"""
political_mobilization.py
=========================
Catholic political mobilisation as a Kulturkampf outcome.

Galloway's seven Reichstag-election files (1871, 1874, 1878, 1881,
1884, 1887, 1890) provide a 7-cross-section panel of Catholic-party
vote shares (Zentrum and Polen) that spans:

  - 1 pre-Kulturkampf election: 1871 (March)
  - 2 enforcement-period elections: 1874 (Jan), 1878 (Jul)
  - 3 rollback-period elections: 1881 (Oct), 1884 (Oct), 1887 (Feb)
  - 1 post-rollback election: 1890 (Feb)

For high-Catholic counties, mean Zentrum vote share jumps from
38.9% in 1871 to 57.6% in 1874 -- the *first* post-Kulturkampf
election -- peaks at 64.5% in 1881, and remains at 58-64% through
1890. The Kulturkampf produced exactly the political backlash that
Catholic political organisation would predict: rather than weakening
Catholic identity, the legislation forged it into the most powerful
opposition bloc in the Reichstag.

This module estimates the political-mobilisation response formally
via DiD on the stacked election-year panel:

  zentrum_share_{it} = beta (cath_share_i x Post_t)
                       + alpha_i + delta_t + epsilon_{it}

where Post is either binary (1[year >= 1874]) or phase-specific
(enforcement / rollback / post-rollback indicators).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

logger = logging.getLogger(__name__)


# Election-year phase tagging.
PRE_KULTURKAMPF = (1871,)
ENFORCEMENT_ELECTIONS = (1874, 1878)
ROLLBACK_ELECTIONS = (1881, 1884, 1887)
POSTROLLBACK_ELECTIONS = (1890,)


def build_election_did_panel(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the 7-election DiD panel from the analysis panel.

    Pulls the time-varying ``zentrum_share_current`` etc. observed at
    panel rows where ``Year`` equals one of the seven election years.

    Returns
    -------
    pd.DataFrame keyed by (Code, Year) with columns:
        zentrum_share, polen_share, catholic_party_share,
        cath_share, post_kulturkampf, cath_share_x_post,
        and three phase dummies (enforcement, rollback, post_rollback).
    """
    needed = [
        "Code", "Year", "Rb", "cath_share", "high_cath",
        "zentrum_share_current", "polen_share_current",
        "catholic_party_share_current",
    ]
    sub = panel.loc[panel["Year"].isin(
        list(PRE_KULTURKAMPF + ENFORCEMENT_ELECTIONS
             + ROLLBACK_ELECTIONS + POSTROLLBACK_ELECTIONS)
    ), needed].copy()
    sub = sub.dropna(subset=["zentrum_share_current"])

    # Rename time-varying columns to drop the "_current" suffix in the
    # election panel; in this context they are the outcome at year t.
    sub = sub.rename(columns={
        "zentrum_share_current": "zentrum_share",
        "polen_share_current": "polen_share",
        "catholic_party_share_current": "catholic_party_share",
    })

    sub["post_kulturkampf"] = (sub["Year"] >= 1874).astype(int)
    sub["cath_share_x_post"] = sub["cath_share"] * sub["post_kulturkampf"]
    sub["enforcement"] = sub["Year"].isin(ENFORCEMENT_ELECTIONS).astype(int)
    sub["rollback"] = sub["Year"].isin(ROLLBACK_ELECTIONS).astype(int)
    sub["post_rollback"] = sub["Year"].isin(POSTROLLBACK_ELECTIONS).astype(int)
    sub["cath_share_x_enforcement"] = sub["cath_share"] * sub["enforcement"]
    sub["cath_share_x_rollback"] = sub["cath_share"] * sub["rollback"]
    sub["cath_share_x_postrollback"] = sub["cath_share"] * sub["post_rollback"]
    return sub.reset_index(drop=True)


def run_political_mobilization_did(
    panel: pd.DataFrame,
    outcome: str = "zentrum_share",
    phase_specific: bool = False,
) -> dict:
    """
    Estimate the political-mobilisation DiD on the 7-election panel.

    Specification (binary Post, ``phase_specific=False``):

      Y_{it} = beta (cath_share_i x Post_t) + alpha_i + delta_t + e_{it}

    Specification (``phase_specific=True``):

      Y_{it} = beta_E (cath_share_i x Enforcement_t)
             + beta_R (cath_share_i x Rollback_t)
             + beta_P (cath_share_i x PostRollback_t)
             + alpha_i + delta_t + e_{it}

    where Post = 1 for elections from 1874 onward and 1871 is the
    omitted reference. Outcome is one of ``zentrum_share``,
    ``polen_share``, ``catholic_party_share``.

    Returns
    -------
    dict with keys ``result`` (PanelOLSResults), ``summary`` (string),
    and the headline coefficient(s).
    """
    df = build_election_did_panel(panel)
    if outcome not in df.columns:
        raise KeyError(f"Outcome {outcome!r} not in election DiD panel.")

    df = df.dropna(subset=[outcome, "cath_share"])
    df = df.set_index(["Code", "Year"])

    if phase_specific:
        exog_cols = [
            "cath_share_x_enforcement",
            "cath_share_x_rollback",
            "cath_share_x_postrollback",
        ]
    else:
        exog_cols = ["cath_share_x_post"]

    mod = PanelOLS(
        df[outcome], df[exog_cols],
        entity_effects=True, time_effects=True,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    out = {
        "result": res,
        "summary": str(res.summary),
        "n_obs": int(res.nobs),
        "n_counties": int(df.index.get_level_values("Code").nunique()),
    }
    for k in exog_cols:
        out[f"{k}_coef"] = float(res.params[k])
        out[f"{k}_se"] = float(res.std_errors[k])
        out[f"{k}_p"] = float(res.pvalues[k])
    return out


def annual_political_mobilization_table(
    panel: pd.DataFrame,
    outcome: str = "zentrum_share",
) -> pd.DataFrame:
    """
    Group means of the political-mobilisation outcome by election year
    and Catholic-share group. Companion to the formal DiD result --
    shows the raw means that drive the regression coefficient.
    """
    df = build_election_did_panel(panel)
    df["group"] = df["high_cath"].map({0: "Low Cath", 1: "High Cath"})
    return (
        df.groupby(["Year", "group"])[outcome]
        .mean()
        .unstack()
        .round(2)
    )
