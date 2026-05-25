"""
magnitudes.py
=============
Translate the IV coefficient on $\\mathrm{CathShare} \\times \\mathrm{Post}$
into economically interpretable magnitudes:

1. Mechanical effect of moving a county from low-Catholic ($\\le$25%) to
   high-Catholic ($\\ge$75%): $\\beta_{IV} \\times \\Delta\\mathrm{cath\\_share}$.
2. Observed differential change in the outcome between high- and
   low-Catholic counties (post minus pre).
3. Share of the observed differential attributable to the Kulturkampf:
   ``IV-implied / observed differential``.

Run as a script to (re)generate ``outputs/tables/magnitudes.tex``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.analysis.regressions import run_iv_did

logger = logging.getLogger(__name__)


def magnitude_decomposition(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
    pre_years: tuple[int, int] = (1862, 1871),
    post_years: tuple[int, int] = (1880, 1889),
    high_cath_threshold: float = 75.0,
    low_cath_threshold: float = 25.0,
) -> pd.DataFrame:
    """Return a DataFrame with one row per outcome summarising magnitudes."""
    pre_a, pre_b = pre_years
    post_a, post_b = post_years
    sub = panel[panel["Year"].between(pre_a, pre_b) | panel["Year"].between(post_a, post_b)].copy()
    sub["period"] = ((sub["Year"] >= post_a) & (sub["Year"] <= post_b)).map(
        {True: "post", False: "pre"}
    )

    # Group by Catholic-share extremes (cleaner contrast than median split).
    sub["cath_group"] = pd.cut(
        sub["cath_share"],
        bins=[-0.001, low_cath_threshold, high_cath_threshold, 100.001],
        labels=["low", "mid", "high"],
    )

    rows = []
    for outcome in outcomes:
        iv = run_iv_did(panel, outcome=outcome, instrument="kmwittenberg")
        beta = iv["iv_coef"]

        means = sub.groupby(["cath_group", "period"], observed=False)[outcome].mean().unstack("period")
        delta_high = means.loc["high", "post"] - means.loc["high", "pre"]
        delta_low = means.loc["low", "post"] - means.loc["low", "pre"]
        observed_gap = delta_high - delta_low

        cath_high_mean = sub[sub["cath_group"] == "high"]["cath_share"].mean()
        cath_low_mean = sub[sub["cath_group"] == "low"]["cath_share"].mean()
        delta_cath = cath_high_mean - cath_low_mean
        iv_implied = beta * delta_cath

        share = (
            iv_implied / observed_gap if abs(observed_gap) > 1e-6 else float("nan")
        )

        rows.append({
            "outcome": outcome,
            "beta_iv": beta,
            "delta_cath_share": delta_cath,
            "iv_implied": iv_implied,
            "delta_high": delta_high,
            "delta_low": delta_low,
            "observed_gap": observed_gap,
            "share_explained": share,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    panel = pd.read_parquet(
        Path(__file__).resolve().parent.parent.parent
        / "data" / "processed" / "analysis_panel.parquet"
    )
    print(magnitude_decomposition(panel).to_string(index=False))
