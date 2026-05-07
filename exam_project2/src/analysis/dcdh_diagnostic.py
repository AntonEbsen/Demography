"""
dcdh_diagnostic.py
==================
de Chaisemartin & D'Haultfoeuille (2020, 2024) negative-weights diagnostic
for two-way fixed-effects regression with continuous treatment intensity.

The TWFE coefficient on $D_{it} = \\mathrm{CathShare}_i \\times \\mathrm{Post}_t$
in the regression

    Y_{it} = \\beta D_{it} + \\alpha_i + \\delta_t + \\varepsilon_{it}

is a weighted average of unit-level effects, with weights

    w_{it} = (D_{it} - \\bar D_i - \\bar D_t + \\bar D) /
             \\sum_{j,s} (D_{js} - \\bar D_j - \\bar D_s + \\bar D)^2

(de Chaisemartin & D'Haultfoeuille, AER 2020). With *heterogeneous*
treatment effects, some weights are negative, so $\\beta$ can have
the wrong sign even when every unit-level effect has the same sign.

Reported:
  - share of negative weights
  - sum of positive vs negative weights
  - "underidentification" statistic: ratio sum_negative / sum_positive
    (closer to 0 = more reassuring; large = robust dCDH/BJS estimator advised)

Run as a script to (re)generate the diagnostic table::

    python -m src.analysis.dcdh_diagnostic
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def dcdh_weights(
    panel: pd.DataFrame,
    treatment: str = "cath_share_x_post",
    entity: str = "Code",
    time: str = "Year",
) -> pd.DataFrame:
    """
    Return ``panel`` with a column ``dcdh_w`` of implicit weights on each
    observation's unit-time treatment effect in the TWFE regression.

    Derivation. For Y = beta D + entity FE + time FE + e with heterogeneous
    treatment effect tau_{it}, the OLS estimate decomposes as
        beta = sum_{it} w_{it} * tau_{it}
    with
        w_{it} = D_{it} * tilde D_{it} / sum_{js} tilde D_{js}^2
    where tilde D is the two-way within-transformed treatment. Observations
    with tilde D_{it} < 0 carry negative weight; if many do, the TWFE
    coefficient can have a different sign than the average treatment effect.
    """
    df = panel.dropna(subset=[treatment, entity, time]).copy()
    D = df[treatment].astype(float).values
    Di_bar = df.groupby(entity)[treatment].transform("mean").values
    Dt_bar = df.groupby(time)[treatment].transform("mean").values
    D_bar = float(np.mean(D))
    within = D - Di_bar - Dt_bar + D_bar
    norm = float((within ** 2).sum())
    if norm <= 0:
        df["dcdh_w"] = 0.0
    else:
        df["dcdh_w"] = (D * within) / norm
    return df


def diagnostic(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    treatment: str = "cath_share_x_post",
) -> pd.DataFrame:
    """Compute the negative-weight diagnostic for the baseline DiD."""
    weighted = dcdh_weights(panel, treatment=treatment)
    w = weighted["dcdh_w"].values
    pos = w > 0
    neg = w < 0
    sum_pos = float(w[pos].sum())
    sum_neg = float(w[neg].sum())  # negative number

    rows = [{
        "n_total": int(len(w)),
        "n_negative": int(neg.sum()),
        "share_negative": float(neg.mean()),
        "sum_pos_weights": sum_pos,
        "sum_neg_weights": sum_neg,
        "ratio_neg_pos": float(abs(sum_neg) / sum_pos) if sum_pos > 0 else float("nan"),
    }]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    panel = pd.read_parquet(
        Path(__file__).resolve().parent.parent.parent
        / "data" / "processed" / "analysis_panel.parquet"
    )
    print(diagnostic(panel).to_string(index=False))
