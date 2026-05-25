"""
variance_decomposition.py
=========================
Variance decomposition of CBR (and other outcomes) into the share
attributable to county fixed effects, year fixed effects, and the
treatment interaction. Reports R^2 of nested specifications:

    R^2_county   : Y ~ county FE
    R^2_year     : Y ~ year FE
    R^2_county_year : Y ~ county FE + year FE
    R^2_full     : Y ~ county FE + year FE + cath_share x post

The marginal contribution of the treatment is
``R^2_full - R^2_county_year``.

For balanced panels and non-collinear regressors,
``R^2_county`` + ``R^2_year`` need not equal ``R^2_county_year``; the
difference reflects the orthogonality of the two sets of FE.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - y_hat) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit_demeaning(y: np.ndarray, code_inv: np.ndarray, year_inv: np.ndarray,
                   include_county: bool, include_year: bool) -> np.ndarray:
    """Fit the FE-only model (county, year, or both) and return predicted Y."""
    n_codes = code_inv.max() + 1 if len(code_inv) else 0
    n_years = year_inv.max() + 1 if len(year_inv) else 0
    code_count = np.bincount(code_inv, minlength=n_codes).astype(float)
    year_count = np.bincount(year_inv, minlength=n_years).astype(float)

    # Iterate until convergence (within-transformation)
    resid = y.astype(float).copy()
    for _ in range(20):
        if include_county:
            cs = np.bincount(code_inv, weights=resid, minlength=n_codes)
            resid = resid - (cs / code_count)[code_inv]
        if include_year:
            ys = np.bincount(year_inv, weights=resid, minlength=n_years)
            resid = resid - (ys / year_count)[year_inv]
        if not (include_county or include_year):
            break
    y_hat = y - resid
    return y_hat


def variance_decomposition(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
) -> pd.DataFrame:
    """Return one row per outcome with R^2 of nested specifications."""
    rows = []
    for outcome in outcomes:
        sub = (
            panel[["Code", "Year", outcome, "cath_share_x_post"]]
            .dropna()
            .reset_index(drop=True)
        )
        y = sub[outcome].values
        codes = sub["Code"].values
        years = sub["Year"].values
        _, code_inv = np.unique(codes, return_inverse=True)
        _, year_inv = np.unique(years, return_inverse=True)

        y_county = _fit_demeaning(y, code_inv, year_inv, True, False)
        y_year = _fit_demeaning(y, code_inv, year_inv, False, True)
        y_cy = _fit_demeaning(y, code_inv, year_inv, True, True)

        # Add the treatment regressor on top of county+year demeaning
        D = sub["cath_share_x_post"].values
        eps_y = y - y_cy
        D_demeaned = _fit_demeaning(D, code_inv, year_inv, True, True)
        eps_D = D - D_demeaned
        denom = float((eps_D ** 2).sum())
        beta = float((eps_D * eps_y).sum() / denom) if denom > 0 else 0.0
        y_full = y_cy + beta * eps_D

        rows.append({
            "outcome": outcome,
            "r2_county": _r2(y, y_county),
            "r2_year": _r2(y, y_year),
            "r2_county_year": _r2(y, y_cy),
            "r2_full": _r2(y, y_full),
            "marginal_treatment": _r2(y, y_full) - _r2(y, y_cy),
        })
    return pd.DataFrame(rows)
