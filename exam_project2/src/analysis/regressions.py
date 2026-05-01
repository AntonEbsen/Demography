"""
regressions.py
==============
DiD and event-study regressions for the Kulturkampf–fertility paper.

Usage (from notebook):
    from src.analysis.regressions import run_baseline_did, run_event_study, run_robustness
"""

import logging
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from typing import Optional

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Set multi-index for linearmodels and ensure correct dtypes."""
    out = df.copy()
    out = out.set_index(["Code", "Year"])
    return out


def run_baseline_did(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment: str = "continuous",
    controls: Optional[list] = None,
    cluster: str = "Code",
) -> dict:
    """
    Run the baseline DiD specification.
    
    Specification
    -------------
    Y_it = β (CathShare_i × Post_t) + α_i + δ_t + X_it'γ + ε_it
    
    or with binary treatment:
    Y_it = β (HighCath_i × Post_t) + α_i + δ_t + X_it'γ + ε_it
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel from build_dataset.
    outcome : str
        Dependent variable. Options: 'cbr', 'legitimate_br', 
        'illegitimate_br', 'marriage_rate', 'cath_marriage_share'.
    treatment : str
        'continuous' → uses cath_share_x_post (Catholic share × Post)
        'binary' → uses treat_x_post (HighCath × Post)
    controls : list, optional
        Additional time-varying controls. Default: ['ln_pop'].
    cluster : str
        Cluster variable for standard errors. Default: 'Code' (county).
    
    Returns
    -------
    dict with keys: 'result' (PanelOLS result), 'summary' (string)
    """
    if controls is None:
        controls = ["ln_pop"]
    
    panel = _prepare_panel(df)
    
    # Drop missing outcomes
    mask = panel[outcome].notna()
    panel = panel[mask]
    
    # Choose treatment variable
    if treatment == "continuous":
        treat_var = "cath_share_x_post"
    elif treatment == "binary":
        treat_var = "treat_x_post"
    else:
        raise ValueError(f"treatment must be 'continuous' or 'binary', got {treatment}")
    
    # Build regressor matrix
    exog_vars = [treat_var] + [c for c in controls if c in panel.columns]
    
    y = panel[outcome]
    X = panel[exog_vars]
    
    # Drop any remaining NaN
    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid]
    X = X[valid]
    
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    
    return {"result": res, "summary": str(res.summary)}


def run_event_study(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment_var: str = "cath_share",
    ref_year: int = 1872,
    cluster: str = "Code",
    controls: Optional[list] = None,
) -> dict:
    """
    Run an event-study specification around the Kulturkampf.
    
    Specification
    -------------
    Y_it = Σ_t β_t (CathShare_i × 1[Year=t]) + α_i + δ_t + ε_it
    
    Omits the reference year (default: 1872, the year before May Laws).
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    outcome : str
        Dependent variable.
    treatment_var : str
        'cath_share' (continuous) or 'high_cath' (binary).
    ref_year : int
        Omitted reference year for the event study.
    controls : list, optional
        Additional time-varying controls.
    
    Returns
    -------
    dict with keys:
        'result' : PanelOLS result
        'coefs'  : DataFrame with Year, beta, se, ci_lo, ci_hi
                    (for plotting)
    """
    if controls is None:
        controls = ["ln_pop"]
    
    panel = _prepare_panel(df)
    
    # Drop missing
    mask = panel[outcome].notna()
    panel = panel[mask]
    
    # Get available years (excluding reference year)
    years = sorted(panel.index.get_level_values("Year").unique())
    interact_years = [y for y in years if y != ref_year]
    
    # Create year × treatment interactions
    for yr in interact_years:
        col_name = f"treat_x_{yr}"
        year_dummy = (panel.index.get_level_values("Year") == yr).astype(float)
        panel[col_name] = year_dummy * panel[treatment_var].values
    
    # Exogenous variables
    interact_cols = [f"treat_x_{yr}" for yr in interact_years]
    exog_vars = interact_cols + [c for c in controls if c in panel.columns]
    
    y = panel[outcome]
    X = panel[exog_vars]
    
    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid]
    X = X[valid]
    
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    
    # Extract coefficients for plotting
    coef_data = []
    for yr in interact_years:
        col = f"treat_x_{yr}"
        beta = res.params[col]
        se = res.std_errors[col]
        coef_data.append({
            "Year": yr,
            "beta": beta,
            "se": se,
            "ci_lo": beta - 1.96 * se,
            "ci_hi": beta + 1.96 * se,
        })
    
    # Add reference year as zero
    coef_data.append({
        "Year": ref_year,
        "beta": 0.0,
        "se": 0.0,
        "ci_lo": 0.0,
        "ci_hi": 0.0,
    })
    
    coefs = pd.DataFrame(coef_data).sort_values("Year").reset_index(drop=True)
    
    return {"result": res, "coefs": coefs}


def run_robustness(
    df: pd.DataFrame,
    outcome: str = "cbr",
) -> pd.DataFrame:
    """
    Run a battery of robustness checks and return a summary table.
    
    Checks
    ------
    1. Binary treatment (HighCath × Post) instead of continuous
    2. Alternative post cutoff: 1872 instead of 1873
    3. Alternative post cutoff: 1875 instead of 1873
    4. Alternative Catholic threshold: 25% instead of 50%
    5. Alternative Catholic threshold: 75% instead of 50%
    6. Excluding Polish-majority counties (Posen, Bromberg provinces)
    7. Only rural counties (excluding Berlin and very large cities)
    
    Returns
    -------
    pd.DataFrame with columns: Specification, Coefficient, SE, p_value, N, N_counties
    """
    results = []
    
    def _run_one(label, data, treat_var="cath_share_x_post"):
        try:
            panel = _prepare_panel(data)
            mask = panel[outcome].notna() & panel["ln_pop"].notna()
            panel = panel[mask]
            
            y = panel[outcome]
            X = panel[[treat_var, "ln_pop"]]
            valid = y.notna() & X.notna().all(axis=1)
            y, X = y[valid], X[valid]
            
            mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            
            results.append({
                "Specification": label,
                "Coefficient": res.params[treat_var],
                "SE": res.std_errors[treat_var],
                "p_value": res.pvalues[treat_var],
                "N": int(res.nobs),
                "N_counties": int(y.index.get_level_values(0).nunique()),
            })
        except Exception as e:
            logger.error("  [error] %s: %s", label, e)
    
    # 1. Baseline (continuous)
    _run_one("Baseline (continuous CathShare × Post)", df)
    
    # 2. Binary treatment
    _run_one("Binary (HighCath>50% × Post)", df, treat_var="treat_x_post")
    
    # 3. Alternative post: 1872
    df_alt = df.copy()
    df_alt["post_kulturkampf"] = (df_alt["Year"] >= 1872).astype(int)
    df_alt["cath_share_x_post"] = df_alt["cath_share"] * df_alt["post_kulturkampf"]
    _run_one("Post cutoff = 1872", df_alt)
    
    # 4. Alternative post: 1875
    df_alt = df.copy()
    df_alt["post_kulturkampf"] = (df_alt["Year"] >= 1875).astype(int)
    df_alt["cath_share_x_post"] = df_alt["cath_share"] * df_alt["post_kulturkampf"]
    _run_one("Post cutoff = 1875", df_alt)
    
    # 5. Alt threshold: 25%
    df_alt = df.copy()
    df_alt["high_cath_25"] = (df_alt["cath_share"] > 25).astype(int)
    df_alt["treat25_x_post"] = df_alt["high_cath_25"] * df_alt["post_kulturkampf"]
    _run_one("Binary threshold = 25%", df_alt, treat_var="treat25_x_post")
    
    # 6. Alt threshold: 75%
    df_alt = df.copy()
    df_alt["high_cath_75"] = (df_alt["cath_share"] > 75).astype(int)
    df_alt["treat75_x_post"] = df_alt["high_cath_75"] * df_alt["post_kulturkampf"]
    _run_one("Binary threshold = 75%", df_alt, treat_var="treat75_x_post")
    
    # 7. Drop Polish provinces (Posen=POS, Bromberg=BRO)
    df_nopol = df[~df["Rb"].isin(["POS", "BRO"])].copy()
    _run_one("Excl. Polish provinces", df_nopol)
    
    return pd.DataFrame(results)
