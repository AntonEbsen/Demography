"""
rollback.py
===========
Extended event study showing both the Kulturkampf enforcement AND the rollback.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from typing import Optional

logger = logging.getLogger(__name__)


def rollback_event_study(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment_var: str = "cath_share",
    ref_year: int = 1872,
    savepath: Optional[str] = None,
):
    """
    Extended event study showing both the Kulturkampf period AND the rollback.

    Key dates to watch on the plot:
        1872-1875: Main Kulturkampf legislation (May Laws in 1873)
        1878: Peace with Vatican begins; Pope Leo XIII elected
        1880-1887: Gradual repeal of Kulturkampf laws ("Milderungsgesetze")
        1887: Final peace with Catholic Church

    If the Kulturkampf had real effects, you'd expect coefficients to:
      - Start near zero (pre-1872)
      - Move away from zero during 1873-1878 (enforcement)
      - Return toward zero during 1880-1887 (rollback)
    """
    df = df.copy()

    cols_needed = ["Code", "Year", outcome, treatment_var]
    sub = df[cols_needed].drop_duplicates(subset=["Code", "Year"]).dropna().copy()
    sub = sub.set_index(["Code", "Year"])

    years = sorted(sub.index.get_level_values("Year").unique())
    interact_years = [y for y in years if y != ref_year]

    for yr in interact_years:
        year_dummy = (sub.index.get_level_values("Year") == yr).astype(float)
        sub[f"treat_x_{yr}"] = year_dummy * sub[treatment_var].values

    interact_cols = [f"treat_x_{yr}" for yr in interact_years]
    exog = list(interact_cols)

    y = sub[outcome]
    X = sub[exog]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    coef_data = []
    for yr in interact_years:
        col = f"treat_x_{yr}"
        beta = res.params[col]
        se = res.std_errors[col]
        coef_data.append({
            "Year": yr, "beta": beta, "se": se,
            "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se,
        })
    coef_data.append({"Year": ref_year, "beta": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
    coefs = pd.DataFrame(coef_data).sort_values("Year").reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvspan(1872, 1878, alpha=0.15, color="#C0392B", label="Kulturkampf enforcement")
    ax.axvspan(1880, 1887, alpha=0.15, color="#2471A3", label="Kulturkampf rollback")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(coefs["Year"], coefs["ci_lo"], coefs["ci_hi"], alpha=0.25, color="#555555")
    ax.plot(coefs["Year"], coefs["beta"], color="#333333", linewidth=2, marker="o", markersize=4)
    ax.scatter([ref_year], [0], color="black", s=80, zorder=5, marker="D", label=f"Reference year ({ref_year})")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"Coefficient on {treatment_var} × Year", fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    pre = coefs[coefs["Year"] < 1872]["beta"].mean()
    enforce = coefs[(coefs["Year"] >= 1873) & (coefs["Year"] <= 1878)]["beta"].mean()
    rollback_mean = coefs[(coefs["Year"] >= 1880) & (coefs["Year"] <= 1887)]["beta"].mean()
    post = coefs[coefs["Year"] >= 1888]["beta"].mean()

    logger.info("Mean coefficients by period:")
    logger.info("  Pre (1862-1871):         %+.4f", pre)
    logger.info("  Enforcement (1873-1878): %+.4f", enforce)
    logger.info("  Rollback (1880-1887):    %+.4f", rollback_mean)
    logger.info("  Post (1888+):            %+.4f", post)

    return {"result": res, "coefs": coefs, "fig": fig}
