"""
trend_and_placebo.py
====================
Trend-adjusted DiD and placebo tests.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def trend_adjusted_did(
    df: pd.DataFrame, outcome: str = "cbr", trend_base_year: int = 1862,
    exclude_war: bool = False, war_years: tuple = (1870, 1871, 1872),
):
    """Estimate a DiD that allows for differential linear pre-trends."""
    logger.info("TREND-ADJUSTED DiD%s", f" (excluding war years {war_years})" if exclude_war else "")

    df = df.copy()
    if exclude_war:
        df = df[~df["Year"].isin(war_years)].copy()

    df["trend"] = df["Year"] - trend_base_year
    df["cath_x_trend"] = df["cath_share"] * df["trend"]

    exog = ["cath_share_x_post", "cath_x_trend", "ln_pop"]
    res = safe_panel_ols(df, outcome, exog)
    res_baseline = safe_panel_ols(df, outcome, ["cath_share_x_post", "ln_pop"])

    b_base = res_baseline.params["cath_share_x_post"]
    b_adj = res.params["cath_share_x_post"]
    p_adj = res.pvalues["cath_share_x_post"]
    g_trend = res.params["cath_x_trend"]

    logger.info("  Baseline b=%.5f, Trend-adjusted b=%.5f (p=%.3f), Trend g=%.5f",
                b_base, b_adj, p_adj, g_trend)

    return {
        "result": res, "result_baseline": res_baseline,
        "coefs": {
            "beta_baseline": b_base, "se_baseline": res_baseline.std_errors["cath_share_x_post"],
            "beta_adjusted": b_adj, "se_adjusted": res.std_errors["cath_share_x_post"],
            "p_adjusted": p_adj, "gamma_trend": g_trend,
            "se_trend": res.std_errors["cath_x_trend"], "p_trend": res.pvalues["cath_x_trend"],
        },
    }


def placebo_test(
    df: pd.DataFrame, outcome: str = "cbr", placebo_years: list = None,
    savepath: str = None,
):
    """Placebo test: pretend the Kulturkampf happened in a different year."""
    if placebo_years is None:
        placebo_years = [1864, 1866, 1868, 1870, 1873, 1876, 1880, 1884]

    logger.info("PLACEBO TEST — FAKE TREATMENT YEARS")

    results = []
    for placebo_year in placebo_years:
        df_pl = df.copy()
        df_pl["post_placebo"] = (df_pl["Year"] >= placebo_year).astype(int)
        df_pl["cath_x_post_placebo"] = df_pl["cath_share"] * df_pl["post_placebo"]
        if df_pl["post_placebo"].var() == 0:
            continue
        try:
            res = safe_panel_ols(df_pl, outcome, ["cath_x_post_placebo", "ln_pop"])
            coef = res.params["cath_x_post_placebo"]
            se = res.std_errors["cath_x_post_placebo"]
            p = res.pvalues["cath_x_post_placebo"]
            is_true = (placebo_year == 1873)
            logger.info("  Placebo year = %d: b = %+.5f (SE = %.5f, p = %.3f)%s",
                        placebo_year, coef, se, p, " <- TRUE" if is_true else "")
            results.append({
                "placebo_year": placebo_year, "coef": coef, "se": se,
                "ci_lo": coef - 1.96*se, "ci_hi": coef + 1.96*se,
                "p_value": p, "is_true": is_true, "n": int(res.nobs),
            })
        except Exception as e:
            logger.warning("  Placebo year = %d: failed (%s)", placebo_year, e)

    res_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in res_df.iterrows():
        color = "#C0392B" if row["is_true"] else "#555555"
        marker = "D" if row["is_true"] else "o"
        ax.errorbar(row["placebo_year"], row["coef"], yerr=1.96*row["se"],
                     fmt=marker, color=color, markersize=8 if row["is_true"] else 6,
                     capsize=4, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(1873, color="#C0392B", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Placebo treatment year", fontsize=11)
    ax.set_ylabel("Coefficient (95% CI)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return {"results_df": res_df, "fig": fig}
