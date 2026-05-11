"""
channels.py
============
Mechanism channel analyses: illegitimacy and infant mortality.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def illegitimacy_analysis(df: pd.DataFrame):
    """
    Did the Kulturkampf affect illegitimacy rates in Catholic counties?

    Logic: Catholic parish oversight of sexuality and marriage weakened
    under the Kulturkampf. Civil marriage replaced church marriage in 1875.
    If institutional oversight mattered for enforcing marital norms,
    illegitimate births should rise in Catholic counties.
    """
    df = df.copy()

    logger.info("=" * 60)
    logger.info("ILLEGITIMACY CHANNEL")
    logger.info("=" * 60)

    logger.info("Mean illegitimacy ratio (%% of births):")
    for period_label, mask in [
        ("Pre-Kulturkampf (1875-1878)", (df["Year"] >= 1875) & (df["Year"] < 1879)),
        ("Rollback (1880-1887)", (df["Year"] >= 1880) & (df["Year"] <= 1887)),
    ]:
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        by_cath = sub.groupby("high_cath")["illegitimacy_ratio"].mean()
        logger.info("  %s:", period_label)
        logger.info("    Low Catholic (<=50%%):  %.2f%%", by_cath.get(0, np.nan))
        logger.info("    High Catholic (>50%%): %.2f%%", by_cath.get(1, np.nan))

    logger.info("DiD: Illegitimacy ratio ~ CathShare × Post")
    res = safe_panel_ols(df, "illegitimacy_ratio", ["cath_share_x_post", "ln_pop"])
    coef = res.params["cath_share_x_post"]
    se = res.std_errors["cath_share_x_post"]
    pval = res.pvalues["cath_share_x_post"]
    logger.info("  β = %.4f (SE = %.4f, p = %.3f)", coef, se, pval)
    logger.info("  N = %d", int(res.nobs))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, "#C0392B"),
        ("Low Catholic (≤50%)", df["high_cath"] == 0, "#2471A3"),
    ]:
        trend = df[mask].groupby("Year")["illegitimacy_ratio"].mean()
        ax.plot(trend.index, trend.values, color=color, linewidth=2, label=label)

    ax.axvspan(1872, 1878, alpha=0.15, color="#E8DAEF")
    ax.axvline(1873, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Illegitimate births (% of total)", fontsize=11)
    ax.set_title("Illegitimacy ratio over time by Catholic share", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return {"result": res, "fig": fig}


def infant_mortality_analysis(df: pd.DataFrame):
    """
    Did disruption of Catholic health services affect infant survival?

    The headline outcome `infant_mortality_rate` is total IMR (= total
    infant deaths / total live births x 1000), the standard demographic
    measure (Princeton EFP / HMD / Galloway, Hammel & Lee 1994
    convention). It is well-defined only from 1875 onwards because
    Galloway's illegitimate-infant-death column `Dth<1bas` does not
    appear in pre-1875 VIT files; we therefore restrict the analysis
    to 1875+ regardless. See fig_imr_break.png for the data-break
    diagnostic on the legitimate-only series.
    """
    df = df[df["Year"] >= 1875].copy()

    logger.info("=" * 60)
    logger.info("INFANT MORTALITY CHANNEL (1875+ only due to data break)")
    logger.info("=" * 60)

    df["post_rollback"] = (df["Year"] >= 1880).astype(int)
    df["cath_x_rollback"] = df["cath_share"] * df["post_rollback"]

    res = safe_panel_ols(df, "infant_mortality_rate", ["cath_x_rollback", "ln_pop"])
    coef = res.params["cath_x_rollback"]
    se = res.std_errors["cath_x_rollback"]
    pval = res.pvalues["cath_x_rollback"]
    logger.info("DiD: Infant Mortality Rate ~ CathShare × (Year>=1880)")
    logger.info("  β = %.4f (SE = %.4f, p = %.3f)", coef, se, pval)
    logger.info("  N = %d", int(res.nobs))

    polish_rbs = ["POS", "BRO"]
    german_cath_rbs = ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]

    logger.info("By sub-region (rollback vs enforcement):")
    for label, mask in [
        ("Polish provinces", df["Rb"].isin(polish_rbs)),
        ("German Catholic provinces", df["Rb"].isin(german_cath_rbs)),
        ("Protestant provinces", ~df["Rb"].isin(polish_rbs + german_cath_rbs)),
    ]:
        sub = df[mask].copy()
        if sub["Code"].nunique() < 10:
            continue
        try:
            res_sub = safe_panel_ols(sub, "infant_mortality_rate", ["cath_x_rollback", "ln_pop"])
            logger.info("  %s (%d counties): β = %.4f (SE = %.4f, p = %.3f)",
                        label, sub["Code"].nunique(),
                        res_sub.params["cath_x_rollback"],
                        res_sub.std_errors["cath_x_rollback"],
                        res_sub.pvalues["cath_x_rollback"])
        except Exception as e:
            logger.warning("  %s: failed (%s)", label, e)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, "#C0392B"),
        ("Low Catholic (≤50%)", df["high_cath"] == 0, "#2471A3"),
    ]:
        trend = df[mask].groupby("Year")["infant_mortality_rate"].mean()
        ax.plot(trend.index, trend.values, color=color, linewidth=2, marker="o", markersize=4, label=label)

    ax.axvspan(1875, 1878, alpha=0.15, color="#C0392B", label="Enforcement")
    ax.axvspan(1880, 1887, alpha=0.15, color="#2471A3", label="Rollback")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Infant mortality rate (per 1,000 legitimate live births)", fontsize=11)
    ax.set_title("Infant mortality 1875-1890 by Catholic share\n(pre-1875 data excluded due to measurement change)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return {"result": res, "fig": fig}
