"""
war_robustness.py
=================
Franco-Prussian War analysis and war-exclusion robustness checks.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from typing import Optional

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def franco_prussian_war_analysis(df: pd.DataFrame):
    """Analyse the 1870-1871 fertility shock (Franco-Prussian War)."""
    df = df.copy()
    logger.info("FRANCO-PRUSSIAN WAR (1870-1871) ANALYSIS")

    df = df.sort_values(["Code", "Year"])
    df["cbr_lag"] = df.groupby("Code")["cbr"].shift(1)
    df["cbr_change"] = df["cbr"] - df["cbr_lag"]

    for year in [1870, 1871, 1872, 1873]:
        row = df[df["Year"] == year].groupby("high_cath")["cbr_change"].mean()
        low = row.get(0, np.nan)
        high = row.get(1, np.nan)
        diff = high - low if (pd.notna(low) and pd.notna(high)) else np.nan
        logger.info("  %d: Low=%+.2f, High=%+.2f, Diff=%+.2f", year, low, high, diff)

    df_war = df[(df["Year"] >= 1868) & (df["Year"] <= 1874)].copy()
    df_war["war_year"] = df_war["Year"].isin([1871, 1872]).astype(int)
    df_war["cath_x_war"] = df_war["cath_share"] * df_war["war_year"]

    try:
        res = safe_panel_ols(df_war, "cbr", ["cath_x_war"])
        logger.info("  Catholic share x War: b=%.4f (SE=%.4f, p=%.3f)",
                     res.params["cath_x_war"], res.std_errors["cath_x_war"], res.pvalues["cath_x_war"])
    except Exception as e:
        logger.warning("  Regression failed: %s", e)
        res = None

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, "#C0392B"),
        ("Low Catholic (<=50%)", df["high_cath"] == 0, "#2471A3"),
    ]:
        trend = df[(mask) & (df["Year"].between(1865, 1878))].groupby("Year")["cbr"].mean()
        ax.plot(trend.index, trend.values, color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.axvspan(1870, 1871, alpha=0.2, color="grey", label="Franco-Prussian War")
    ax.axvspan(1871, 1878, alpha=0.15, color="#E8DAEF",
               label="Kulturkampf (1871-78)")
    ax.axvline(1873, color="#7B1A1A", linestyle="--", linewidth=1.2, alpha=0.9,
               label="May Laws (treatment year)")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Crude birth rate (per 1,000)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return {"result": res, "fig": fig}


def province_war_effect(
    df: pd.DataFrame,
    war_years: tuple[int, ...] = (1866, 1870, 1871),
    ref_years: tuple[int, ...] = (1864, 1865, 1868, 1869),
    outcome: str = "cbr",
    min_counties: int = 3,
) -> pd.DataFrame:
    """
    By-Regierungsbezirk comparison of mean outcome (default: CBR) in
    war years vs flanking non-war years. Used to test whether the
    pre-1873 CBR trend in the event study is driven by differential
    Prussian-Army recruitment burden across Protestant- vs
    Catholic-majority provinces.

    The Austro-Prussian War (1866) and Franco-Prussian War (1870-71)
    mobilised young men from each Regierungsbezirk in proportion to
    its population and conscription quotas. If Protestant provinces
    shouldered more of the recruitment burden, their CBR would dip in
    war years (men away from wives, partial cohort mortality) while
    Catholic provinces -- especially Polish-Catholic areas where
    conscription was less aggressive politically -- would dip less.
    The differential dip then opens a Catholic-Protestant CBR gap
    during the war years that mechanically *closes* in the years
    immediately after, generating an apparent pre-trend in the
    event study without any behavioural Catholic-Protestant fertility
    difference.

    Returns a DataFrame keyed by Rb with columns:
      - mean_war_years        : mean outcome in war_years
      - mean_nonwar_years     : mean outcome in ref_years
      - diff                  : mean_war - mean_nonwar (negative = dip)
      - cath_share_rb_mean    : Rb-mean of cath_share (1871 census)
      - n_counties            : number of counties in the Rb
    sorted ascending by `diff` (most-negative war dip first).

    Rbs with fewer than `min_counties` counties are dropped.
    """
    if outcome not in df.columns:
        raise KeyError(f"Outcome {outcome!r} not in panel.")

    war_mask = df["Year"].isin(war_years)
    ref_mask = df["Year"].isin(ref_years)

    war_means = (
        df.loc[war_mask, ["Rb", outcome]]
        .groupby("Rb")[outcome]
        .mean()
        .rename("mean_war_years")
    )
    ref_means = (
        df.loc[ref_mask, ["Rb", outcome]]
        .groupby("Rb")[outcome]
        .mean()
        .rename("mean_nonwar_years")
    )
    cath_means = (
        df[["Rb", "cath_share"]]
        .drop_duplicates(subset=["Rb", "cath_share"])
        .groupby("Rb")["cath_share"]
        .mean()
        .rename("cath_share_rb_mean")
    )
    n_counties = (
        df.groupby("Rb")["Code"]
        .nunique()
        .rename("n_counties")
    )

    out = (
        pd.concat([war_means, ref_means, cath_means, n_counties], axis=1)
        .dropna(subset=["mean_war_years", "mean_nonwar_years"])
    )
    out["diff"] = out["mean_war_years"] - out["mean_nonwar_years"]
    out = out[out["n_counties"] >= min_counties]
    return out[
        ["mean_war_years", "mean_nonwar_years", "diff",
         "cath_share_rb_mean", "n_counties"]
    ].sort_values("diff").reset_index()


def robustness_exclude_war(
    df: pd.DataFrame, outcome: str = "cbr",
    war_years: tuple = (1870, 1871, 1872), ref_year: int = 1869,
    savepath: Optional[str] = None,
):
    """Re-estimate DiD and event study excluding Franco-Prussian War years."""
    logger.info("ROBUSTNESS: EXCLUDING WAR YEARS %s", war_years)
    df_clean = df[~df["Year"].isin(war_years)].copy()

    res_did = safe_panel_ols(df_clean, outcome, ["cath_share_x_post"])
    logger.info("  cath_share x post: b=%.4f (SE=%.4f, p=%.3f)",
                 res_did.params["cath_share_x_post"], res_did.std_errors["cath_share_x_post"],
                 res_did.pvalues["cath_share_x_post"])

    cols_needed = ["Code", "Year", outcome, "cath_share"]
    sub = df_clean[cols_needed].drop_duplicates(subset=["Code", "Year"]).dropna().copy()
    sub = sub.set_index(["Code", "Year"])
    years = sorted(sub.index.get_level_values("Year").unique())
    interact_years = [y for y in years if y != ref_year]
    for yr in interact_years:
        year_dummy = (sub.index.get_level_values("Year") == yr).astype(float)
        sub[f"treat_x_{yr}"] = year_dummy * sub["cath_share"].values
    interact_cols = [f"treat_x_{yr}" for yr in interact_years]
    y_var = sub[outcome]
    X = sub[interact_cols]
    mod = PanelOLS(y_var, X, entity_effects=True, time_effects=True)
    res_event = mod.fit(cov_type="clustered", cluster_entity=True)

    coef_data = []
    for yr in interact_years:
        col = f"treat_x_{yr}"
        b, s = res_event.params[col], res_event.std_errors[col]
        coef_data.append({"Year": yr, "beta": b, "se": s, "ci_lo": b - 1.96*s, "ci_hi": b + 1.96*s})
    coef_data.append({"Year": ref_year, "beta": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
    coefs = pd.DataFrame(coef_data).sort_values("Year").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for wy in war_years:
        ax.axvspan(wy - 0.4, wy + 0.4, alpha=0.1, color="grey")
    ax.axvspan(1871, 1878, alpha=0.15, color="#C0392B",
               label="Kulturkampf enforcement (1871-78)")
    ax.axvspan(1880, 1887, alpha=0.15, color="#2471A3",
               label="Kulturkampf rollback (1880-87)")
    ax.axvline(1873, color="#7B1A1A", linestyle="--", linewidth=1.2, alpha=0.9,
               label="May Laws (treatment year)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(coefs["Year"], coefs["ci_lo"], coefs["ci_hi"], alpha=0.25, color="#333333")
    ax.plot(coefs["Year"], coefs["beta"], color="#333333", linewidth=2, marker="o", markersize=5)
    ax.scatter([ref_year], [0], color="black", s=80, zorder=5, marker="D", label=f"Reference year ({ref_year})")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Coefficient on cath_share x Year", fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return {"did_result": res_did, "event_result": res_event, "coefs": coefs, "fig": fig}
