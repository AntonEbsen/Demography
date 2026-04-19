"""
advanced.py
===========
Additional analyses building on the baseline and Polish/German heterogeneity.

Analyses included:
    1. Rollback event study — did the gradual end of Kulturkampf reverse effects?
    2. Illegitimacy channel — did Catholic parish oversight weakening change
       illegitimacy patterns?
    3. Infant mortality channel — did disrupted Catholic health services
       raise infant mortality?
    4. Franco-Prussian War analysis — what drove the 1870 spike?
    5. War-excluded robustness — baseline DiD + event study without 1870-72.

Usage (from notebook):
    from src.advanced import (
        rollback_event_study,
        illegitimacy_analysis,
        infant_mortality_analysis,
        franco_prussian_war_analysis,
        robustness_exclude_war,
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from typing import Optional


def _safe_panel_ols(df, outcome, exog_vars, entity="Code", time="Year"):
    """Deduplicate, drop NaN, set index, fit PanelOLS with clustered SEs."""
    cols_needed = [entity, time, outcome] + exog_vars
    sub = df[cols_needed].drop_duplicates(subset=[entity, time]).dropna().copy()
    sub = sub.set_index([entity, time])
    
    y = sub[outcome]
    X = sub[exog_vars]
    
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


# ===================================================================
# 1. Rollback event study
# ===================================================================

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
    
    A flat null pattern throughout rules out even temporary effects.
    """
    df = df.copy()
    
    # Set multi-index via safe method
    cols_needed = ["Code", "Year", outcome, treatment_var, "ln_pop"]
    sub = df[cols_needed].drop_duplicates(subset=["Code", "Year"]).dropna().copy()
    sub = sub.set_index(["Code", "Year"])
    
    years = sorted(sub.index.get_level_values("Year").unique())
    interact_years = [y for y in years if y != ref_year]
    
    # Create year × treatment interactions
    for yr in interact_years:
        year_dummy = (sub.index.get_level_values("Year") == yr).astype(float)
        sub[f"treat_x_{yr}"] = year_dummy * sub[treatment_var].values
    
    interact_cols = [f"treat_x_{yr}" for yr in interact_years]
    exog = interact_cols + ["ln_pop"]
    
    y = sub[outcome]
    X = sub[exog]
    
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    
    # Extract coefficients
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
    coef_data.append({"Year": ref_year, "beta": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
    coefs = pd.DataFrame(coef_data).sort_values("Year").reset_index(drop=True)
    
    # Plot with both periods marked
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Shade the two periods
    ax.axvspan(1872, 1878, alpha=0.15, color="#C0392B", label="Kulturkampf enforcement")
    ax.axvspan(1880, 1887, alpha=0.15, color="#2471A3", label="Kulturkampf rollback")
    
    ax.axhline(0, color="black", linewidth=0.8)
    
    ax.fill_between(coefs["Year"], coefs["ci_lo"], coefs["ci_hi"],
                    alpha=0.25, color="#555555")
    ax.plot(coefs["Year"], coefs["beta"], color="#333333", linewidth=2,
            marker="o", markersize=4)
    
    ax.scatter([ref_year], [0], color="black", s=80, zorder=5,
               marker="D", label=f"Reference year ({ref_year})")
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"Coefficient on {treatment_var} × Year", fontsize=11)
    ax.set_title("Event study with Kulturkampf enforcement AND rollback periods",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    # Summarise in three periods
    print("Mean coefficients by period:")
    pre = coefs[coefs["Year"] < 1872]["beta"].mean()
    enforce = coefs[(coefs["Year"] >= 1873) & (coefs["Year"] <= 1878)]["beta"].mean()
    rollback = coefs[(coefs["Year"] >= 1880) & (coefs["Year"] <= 1887)]["beta"].mean()
    post = coefs[coefs["Year"] >= 1888]["beta"].mean()
    
    print(f"  Pre (1862-1871):         {pre:+.4f}")
    print(f"  Enforcement (1873-1878): {enforce:+.4f}")
    print(f"  Rollback (1880-1887):    {rollback:+.4f}")
    print(f"  Post (1888+):            {post:+.4f}")
    
    return {"result": res, "coefs": coefs, "fig": fig}


# ===================================================================
# 2. Illegitimacy channel
# ===================================================================

def illegitimacy_analysis(df: pd.DataFrame):
    """
    Did the Kulturkampf affect illegitimacy rates in Catholic counties?
    
    Logic: Catholic parish oversight of sexuality and marriage weakened
    under the Kulturkampf. Civil marriage replaced church marriage in 1875.
    If institutional oversight mattered for enforcing marital norms,
    illegitimate births should rise in Catholic counties.
    
    If there's NO effect, it suggests that informal social pressure
    (family, community) mattered more than institutional oversight.
    """
    df = df.copy()
    
    # Illegitimacy ratio is % of births outside marriage
    # Only available from VIT files that have the split (mostly 1875+)
    
    print("=" * 60)
    print("ILLEGITIMACY CHANNEL")
    print("=" * 60)
    
    # Descriptive: mean illegitimacy ratio by Catholic share and period
    print("\nMean illegitimacy ratio (% of births):")
    for period_label, mask in [
        ("Pre-Kulturkampf (1875-1878)", (df["Year"] >= 1875) & (df["Year"] < 1879)),
        ("Rollback (1880-1887)", (df["Year"] >= 1880) & (df["Year"] <= 1887)),
    ]:
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        by_cath = sub.groupby("high_cath")["illegitimacy_ratio"].mean()
        print(f"  {period_label}:")
        print(f"    Low Catholic (≤50%):  {by_cath.get(0, np.nan):.2f}%")
        print(f"    High Catholic (>50%): {by_cath.get(1, np.nan):.2f}%")
    
    # DiD with illegitimacy ratio as outcome
    # Note: this has limited pre-period since Birbastot is sparse before 1875
    print("\nDiD: Illegitimacy ratio ~ CathShare × Post")
    res = _safe_panel_ols(df, "illegitimacy_ratio", ["cath_share_x_post", "ln_pop"])
    coef = res.params["cath_share_x_post"]
    se = res.std_errors["cath_share_x_post"]
    pval = res.pvalues["cath_share_x_post"]
    print(f"  β = {coef:.4f} (SE = {se:.4f}, p = {pval:.3f})")
    print(f"  N = {int(res.nobs)}")
    
    # Plot trends
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
    ax.set_title("Illegitimacy ratio over time by Catholic share",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    return {"result": res, "fig": fig}


# ===================================================================
# 3. Infant mortality channel
# ===================================================================

def infant_mortality_analysis(df: pd.DataFrame):
    """
    Did disruption of Catholic health services affect infant survival?
    
    IMPORTANT: Galloway's infant mortality measure changes definition in 1875.
    Pre-1875 files use 'Dthyoung' (deaths of young children, mixed definition),
    while 1875+ files use 'Dth<1leg' (deaths under 1 year, legitimate births).
    
    We therefore restrict this analysis to 1875+ only, which means we cannot
    compare pre- vs post-Kulturkampf. Instead we compare:
      - Enforcement period: 1875-1878
      - Rollback period:    1880-1887
    
    This is not a true DiD but a post-only comparison, and results should
    be interpreted cautiously.
    """
    df = df[df["Year"] >= 1875].copy()
    
    print("=" * 60)
    print("INFANT MORTALITY CHANNEL (1875+ only due to data break)")
    print("=" * 60)
    
    # Redefine post-treatment to mean "rollback" within the 1875+ window
    df["post_rollback"] = (df["Year"] >= 1880).astype(int)
    df["cath_x_rollback"] = df["cath_share"] * df["post_rollback"]
    
    # DiD comparing rollback vs enforcement period
    res = _safe_panel_ols(
        df, "infant_mortality_rate", ["cath_x_rollback", "ln_pop"]
    )
    coef = res.params["cath_x_rollback"]
    se = res.std_errors["cath_x_rollback"]
    pval = res.pvalues["cath_x_rollback"]
    print(f"\nDiD: Infant Mortality Rate ~ CathShare × (Year>=1880)")
    print(f"  β = {coef:.4f} (SE = {se:.4f}, p = {pval:.3f})")
    print(f"  N = {int(res.nobs)}")
    print(f"  Interpretation: relative change in rollback vs enforcement period")
    
    # By sub-region
    print("\nBy sub-region (rollback vs enforcement):")
    polish_rbs = ["POS", "BRO"]
    german_cath_rbs = ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]
    
    for label, mask in [
        ("Polish provinces", df["Rb"].isin(polish_rbs)),
        ("German Catholic provinces", df["Rb"].isin(german_cath_rbs)),
        ("Protestant provinces", ~df["Rb"].isin(polish_rbs + german_cath_rbs)),
    ]:
        sub = df[mask].copy()
        if sub["Code"].nunique() < 10:
            continue
        try:
            res_sub = _safe_panel_ols(
                sub, "infant_mortality_rate", ["cath_x_rollback", "ln_pop"]
            )
            print(f"  {label} ({sub['Code'].nunique()} counties): "
                  f"β = {res_sub.params['cath_x_rollback']:.4f} "
                  f"(SE = {res_sub.std_errors['cath_x_rollback']:.4f}, "
                  f"p = {res_sub.pvalues['cath_x_rollback']:.3f})")
        except Exception as e:
            print(f"  {label}: failed ({e})")
    
    # Plot trends (1875+ only)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, "#C0392B"),
        ("Low Catholic (≤50%)", df["high_cath"] == 0, "#2471A3"),
    ]:
        trend = df[mask].groupby("Year")["infant_mortality_rate"].mean()
        ax.plot(trend.index, trend.values, color=color, linewidth=2,
                marker="o", markersize=4, label=label)
    
    ax.axvspan(1875, 1878, alpha=0.15, color="#C0392B", label="Enforcement")
    ax.axvspan(1880, 1887, alpha=0.15, color="#2471A3", label="Rollback")
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Infant mortality rate (per 1,000 legitimate live births)", fontsize=11)
    ax.set_title("Infant mortality 1875-1890 by Catholic share\n"
                 "(pre-1875 data excluded due to measurement change)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    return {"result": res, "fig": fig}


# ===================================================================
# 4. Franco-Prussian War analysis
# ===================================================================

def franco_prussian_war_analysis(df: pd.DataFrame):
    """
    Analyse the 1870-1871 fertility shock (Franco-Prussian War).
    
    The war caused a large drop in births 9-12 months later as men
    were mobilised. Did Catholic and Protestant counties differ in
    their war response? This matters because:
    
    1. It explains the pre-trend volatility in your main event study
    2. It's historically interesting: Catholics were over-represented
       in the army (Rhineland, Bavarian troops in federal army)
    3. If Catholic counties had larger war-related fertility drops,
       the 1870-1872 period can't be used as a clean "baseline"
    """
    df = df.copy()
    
    print("=" * 60)
    print("FRANCO-PRUSSIAN WAR (1870-1871) ANALYSIS")
    print("=" * 60)
    
    # Compute year-on-year change in CBR
    df = df.sort_values(["Code", "Year"])
    df["cbr_lag"] = df.groupby("Code")["cbr"].shift(1)
    df["cbr_change"] = df["cbr"] - df["cbr_lag"]
    
    print("\nMean year-on-year change in CBR by Catholic share group:")
    for year in [1870, 1871, 1872, 1873]:
        row = df[df["Year"] == year].groupby("high_cath")["cbr_change"].mean()
        low = row.get(0, np.nan)
        high = row.get(1, np.nan)
        diff = high - low if (pd.notna(low) and pd.notna(high)) else np.nan
        print(f"  {year}: Low={low:+.2f}, High={high:+.2f}, Diff={diff:+.2f}")
    
    # Isolate war shock effect: was it different across Catholic share?
    # Run a regression on the war-shock window
    df_war = df[(df["Year"] >= 1868) & (df["Year"] <= 1874)].copy()
    df_war["war_year"] = df_war["Year"].isin([1871, 1872]).astype(int)
    df_war["cath_x_war"] = df_war["cath_share"] * df_war["war_year"]
    
    print("\nDiD around war shock (1868-1874, war years = 1871-1872):")
    try:
        res = _safe_panel_ols(df_war, "cbr", ["cath_x_war", "ln_pop"])
        coef = res.params["cath_x_war"]
        se = res.std_errors["cath_x_war"]
        pval = res.pvalues["cath_x_war"]
        print(f"  Catholic share × War: β = {coef:.4f} "
              f"(SE = {se:.4f}, p = {pval:.3f})")
        print(f"  → Catholic counties' births fell by {coef:.3f} more per "
              f"1,000 per pp Catholic during war")
    except Exception as e:
        print(f"  Regression failed: {e}")
        res = None
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, "#C0392B"),
        ("Low Catholic (≤50%)", df["high_cath"] == 0, "#2471A3"),
    ]:
        trend = df[(mask) & (df["Year"].between(1865, 1878))].groupby("Year")["cbr"].mean()
        ax.plot(trend.index, trend.values, color=color, linewidth=2, 
                marker="o", markersize=5, label=label)
    
    ax.axvspan(1870, 1871, alpha=0.2, color="grey", label="Franco-Prussian War")
    ax.axvspan(1872, 1878, alpha=0.15, color="#E8DAEF", label="Kulturkampf")
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Crude birth rate (per 1,000)", fontsize=11)
    ax.set_title("Franco-Prussian War shock vs Kulturkampf",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    return {"result": res, "fig": fig}


# ===================================================================
# 5. Robustness: exclude the Franco-Prussian War years
# ===================================================================

def robustness_exclude_war(
    df: pd.DataFrame,
    outcome: str = "cbr",
    war_years: tuple = (1870, 1871, 1872),
    ref_year: int = 1869,
    savepath: Optional[str] = None,
):
    """
    Re-estimate the baseline DiD and event study excluding the 
    Franco-Prussian War years (default: 1870, 1871, 1872).
    
    Rationale: The war caused sharp but transitory fertility drops in 
    both Catholic and Protestant counties. Including these years 
    inflates the variance of pre-treatment event-study coefficients 
    and creates apparent pre-trend volatility that has nothing to do 
    with the Kulturkampf.
    
    The reference year must move accordingly: we use 1869 (last clean 
    pre-war year) instead of 1872 for the event study.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full analysis panel.
    outcome : str
        Dependent variable.
    war_years : tuple of int
        Years to exclude.
    ref_year : int
        Reference year for the event study (should be pre-war and 
        pre-Kulturkampf).
    savepath : str, optional
        If provided, save the plot here.
    
    Returns
    -------
    dict with:
        'did_result'     : baseline DiD result on war-excluded sample
        'event_result'   : event study result
        'coefs'          : DataFrame of event-study coefficients
        'fig'            : the event-study figure
    """
    print("=" * 60)
    print(f"ROBUSTNESS: EXCLUDING WAR YEARS {war_years}")
    print("=" * 60)
    
    df_clean = df[~df["Year"].isin(war_years)].copy()
    
    print(f"\nOriginal panel: {len(df)} obs")
    print(f"After excluding {war_years}: {len(df_clean)} obs "
          f"(dropped {len(df) - len(df_clean)})")
    
    # ---------- Baseline DiD on the war-excluded sample ----------
    print("\n--- BASELINE DiD (war excluded) ---")
    res_did = _safe_panel_ols(
        df_clean, outcome, ["cath_share_x_post", "ln_pop"]
    )
    coef = res_did.params["cath_share_x_post"]
    se = res_did.std_errors["cath_share_x_post"]
    pval = res_did.pvalues["cath_share_x_post"]
    print(f"  cath_share × post: β = {coef:.4f} "
          f"(SE = {se:.4f}, p = {pval:.3f})")
    print(f"  N = {int(res_did.nobs)}")
    
    # Also compare to baseline with war years included (for reference)
    res_full = _safe_panel_ols(
        df, outcome, ["cath_share_x_post", "ln_pop"]
    )
    print(f"\n  For comparison (war included): "
          f"β = {res_full.params['cath_share_x_post']:.4f} "
          f"(SE = {res_full.std_errors['cath_share_x_post']:.4f}, "
          f"p = {res_full.pvalues['cath_share_x_post']:.3f})")
    
    # ---------- Event study on war-excluded sample ----------
    print("\n--- EVENT STUDY (war excluded, reference year = {}) ---".format(ref_year))
    
    cols_needed = ["Code", "Year", outcome, "cath_share", "ln_pop"]
    sub = df_clean[cols_needed].drop_duplicates(subset=["Code", "Year"]).dropna().copy()
    sub = sub.set_index(["Code", "Year"])
    
    years = sorted(sub.index.get_level_values("Year").unique())
    interact_years = [y for y in years if y != ref_year]
    
    for yr in interact_years:
        year_dummy = (sub.index.get_level_values("Year") == yr).astype(float)
        sub[f"treat_x_{yr}"] = year_dummy * sub["cath_share"].values
    
    interact_cols = [f"treat_x_{yr}" for yr in interact_years]
    exog = interact_cols + ["ln_pop"]
    
    y = sub[outcome]
    X = sub[exog]
    
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res_event = mod.fit(cov_type="clustered", cluster_entity=True)
    
    # Build coefficient table
    coef_data = []
    for yr in interact_years:
        col = f"treat_x_{yr}"
        b = res_event.params[col]
        s = res_event.std_errors[col]
        coef_data.append({
            "Year": yr,
            "beta": b,
            "se": s,
            "ci_lo": b - 1.96 * s,
            "ci_hi": b + 1.96 * s,
        })
    coef_data.append({"Year": ref_year, "beta": 0.0, "se": 0.0,
                      "ci_lo": 0.0, "ci_hi": 0.0})
    coefs = pd.DataFrame(coef_data).sort_values("Year").reset_index(drop=True)
    
    # ---------- Pre-trend check ----------
    pre_coefs = coefs[coefs["Year"] < 1872]
    print(f"\nPre-trend coefficients (before 1872):")
    print(pre_coefs[["Year", "beta", "se"]].to_string(index=False))
    print(f"\n  Range of pre-trend betas: "
          f"[{pre_coefs['beta'].min():.4f}, {pre_coefs['beta'].max():.4f}]")
    print(f"  Mean pre-trend beta:       {pre_coefs['beta'].mean():+.4f}")
    
    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Shade war years (even though excluded, show for context)
    for wy in war_years:
        ax.axvspan(wy - 0.4, wy + 0.4, alpha=0.1, color="grey")
    
    # Shade Kulturkampf periods
    ax.axvspan(1872, 1878, alpha=0.15, color="#C0392B", label="Kulturkampf enforcement")
    ax.axvspan(1880, 1887, alpha=0.15, color="#2471A3", label="Kulturkampf rollback")
    
    ax.axhline(0, color="black", linewidth=0.8)
    
    # Connect coefficients across the excluded years with dashed line for clarity
    ax.fill_between(coefs["Year"], coefs["ci_lo"], coefs["ci_hi"],
                    alpha=0.25, color="#333333")
    ax.plot(coefs["Year"], coefs["beta"], color="#333333", linewidth=2,
            marker="o", markersize=5)
    
    ax.scatter([ref_year], [0], color="black", s=80, zorder=5,
               marker="D", label=f"Reference year ({ref_year})")
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"Coefficient on cath_share × Year", fontsize=11)
    ax.set_title(f"Event study excluding Franco-Prussian War years {war_years}\n"
                 f"(grey bands = excluded years)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    return {
        "did_result": res_did,
        "event_result": res_event,
        "coefs": coefs,
        "fig": fig,
    }


# ===================================================================
# 6. Trend-adjusted DiD
# ===================================================================

def trend_adjusted_did(
    df: pd.DataFrame,
    outcome: str = "cbr",
    trend_base_year: int = 1862,
    exclude_war: bool = False,
    war_years: tuple = (1870, 1871, 1872),
):
    """
    Estimate a DiD that allows for differential linear pre-trends
    between counties with different Catholic shares.
    
    Specification
    -------------
    Y_it = β (CathShare × Post) + γ (CathShare × Trend) 
           + controls + α_i + δ_t + ε_it
    
    where Trend = Year − trend_base_year.
    
    The γ term absorbs any linear differential trend correlated with
    Catholic share. The β term is then identified from *deviations*
    from that trend — i.e. a discontinuous shift at the Kulturkampf.
    
    Interpretation
    --------------
    If your raw event study shows a smooth downward trend through the
    whole period (as in our results), the baseline DiD picks up this
    trend as if it were a treatment effect. By explicitly controlling
    for the trend, we isolate any Kulturkampf-specific *break*.
    
    A null β here is a much stronger result than a null baseline DiD,
    because it means: after accounting for the pre-existing convergence
    between Catholic and Protestant counties, there is no additional
    fertility change attributable to the Kulturkampf.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    outcome : str
        Dependent variable (default: 'cbr').
    trend_base_year : int
        Year from which linear trend is measured.
    exclude_war : bool
        If True, drop the Franco-Prussian War years.
    war_years : tuple
        Years to drop if exclude_war is True.
    
    Returns
    -------
    dict with keys 'result' (PanelOLS fit), 'coefs' (dict of β, γ, SEs).
    """
    print("=" * 60)
    print("TREND-ADJUSTED DiD")
    if exclude_war:
        print(f"(excluding war years {war_years})")
    print("=" * 60)
    
    df = df.copy()
    if exclude_war:
        df = df[~df["Year"].isin(war_years)].copy()
    
    # Construct the linear trend interaction
    df["trend"] = df["Year"] - trend_base_year
    df["cath_x_trend"] = df["cath_share"] * df["trend"]
    
    # Run the regression
    exog = ["cath_share_x_post", "cath_x_trend", "ln_pop"]
    res = _safe_panel_ols(df, outcome, exog)
    
    # Also run baseline (without trend adjustment) for comparison
    res_baseline = _safe_panel_ols(df, outcome, ["cath_share_x_post", "ln_pop"])
    
    # Report
    print("\n--- BASELINE (no trend adjustment) ---")
    b_base = res_baseline.params["cath_share_x_post"]
    s_base = res_baseline.std_errors["cath_share_x_post"]
    p_base = res_baseline.pvalues["cath_share_x_post"]
    print(f"  cath_share × post: β = {b_base:+.5f} "
          f"(SE = {s_base:.5f}, p = {p_base:.3f})")
    
    print("\n--- TREND-ADJUSTED ---")
    b_adj = res.params["cath_share_x_post"]
    s_adj = res.std_errors["cath_share_x_post"]
    p_adj = res.pvalues["cath_share_x_post"]
    print(f"  cath_share × post:  β = {b_adj:+.5f} "
          f"(SE = {s_adj:.5f}, p = {p_adj:.3f})")
    
    g_trend = res.params["cath_x_trend"]
    s_trend = res.std_errors["cath_x_trend"]
    p_trend = res.pvalues["cath_x_trend"]
    print(f"  cath_share × trend: γ = {g_trend:+.5f} "
          f"(SE = {s_trend:.5f}, p = {p_trend:.3f})")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"  γ = {g_trend:+.5f} means that per year of trend, a 10-pp")
    print(f"  more-Catholic county's CBR changed by {10*g_trend:+.4f} per 1,000")
    print(f"  relative to a less-Catholic county, independent of the Kulturkampf.")
    print(f"  Over 30 years that compounds to {30*10*g_trend:+.2f} per 1,000.")
    print(f"")
    if p_adj > 0.10:
        print(f"  β (Kulturkampf break) is NOT significant after trend adjustment.")
        print(f"  → No evidence of a Kulturkampf-specific fertility shock.")
    else:
        print(f"  β (Kulturkampf break) IS significant after trend adjustment.")
        print(f"  → Evidence of a discontinuous shift beyond the linear trend.")
    
    return {
        "result": res,
        "result_baseline": res_baseline,
        "coefs": {
            "beta_baseline": b_base, "se_baseline": s_base, "p_baseline": p_base,
            "beta_adjusted": b_adj, "se_adjusted": s_adj, "p_adjusted": p_adj,
            "gamma_trend": g_trend, "se_trend": s_trend, "p_trend": p_trend,
        },
    }


# ===================================================================
# 7. Polish vs German × Rollback interaction
# ===================================================================

def polish_german_rollback(
    df: pd.DataFrame,
    outcome: str = "cbr",
    savepath: str = None,
):
    """
    Does the Polish-vs-German divergence reverse during the rollback period?
    
    Rationale
    ---------
    You already found that during the full post-Kulturkampf era
    (1873+), Polish Catholic provinces saw significantly lower Catholic
    fertility while German Catholic provinces saw slightly higher
    fertility. This analysis splits the "post" period into two phases
    to see whether these effects persisted through the Kulturkampf
    rollback (1880-1887):
    
      - Enforcement (1873-1878): Kulturkampf in full force
      - Rollback    (1880-1887): laws gradually repealed
    
    Three possible patterns:
      1. Polish negative effect DISAPPEARS in rollback
         → Strong evidence Kulturkampf caused it.
      2. Polish negative effect PERSISTS in rollback
         → Suggests something else (e.g. ongoing Germanization)
            drove the result, not the religious legislation per se.
      3. Polish effect GROWS in rollback
         → The slow fertility transition catching up.
    
    Specification
    -------------
    Interact cath_share with period dummies (enforcement / rollback),
    and estimate separately on the Polish and German sub-samples.
    
    Returns
    -------
    dict with sub-sample results and a coefficient plot.
    """
    print("=" * 60)
    print("POLISH × ROLLBACK vs GERMAN × ROLLBACK")
    print("=" * 60)
    
    df = df.copy()
    
    # Create enforcement / rollback / post-rollback indicators
    df["enforcement"] = df["Year"].between(1873, 1878).astype(int)
    df["rollback"] = df["Year"].between(1880, 1887).astype(int)
    df["post_rollback"] = (df["Year"] >= 1888).astype(int)
    
    df["cath_x_enforcement"] = df["cath_share"] * df["enforcement"]
    df["cath_x_rollback"] = df["cath_share"] * df["rollback"]
    df["cath_x_postrollback"] = df["cath_share"] * df["post_rollback"]
    
    polish_rbs = ["POS", "BRO"]
    german_cath_rbs = ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]
    
    sub_results = {}
    
    for label, mask in [
        ("Polish provinces", df["Rb"].isin(polish_rbs)),
        ("German Catholic provinces", df["Rb"].isin(german_cath_rbs)),
        ("Protestant provinces (rest)", ~df["Rb"].isin(polish_rbs + german_cath_rbs)),
    ]:
        sub = df[mask].copy()
        if sub["Code"].nunique() < 10:
            print(f"  {label}: too few counties, skipping")
            continue
        
        exog = ["cath_x_enforcement", "cath_x_rollback", "cath_x_postrollback", "ln_pop"]
        res = _safe_panel_ols(sub, outcome, exog)
        
        sub_results[label] = {
            "enforcement": {
                "coef": res.params["cath_x_enforcement"],
                "se":   res.std_errors["cath_x_enforcement"],
                "p":    res.pvalues["cath_x_enforcement"],
            },
            "rollback": {
                "coef": res.params["cath_x_rollback"],
                "se":   res.std_errors["cath_x_rollback"],
                "p":    res.pvalues["cath_x_rollback"],
            },
            "post_rollback": {
                "coef": res.params["cath_x_postrollback"],
                "se":   res.std_errors["cath_x_postrollback"],
                "p":    res.pvalues["cath_x_postrollback"],
            },
            "n_counties": sub["Code"].nunique(),
        }
        
        print(f"\n  {label} ({sub['Code'].nunique()} counties):")
        for period_label, key in [
            ("Enforcement  (1873-1878)", "enforcement"),
            ("Rollback     (1880-1887)", "rollback"),
            ("Post-rollback (1888+)   ", "post_rollback"),
        ]:
            r = sub_results[label][key]
            stars = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
            print(f"    {period_label}: β = {r['coef']:+.4f} "
                  f"(SE = {r['se']:.4f}, p = {r['p']:.3f}) {stars}")
    
    # Plot coefficients by period for each sub-sample
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "Polish provinces": "#C0392B",
        "German Catholic provinces": "#2471A3",
        "Protestant provinces (rest)": "#555555",
    }
    
    period_labels = ["Enforcement\n(1873-1878)", "Rollback\n(1880-1887)", "Post-rollback\n(1888+)"]
    period_keys = ["enforcement", "rollback", "post_rollback"]
    
    width = 0.25
    x = np.arange(len(period_labels))
    
    for i, (label, r) in enumerate(sub_results.items()):
        if label not in colors:
            continue
        coefs = [r[k]["coef"] for k in period_keys]
        ses = [r[k]["se"] for k in period_keys]
        xpos = x + (i - 1) * width
        ax.bar(xpos, coefs, width=width, yerr=[1.96 * s for s in ses],
               color=colors[label], alpha=0.8, label=label, capsize=4)
    
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(period_labels)
    ax.set_ylabel("Coefficient on cath_share × period", fontsize=11)
    ax.set_title("Kulturkampf effects by sub-region and period\n(95% CI)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    return {"results": sub_results, "fig": fig}


# ===================================================================
# 8. Placebo test with fake treatment year
# ===================================================================

def placebo_test(
    df: pd.DataFrame,
    outcome: str = "cbr",
    placebo_years: list = None,
    savepath: str = None,
):
    """
    Placebo test: pretend the Kulturkampf happened in a different year
    and re-run the baseline DiD.
    
    Rationale
    ---------
    If the baseline DiD (β ≈ 0 at true treatment year 1873) is really
    picking up the absence of a Kulturkampf effect, then:
      - β at fake treatment years in the PRE-period should also be ≈ 0
        (nothing special happened there)
      - β at fake treatment years in the POST-period should also be ≈ 0
        (nothing special happened there either)
    
    If instead some placebo years produce significant coefficients,
    then the DiD is sensitive to the choice of cutoff and its null
    result may be coincidental.
    
    Parameters
    ----------
    placebo_years : list of int
        Years to treat as fake Kulturkampf onset.
        Default: 1864, 1866, 1868, 1870, 1873 (true), 1876, 1880, 1884.
    
    Returns
    -------
    DataFrame with placebo_year, coef, se, p_value, sample restriction.
    """
    if placebo_years is None:
        placebo_years = [1864, 1866, 1868, 1870, 1873, 1876, 1880, 1884]
    
    print("=" * 60)
    print("PLACEBO TEST — FAKE TREATMENT YEARS")
    print("=" * 60)
    print("Re-estimating the DiD with the 'post' indicator set at various")
    print("different years. Only the true treatment year (1873) should")
    print("produce a meaningful result if the design is correct.\n")
    
    results = []
    
    for placebo_year in placebo_years:
        df_pl = df.copy()
        # Override post_kulturkampf with the placebo year
        df_pl["post_placebo"] = (df_pl["Year"] >= placebo_year).astype(int)
        df_pl["cath_x_post_placebo"] = df_pl["cath_share"] * df_pl["post_placebo"]
        
        # Need variation in post_placebo → trim sample if needed
        if df_pl["post_placebo"].var() == 0:
            continue
        
        try:
            res = _safe_panel_ols(
                df_pl, outcome, ["cath_x_post_placebo", "ln_pop"]
            )
            coef = res.params["cath_x_post_placebo"]
            se = res.std_errors["cath_x_post_placebo"]
            p = res.pvalues["cath_x_post_placebo"]
            
            is_true = (placebo_year == 1873)
            marker = " ← TRUE" if is_true else ""
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            
            print(f"  Placebo year = {placebo_year}: β = {coef:+.5f} "
                  f"(SE = {se:.5f}, p = {p:.3f}) {stars}{marker}")
            
            results.append({
                "placebo_year": placebo_year,
                "coef": coef,
                "se": se,
                "ci_lo": coef - 1.96 * se,
                "ci_hi": coef + 1.96 * se,
                "p_value": p,
                "is_true": is_true,
                "n": int(res.nobs),
            })
        except Exception as e:
            print(f"  Placebo year = {placebo_year}: failed ({e})")
    
    res_df = pd.DataFrame(results)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#C0392B" if t else "#555555" for t in res_df["is_true"]]
    markers = ["D" if t else "o" for t in res_df["is_true"]]
    
    for _, row in res_df.iterrows():
        color = "#C0392B" if row["is_true"] else "#555555"
        marker = "D" if row["is_true"] else "o"
        size = 80 if row["is_true"] else 50
        ax.errorbar(
            row["placebo_year"], row["coef"],
            yerr=1.96 * row["se"],
            fmt=marker, color=color, markersize=8 if row["is_true"] else 6,
            capsize=4, linewidth=1.5,
        )
    
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(1873, color="#C0392B", linestyle="--", linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel("Placebo treatment year", fontsize=11)
    ax.set_ylabel("Coefficient on cath_share × post_placebo (95% CI)", fontsize=11)
    ax.set_title("Placebo test: coefficients at fake treatment years\n"
                 "(red diamond = true Kulturkampf year)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    # Summary assessment
    n_sig = (res_df[~res_df["is_true"]]["p_value"] < 0.05).sum()
    n_placebo = (~res_df["is_true"]).sum()
    print(f"\nSummary:")
    print(f"  {n_sig} out of {n_placebo} placebo years produced p < 0.05")
    if n_sig == 0:
        print(f"  → Consistent with a clean null: no year shows a significant effect.")
    elif n_sig <= 1:
        print(f"  → One placebo is significant (within chance expectations at 5% level).")
    else:
        print(f"  → Multiple placebos are significant — design is noisy.")
    
    return {"results_df": res_df, "fig": fig}