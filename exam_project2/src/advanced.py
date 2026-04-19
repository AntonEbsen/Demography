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