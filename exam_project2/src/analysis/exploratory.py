"""
exploratory.py
==============
Additional analyses for nerdy fun.

Usage (from notebook):
    from src.analysis.exploratory import (
        heterogeneity_by_urbanization,
        polish_vs_german_catholics,
        fertility_convergence,
        marriage_to_birth_pipeline,
        dose_response_plot,
        infant_mortality_did,
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from typing import Optional


def _safe_panel_ols(df, outcome, exog_vars, entity="Code", time="Year"):
    """
    Safely run PanelOLS: drop duplicates, handle NaN, set index.
    Returns the fitted result.
    """
    cols_needed = [entity, time, outcome] + exog_vars
    sub = df[cols_needed].drop_duplicates(subset=[entity, time]).dropna().copy()
    sub = sub.set_index([entity, time])
    
    y = sub[outcome]
    X = sub[exog_vars]
    
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


# ===================================================================
# 1. Heterogeneity: Urban vs Rural
# ===================================================================

def heterogeneity_by_urbanization(df: pd.DataFrame, outcome: str = "cbr"):
    """
    Split the sample into urban and rural counties and run DiD separately.
    Also runs a triple-difference: CathShare × Post × Urban.
    """
    pop_1871 = df.groupby("Code")["Poptot"].first()
    median_pop = pop_1871.median()
    urban_codes = set(pop_1871[pop_1871 > median_pop].index)
    
    df = df.copy()
    df["urban"] = df["Code"].isin(urban_codes).astype(int)
    
    results = {}
    
    for label, mask in [("Urban", df["urban"] == 1), ("Rural", df["urban"] == 0)]:
        sub = df[mask].copy()
        res = _safe_panel_ols(sub, outcome, ["cath_share_x_post", "ln_pop"])
        
        results[label] = {
            "coef": res.params["cath_share_x_post"],
            "se": res.std_errors["cath_share_x_post"],
            "pval": res.pvalues["cath_share_x_post"],
            "n": int(res.nobs),
        }
        print(f"{label}: β = {results[label]['coef']:.4f} "
              f"(SE = {results[label]['se']:.4f}, p = {results[label]['pval']:.3f})")
    
    # Triple difference
    df["cath_x_post_x_urban"] = df["cath_share"] * df["post_kulturkampf"] * df["urban"]
    df["post_x_urban"] = df["post_kulturkampf"] * df["urban"]
    
    exog = ["cath_share_x_post", "cath_x_post_x_urban", "post_x_urban", "ln_pop"]
    res = _safe_panel_ols(df, outcome, exog)
    
    print(f"\nTriple difference (CathShare × Post × Urban):")
    print(f"  β = {res.params['cath_x_post_x_urban']:.4f} "
          f"(SE = {res.std_errors['cath_x_post_x_urban']:.4f}, "
          f"p = {res.pvalues['cath_x_post_x_urban']:.3f})")
    
    results["triple_diff"] = res
    return results


# ===================================================================
# 2. Polish vs German Catholics
# ===================================================================

def polish_vs_german_catholics(df: pd.DataFrame, outcome: str = "cbr"):
    """
    Compare Kulturkampf effects on Polish-Catholic vs German-Catholic counties.
    """
    df = df.copy()
    
    polish_rbs = ["POS", "BRO"]
    german_cath_rbs = ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]
    
    results = {}
    
    for label, mask in [
        ("Polish Catholic provinces", df["Rb"].isin(polish_rbs)),
        ("German Catholic provinces", df["Rb"].isin(german_cath_rbs)),
        ("Protestant provinces (rest)", ~df["Rb"].isin(polish_rbs + german_cath_rbs)),
    ]:
        sub = df[mask].copy()
        if sub["Code"].nunique() < 10:
            print(f"{label}: too few counties ({sub['Code'].nunique()}), skipping")
            continue
        
        res = _safe_panel_ols(sub, outcome, ["cath_share_x_post", "ln_pop"])
        
        results[label] = {
            "coef": res.params["cath_share_x_post"],
            "se": res.std_errors["cath_share_x_post"],
            "pval": res.pvalues["cath_share_x_post"],
            "n_counties": sub["Code"].nunique(),
        }
        print(f"{label} ({sub['Code'].nunique()} counties): "
              f"β = {results[label]['coef']:.4f} "
              f"(SE = {results[label]['se']:.4f}, p = {results[label]['pval']:.3f})")
    
    return results


# ===================================================================
# 3. Fertility convergence
# ===================================================================

def fertility_convergence(df: pd.DataFrame):
    """
    Test whether initial Catholic share predicts the rate of fertility decline.
    """
    df = df.copy()
    
    early = df[df["Year"] <= 1864].groupby("Code")["cbr"].mean()
    late = df[df["Year"] >= 1888].groupby("Code")["cbr"].mean()
    
    change = (late - early).dropna()
    change.name = "cbr_change"
    
    cath = df.groupby("Code")["cath_share"].first()
    
    merged = pd.DataFrame({"cbr_change": change, "cath_share": cath}).dropna()
    
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(merged["cath_share"], merged["cbr_change"])
    
    print(f"β-convergence regression: ΔCBR = {intercept:.2f} + {slope:.4f} × CathShare")
    print(f"  SE = {se:.4f}, p = {p:.3f}, R² = {r**2:.3f}")
    print(f"  N = {len(merged)} counties")
    
    if slope < 0:
        print("  → Higher Catholic share predicts LARGER fertility decline (convergence)")
    else:
        print("  → Higher Catholic share predicts SMALLER fertility decline (divergence)")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["cath_share"], merged["cbr_change"], alpha=0.4, s=15, color="#C0392B")
    
    x_line = np.linspace(0, 100, 100)
    ax.plot(x_line, intercept + slope * x_line, color="black", linewidth=2)
    
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Catholic share 1871 (%)", fontsize=11)
    ax.set_ylabel("Change in CBR (late 1880s − early 1860s)", fontsize=11)
    ax.set_title("Fertility convergence: Did Catholic counties decline faster?",
                 fontsize=12, fontweight="bold")
    ax.text(0.05, 0.95, f"β = {slope:.4f} (p = {p:.3f})",
            transform=ax.transAxes, fontsize=10, va="top")
    
    plt.tight_layout()
    return fig, merged


# ===================================================================
# 4. Marriage → Birth pipeline (lagged effects)
# ===================================================================

def marriage_to_birth_pipeline(df: pd.DataFrame):
    """
    Test whether the marriage rate effect leads to a lagged birth effect.
    """
    df = df.copy().sort_values(["Code", "Year"])
    
    df["cath_x_post_lag1"] = df.groupby("Code")["cath_share_x_post"].shift(1)
    df["cath_x_post_lag2"] = df.groupby("Code")["cath_share_x_post"].shift(2)
    
    results = {}
    
    for lag_var, label in [
        ("cath_share_x_post", "Contemporaneous"),
        ("cath_x_post_lag1", "1-year lag"),
        ("cath_x_post_lag2", "2-year lag"),
    ]:
        res = _safe_panel_ols(df, "cbr", [lag_var, "ln_pop"])
        
        results[label] = {
            "coef": res.params[lag_var],
            "se": res.std_errors[lag_var],
            "pval": res.pvalues[lag_var],
        }
        print(f"{label}: β = {results[label]['coef']:.4f} "
              f"(SE = {results[label]['se']:.4f}, p = {results[label]['pval']:.3f})")
    
    return results


# ===================================================================
# 5. Dose-response plot
# ===================================================================

def dose_response_plot(df: pd.DataFrame, savepath: str = None):
    """
    Bin counties by Catholic share and plot fertility change for each bin.
    """
    df = df.copy()
    
    bins = [0, 5, 20, 50, 80, 95, 100]
    labels = ["0-5%", "5-20%", "20-50%", "50-80%", "80-95%", "95-100%"]
    
    df["cath_bin"] = pd.cut(
        df["cath_share"], bins=bins, labels=labels, include_lowest=True
    )
    
    results = []
    for bin_label in labels:
        sub = df[df["cath_bin"] == bin_label]
        if sub["Code"].nunique() < 5:
            continue
        
        pre = sub[sub["Year"] < 1873]["cbr"].mean()
        post = sub[sub["Year"] >= 1873]["cbr"].mean()
        n = sub["Code"].nunique()
        
        results.append({
            "bin": bin_label,
            "pre_cbr": pre,
            "post_cbr": post,
            "change": post - pre,
            "n_counties": n,
        })
    
    res_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: levels
    ax = axes[0]
    x = range(len(res_df))
    ax.bar([i - 0.15 for i in x], res_df["pre_cbr"], width=0.3,
           color="#2471A3", alpha=0.7, label="Pre-Kulturkampf")
    ax.bar([i + 0.15 for i in x], res_df["post_cbr"], width=0.3,
           color="#C0392B", alpha=0.7, label="Post-Kulturkampf")
    ax.set_xticks(list(x))
    ax.set_xticklabels(res_df["bin"], rotation=45)
    ax.set_ylabel("Mean CBR (per 1,000)")
    ax.set_title("Fertility levels by Catholic share bin")
    ax.legend()
    
    # Right panel: changes
    ax = axes[1]
    colors = ["#C0392B" if c > 0 else "#2471A3" for c in res_df["change"]]
    ax.bar(list(x), res_df["change"], color=colors, alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(res_df["bin"], rotation=45)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Change in CBR (post − pre)")
    ax.set_title("Fertility change by Catholic share bin")
    
    for i, row in enumerate(res_df.itertuples()):
        ax.text(i, row.change + 0.1, f"n={row.n_counties}",
                ha="center", fontsize=8)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    print(res_df.to_string(index=False))
    return fig, res_df


# ===================================================================
# 6. Infant mortality channel
# ===================================================================

def infant_mortality_did(df: pd.DataFrame):
    """
    Run the baseline DiD with infant mortality rate as outcome.
    """
    res = _safe_panel_ols(df, "infant_mortality_rate", ["cath_share_x_post", "ln_pop"])
    
    print("DiD: Infant Mortality Rate ~ CathShare × Post")
    print(f"  β = {res.params['cath_share_x_post']:.4f} "
          f"(SE = {res.std_errors['cath_share_x_post']:.4f}, "
          f"p = {res.pvalues['cath_share_x_post']:.3f})")
    print(f"  N = {int(res.nobs)}")
    
    return res