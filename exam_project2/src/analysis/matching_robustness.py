"""
matching_robustness.py
======================
Two analyses that use the 1849 pre-treatment variables WITHOUT putting
them in the regression as time-invariant controls (which causes the 
within-FE problems we saw).

Option B: matched-sample robustness
   Restrict the sample to counties with similar pre-treatment 
   characteristics (matching on 1849 variables), then re-run the 
   baseline DiD. Tests whether the null holds when high- and 
   low-Catholic counties are made comparable on observables.

Option C: pre-treatment balance table
   Compare high- vs low-Catholic counties on baseline 1849 / 1816-21
   characteristics. Goes in the data section of the paper.

Usage (from notebook):
    from src.analysis.matching_robustness import (
        pretreatment_balance_table,
        matched_sample_did,
    )
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.analysis.regressions import run_baseline_did


# ===================================================================
# Option C: Pre-treatment balance table
# ===================================================================

def pretreatment_balance_table(
    panel: pd.DataFrame,
    variables: Optional[dict] = None,
    cath_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Compare high- vs low-Catholic counties on pre-treatment characteristics.
    
    Returns a table suitable for the paper's data section: shows means,
    differences, and t-test p-values for the two groups.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Analysis panel with 1849/1816-21 variables already merged.
    variables : dict
        Map of {label: column_name}. Defaults to a curated set.
    cath_threshold : float
        Catholic share cutoff for grouping (default: 50%).
    
    Returns
    -------
    pd.DataFrame with columns:
        Variable, High-Catholic mean, Low-Catholic mean, Difference, p-value, N
    """
    if variables is None:
        variables = {
            # 1871 (treatment-period) characteristics
            "Population (log, 1871)": "lnpop",
            "Urban share (%)": "f_urban",
            "Jewish share (%)": "f_jew",
            "Young (under 15) share (%)": "f_young",
            "Female share (%)": "f_fem",
            "Born in locality (%)": "f_ortsgeb",
            "Household size": "hhsize",
            # Pre-Kulturkampf 1849 baseline
            "Fertile female share, 1849 (%)": "share_female_fertile_1849",
            "School attendance, 1849": "school_attendance_1849",
            "Catholic priests / 1000, 1849": "cath_priests_per_1000_1849",
            "Married men share, 1849": "share_married_men_1849",
            "Mean family size, 1849": "mean_family_size_1849",
            # Long-run baseline
            "Illegitimacy rate, 1816-21 (%)": "oow_share_1816_21",
        }
    
    from scipy import stats
    
    # Use one observation per county
    cross = panel.groupby("Code").agg({
        **{v: "first" for v in variables.values() if v in panel.columns},
        "cath_share": "first",
    }).reset_index()
    
    cross["high_cath"] = (cross["cath_share"] > cath_threshold).astype(int)
    
    rows = []
    for label, col in variables.items():
        if col not in cross.columns:
            continue
        
        sub = cross[[col, "high_cath"]].dropna()
        high = sub[sub["high_cath"] == 1][col]
        low = sub[sub["high_cath"] == 0][col]
        
        if len(high) < 5 or len(low) < 5:
            continue
        
        # Welch's t-test (unequal variance)
        tstat, pval = stats.ttest_ind(high, low, equal_var=False)
        
        rows.append({
            "Variable": label,
            "High-Cath mean": round(high.mean(), 3),
            "Low-Cath mean": round(low.mean(), 3),
            "Difference": round(high.mean() - low.mean(), 3),
            "p-value": round(pval, 3),
            "N high": len(high),
            "N low": len(low),
        })
    
    table = pd.DataFrame(rows)

    print("=" * 70)
    print(f"PRE-TREATMENT BALANCE TABLE")
    print(f"(High-Catholic = >{cath_threshold:.0f}% Catholic, "
          f"Low-Catholic = <={cath_threshold:.0f}% Catholic)")
    print("=" * 70)

    if table.empty:
        missing = [v for v in variables.values() if v not in cross.columns]
        print("No variables available for the balance table.")
        print(f"Missing columns in panel: {missing}")
        print("→ Did you call `merge_ipehd_controls(panel)` before this?")
        return table

    print(table.to_string(index=False))

    n_imbalanced = (table["p-value"] < 0.05).sum()
    print(f"\n{n_imbalanced} of {len(table)} variables show significant "
          f"imbalance (p < 0.05)")
    print("→ This motivates the matched-sample robustness check.")

    return table


# ===================================================================
# Option B: Matched-sample DiD
# ===================================================================

def matched_sample_did(
    panel: pd.DataFrame,
    matching_vars: Optional[list] = None,
    outcome: str = "cbr",
    n_bins: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Re-run the baseline DiD on a sample where high- and low-Catholic
    counties have been matched on pre-treatment characteristics.
    
    Method
    ------
    1. Compute a propensity score: predict P(high_cath=1) from 
       pre-treatment variables using logistic regression.
    2. Bin counties into n_bins propensity-score strata.
    3. Within each stratum, keep only counties from strata that contain 
       BOTH high- and low-Catholic counties (common support).
    4. Run the baseline DiD on this restricted sample.
    
    The intuition: instead of putting the controls IN the regression
    (which fails because they are time-invariant), we use them to define
    a sub-sample where high- and low-Catholic counties are similar on
    observables. The DiD on this matched sample is then less sensitive
    to confounding by these characteristics.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Analysis panel with iPEHD + 1849 variables merged.
    matching_vars : list
        Pre-treatment variables to match on. Default: a curated set.
    outcome : str
        Dependent variable.
    n_bins : int
        Number of propensity-score strata.
    
    Returns
    -------
    dict with keys: 'baseline_full', 'baseline_matched', 'matched_panel',
                    'common_support_share'
    """
    from sklearn.linear_model import LogisticRegression
    
    if matching_vars is None:
        # Use only variables with broad coverage to avoid losing the sample
        matching_vars = [
            "lnpop",        # log population (very broad coverage)
            "f_urban",      # urbanization
            "f_jew",        # Jewish population share
            "f_young",      # demographic structure
            "hhsize",       # household structure
        ]
    
    if verbose:
        print("=" * 70)
        print("MATCHED-SAMPLE DiD ROBUSTNESS")
        print("=" * 70)
        print(f"Matching variables: {matching_vars}")
    
    # ---------- Step 1: Get cross-section for matching ----------
    cross = panel.groupby("Code").agg({
        **{v: "first" for v in matching_vars if v in panel.columns},
        "cath_share": "first",
    }).reset_index()
    
    available_match_vars = [v for v in matching_vars if v in cross.columns]
    if len(available_match_vars) < len(matching_vars):
        missing = set(matching_vars) - set(available_match_vars)
        if verbose:
            print(f"Note: missing matching variables {missing}")
    
    cross_clean = cross.dropna(subset=available_match_vars)
    cross_clean["high_cath"] = (cross_clean["cath_share"] > 50).astype(int)
    
    if verbose:
        print(f"\nCounties with all matching vars: "
              f"{len(cross_clean)} of {len(cross)}")
        print(f"  High-Catholic: {cross_clean['high_cath'].sum()}")
        print(f"  Low-Catholic:  {(1-cross_clean['high_cath']).sum()}")
    
    # ---------- Step 2: Estimate propensity score ----------
    X = cross_clean[available_match_vars].values
    y = cross_clean["high_cath"].values
    
    # Standardise for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std
    
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_scaled, y)
    cross_clean["pscore"] = lr.predict_proba(X_scaled)[:, 1]
    
    if verbose:
        print(f"\nPropensity score range:")
        print(f"  Among high-Cath: "
              f"{cross_clean[cross_clean['high_cath']==1]['pscore'].min():.3f} "
              f"to {cross_clean[cross_clean['high_cath']==1]['pscore'].max():.3f}")
        print(f"  Among low-Cath:  "
              f"{cross_clean[cross_clean['high_cath']==0]['pscore'].min():.3f} "
              f"to {cross_clean[cross_clean['high_cath']==0]['pscore'].max():.3f}")
    
    # ---------- Step 3: Bin into propensity strata, keep common support ----------
    cross_clean["pscore_bin"] = pd.qcut(
        cross_clean["pscore"], q=n_bins, labels=False, duplicates="drop"
    )
    
    bin_counts = cross_clean.groupby("pscore_bin")["high_cath"].agg(
        ["sum", "count"]
    )
    bin_counts["n_low"] = bin_counts["count"] - bin_counts["sum"]
    bin_counts = bin_counts.rename(columns={"sum": "n_high"})
    
    # Keep only bins with at least 3 of each type
    common_support_bins = bin_counts[
        (bin_counts["n_high"] >= 3) & (bin_counts["n_low"] >= 3)
    ].index.tolist()
    
    matched_codes = cross_clean[
        cross_clean["pscore_bin"].isin(common_support_bins)
    ]["Code"].tolist()
    
    if verbose:
        print(f"\nPropensity score bins:")
        print(bin_counts.to_string())
        print(f"\nBins with common support ({n_bins} requested, "
              f"{len(common_support_bins)} kept): "
              f"{common_support_bins}")
        print(f"Matched sample: {len(matched_codes)} counties "
              f"(from {len(cross_clean)} eligible)")
    
    matched_panel = panel[panel["Code"].isin(matched_codes)].copy()
    common_support_share = len(matched_codes) / len(cross_clean) if len(cross_clean) else 0
    
    # ---------- Step 4: Run DiD on full and matched samples ----------
    if verbose:
        print("\n" + "=" * 70)
        print("DiD ON FULL SAMPLE (for comparison)")
        print("=" * 70)
    full_result = run_baseline_did(
        panel, outcome=outcome, treatment="continuous", controls=["ln_pop"],
    )
    print(full_result["summary"])
    
    if verbose:
        print("\n" + "=" * 70)
        print("DiD ON MATCHED SAMPLE")
        print("=" * 70)
    matched_result = run_baseline_did(
        matched_panel, outcome=outcome, treatment="continuous", controls=["ln_pop"],
    )
    print(matched_result["summary"])
    
    # Side-by-side
    if verbose:
        b_full = full_result["result"].params["cath_share_x_post"]
        s_full = full_result["result"].std_errors["cath_share_x_post"]
        p_full = full_result["result"].pvalues["cath_share_x_post"]
        b_match = matched_result["result"].params["cath_share_x_post"]
        s_match = matched_result["result"].std_errors["cath_share_x_post"]
        p_match = matched_result["result"].pvalues["cath_share_x_post"]
        
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"{'Specification':<25} {'β':>10} {'SE':>10} {'p':>8} {'N counties':>12}")
        print("-" * 70)
        print(f"{'Full sample':<25} {b_full:>10.4f} {s_full:>10.4f} "
              f"{p_full:>8.3f} {panel['Code'].nunique():>12}")
        print(f"{'Matched sample':<25} {b_match:>10.4f} {s_match:>10.4f} "
              f"{p_match:>8.3f} {matched_panel['Code'].nunique():>12}")
        
        if abs(b_full - b_match) < max(s_full, s_match):
            print("\n→ Coefficients are similar (within one SE).")
            print("  Null result is robust to matching on observables.")
        else:
            print("\n→ Coefficients differ meaningfully — matching changed the result.")
    
    return {
        "baseline_full": full_result,
        "baseline_matched": matched_result,
        "matched_panel": matched_panel,
        "matched_codes": matched_codes,
        "common_support_share": common_support_share,
    }