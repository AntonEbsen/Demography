"""
plots.py
========
Visualization functions for the Kulturkampf–fertility paper.

Usage (from notebook):
    from src.visualization.plots import plot_event_study, plot_fertility_trends, plot_cath_distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Consistent style
COLORS = {
    "catholic": "#C0392B",
    "protestant": "#2471A3",
    "neutral": "#555555",
    "ci": "#D5E8D4",
    "kulturkampf": "#E8DAEF",
}


def plot_counterfactual_paths(
    df: pd.DataFrame,
    iv_coef: float,
    outcome: str = "cbr",
    ylabel: str = "Crude birth rate (per 1,000)",
    title: str = "Observed vs.\\ counterfactual fertility paths (no Kulturkampf)",
    high_threshold: float = 75.0,
    low_threshold: float = 25.0,
    savepath: str | None = None,
):
    """
    Observed and IV-counterfactual outcome paths by Catholic share.

    For each county-year observation, the counterfactual outcome is

        Y_{it}^{cf} = Y_{it} - beta_IV * cath_share_i * 1[t >= 1873]

    i.e. the realised outcome with the Kulturkampf-attributable component
    netted out using the 2SLS estimate from ``run_iv_did``. Plots the
    observed and counterfactual annual means for high-Catholic
    (cath_share > high_threshold) and low-Catholic (cath_share < low_threshold)
    counties side by side.
    """
    df = df.copy()
    df["post"] = (df["Year"] >= 1873).astype(int)
    df["cf"] = df[outcome] - iv_coef * df["cath_share"] * df["post"]

    df["group"] = pd.cut(
        df["cath_share"],
        bins=[-0.001, low_threshold, high_threshold, 100.001],
        labels=["low", "mid", "high"],
    )
    keep = df[df["group"].isin(["low", "high"])].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "high": {"color": COLORS["catholic"], "label_obs": "High Catholic ($>$75\\%)"},
        "low": {"color": COLORS["protestant"], "label_obs": "Low Catholic ($<$25\\%)"},
    }
    for grp, st in styles.items():
        sub = keep[keep["group"] == grp]
        obs = sub.groupby("Year")[outcome].mean()
        cf = sub.groupby("Year")["cf"].mean()
        ax.plot(obs.index, obs.values, color=st["color"], linewidth=2,
                label=f"{st['label_obs']}, observed")
        ax.plot(cf.index, cf.values, color=st["color"], linewidth=1.5,
                linestyle="--", alpha=0.85,
                label=f"{st['label_obs']}, counterfactual")

    ax.axvspan(1872, 1878, alpha=0.15, color=COLORS["kulturkampf"],
               label="Kulturkampf (1872--1878)")
    ax.axvline(1873, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_fertility_trends(
    df: pd.DataFrame,
    outcome: str = "cbr",
    ylabel: str = "Crude birth rate (per 1,000)",
    title: str = "Fertility trends by Catholic share",
    savepath: str = None,
):
    """
    Plot average fertility over time for high- vs low-Catholic counties.
    Shades the Kulturkampf period (1872-1878).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, COLORS["catholic"]),
        ("Low Catholic (≤50%)", df["high_cath"] == 0, COLORS["protestant"]),
    ]:
        group = df[mask].groupby("Year")[outcome].mean()
        ax.plot(group.index, group.values, color=color, linewidth=2, label=label)
    
    # Shade Kulturkampf period
    ax.axvspan(1872, 1878, alpha=0.15, color=COLORS["kulturkampf"], 
               label="Kulturkampf (1872–1878)")
    
    # Reference line for May Laws
    ax.axvline(1873, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1873.2, ax.get_ylim()[1] * 0.98, "May Laws\n1873", 
            fontsize=8, color="grey", va="top")
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_event_study(
    coefs: pd.DataFrame,
    ref_year: int = 1872,
    title: str = "Event study: Catholic share × Year",
    ylabel: str = "Coefficient on CathShare × Year",
    savepath: str = None,
):
    """
    Plot event-study coefficients with 95% confidence intervals.
    
    Parameters
    ----------
    coefs : pd.DataFrame
        Output from regressions.run_event_study()['coefs'].
        Must have columns: Year, beta, ci_lo, ci_hi.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Shade Kulturkampf
    ax.axvspan(1872, 1878, alpha=0.12, color=COLORS["kulturkampf"])
    
    # Zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    
    # Confidence intervals
    ax.fill_between(
        coefs["Year"], coefs["ci_lo"], coefs["ci_hi"],
        alpha=0.25, color=COLORS["catholic"],
    )
    
    # Point estimates
    ax.plot(
        coefs["Year"], coefs["beta"],
        color=COLORS["catholic"], linewidth=2, marker="o", markersize=4,
    )
    
    # Mark reference year
    ax.scatter(
        [ref_year], [0], color="black", s=80, zorder=5, 
        marker="D", label=f"Reference year ({ref_year})",
    )
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_cath_distribution(
    df: pd.DataFrame,
    savepath: str = None,
):
    """
    Histogram of Catholic population shares across counties.
    Shows the bimodal distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # One observation per county
    county_cath = df.groupby("Code")["cath_share"].first()
    
    ax.hist(
        county_cath, bins=30, color=COLORS["catholic"], 
        alpha=0.7, edgecolor="white", linewidth=0.5,
    )
    
    ax.axvline(50, color="black", linestyle="--", linewidth=1.2,
               label="50% threshold")
    
    ax.set_xlabel("Catholic population share (%)", fontsize=11)
    ax.set_ylabel("Number of counties", fontsize=11)
    ax.set_title("Distribution of Catholic shares across Prussian counties (1871)", 
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    
    # Annotate counts
    n_high = (county_cath > 50).sum()
    n_low = (county_cath <= 50).sum()
    ax.text(0.02, 0.95, f"≤50%: {n_low} counties\n>50%: {n_high} counties",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_robustness_table(
    rob_df: pd.DataFrame,
    savepath: str = None,
):
    """
    Coefficient plot for robustness checks.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    y_pos = range(len(rob_df))
    
    ax.errorbar(
        rob_df["Coefficient"], y_pos,
        xerr=1.96 * rob_df["SE"],
        fmt="o", color=COLORS["catholic"], 
        capsize=4, markersize=6, linewidth=1.5,
    )
    
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(rob_df["Specification"], fontsize=9)
    ax.set_xlabel("Coefficient (95% CI)", fontsize=11)
    ax.set_title("Robustness checks", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax
