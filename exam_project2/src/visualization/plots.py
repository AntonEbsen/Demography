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


def plot_population_and_migration(
    panel: pd.DataFrame,
    enforcement_years: tuple[int, int] = (1872, 1878),
    rollback_years: tuple[int, int] = (1880, 1887),
    savepath: str | None = None,
):
    """
    Two-panel figure: (left) regional population indices (1862 = 100),
    (right) implied net migration rate per 1{,}000 population, by sub-region.

    Net migration is computed residually: implied_migration = pop_change -
    natural_increase, where natural_increase = Birtot - Dthtot. This is
    standard in historical demography when direct migration registers are
    unavailable.

    Polish provinces (POS, BRO), German Catholic provinces (KOL, KOB, TRI,
    AAC, OPP, MUN), and Protestant (rest) are shown separately so the
    reader can see whether the post-1873 demographic response in Polish
    Catholic counties is driven by emigration rather than fertility.
    """
    polish_rbs = ("POS", "BRO")
    german_cath_rbs = ("KOL", "KOB", "TRI", "AAC", "OPP", "MUN")

    df = panel.copy()
    df["region"] = np.where(
        df["Rb"].isin(polish_rbs), "Polish",
        np.where(df["Rb"].isin(german_cath_rbs), "German Catholic",
                 "Protestant (rest)"),
    )

    # Net migration estimate: pop change - natural increase.
    df = df.sort_values(["Code", "Year"])
    df["pop_change"] = df.groupby("Code")["Poptot"].diff()
    df["natural_increase"] = df["Birtot"] - df["Dthtot"]
    df["implied_migration"] = df["pop_change"] - df["natural_increase"]
    df["migration_rate"] = df["implied_migration"] / df["Poptot"] * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    region_colors = {
        "Polish": COLORS["catholic"],
        "German Catholic": COLORS["protestant"],
        "Protestant (rest)": COLORS["neutral"],
    }

    # Left panel: population index (1862 = 100)
    ax_l = axes[0]
    pop = df.groupby(["region", "Year"])["Poptot"].sum().unstack("region")
    base = pop.loc[pop.index.min()]
    pop_index = pop / base * 100.0
    for region in ("Polish", "German Catholic", "Protestant (rest)"):
        ax_l.plot(pop_index.index, pop_index[region], color=region_colors[region],
                  linewidth=2, marker="o", markersize=3, label=region)
    ax_l.axvspan(*enforcement_years, alpha=0.15, color=COLORS["kulturkampf"])
    ax_l.axvspan(*rollback_years, alpha=0.12, color=COLORS["protestant"])
    ax_l.axhline(100, color="black", linewidth=0.6, linestyle=":")
    ax_l.set_xlabel("Year", fontsize=11)
    ax_l.set_ylabel("Population index (1862 = 100)", fontsize=11)
    ax_l.set_title("Population trajectory by sub-region", fontsize=12, fontweight="bold")
    ax_l.legend(loc="upper left", fontsize=9)
    ax_l.grid(axis="y", alpha=0.3)

    # Right panel: 5-year rolling median net migration rate by region
    ax_r = axes[1]
    # Use median to suppress year-on-year noise from county-boundary edits.
    mig_by_year = (
        df.groupby(["region", "Year"])["migration_rate"]
        .median()
        .unstack("region")
    )
    smoothed = mig_by_year.rolling(window=3, center=True, min_periods=1).mean()
    for region in ("Polish", "German Catholic", "Protestant (rest)"):
        ax_r.plot(smoothed.index, smoothed[region], color=region_colors[region],
                  linewidth=2, marker="o", markersize=3, label=region)
    ax_r.axvspan(*enforcement_years, alpha=0.15, color=COLORS["kulturkampf"],
                 label=f"Kulturkampf ({enforcement_years[0]}--{enforcement_years[1]})")
    ax_r.axvspan(*rollback_years, alpha=0.12, color=COLORS["protestant"],
                 label=f"Rollback ({rollback_years[0]}--{rollback_years[1]})")
    ax_r.axhline(0, color="black", linewidth=0.6, linestyle="-")
    ax_r.set_xlabel("Year", fontsize=11)
    ax_r.set_ylabel("Implied net migration rate (per 1{,}000 pop)", fontsize=11)
    ax_r.set_title("Net migration: pop change − natural increase",
                   fontsize=12, fontweight="bold")
    ax_r.legend(loc="lower left", fontsize=8, ncol=1)
    ax_r.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_lexis_diagram(
    title: str = "Lexis diagram: cohorts and the Kulturkampf",
    year_range: tuple[int, int] = (1822, 1900),
    age_range: tuple[int, int] = (0, 50),
    repro_age_range: tuple[int, int] = (15, 49),
    enforcement_years: tuple[int, int] = (1872, 1878),
    rollback_years: tuple[int, int] = (1880, 1887),
    cohort_step: int = 5,
    savepath: str | None = None,
):
    """
    Lexis diagram showing cohorts crossing the Kulturkampf and the rollback
    period.

    The diagram plots calendar year (x-axis) against age (y-axis). Each
    diagonal line represents a single birth cohort progressing through life.
    The reproductive-age band (15--49) and the two policy windows are shaded
    so that the reader sees at a glance which cohorts had reproductive
    careers intersecting either or both.

    Useful for the demographic-mechanism narrative: the cohorts in the
    intersection (born roughly 1823--1872) are the ones whose marriage and
    fertility decisions could be affected by the Kulturkampf.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Reproductive-age band (highlighted)
    ax.axhspan(repro_age_range[0], repro_age_range[1], alpha=0.10,
               color="#888888", zorder=0)

    # Policy windows
    ax.axvspan(*enforcement_years, alpha=0.20, color=COLORS["catholic"],
               label=f"Kulturkampf enforcement ({enforcement_years[0]}–{enforcement_years[1]})")
    ax.axvspan(*rollback_years, alpha=0.18, color=COLORS["protestant"],
               label=f"Rollback ({rollback_years[0]}–{rollback_years[1]})")

    # Cohort diagonal lines: a person born in year c is at age (year - c).
    cohort_first = year_range[0] - age_range[1]
    cohort_last = year_range[1] - age_range[0]
    cohorts = np.arange(cohort_first, cohort_last + 1)
    annotated_cohorts = list(range(cohort_first - cohort_first % cohort_step,
                                    cohort_last, cohort_step))

    for c in cohorts:
        years = np.arange(max(year_range[0], c + age_range[0]),
                          min(year_range[1], c + age_range[1]) + 1)
        if len(years) < 2:
            continue
        ages = years - c
        is_annotated = c in annotated_cohorts
        ax.plot(years, ages,
                color="#444444" if is_annotated else "#cccccc",
                linewidth=1.2 if is_annotated else 0.4,
                alpha=0.95 if is_annotated else 0.6,
                zorder=2 if is_annotated else 1)
        if is_annotated and years[-1] >= year_range[1] - cohort_step:
            ax.annotate(
                f"b. {c}",
                xy=(years[-1], ages[-1]),
                xytext=(2, 0), textcoords="offset points",
                fontsize=7, color="#444444", va="center",
            )

    # Highlight the cohorts whose reproductive years intersect the Kulturkampf
    enforce_lo = enforcement_years[0] - repro_age_range[1]  # b. 1823 turns 49 in 1872
    enforce_hi = enforcement_years[1] - repro_age_range[0]  # b. 1863 turns 15 in 1878
    rollback_lo = rollback_years[0] - repro_age_range[1]
    rollback_hi = rollback_years[1] - repro_age_range[0]
    intersect_lo = min(enforce_lo, rollback_lo)
    intersect_hi = max(enforce_hi, rollback_hi)

    for c in (intersect_lo, intersect_hi):
        years = np.arange(max(year_range[0], c + age_range[0]),
                          min(year_range[1], c + age_range[1]) + 1)
        if len(years) < 2:
            continue
        ages = years - c
        ax.plot(years, ages, color="#000000", linewidth=2.0, alpha=0.9,
                zorder=4, label=f"Bounding cohort (b. {c})" if c == intersect_lo else None)

    # Reference horizontal lines at boundaries of the reproductive interval
    for y in repro_age_range:
        ax.axhline(y, color="#444444", linewidth=0.6, linestyle=":")

    # Labels and aesthetics
    ax.set_xlim(*year_range)
    ax.set_ylim(*age_range)
    ax.set_xlabel("Calendar year", fontsize=11)
    ax.set_ylabel("Age", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(axis="both", alpha=0.2)

    # Annotation: which cohorts intersect
    note = (
        f"Cohorts born {intersect_lo}–{intersect_hi} had at least part of\n"
        f"their reproductive career (ages {repro_age_range[0]}–{repro_age_range[1]})\n"
        f"intersect the Kulturkampf and/or rollback window."
    )
    ax.annotate(
        note,
        xy=(year_range[0] + 2, age_range[1] - 8),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFEE",
                  edgecolor="#888888"),
        ha="left", va="top",
    )

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


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


def plot_event_study_cbr_gfr(
    coefs_cbr: pd.DataFrame,
    coefs_gfr: pd.DataFrame,
    pretrends_cbr: dict | None = None,
    pretrends_gfr: dict | None = None,
    ref_year: int = 1872,
    savepath: str | None = None,
):
    """
    Side-by-side event-study figure: CBR (left) and the static-1871 GFR
    (right). The GFR panel addresses the standard demographic critique that
    CBR is mechanically affected by age structure -- a Demography reader
    can verify directly that the event-study shape, the timing of the
    departure from zero, and the pre-trends test all carry over to the
    age-standardised outcome.

    Each panel shades the Kulturkampf enforcement window (1872--1878),
    plots the 95% CI ribbon, the point estimates, and the omitted
    reference year. If pre-trends Wald dictionaries (output of
    ``regressions.pretrends_wald_test``) are passed, the $\\chi^2$
    statistic, df, and $p$-value are annotated in the upper-left corner of
    each panel.

    Parameters
    ----------
    coefs_cbr, coefs_gfr : pd.DataFrame
        Output of ``run_event_study(...)['coefs']`` for each outcome.
        Must have columns ``Year, beta, ci_lo, ci_hi``.
    pretrends_cbr, pretrends_gfr : dict, optional
        Output of ``pretrends_wald_test(...)``. If provided, annotated.
    ref_year : int
        Reference (omitted) event-study year.
    savepath : str, optional
        Where to write the PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    panels = [
        (axes[0], coefs_cbr, pretrends_cbr,
         "Crude birth rate",
         "Coefficient on CathShare $\\times$ Year (per 1,000 pop)"),
        (axes[1], coefs_gfr, pretrends_gfr,
         "GFR (1871 base)",
         "Coefficient on CathShare $\\times$ Year (per 1,000 women 15--49)"),
    ]
    for ax, coefs, pre, title, ylabel in panels:
        ax.axvspan(1872, 1878, alpha=0.12, color=COLORS["kulturkampf"])
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.fill_between(
            coefs["Year"], coefs["ci_lo"], coefs["ci_hi"],
            alpha=0.25, color=COLORS["catholic"],
        )
        ax.plot(
            coefs["Year"], coefs["beta"],
            color=COLORS["catholic"], linewidth=2, marker="o", markersize=4,
        )
        ax.scatter(
            [ref_year], [0], color="black", s=80, zorder=5,
            marker="D", label=f"Reference year ({ref_year})",
        )
        if pre is not None:
            txt = (
                f"Pre-trends Wald $\\chi^2$ = {pre['wald_chi2']:.2f} "
                f"(df = {pre['df']})\n"
                f"$p$-value = {pre['p_value']:.3f}"
            )
            ax.text(
                0.02, 0.98, txt,
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="grey", alpha=0.85),
            )
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", frameon=True, fontsize=8)

    fig.suptitle(
        "Event study: CBR vs General Fertility Rate (1871 denominator)",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, axes


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
