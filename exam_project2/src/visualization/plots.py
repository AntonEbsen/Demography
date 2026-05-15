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
    enforcement_years: tuple[int, int] = (1873, 1878),
    rollback_years: tuple[int, int] = (1880, 1887),
    savepath: str = None,
):
    """
    Plot event-study coefficients with 95% confidence intervals.

    Shades the two Kulturkampf policy phases distinctly so the reader
    can see whether coefficients move during legislative *enforcement*
    (1873-1878, light purple) or during the gradual *rollback*
    (1880-1887, neutral grey), or both. Year 1879 (transition) and
    1888+ (post-rollback) are unshaded.

    Parameters
    ----------
    coefs : pd.DataFrame
        Output from regressions.run_event_study()['coefs'].
        Must have columns: Year, beta, ci_lo, ci_hi.
    ref_year : int
        Reference (omitted) event-study year, marked with a diamond.
    enforcement_years, rollback_years : (int, int)
        Inclusive year ranges for the two shaded policy windows.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Two-phase Kulturkampf shading.
    ax.axvspan(
        enforcement_years[0] - 0.5, enforcement_years[1] + 0.5,
        alpha=0.15, color="#9B59B6",
        label=f"Enforcement ({enforcement_years[0]}-{enforcement_years[1]})",
    )
    ax.axvspan(
        rollback_years[0] - 0.5, rollback_years[1] + 0.5,
        alpha=0.18, color="#7F8C8D",
        label=f"Rollback ({rollback_years[0]}-{rollback_years[1]})",
    )

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
        label="Point estimate (95% CI ribbon)",
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


def plot_event_study_cbr_ig(
    coefs_cbr: pd.DataFrame,
    coefs_ig: pd.DataFrame,
    pretrends_cbr: dict | None = None,
    pretrends_ig: dict | None = None,
    ref_year: int = 1872,
    enforcement_years: tuple[int, int] = (1873, 1878),
    rollback_years: tuple[int, int] = (1880, 1887),
    savepath: str | None = None,
):
    """
    Side-by-side event-study figure: CBR (left) and Coale's $I_g$
    (right). $I_g$ is the Princeton EFP marital-fertility index --
    legitimate births per married woman 15--49, normalised against the
    Hutterite natural-fertility maximum -- and the headline outcome in
    Galloway, Hammel & Lee (1994). Reading the two panels jointly tells
    a Demography-aware reader whether the event-study shape and
    pre-trends conclusion hold both for the broad CBR and for the
    nuptiality-netted marital-fertility index.

    Each panel shades the two policy phases distinctly:

    - **Enforcement (1873--1878)** -- the May Laws are in force, Catholic
      clergy face expulsions and incarceration, civil marriage replaces
      Catholic-parish marriage (1875), parochial-school inspection is
      transferred to the state.
    - **Rollback (1880--1887)** -- progressive repeal beginning under
      Leo XIII's diplomatic settlement; legislation still on the books
      but weakened; Polish-Catholic restrictions persist longer than
      German-Catholic ones (Polenausweisungen, 1885--86).
    - Years 1879 and 1888+ are the transition / post-rollback periods.

    The two shaded blocks let the reader see at a glance whether the
    event-study coefficients move during legislative enforcement, during
    rollback, or both -- the standard demographic-transition question
    for an institutional shock that was *gradually* reversed.

    Parameters
    ----------
    coefs_cbr, coefs_ig : pd.DataFrame
        Output of ``run_event_study(...)['coefs']`` for each outcome.
        Must have columns ``Year, beta, ci_lo, ci_hi``.
    pretrends_cbr, pretrends_ig : dict, optional
        Output of ``pretrends_wald_test(...)``. If provided, annotated.
    ref_year : int
        Reference (omitted) event-study year.
    enforcement_years, rollback_years : (int, int)
        Inclusive year ranges for the two shaded policy windows.
    savepath : str, optional
        Where to write the PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    panels = [
        (axes[0], coefs_cbr, pretrends_cbr,
         "Crude birth rate",
         "Coefficient on CathShare $\\times$ Year (per 1,000 mid-year pop)"),
        (axes[1], coefs_ig, pretrends_ig,
         "$I_g$ (Coale marital fertility)",
         "Coefficient on CathShare $\\times$ Year ($I_g$ units)"),
    ]
    enf_color = "#9B59B6"   # light purple for enforcement
    roll_color = "#7F8C8D"  # neutral grey for rollback

    for ax, coefs, pre, title, ylabel in panels:
        # Enforcement shading (light purple, 1873-1878). Use axvspan from
        # enf_start - 0.5 to enf_end + 0.5 to align block edges with
        # year-tick centres.
        ax.axvspan(
            enforcement_years[0] - 0.5, enforcement_years[1] + 0.5,
            alpha=0.15, color=enf_color,
            label=f"Enforcement ({enforcement_years[0]}-{enforcement_years[1]})",
        )
        # Rollback shading (neutral grey, 1880-1887).
        ax.axvspan(
            rollback_years[0] - 0.5, rollback_years[1] + 0.5,
            alpha=0.18, color=roll_color,
            label=f"Rollback ({rollback_years[0]}-{rollback_years[1]})",
        )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.fill_between(
            coefs["Year"], coefs["ci_lo"], coefs["ci_hi"],
            alpha=0.25, color=COLORS["catholic"],
        )
        ax.plot(
            coefs["Year"], coefs["beta"],
            color=COLORS["catholic"], linewidth=2, marker="o", markersize=4,
            label="Point estimate (95\\% CI ribbon)",
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
        "Event study: CBR vs $I_g$ (Coale marital fertility) "
        "-- enforcement (1873-78) and rollback (1880-87) shaded",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_cbr_war_context(
    panel: pd.DataFrame,
    outcome: str = "cbr",
    austro_prussian_year: int = 1866,
    franco_prussian_years: tuple[int, int] = (1870, 1871),
    kulturkampf_year: int = 1873,
    ylabel: str | None = None,
    title: str | None = None,
    savepath: str | None = None,
):
    """
    Raw-means time-series of CBR (or any outcome) by high-/low-Catholic
    group, 1862--1890, with wartime years shaded.

    Diagnostic for the pre-1873 CBR trend: if Protestant CBR dips
    during the Austro-Prussian War (1866) and the Franco-Prussian War
    (1870-71) but Catholic CBR does not, then the apparent "Catholic
    counties trending upward relative to Protestant counties" in
    1865-1872 is mechanical war-cohort effect rather than a behavioural
    pre-trend that threatens DiD identification.

    Cleanly: war-year shading on a means plot answers, at a glance,
    whether the pre-trend in the event-study figure is a parallel-
    trends violation or a Prussian-Army-recruitment-burden artefact.
    """
    if outcome not in panel.columns:
        raise KeyError(f"Outcome {outcome!r} not in panel.")

    df = panel.dropna(subset=[outcome, "high_cath"]).copy()
    df["group"] = df["high_cath"].map({0: "Low Catholic", 1: "High Catholic"})
    annual = (
        df.groupby(["Year", "group"])[outcome]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    y_max = max(annual.max()) * 1.06
    y_min = min(annual.min()) * 0.94

    # Shade the two Prussian wars.
    ax.axvspan(
        austro_prussian_year - 0.5, austro_prussian_year + 0.5,
        alpha=0.25, color="#7F8C8D",
        label=f"Austro-Prussian War ({austro_prussian_year})",
    )
    ax.axvspan(
        franco_prussian_years[0] - 0.5, franco_prussian_years[1] + 0.5,
        alpha=0.25, color="#34495E",
        label=f"Franco-Prussian War ({franco_prussian_years[0]}-{franco_prussian_years[1]})",
    )

    # Vertical line at the Kulturkampf May Laws.
    ax.axvline(
        kulturkampf_year, color="#C0392B", linestyle="--", linewidth=1.3,
        label=f"May Laws ({kulturkampf_year})",
    )

    # Two group lines.
    for grp, color, marker in [
        ("High Catholic", COLORS["catholic"], "o"),
        ("Low Catholic", COLORS["protestant"], "s"),
    ]:
        if grp not in annual.columns:
            continue
        ax.plot(
            annual.index, annual[grp],
            color=color, linewidth=2, marker=marker, markersize=5, label=grp,
        )

    ax.set_xlim(annual.index.min() - 0.5, annual.index.max() + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Year", fontsize=11)
    if ylabel is None:
        ylabel = "Mean " + outcome.upper().replace("_", " ")
    ax.set_ylabel(ylabel, fontsize=11)
    if title is None:
        title = (
            f"Mean {outcome.upper()} by Catholic-share group, "
            "1862-1890, with Prussian wars shaded"
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_zentrum_event_study(
    coefs: pd.DataFrame,
    enforcement_years: tuple[int, int] = (1873, 1878),
    rollback_years: tuple[int, int] = (1880, 1887),
    title: str | None = None,
    ylabel: str | None = None,
    savepath: str | None = None,
):
    """
    Event-study plot for the political-mobilisation DiD: year-by-year
    coefficients on ``cath_share x 1{Year = t}`` with 1871 as the
    omitted reference, plotted across the 7 Reichstag elections
    1871--1890.

    Each point is the additional Zentrum vote share (in percentage
    points of valid votes) per percentage point of `cath_share` at
    that election year, relative to the 1871 baseline. So a coefficient
    of $+0.24$ at 1874 means: comparing two counties differing by 1pp
    of `cath_share`, the gap in Zentrum vote share grew by 0.24 pp
    between 1871 and 1874.

    The Kulturkampf enforcement (1873--78) and rollback (1880--87)
    windows are shaded distinctly so the reader can see in which
    policy phase the political-mobilisation response peaked.

    Parameters
    ----------
    coefs : pd.DataFrame
        Output from ``run_political_mobilization_event_study(...)['coefs']``.
        Must have columns: Year, beta, ci_lo, ci_hi.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    y_max = max(coefs["ci_hi"]) * 1.10
    y_min = min(min(coefs["ci_lo"]) * 1.10, -0.02)

    # Two-phase Kulturkampf shading.
    ax.axvspan(
        enforcement_years[0] - 0.5, enforcement_years[1] + 0.5,
        alpha=0.15, color="#9B59B6",
        label=f"Enforcement ({enforcement_years[0]}-{enforcement_years[1]})",
    )
    ax.axvspan(
        rollback_years[0] - 0.5, rollback_years[1] + 0.5,
        alpha=0.18, color="#7F8C8D",
        label=f"Rollback ({rollback_years[0]}-{rollback_years[1]})",
    )

    # Zero line.
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    # 95% CI ribbon + point estimates.
    ax.fill_between(
        coefs["Year"], coefs["ci_lo"], coefs["ci_hi"],
        alpha=0.25, color=COLORS["catholic"],
    )
    ax.plot(
        coefs["Year"], coefs["beta"],
        color=COLORS["catholic"], linewidth=2.2, marker="o", markersize=7,
        label="Coef. on CathShare $\\times$ 1[Year$=t$] (95\\% CI)",
    )

    # Annotate each point.
    for _, row in coefs.iterrows():
        if abs(row["beta"]) < 1e-9:
            label = "ref."
        else:
            label = f"{row['beta']:+.3f}"
        ax.annotate(
            label,
            (row["Year"], row["beta"]),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8, fontweight="bold",
            color=COLORS["catholic"] if abs(row["beta"]) > 1e-9 else "black",
        )

    # Mark the 1871 reference with a black diamond.
    if (coefs["beta"].abs() < 1e-9).any():
        ref_year = int(coefs.loc[coefs["beta"].abs() < 1e-9, "Year"].iloc[0])
        ax.scatter(
            [ref_year], [0], color="black", s=80, zorder=5,
            marker="D", label=f"Reference year ({ref_year})",
        )

    ax.set_xticks(coefs["Year"].tolist())
    ax.set_xticklabels(coefs["Year"].astype(int).tolist())
    ax.set_xlim(coefs["Year"].min() - 1, coefs["Year"].max() + 1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Election year (Reichstag)", fontsize=11)
    if ylabel is None:
        ylabel = (
            "Coefficient on CathShare $\\times$ 1[Year$=t$] "
            "(Zentrum vote-share units)"
        )
    ax.set_ylabel(ylabel, fontsize=11)
    if title is None:
        title = (
            "Event study: Catholic political mobilisation, 1871-1890\n"
            "(1871 omitted; coefficients show post-Kulturkampf "
            "Zentrum-share gap per pp of cath\\_share)"
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_zentrum_mobilization(
    panel: pd.DataFrame,
    election_years: tuple[int, ...] = (1871, 1874, 1878, 1881, 1884, 1887, 1890),
    enforcement_years: tuple[int, int] = (1873, 1878),
    rollback_years: tuple[int, int] = (1880, 1887),
    savepath: str | None = None,
):
    """
    Time-varying Zentrum (Catholic Centre Party) vote share by
    Catholic-share group, across the 7 Reichstag elections 1871--1890.

    Galloway's seven election cross-sections span one pre-Kulturkampf
    election (1871) and six post-Kulturkampf elections covering
    enforcement (1874, 1878) and rollback (1881, 1884, 1887) plus one
    post-rollback (1890). Plotting mean Zentrum vote share at each
    election for High-Catholic vs Low-Catholic counties reveals the
    Catholic political-mobilisation response: in high-Catholic
    counties, Zentrum vote share roughly doubles between 1871 and 1874
    -- the *first* post-Kulturkampf election -- peaks in 1881 (early
    rollback), and never reverts to its pre-treatment level. This is
    the textbook backfire-effect of an institutional shock: rather
    than weakening Catholic identity, the Kulturkampf consolidated it
    into the Reichstag's most disciplined opposition bloc.

    The figure pairs with the formal political-mobilisation DiD in
    ``political_mobilization.run_political_mobilization_did``.
    """
    df = panel.dropna(subset=["zentrum_share_current"]).copy()
    df = df[df["Year"].isin(list(election_years))]
    df["group"] = df["high_cath"].map({0: "Low Catholic", 1: "High Catholic"})
    annual = (
        df.groupby(["Year", "group"])["zentrum_share_current"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    y_min, y_max = 0, max(annual.max()) * 1.10

    # Two-phase Kulturkampf shading.
    ax.axvspan(
        enforcement_years[0] - 0.5, enforcement_years[1] + 0.5,
        alpha=0.15, color="#9B59B6",
        label=f"Enforcement ({enforcement_years[0]}-{enforcement_years[1]})",
    )
    ax.axvspan(
        rollback_years[0] - 0.5, rollback_years[1] + 0.5,
        alpha=0.18, color="#7F8C8D",
        label=f"Rollback ({rollback_years[0]}-{rollback_years[1]})",
    )

    # Reference line at the 1871 pre-Kulturkampf level for each group.
    for grp, color in [("High Catholic", COLORS["catholic"]),
                       ("Low Catholic", COLORS["protestant"])]:
        if grp in annual.columns:
            base = annual.loc[1871, grp] if 1871 in annual.index else annual[grp].iloc[0]
            ax.axhline(base, color=color, linestyle=":", linewidth=1.0, alpha=0.6)

    # Two group lines.
    for grp, color, marker in [
        ("High Catholic", COLORS["catholic"], "o"),
        ("Low Catholic", COLORS["protestant"], "s"),
    ]:
        if grp not in annual.columns:
            continue
        ax.plot(
            annual.index, annual[grp],
            color=color, linewidth=2.2, marker=marker, markersize=7, label=grp,
        )
        # Annotate each point with its value
        for x, y in zip(annual.index, annual[grp]):
            if pd.notna(y):
                ax.annotate(
                    f"{y:.1f}",
                    (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=color, fontweight="bold",
                )

    ax.set_xlim(min(election_years) - 0.5, max(election_years) + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(list(election_years))
    ax.set_xticklabels(list(election_years))
    ax.set_xlabel("Election year (Reichstag)", fontsize=11)
    ax.set_ylabel("Mean Zentrum vote share (\\% of valid votes)", fontsize=11)
    ax.set_title(
        "Catholic political mobilisation: Zentrum vote share by Catholic-share group, 1871-1890\n"
        "(dotted lines = 1871 pre-Kulturkampf baseline for each group)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="center right", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_imr_break(
    panel: pd.DataFrame,
    break_year: int = 1875,
    savepath: str | None = None,
):
    """
    Time-series plot of mean infant mortality rate (IMR) by year,
    1862--1890, documenting the Galloway data break at 1875.

    Uses the diagnostic legitimate-only series
    ``infant_mortality_rate_leg`` (= Dth_infant_leg / Birlegtot x 1000)
    because that is where the 1875 break is visible. Pre-1875 the
    numerator falls back to Dthyoung; from 1875 it switches to
    Dth<1leg, producing the ~3-4x level discontinuity in the figure.
    The headline analytical variable ``infant_mortality_rate`` (total
    IMR) is restricted to 1875+ by construction (Galloway's Dth<1bas
    column does not appear earlier) and so cannot show the break --
    that is why we plot the legitimate-only diagnostic here.
    """
    imr_col = "infant_mortality_rate_leg"
    annual = (
        panel.dropna(subset=[imr_col])
        .groupby("Year")[imr_col]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    y_max = max(annual[imr_col]) * 1.18
    # Shade pre/post regions to emphasise the two definitional regimes.
    ax.axvspan(
        annual["Year"].min() - 0.5, break_year - 0.5,
        alpha=0.10, color="#C0392B",
        label=r"Pre-1875: $\mathrm{Dth_{young}}$ fallback (broader young-age deaths)",
    )
    ax.axvspan(
        break_year - 0.5, annual["Year"].max() + 0.5,
        alpha=0.10, color="#27AE60",
        label=r"Post-1875: $\mathrm{Dth_{<1\,leg}}$ (true infant deaths)",
    )

    ax.plot(
        annual["Year"], annual[imr_col],
        color=COLORS["catholic"], linewidth=2, marker="o", markersize=4,
        label="Mean legitimate-IMR (diagnostic series)",
    )

    ax.axvline(break_year, color="black", linestyle="--", linewidth=1.3)
    ax.text(
        break_year - 0.3, y_max * 0.97,
        "Galloway IMR definition\nchange at 1875",
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.9),
    )

    ax.set_xlim(annual["Year"].min() - 0.5, annual["Year"].max() + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(
        "Mean infant mortality rate (per 1,000 legitimate live births)",
        fontsize=10,
    )
    ax.set_title(
        "Infant mortality rate, Prussian counties 1862-1890\n"
        "(Galloway data-definition break at 1875)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_imr_by_group(
    panel: pd.DataFrame,
    break_year: int = 1875,
    enforcement_years: tuple[int, int] = (1872, 1878),
    rollback_years: tuple[int, int] = (1880, 1887),
    savepath: str | None = None,
):
    """
    Time-series plot of total infant mortality rate by year and
    by Catholic-share group (high vs low), 1875--1890. The
    Kulturkampf enforcement (1872--78) and rollback (1880--87)
    windows are shaded for context.

    Uses the headline analytical variable ``infant_mortality_rate``
    (total IMR = total infant deaths / total live births x 1000),
    which is well-defined only from 1875 onwards because Galloway's
    illegitimate-infant-death column ``Dth<1bas`` does not appear
    earlier. Pre-1875 values are therefore omitted entirely; the
    companion plot ``plot_imr_break`` documents the data-break issue
    using the legitimate-only diagnostic series.

    The two-line layout shows that post-1875 High-Catholic and
    Low-Catholic IMR series track each other closely with no obvious
    divergence around the rollback or post-rollback periods -- visual
    confirmation of the IMR null result in
    ``channels.infant_mortality_analysis``.
    """
    df = panel.dropna(subset=["infant_mortality_rate"]).copy()
    df["group"] = df["high_cath"].map({0: "Low Cath", 1: "High Cath"})
    annual = (
        df.groupby(["Year", "group"])["infant_mortality_rate"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_min = max(0, min(annual.min()) - 30)
    y_max = max(annual.max()) * 1.08

    # Kulturkampf enforcement window (light purple) -- only the portion
    # that overlaps the visible 1875+ range.
    ax.axvspan(
        max(enforcement_years[0], annual.index.min()),
        enforcement_years[1],
        alpha=0.12, color=COLORS["kulturkampf"],
        label=f"Kulturkampf enforcement ({enforcement_years[0]}-{enforcement_years[1]})",
    )
    # Rollback window (light grey).
    ax.axvspan(
        rollback_years[0], rollback_years[1],
        alpha=0.10, color="#7F8C8D",
        label=f"Rollback ({rollback_years[0]}-{rollback_years[1]})",
    )

    # Two group lines.
    for grp, color, label in [
        ("High Cath", COLORS["catholic"], r"High Catholic ($>$50%)"),
        ("Low Cath", COLORS["protestant"], r"Low Catholic ($\leq$50%)"),
    ]:
        if grp not in annual.columns:
            continue
        ax.plot(
            annual.index, annual[grp],
            color=color, linewidth=2, marker="o", markersize=4, label=label,
        )

    ax.set_xlim(annual.index.min() - 0.5, annual.index.max() + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(
        "Mean total infant mortality rate (per 1,000 total live births)",
        fontsize=10,
    )
    ax.set_title(
        "Total infant mortality rate by Catholic-share group, 1875-1890\n"
        "(no Catholic-specific IMR response to the Kulturkampf)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
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
