from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def simulate_did_panel(
    n_units: int = 200,
    n_periods: int = 12,
    treat_share: float = 0.5,
    treat_start: int = 7,
    effect: float = 1.5,
    seed: int = 42,
    add_violation: bool = False,
) -> pd.DataFrame:
    """
    Simulate a classic DID panel with unit & time effects.
    - Some units are treated (treated_i=1) and treatment turns on after treat_start.
    - Outcome includes unit FE + time FE + noise + treatment effect in post period for treated units.
    - Optional add_violation=True adds a differential pre-trend to treated units (to test diagnostics).

    Returns df with columns: unit, time, treated, post, did, x, y
    """
    rng = np.random.default_rng(seed)

    units = np.arange(n_units)
    times = np.arange(1, n_periods + 1)

    treated_units = rng.choice(
        units, size=int(np.round(n_units * treat_share)), replace=False
    )
    treated = np.isin(units, treated_units).astype(int)

    # Expand to panel
    df = pd.MultiIndex.from_product([units, times], names=["unit", "time"]).to_frame(
        index=False
    )
    df["treated"] = df["unit"].map(pd.Series(treated, index=units)).astype(int)
    df["post"] = (df["time"] >= treat_start).astype(int)
    df["did"] = df["treated"] * df["post"]

    # Controls (optional)
    df["x"] = rng.normal(0, 1, size=len(df))

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, size=n_units)
    time_fe = rng.normal(0, 0.6, size=n_periods)

    df["unit_fe"] = df["unit"].map(pd.Series(unit_fe, index=units))
    df["time_fe"] = df["time"].map(pd.Series(time_fe, index=times))

    # Optional: violate parallel trends (treated pre-trend)
    pretrend = 0.0
    if add_violation:
        # treated units trend upward even before treatment
        pretrend = 0.08

    eps = rng.normal(0, 1.0, size=len(df))
    df["y"] = (
        2.0
        + df["unit_fe"]
        + df["time_fe"]
        + 0.3 * df["x"]
        + effect * df["did"]
        + pretrend * df["treated"] * df["time"]  # pre-trend violation knob
        + eps
    )

    return df.drop(columns=["unit_fe", "time_fe"])


def did_twfe(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treated: str,
    post: str,
    controls: list[str] | None = None,
    cluster: str | None = None,
):
    """
    Two-way fixed effects DiD:
        y_it = alpha_i + gamma_t + beta * (treated_i * post_t) + controls + error

    Returns fitted statsmodels result.
    """
    controls = controls or []
    # did term
    df = df.copy()
    df["did"] = df[treated] * df[post]

    # FE formula
    rhs = ["did"] + controls + [f"C({unit})", f"C({time})"]
    formula = f"{y} ~ " + " + ".join(rhs)

    model = smf.ols(formula, data=df)

    if cluster is None:
        return model.fit()
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})


def event_study(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treated: str,
    treat_start: int,
    relative_to: int = -1,
    controls: list[str] | None = None,
    cluster: str | None = None,
):
    """
    Run an event-study DiD (dynamic DiD):
        y_it = alpha_i + gamma_t + Sum_{k != relative_to} beta_k * D_{i, t+k} + controls + error
    where D_{i, t+k} is an indicator for being in the 'k'th period relative to treatment start.

    Args:
        df: The panel dataframe.
        y: Dependent variable name.
        unit: Unit fixed effect variable name.
        time: Time fixed effect variable name.
        treated: Indicator for treated units (ever treated).
        treat_start: The period when treatment begins.
        relative_to: The reference period (usually -1).
        controls: Optional list of control variables.
        cluster: Variable name to cluster standard errors by.

    Returns:
        The fitted statsmodels regression object.
    """
    df = df.copy()
    df["rel_time"] = df[time] - treat_start

    # Create dummies for each relative period, excluding the reference period
    rel_periods = sorted(df["rel_time"].unique())
    rel_periods = [p for p in rel_periods if p != relative_to]

    event_dummies = []
    for p in rel_periods:
        # Use 'm' for minus and 'p' for plus to avoid formula issues with '-'
        prefix = "rel_m" if p < 0 else "rel_p"
        col_name = f"{prefix}{abs(p)}"
        # Dummy = 1 if treated unit AND in that relative period
        df[col_name] = ((df["rel_time"] == p) & (df[treated] == 1)).astype(int)
        event_dummies.append(col_name)

    controls = controls or []
    rhs = event_dummies + controls + [f"C({unit})", f"C({time})"]
    formula = f"{y} ~ " + " + ".join(rhs)

    model = smf.ols(formula, data=df)

    if cluster is None:
        return model.fit()
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})


def simulate_staggered_did(
    n_units: int = 200,
    n_periods: int = 15,
    treat_share: float = 0.5,
    min_treat_start: int = 5,
    max_treat_start: int = 10,
    effect: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a staggered DiD panel where units are treated at different times.
    """
    rng = np.random.default_rng(seed)

    units = np.arange(n_units)
    times = np.arange(1, n_periods + 1)

    # Assign each unit a treatment start time (or np.inf if never treated)
    treat_starts = rng.integers(min_treat_start, max_treat_start + 1, size=n_units)
    # Only some units are ever treated
    is_treated_unit = rng.random(n_units) < treat_share
    treat_starts = np.where(is_treated_unit, treat_starts, np.inf)

    # Expand to panel
    df = pd.MultiIndex.from_product([units, times], names=["unit", "time"]).to_frame(
        index=False
    )
    df["treat_start"] = df["unit"].map(pd.Series(treat_starts, index=units))
    df["treated"] = (df["treat_start"] != np.inf).astype(int)
    df["post"] = (df["time"] >= df["treat_start"]).astype(int)
    df["did"] = df["post"]  # In staggered DiD, 'post' effectively is the DID term

    # Fixed effects
    unit_fe = rng.normal(0, 1.0, size=n_units)
    time_fe = rng.normal(0, 0.6, size=n_periods)

    df["unit_fe"] = df["unit"].map(pd.Series(unit_fe, index=units))
    df["time_fe"] = df["time"].map(pd.Series(time_fe, index=times))

    eps = rng.normal(0, 1.0, size=len(df))
    df["y"] = (
        2.0
        + df["unit_fe"]
        + df["time_fe"]
        + effect * df["did"]
        + eps
    )

    return df.drop(columns=["unit_fe", "time_fe"])


def run_placebo_test(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treated: str,
    post: str,
    n_iterations: int = 50,
    cluster: str | None = None,
):
    """
    Run a placebo test by randomly permuting treatment status.
    Returns a list of estimated coefficients.
    """
    df = df.copy()
    unit_ids = df[unit].unique()
    estimated_effects = []

    for _ in range(n_iterations):
        # Randomly shuffle which units are considered 'treated'
        shuffled_treated = np.random.permutation(df[treated].unique()) # This is not quite right for panel
        # Better: shuffle treatment status at the unit level
        treated_status = df.groupby(unit)[treated].first()
        shuffled_status = np.random.permutation(treated_status.values)
        lookup = pd.Series(shuffled_status, index=treated_status.index)
        
        df["placebo_treated"] = df[unit].map(lookup)
        df["placebo_did"] = df["placebo_treated"] * df[post]
        
        # Fit a simple TWFE on placebo
        rhs = ["placebo_did", f"C({unit})", f"C({time})"]
        formula = f"{y} ~ " + " + ".join(rhs)
        res = smf.ols(formula, data=df).fit()
        estimated_effects.append(res.params["placebo_did"])

    return pd.Series(estimated_effects)


def run_leave_one_out_sensitivity(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treated: str,
    post: str,
    cluster: str | None = None,
):
    """
    Run a leave-one-out sensitivity analysis by dropping one unit at a time.
    Returns a dataframe of results.
    """
    unit_ids = df[unit].unique()
    results = []

    for u in unit_ids:
        sub_df = df[df[unit] != u]
        res = did_twfe(sub_df, y, unit, time, treated, post, cluster=cluster)
        results.append({"dropped_unit": u, "effect": res.params["did"], "se": res.bse["did"]})

    return pd.DataFrame(results)


def visualize_sensitivity(
    placebo_effects: pd.Series | None = None,
    true_effect: float | None = None,
    loo_results: pd.DataFrame | None = None,
    title: str = "Sensitivity Analysis",
):
    """
    Visualize placebo distribution or Leave-One-Out results.
    """
    if placebo_effects is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(placebo_effects, bins=20, alpha=0.6, label="Placebo Effects", color='gray')
        if true_effect is not None:
            ax.axvline(true_effect, color='red', linestyle='--', label=f"Actual Effect: {true_effect:.3f}")
        ax.set_title(f"{title}: Placebo Distribution")
        ax.set_xlabel("Estimated Coefficient")
        ax.legend()
        return ax

    if loo_results is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(
            range(len(loo_results)),
            loo_results["effect"],
            yerr=1.96 * loo_results["se"],
            fmt='o',
            alpha=0.5,
            markersize=3
        )
        if true_effect is not None:
            ax.axhline(true_effect, color='red', linestyle='--', label="Full Sample Effect")
        ax.set_title(f"{title}: Leave-One-Out Results")
        ax.set_ylabel("Effect Estimate")
        ax.set_xlabel("Iteration (dropped unit)")
        return ax
def run_stacked_did(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treat_start_col: str,
    cluster: str | None = None,
):
    """
    Implement a Stacked DiD to handle staggered treatment timing.
    Stacks 'sub-experiments' for each treatment cohort to avoid bias.
    """
    df = df.copy()
    cohorts = df[df[treat_start_col] != np.inf][treat_start_col].unique()
    stacked_data = []

    for event_time in cohorts:
        # Units treated at 'event_time'
        treated_cohort = df[df[treat_start_col] == event_time].copy()
        
        # 'Clean' controls: never treated or not yet treated by event_time + buffer
        # A simple version: never treated units
        never_treated = df[df[treat_start_col] == np.inf].copy()
        
        # Combine
        sub_df = pd.concat([treated_cohort, never_treated])
        sub_df["cohort_id"] = int(event_time)
        sub_df["post"] = (sub_df[time] >= event_time).astype(int)
        sub_df["did"] = (sub_df[treat_start_col] == event_time).astype(int) * sub_df["post"]
        
        stacked_data.append(sub_df)

    big_df = pd.concat(stacked_data)
    
    # Interaction FE for cohort-specific unit and time effects
    big_df["unit_cohort"] = big_df[unit].astype(str) + "_" + big_df["cohort_id"].astype(str)
    big_df["time_cohort"] = big_df[time].astype(str) + "_" + big_df["cohort_id"].astype(str)

    formula = f"{y} ~ did + C(unit_cohort) + C(time_cohort)"
    model = smf.ols(formula, data=big_df)
    
    if cluster is None:
        return model.fit()
    return model.fit(cov_type="cluster", cov_kwds={"groups": big_df[cluster]})


def group_time_means(df: pd.DataFrame, y: str, treated: str, time: str) -> pd.DataFrame:
    """Mean outcome by treated group and time (for plotting parallel trends)."""
    return (
        df.groupby([treated, time])[y].mean().reset_index().sort_values([treated, time])
    )


def visualize_trends(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treated: str,
    treat_start: int | None = None,
    title: str = "Parallel Trends Analysis",
    ax: plt.Axes | None = None,
):
    """
    Plot the average outcome over time for treated and control groups.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    means = group_time_means(df, y, treated, time)

    for group in means[treated].unique():
        subset = means[means[treated] == group]
        label = "Treated" if group == 1 else "Control"
        ax.plot(subset[time], subset[y], marker="o", label=label)

    if treat_start is not None:
        ax.axvline(
            treat_start - 0.5,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Treatment Start",
        )

    ax.set_title(title)
    ax.set_xlabel(time)
    ax.set_ylabel(f"Mean {y}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def visualize_event_study(
    result,
    relative_to: int = -1,
    title: str = "Event Study: Dynamic Treatment Effects",
    ax: plt.Axes | None = None,
):
    """
    Plot coefficients and 95% confidence intervals from an event study model.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Extract coefficients that look like 'rel_X'
    params = result.params
    conf = result.conf_int()

    coefs = []
    lower = []
    upper = []
    periods = []

    # Add the reference period (zero effect by design)
    periods.append(relative_to)
    coefs.append(0)
    lower.append(0)
    upper.append(0)

    for idx in params.index:
        if idx.startswith("rel_"):
            try:
                # Handle 'rel_mX' and 'rel_pX'
                if "rel_m" in idx:
                    p = -int(idx.replace("rel_m", ""))
                elif "rel_p" in idx:
                    p = int(idx.replace("rel_p", ""))
                else:
                    # fallback for old format if any
                    p = int(idx.split("_")[1])
                
                periods.append(p)
                coefs.append(params[idx])
                lower.append(conf.loc[idx, 0])
                upper.append(conf.loc[idx, 1])
            except (ValueError, IndexError):
                continue

    # Sort by period
    plot_df = pd.DataFrame(
        {"period": periods, "coef": coefs, "lower": lower, "upper": upper}
    ).sort_values("period")

    ax.errorbar(
        plot_df["period"],
        plot_df["coef"],
        yerr=[plot_df["coef"] - plot_df["lower"], plot_df["upper"] - plot_df["coef"]],
        fmt="o-",
        capsize=5,
        color="black",
        label="Est. Coefficient",
    )

    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(relative_to, color="gray", linestyle=":", alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Periods Relative to Treatment")
    ax.set_ylabel("Effect Size")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax
