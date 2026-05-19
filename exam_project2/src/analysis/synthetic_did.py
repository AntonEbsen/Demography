"""
synthetic_did.py
================
Synthetic Difference-in-Differences (Arkhangelsky, Athey, Hirshberg,
Imbens, Wager 2021, *American Economic Review*).

Motivation
----------
The baseline TWFE/DiD on this panel compares high-Catholic counties
(treated) against low-Catholic counties (control), assuming parallel
trends in the pre-1873 period. If the two groups drift differentially
before the Kulturkampf, the post-1873 estimate absorbs that drift.

SDID relaxes parallel trends *by construction*: it picks unit weights
$\\hat\\omega_i \\ge 0$, $\\sum_i \\hat\\omega_i = 1$ over the control
counties so that the weighted control trend matches the treated trend
in the pre-period, plus time weights $\\hat\\lambda_t \\ge 0$,
$\\sum_t \\hat\\lambda_t = 1$ over pre-period years emphasising the
years closest to the discontinuity. The estimator is then a weighted
TWFE regression:

.. math::

    (\\hat\\tau, \\hat\\mu, \\hat\\alpha, \\hat\\beta)
    \\;=\\; \\arg\\min_{\\tau, \\mu, \\alpha, \\beta}
    \\sum_{i,t} (Y_{it} - \\mu - \\alpha_i - \\beta_t - \\tau W_{it})^2
    \\cdot \\hat\\omega_i^{\\text{sdid}} \\hat\\lambda_t^{\\text{sdid}}

with $\\hat\\omega_i^{\\text{sdid}} = \\hat\\omega_i$ for controls
(else $1/N_{\\text{tr}}$) and $\\hat\\lambda_t^{\\text{sdid}} =
\\hat\\lambda_t$ for pre-period (else $1/T_{\\text{post}}$).

References
----------
Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic
Review*, 111(12), 4088–4118.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Panel preparation
# ----------------------------------------------------------------------

def _prepare_balanced_panel(
    df: pd.DataFrame,
    outcome: str,
    treat_col: str,
    year_start: int,
    year_end: int,
    treatment_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Reshape a long county-year frame into the SDID matrices.

    Returns
    -------
    Y : (N, T) array of outcomes, rows ordered controls first
    W : (N, T) treatment indicator (1 for treated unit × post period)
    treated_mask : (N,) bool, True for treated rows
    years : (T,) list of years
    codes : (N,) list of county codes in row order
    """
    cols_needed = ["Code", "Year", outcome, treat_col]
    sub = (
        df[cols_needed]
        .dropna()
        .drop_duplicates(subset=["Code", "Year"])
        .copy()
    )
    sub = sub[(sub["Year"] >= year_start) & (sub["Year"] <= year_end)]

    # Each county must have a constant treatment status (block design)
    cnt = sub.groupby("Code")[treat_col].nunique()
    bad = cnt[cnt > 1].index
    if len(bad) > 0:
        logger.warning(
            "Dropping %d counties with non-constant %s in [%d, %d]",
            len(bad), treat_col, year_start, year_end,
        )
        sub = sub[~sub["Code"].isin(bad)]

    # Require a balanced rectangle over the full window
    years = sorted(sub["Year"].unique().tolist())
    expected = len(years)
    keep = (
        sub.groupby("Code")["Year"].nunique().loc[lambda s: s == expected].index
    )
    sub = sub[sub["Code"].isin(keep)].copy()

    if sub["Code"].nunique() == 0:
        raise ValueError(
            f"No counties have a complete {year_start}-{year_end} panel "
            f"for outcome={outcome!r}."
        )

    # Pivot to wide
    Y_wide = (
        sub.pivot(index="Code", columns="Year", values=outcome)
        .reindex(columns=years)
    )
    treat_status = (
        sub.groupby("Code")[treat_col].first().astype(int).reindex(Y_wide.index)
    )
    treated_mask = (treat_status == 1).to_numpy()

    # Order rows: controls first, treated after
    order = np.argsort(treated_mask.astype(int), kind="stable")
    Y = Y_wide.to_numpy()[order]
    treated_mask = treated_mask[order]
    codes = Y_wide.index.to_numpy()[order].tolist()

    # Block treatment matrix
    W = np.zeros_like(Y, dtype=int)
    post_idx = np.array([y >= treatment_year for y in years], dtype=bool)
    W[treated_mask[:, None] & post_idx[None, :]] = 1

    return Y, W, treated_mask, years, codes


# ----------------------------------------------------------------------
# Weight optimisation (simplex-constrained ridge least squares)
# ----------------------------------------------------------------------

def _solve_simplex_qp(
    A: np.ndarray,
    b: np.ndarray,
    zeta: float,
    has_intercept: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Solve  min_{w >= 0, 1'w = 1, w0 free}
           || A w + w0 1 - b ||^2  +  zeta^2 * n_rows * ||w||^2

    Parameters
    ----------
    A : (n_rows, n_w) regressors (e.g. control-unit pre-period matrix)
    b : (n_rows,) target (e.g. treated-unit pre-period average)
    zeta : ridge penalty
    has_intercept : if True, fit an unrestricted intercept w0
    seed : reproducible starting point

    Returns
    -------
    w : (n_w,) simplex weights
    w0 : intercept (0.0 if has_intercept=False)
    """
    n_rows, n_w = A.shape
    rng = np.random.default_rng(seed)
    pen = (zeta ** 2) * n_rows

    def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        if has_intercept:
            w0 = theta[0]
            w = theta[1:]
        else:
            w0 = 0.0
            w = theta
        resid = A @ w + w0 - b
        loss = 0.5 * resid @ resid + 0.5 * pen * (w @ w)
        gw = A.T @ resid + pen * w
        if has_intercept:
            g0 = resid.sum()
            return float(loss), np.concatenate([[g0], gw])
        return float(loss), gw

    # Uniform start on the simplex
    w_init = np.full(n_w, 1.0 / n_w)
    if has_intercept:
        # Centre intercept at b̄ - Aw̄ for a sane warm start
        w0_init = float(b.mean() - (A @ w_init).mean())
        theta0 = np.concatenate([[w0_init], w_init])
        # Add small jitter to break ties in degenerate problems
        theta0[1:] += rng.uniform(-1e-6, 1e-6, n_w)
        theta0[1:] = np.clip(theta0[1:], 1e-9, None)
        theta0[1:] /= theta0[1:].sum()
    else:
        theta0 = w_init

    # Constraints: w >= 0, sum(w) == 1
    if has_intercept:
        bounds = [(None, None)] + [(0.0, 1.0)] * n_w
        cons = {
            "type": "eq",
            "fun": lambda t: t[1:].sum() - 1.0,
            "jac": lambda t: np.concatenate([[0.0], np.ones(n_w)]),
        }
    else:
        bounds = [(0.0, 1.0)] * n_w
        cons = {
            "type": "eq",
            "fun": lambda t: t.sum() - 1.0,
            "jac": lambda t: np.ones(n_w),
        }

    res = minimize(
        loss_and_grad,
        theta0,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[cons],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not res.success:
        logger.debug("SLSQP did not converge cleanly: %s", res.message)

    if has_intercept:
        w0 = float(res.x[0])
        w = np.clip(res.x[1:], 0.0, None)
    else:
        w0 = 0.0
        w = np.clip(res.x, 0.0, None)
    s = w.sum()
    if s > 0:
        w = w / s
    return w, w0


def _compute_zeta(Y_co_pre: np.ndarray, n_tr: int, T_post: int) -> float:
    """
    Noise-based regularisation parameter from Arkhangelsky et al. (2021):

        zeta = (N_tr * T_post)^(1/4) * sigma_hat,

    with sigma_hat = std of within-unit first differences over the
    pre-period control matrix.
    """
    if Y_co_pre.shape[1] < 2:
        return 1e-6
    diffs = np.diff(Y_co_pre, axis=1).ravel()
    diffs = diffs[np.isfinite(diffs)]
    sigma = diffs.std(ddof=1) if diffs.size > 1 else 1e-6
    return float((n_tr * T_post) ** 0.25 * max(sigma, 1e-12))


def _unit_weights(
    Y: np.ndarray, treated_mask: np.ndarray, T_pre: int, T_post: int,
) -> tuple[np.ndarray, float, float]:
    """
    Solve for control-unit weights ω̂ matching the average treated
    pre-period trajectory. Returns (omega, omega0, zeta).

    The optimisation regressors A are the *pre-period* outcomes of the
    control units arranged as (T_pre, N_co); the target b is the
    cross-treated average per pre-period year.
    """
    Y_co = Y[~treated_mask]
    Y_tr = Y[treated_mask]
    Y_co_pre = Y_co[:, :T_pre]           # (N_co, T_pre)
    Y_tr_pre = Y_tr[:, :T_pre]           # (N_tr, T_pre)

    A = Y_co_pre.T                       # (T_pre, N_co)
    b = Y_tr_pre.mean(axis=0)            # (T_pre,)
    n_tr = int(treated_mask.sum())
    zeta = _compute_zeta(Y_co_pre, n_tr=n_tr, T_post=T_post)
    omega, omega0 = _solve_simplex_qp(A, b, zeta=zeta, has_intercept=True)
    return omega, omega0, zeta


def _time_weights(
    Y: np.ndarray, treated_mask: np.ndarray, T_pre: int,
) -> tuple[np.ndarray, float]:
    """
    Solve for pre-period time weights λ̂ that make each control unit's
    weighted pre-period level match its own post-period average.

    Returns (lam, lam0). The paper uses a much smaller regularisation
    for time weights than for unit weights — we use 1e-6 * σ̂.
    """
    Y_co = Y[~treated_mask]
    Y_co_pre = Y_co[:, :T_pre]           # (N_co, T_pre)
    Y_co_post = Y_co[:, T_pre:]          # (N_co, T_post)

    A = Y_co_pre                          # (N_co, T_pre)
    b = Y_co_post.mean(axis=1)           # (N_co,)
    # essentially unregularised
    diffs = np.diff(Y_co_pre, axis=1).ravel()
    diffs = diffs[np.isfinite(diffs)]
    sigma = diffs.std(ddof=1) if diffs.size > 1 else 1e-6
    zeta_lam = 1e-6 * max(sigma, 1e-12)
    lam, lam0 = _solve_simplex_qp(A, b, zeta=zeta_lam, has_intercept=True)
    return lam, lam0


# ----------------------------------------------------------------------
# Weighted TWFE point estimate
# ----------------------------------------------------------------------

def _weighted_twfe_tau(
    Y: np.ndarray,
    W: np.ndarray,
    omega: np.ndarray,
    lam: np.ndarray,
    treated_mask: np.ndarray,
    T_pre: int,
) -> float:
    """
    Compute the SDID point estimate as the closed-form weighted 2x2 DiD
    induced by the optimal unit and time weights. Equivalent to running

        argmin_{tau, mu, alpha, beta} sum_{i,t}
            (Y_it - mu - alpha_i - beta_t - tau W_it)^2
            * w_i^sdid * w_t^sdid

    See Arkhangelsky et al. (2021), § 2.1, eq. 2.8.
    """
    n_tr = int(treated_mask.sum())
    n_co = int((~treated_mask).sum())
    T_post = Y.shape[1] - T_pre

    w_unit = np.empty(Y.shape[0])
    w_unit[~treated_mask] = omega
    w_unit[treated_mask] = 1.0 / n_tr

    w_time = np.empty(Y.shape[1])
    w_time[:T_pre] = lam
    w_time[T_pre:] = 1.0 / T_post

    # Treated post-period mean (uniform 1/(N_tr T_post))
    Y_tr_post = Y[treated_mask][:, T_pre:].mean()

    # Treated pre-period: weighted by λ over pre years, uniform over treated units
    Y_tr_pre = (Y[treated_mask][:, :T_pre] * lam[None, :]).sum(axis=1).mean()

    # Control post-period: uniform over post years, weighted by ω over units
    Y_co_post = (Y[~treated_mask][:, T_pre:].mean(axis=1) * omega).sum()

    # Control pre-period: ω over units, λ over years
    Y_co_pre = (
        (Y[~treated_mask][:, :T_pre] * lam[None, :]).sum(axis=1) * omega
    ).sum()

    tau = (Y_tr_post - Y_tr_pre) - (Y_co_post - Y_co_pre)
    return float(tau)


# ----------------------------------------------------------------------
# Public estimator
# ----------------------------------------------------------------------

@dataclass
class SDIDResult:
    """Container for a single SDID estimate."""
    tau_hat: float
    se: Optional[float]
    p_value: Optional[float]
    n_treated: int
    n_control: int
    T_pre: int
    T_post: int
    omega: np.ndarray
    omega0: float
    lam: np.ndarray
    lam0: float
    zeta: float
    years: list = field(default_factory=list)
    codes: list = field(default_factory=list)
    treated_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    Y: np.ndarray = field(default_factory=lambda: np.array([[]]))
    placebo_taus: Optional[np.ndarray] = None
    outcome: str = ""
    treat_col: str = ""
    treatment_year: int = 0

    def summary_line(self) -> str:
        """One-line text summary for logs / quick inspection."""
        se = f"{self.se:.4f}" if self.se is not None else "n/a"
        p = f"{self.p_value:.3f}" if self.p_value is not None else "n/a"
        return (
            f"SDID τ̂={self.tau_hat:+.4f}  SE={se}  p={p}  "
            f"(N_tr={self.n_treated}, N_co={self.n_control}, "
            f"T_pre={self.T_pre}, T_post={self.T_post})"
        )


def run_sdid(
    df: pd.DataFrame,
    outcome: str = "marriage_rate",
    treat_col: str = "high_cath",
    treatment_year: int = 1873,
    year_start: int = 1862,
    year_end: int = 1885,
    n_placebo: int = 500,
    seed: int = 42,
    compute_se: bool = True,
) -> SDIDResult:
    """
    Estimate the Synthetic DiD ATT for a single outcome and treatment
    definition, with optional placebo standard errors.

    Parameters
    ----------
    df : pd.DataFrame
        Long county-year panel (must contain ``Code``, ``Year``,
        ``outcome``, ``treat_col``).
    outcome : str
        Outcome column (e.g. ``"marriage_rate"``, ``"cbr"``).
    treat_col : str
        Binary treatment column (e.g. ``"high_cath"``, or a custom
        threshold indicator built upstream).
    treatment_year : int
        First post-treatment year (post = Year >= treatment_year).
    year_start, year_end : int
        Sample window. The panel is restricted to counties balanced
        across this window for ``outcome``.
    n_placebo : int
        Number of placebo permutations for SE inference. Set to 0 to
        skip and return ``se=None``.
    seed : int
        RNG seed for placebo draws.
    compute_se : bool
        Convenience flag: if False, skip placebo inference.

    Returns
    -------
    SDIDResult
    """
    Y, W, treated_mask, years, codes = _prepare_balanced_panel(
        df, outcome=outcome, treat_col=treat_col,
        year_start=year_start, year_end=year_end,
        treatment_year=treatment_year,
    )
    T = len(years)
    T_pre = int(sum(y < treatment_year for y in years))
    T_post = T - T_pre
    if T_pre < 2 or T_post < 1:
        raise ValueError(
            f"Need T_pre>=2 and T_post>=1 (got T_pre={T_pre}, T_post={T_post})."
        )

    n_tr = int(treated_mask.sum())
    n_co = int((~treated_mask).sum())
    if n_tr < 1:
        raise ValueError(f"No treated units under {treat_col!r}.")
    if n_co < 2:
        raise ValueError(f"Need at least 2 control units (got {n_co}).")

    omega, omega0, zeta = _unit_weights(Y, treated_mask, T_pre, T_post)
    lam, lam0 = _time_weights(Y, treated_mask, T_pre)
    tau_hat = _weighted_twfe_tau(Y, W, omega, lam, treated_mask, T_pre)

    se: Optional[float] = None
    p_value: Optional[float] = None
    placebo_taus: Optional[np.ndarray] = None

    if compute_se and n_placebo > 0:
        if n_co <= n_tr + 1:
            logger.warning(
                "Too few controls (%d) for n_tr=%d placebo SEs; skipping.",
                n_co, n_tr,
            )
        else:
            placebo_taus = _placebo_distribution(
                Y, treated_mask, T_pre, T_post,
                n_placebo=n_placebo, seed=seed,
            )
            se = float(placebo_taus.std(ddof=1))
            # Two-sided exact-Fisher p (≥-extreme share among placebos)
            p_value = float(
                (np.abs(placebo_taus) >= abs(tau_hat)).mean()
            )

    res = SDIDResult(
        tau_hat=tau_hat, se=se, p_value=p_value,
        n_treated=n_tr, n_control=n_co,
        T_pre=T_pre, T_post=T_post,
        omega=omega, omega0=omega0,
        lam=lam, lam0=lam0, zeta=zeta,
        years=years, codes=codes,
        treated_mask=treated_mask, Y=Y,
        placebo_taus=placebo_taus,
        outcome=outcome, treat_col=treat_col,
        treatment_year=treatment_year,
    )
    logger.info(res.summary_line())
    return res


# ----------------------------------------------------------------------
# Placebo inference
# ----------------------------------------------------------------------

def _placebo_distribution(
    Y: np.ndarray,
    treated_mask: np.ndarray,
    T_pre: int,
    T_post: int,
    n_placebo: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Draw the SDID placebo distribution: discard the true treated units,
    randomly relabel n_tr of the surviving controls as pseudo-treated,
    rerun the full estimator.

    Returns an array of n_placebo placebo τ̂s.
    """
    rng = np.random.default_rng(seed)
    Y_co = Y[~treated_mask]
    n_co = Y_co.shape[0]
    n_tr = int(treated_mask.sum())

    placebos = np.empty(n_placebo)
    for b in range(n_placebo):
        idx = rng.choice(n_co, size=n_tr, replace=False)
        mask = np.zeros(n_co, dtype=bool)
        mask[idx] = True
        # Reorder: controls first, pseudo-treated last
        order = np.argsort(mask.astype(int), kind="stable")
        Y_b = Y_co[order]
        mask_b = mask[order]
        try:
            om, _, _ = _unit_weights(Y_b, mask_b, T_pre, T_post)
            lm, _ = _time_weights(Y_b, mask_b, T_pre)
            placebos[b] = _weighted_twfe_tau(
                Y_b, None, om, lm, mask_b, T_pre,
            )
        except Exception as exc:                # noqa: BLE001
            logger.debug("placebo draw %d failed: %s", b, exc)
            placebos[b] = np.nan
    placebos = placebos[np.isfinite(placebos)]
    if placebos.size == 0:
        raise RuntimeError("All placebo draws failed.")
    return placebos


# ----------------------------------------------------------------------
# Threshold sweep
# ----------------------------------------------------------------------

def run_sdid_threshold_sweep(
    df: pd.DataFrame,
    thresholds: Sequence[float] = (40.0, 50.0, 60.0),
    outcomes: Sequence[str] = ("marriage_rate", "cbr"),
    treatment_year: int = 1873,
    year_start: int = 1862,
    year_end: int = 1885,
    n_placebo: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run SDID across (threshold, outcome) cells. For each threshold the
    treatment indicator is rebuilt as ``cath_share > threshold``.

    Returns
    -------
    pd.DataFrame with one row per (outcome, threshold) cell.
    """
    rows = []
    for thr in thresholds:
        col = f"_high_cath_thr_{int(thr)}"
        work = df.copy()
        work[col] = (work["cath_share"] > thr).astype(int)
        for out in outcomes:
            try:
                res = run_sdid(
                    work, outcome=out, treat_col=col,
                    treatment_year=treatment_year,
                    year_start=year_start, year_end=year_end,
                    n_placebo=n_placebo, seed=seed,
                )
                rows.append({
                    "outcome": out,
                    "threshold": thr,
                    "tau_hat": res.tau_hat,
                    "se": res.se,
                    "p_value": res.p_value,
                    "n_treated": res.n_treated,
                    "n_control": res.n_control,
                    "T_pre": res.T_pre,
                    "T_post": res.T_post,
                })
            except Exception as exc:            # noqa: BLE001
                logger.warning("threshold=%s outcome=%s failed: %s",
                               thr, out, exc)
                rows.append({
                    "outcome": out, "threshold": thr,
                    "tau_hat": np.nan, "se": np.nan, "p_value": np.nan,
                    "n_treated": np.nan, "n_control": np.nan,
                    "T_pre": np.nan, "T_post": np.nan,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------

def plot_synthetic_vs_treated(
    res: SDIDResult,
    ax=None,
    show_lambda: bool = True,
    title: Optional[str] = None,
):
    """
    Plot the treated-mean trajectory against the synthetic control
    trajectory (control units weighted by ω̂), and shade the pre-period
    years by time-weight λ̂. Designed to be the *visual proof* that
    pre-trends line up by construction.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    Y_tr = res.Y[res.treated_mask].mean(axis=0)
    Y_co_syn = (res.Y[~res.treated_mask] * res.omega[:, None]).sum(axis=0)
    Y_co_syn = Y_co_syn + res.omega0  # apply fitted intercept shift

    years = np.array(res.years)
    ax.plot(years, Y_tr, marker="o", color="C3", label="Treated (mean)")
    ax.plot(years, Y_co_syn, marker="s", color="C0",
            label="Synthetic control (ω̂)")
    ax.axvline(res.treatment_year, color="black", linestyle="--",
               linewidth=1, label=f"Treatment ({res.treatment_year})")

    if show_lambda and res.T_pre > 0:
        # Shade pre-period bars proportionally to λ̂ (visualises which
        # pre-years the time weights up-weight).
        pre_years = years[: res.T_pre]
        ymin, ymax = ax.get_ylim()
        height = (ymax - ymin) * 0.04
        for y, w in zip(pre_years, res.lam):
            ax.bar(
                y, height, bottom=ymin, width=0.6,
                color="grey", alpha=0.15 + 0.65 * w / max(res.lam.max(), 1e-9),
                edgecolor="none", zorder=0,
            )

    se = f"{res.se:.3f}" if res.se is not None else "n/a"
    if title is None:
        title = (
            f"SDID — {res.outcome} | {res.treat_col} | "
            f"τ̂={res.tau_hat:+.3f} (SE {se})"
        )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(res.outcome)
    ax.legend(loc="best", frameon=False)
    return ax
