"""Unit tests for the hand-rolled Synthetic DiD estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exam_project2.src.analysis.synthetic_did import (
    _solve_simplex_qp,
    _unit_weights,
    _time_weights,
    _weighted_twfe_tau,
    run_sdid,
    run_sdid_threshold_sweep,
)


def _simulate_panel(
    n_co: int = 30,
    n_tr: int = 8,
    T_pre: int = 8,
    T_post: int = 5,
    tau: float = -1.5,
    nonlinear_trend: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate a block-treatment panel where treated and control units
    share a common (possibly nonlinear) time trend plus unit FEs, so
    that SDID should recover ``tau`` while a naive DiD with linear
    extrapolation would be biased when ``nonlinear_trend=True``.
    """
    rng = np.random.default_rng(seed)
    N = n_co + n_tr
    T = T_pre + T_post

    alpha = rng.normal(0, 1, N)
    t_idx = np.arange(T)
    if nonlinear_trend:
        beta = 0.2 * t_idx + 0.5 * np.sin(t_idx / 2.0)
    else:
        beta = 0.2 * t_idx

    Y = alpha[:, None] + beta[None, :] + 0.3 * rng.normal(size=(N, T))
    treated = np.array([0] * n_co + [1] * n_tr)
    W = np.zeros((N, T), dtype=int)
    W[treated == 1, T_pre:] = 1
    Y = Y + tau * W

    codes = [f"C{i:03d}" for i in range(N)]
    year_start = 1860
    years = list(range(year_start, year_start + T))
    treatment_year = year_start + T_pre

    rows = []
    for i, code in enumerate(codes):
        for j, yr in enumerate(years):
            rows.append({
                "Code": code,
                "Year": yr,
                "y": Y[i, j],
                "treat": int(treated[i]),
            })
    df = pd.DataFrame(rows)
    df.attrs["treatment_year"] = treatment_year
    df.attrs["tau"] = tau
    return df


def test_simplex_qp_basic_recovery():
    """Simplex QP should reproduce a known target made from a convex
    combination of regressors when no regularisation is applied."""
    rng = np.random.default_rng(0)
    n_rows, n_w = 12, 6
    A = rng.normal(size=(n_rows, n_w))
    true_w = rng.dirichlet(np.ones(n_w))
    b = A @ true_w
    w_hat, w0_hat = _solve_simplex_qp(A, b, zeta=0.0, has_intercept=True)
    assert np.isclose(w_hat.sum(), 1.0, atol=1e-6)
    assert (w_hat >= -1e-9).all()
    # With zero noise and zero penalty, the fitted ω should match the
    # target up to numerical tolerance.
    np.testing.assert_allclose(A @ w_hat + w0_hat, b, atol=1e-4)


def test_unit_and_time_weights_on_simplex():
    """Weights from the panel solvers must live on the simplex."""
    df = _simulate_panel(seed=1)
    Y = (
        df.pivot(index="Code", columns="Year", values="y")
        .sort_index()
    )
    treated = (
        df.groupby("Code")["treat"].first().reindex(Y.index).to_numpy().astype(bool)
    )
    # Reorder controls first
    order = np.argsort(treated.astype(int), kind="stable")
    Y_mat = Y.to_numpy()[order]
    treated_mask = treated[order]
    T_pre = df.attrs["treatment_year"] - df["Year"].min()
    T_post = Y_mat.shape[1] - T_pre

    omega, omega0, zeta = _unit_weights(Y_mat, treated_mask, T_pre, T_post)
    lam, lam0 = _time_weights(Y_mat, treated_mask, T_pre)

    assert omega.shape[0] == (~treated_mask).sum()
    assert lam.shape[0] == T_pre
    assert np.isclose(omega.sum(), 1.0, atol=1e-6)
    assert np.isclose(lam.sum(), 1.0, atol=1e-6)
    assert (omega >= -1e-9).all()
    assert (lam >= -1e-9).all()
    assert zeta > 0.0


def test_sdid_recovers_known_tau_nonlinear_trend():
    """SDID should recover τ within ~0.3 even under a nonlinear common
    trend that would bias a naive linear-trend DiD."""
    df = _simulate_panel(tau=-1.5, nonlinear_trend=True, seed=42)
    res = run_sdid(
        df, outcome="y", treat_col="treat",
        treatment_year=df.attrs["treatment_year"],
        year_start=df["Year"].min(), year_end=df["Year"].max(),
        n_placebo=200, seed=42,
    )
    assert res.n_treated == 8
    assert res.n_control == 30
    assert abs(res.tau_hat - df.attrs["tau"]) < 0.3, (
        f"τ̂={res.tau_hat} vs true τ={df.attrs['tau']}"
    )
    assert res.se is not None and res.se > 0
    assert 0.0 <= res.p_value <= 1.0


def test_sdid_zero_effect_under_no_treatment():
    """If τ=0, SDID should produce an estimate near zero and a
    non-significant placebo p-value."""
    df = _simulate_panel(tau=0.0, nonlinear_trend=True, seed=7)
    res = run_sdid(
        df, outcome="y", treat_col="treat",
        treatment_year=df.attrs["treatment_year"],
        year_start=df["Year"].min(), year_end=df["Year"].max(),
        n_placebo=200, seed=7,
    )
    assert abs(res.tau_hat) < 0.3
    # With B=200 placebos, the null p-value should rarely fall below 0.05
    # under this DGP — but we only assert it isn't crazy-small.
    assert res.p_value > 0.01


def test_weighted_twfe_closed_form_matches_manual():
    """The closed-form SDID estimator should agree with a manual
    weighted 2×2 DiD computation."""
    df = _simulate_panel(tau=-2.0, nonlinear_trend=False, seed=3)
    Y = (
        df.pivot(index="Code", columns="Year", values="y")
        .sort_index()
    )
    treated = (
        df.groupby("Code")["treat"].first().reindex(Y.index).to_numpy().astype(bool)
    )
    order = np.argsort(treated.astype(int), kind="stable")
    Y_mat = Y.to_numpy()[order]
    treated_mask = treated[order]
    T_pre = df.attrs["treatment_year"] - df["Year"].min()
    T_post = Y_mat.shape[1] - T_pre

    omega, _, _ = _unit_weights(Y_mat, treated_mask, T_pre, T_post)
    lam, _ = _time_weights(Y_mat, treated_mask, T_pre)

    tau = _weighted_twfe_tau(Y_mat, None, omega, lam, treated_mask, T_pre)

    # Manual 2×2
    tr = Y_mat[treated_mask]
    co = Y_mat[~treated_mask]
    Y_tr_post = tr[:, T_pre:].mean()
    Y_tr_pre = (tr[:, :T_pre] * lam[None, :]).sum(axis=1).mean()
    Y_co_post = (co[:, T_pre:].mean(axis=1) * omega).sum()
    Y_co_pre = ((co[:, :T_pre] * lam[None, :]).sum(axis=1) * omega).sum()
    tau_manual = (Y_tr_post - Y_tr_pre) - (Y_co_post - Y_co_pre)

    assert np.isclose(tau, tau_manual, atol=1e-10)


def test_threshold_sweep_schema():
    """Threshold sweep returns one row per (outcome, threshold)."""
    rng = np.random.default_rng(0)
    df = _simulate_panel(tau=-1.0, seed=11)
    # Build a fake cath_share so the threshold logic is exercised.
    cath = {
        code: float(rng.uniform(0, 100))
        for code in df["Code"].unique()
    }
    df["cath_share"] = df["Code"].map(cath)
    # Force the treatment indicator to align with the >50 threshold so
    # the sweep produces at least one non-empty cell.
    df.loc[df["cath_share"] > 50, "treat"] = 1
    df.loc[df["cath_share"] <= 50, "treat"] = 0

    out = run_sdid_threshold_sweep(
        df, thresholds=(40, 50, 60),
        outcomes=("y",),
        treatment_year=df.attrs["treatment_year"],
        year_start=df["Year"].min(),
        year_end=df["Year"].max(),
        n_placebo=50,
    )
    assert set(out.columns) >= {
        "outcome", "threshold", "tau_hat", "se", "p_value",
        "n_treated", "n_control", "T_pre", "T_post",
    }
    assert len(out) == 3
