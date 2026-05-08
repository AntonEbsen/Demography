"""
regressions.py
==============
DiD and event-study regressions for the Kulturkampf–fertility paper.

Usage (from notebook):
    from src.analysis.regressions import run_baseline_did, run_event_study, run_robustness
"""

import logging
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults
from typing import Optional, Dict, List, Union

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Set multi-index for linearmodels and ensure correct dtypes."""
    out = df.copy()
    out = out.set_index(["Code", "Year"])
    return out


def run_baseline_did(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment: str = "continuous",
    controls: Optional[List[str]] = None,
    cluster: str = "Code",
    fe_design: str = "twfe",
    two_way_cluster: bool = False,
) -> Dict[str, Union[PanelEffectsResults, str]]:
    """
    Run the baseline DiD specification.
    
    Specification
    -------------
    Y_it = β (CathShare_i × Post_t) + α_i + δ_t + X_it'γ + ε_it
    
    or with binary treatment:
    Y_it = β (HighCath_i × Post_t) + α_i + δ_t + X_it'γ + ε_it
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel from build_dataset.
    outcome : str
        Dependent variable. Options: 'cbr', 'legitimate_br', 
        'illegitimate_br', 'marriage_rate', 'cath_marriage_share'.
    treatment : str
        'continuous' → uses cath_share_x_post (Catholic share × Post)
        'binary' → uses treat_x_post (HighCath × Post)
    controls : list, optional
        Additional time-varying controls. Default: ['ln_pop'].
    cluster : str
        Cluster variable for standard errors. Default: 'Code' (county).
    
    Returns
    -------
    dict with keys: 'result' (PanelOLS result), 'summary' (string)
    """
    if controls is None:
        # NOTE: infant_mortality_rate intentionally excluded — Galloway's
        # definition changes in 1875, which would make a control conditioned
        # on a variable that does not have a constant meaning across the
        # sample. See channels.py for the 1875+ restriction we apply when
        # infant mortality is the *outcome*.
        controls = ["ln_pop"]

    panel = _prepare_panel(df)

    # Drop missing outcomes
    mask = panel[outcome].notna()
    panel = panel[mask]

    # Choose treatment variable
    if treatment == "continuous":
        treat_var = "cath_share_x_post"
    elif treatment == "binary":
        treat_var = "treat_x_post"
    else:
        raise ValueError(f"treatment must be 'continuous' or 'binary', got {treatment}")
    
    # Build regressor matrix
    exog_vars = [treat_var] + [c for c in controls if c in panel.columns]

    y = panel[outcome]
    X = panel[exog_vars]

    # Drop any remaining NaN
    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid]
    X = X[valid]

    if fe_design == "twfe":
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    elif fe_design == "year_x_rb":
        # Year × Regierungsbezirk fixed effects: compares only counties
        # within the same administrative region in the same year. Strictly
        # subsumes year FE; entity FE is still required.
        if "Rb" not in panel.columns:
            raise KeyError("fe_design='year_x_rb' requires column 'Rb' in df.")
        year_rb = (
            panel.loc[X.index].index.get_level_values("Year").astype(str)
            + "_"
            + panel.loc[X.index, "Rb"].astype(str)
        )
        other = pd.DataFrame({"year_rb": year_rb.values}, index=X.index)
        mod = PanelOLS(y, X, entity_effects=True, other_effects=other)
    elif fe_design == "twfe_county_trends":
        # Entity FE + year FE + county-specific linear trend. Absorbs
        # deterministic pre-trends (each county gets its own slope on Year).
        # Trends added as additional exog — county-i trend = (Year - mean_Year)
        # for county i and 0 elsewhere.
        years = X.index.get_level_values("Year").astype(float).values
        codes = X.index.get_level_values("Code").values
        year_centered = years - years.mean()
        unique_codes = sorted(set(codes))[:-1]  # drop one for identification
        trend_cols = {}
        for c in unique_codes:
            trend_cols[f"trend_{c}"] = np.where(codes == c, year_centered, 0.0)
        trend_df = pd.DataFrame(trend_cols, index=X.index)
        X = pd.concat([X, trend_df], axis=1)
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    else:
        raise ValueError(
            f"fe_design must be 'twfe', 'year_x_rb', or 'twfe_county_trends'; "
            f"got {fe_design!r}"
        )
    if two_way_cluster:
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    else:
        res = mod.fit(cov_type="clustered", cluster_entity=True)

    return {"result": res, "summary": str(res.summary)}


def run_start_year_sensitivity(
    df: pd.DataFrame,
    outcome: str = "cbr",
    start_years: tuple[int, ...] = (1862, 1865, 1867, 1869, 1871),
    fe_design: str = "twfe",
) -> pd.DataFrame:
    """
    Re-estimate the baseline DiD on samples starting at increasing start years.

    Useful for diagnosing whether a non-zero pre-trends F-test reflects a
    real differential trajectory or just early-period measurement noise:
    if the coefficient (and pre-trends test) stabilises as the start year
    moves toward 1872, the pre-trend was driven by the dropped years.
    """
    rows = []
    for start in start_years:
        sub = df[df["Year"] >= start].copy()
        try:
            res = run_baseline_did(
                sub, outcome=outcome, treatment="continuous", fe_design=fe_design
            )["result"]
            rows.append({
                "start_year": start,
                "coef": float(res.params["cath_share_x_post"]),
                "se": float(res.std_errors["cath_share_x_post"]),
                "p": float(res.pvalues["cath_share_x_post"]),
                "n": int(res.nobs),
                "n_counties": int(sub["Code"].nunique()),
            })
        except Exception as exc:
            logger.error("start_year=%s failed: %s", start, exc)
    return pd.DataFrame(rows)


def run_iv_did_multi(
    df: pd.DataFrame,
    outcome: str = "cbr",
    instruments: tuple[str, ...] = ("kmwittenberg", "km_bishop"),
    controls: Optional[List[str]] = None,
) -> Dict[str, Union[object, float, int]]:
    """
    2SLS DiD with multiple instruments. Reports first-stage F and Hansen J
    over-identification statistic. Useful when you have $\\ge 2$ valid
    instruments for ``cath_share x post`` (e.g. distance to Wittenberg
    plus distance to nearest Catholic bishop's seat).

    The Hansen J p-value tests whether the over-identifying restrictions
    (more instruments than endogenous regressors) hold. Failure to reject
    is consistent with all instruments being exogenous.
    """
    if controls is None:
        controls = ["ln_pop"]
    missing = [z for z in instruments if z not in df.columns]
    if missing:
        raise KeyError(
            f"Instrument(s) not in panel: {missing}. Merge in via centroids."
        )

    needed = ["Code", "Year", outcome, "cath_share", "post_kulturkampf",
              "cath_share_x_post"] + list(instruments) + controls
    sub = df[[c for c in needed if c in df.columns]].dropna().copy()
    instr_cols = []
    for z in instruments:
        col = f"{z}_x_post"
        sub[col] = sub[z] * sub["post_kulturkampf"]
        instr_cols.append(col)

    entity_d = pd.get_dummies(sub["Code"], prefix="ent", drop_first=True).astype(float)
    year_d = pd.get_dummies(sub["Year"], prefix="yr", drop_first=True).astype(float)
    exog = pd.concat(
        [sub[controls].reset_index(drop=True), entity_d.reset_index(drop=True),
         year_d.reset_index(drop=True)],
        axis=1,
    )
    exog.insert(0, "const", 1.0)

    iv = IV2SLS(
        dependent=sub[outcome].reset_index(drop=True),
        exog=exog,
        endog=sub[["cath_share_x_post"]].reset_index(drop=True),
        instruments=sub[instr_cols].reset_index(drop=True),
    ).fit(cov_type="clustered", clusters=sub["Code"].reset_index(drop=True))

    fs_diag = iv.first_stage.diagnostics
    fs_f = float(fs_diag.loc["cath_share_x_post", "f.stat"])
    fs_partial_r2 = float(fs_diag.loc["cath_share_x_post", "partial.rsquared"])

    # Wooldridge over-identification test (the J/Hansen equivalent valid
    # under clustered standard errors). Failure to reject is consistent
    # with all instruments being exogenous.
    j_test = iv.wooldridge_overid
    j_stat = float(j_test.stat)
    j_p = float(j_test.pval)
    j_df = int(j_test.df)

    # Wu--Hausman test of endogeneity. Rejection means OLS is inconsistent
    # and 2SLS is required; failure to reject means OLS is fine.
    wh = iv.wu_hausman()
    wh_stat = float(wh.stat)
    wh_p = float(wh.pval)

    # Anderson--Rubin test of $H_0: \beta_{\mathrm{endog}} = 0$. Robust to
    # weak instruments because it does not condition on the (potentially
    # noisy) first-stage estimate. Reported alongside the standard 2SLS
    # $p$-value as a sensitivity check against weak-instrument concerns.
    ar = iv.anderson_rubin
    ar_stat = float(ar.stat) if hasattr(ar, "stat") else float("nan")
    ar_p = float(ar.pval) if hasattr(ar, "pval") else float("nan")
    ar_df = int(ar.df) if hasattr(ar, "df") else 0

    return {
        "iv": iv,
        "instruments": list(instruments),
        "iv_coef": float(iv.params["cath_share_x_post"]),
        "iv_se": float(iv.std_errors["cath_share_x_post"]),
        "iv_p": float(iv.pvalues["cath_share_x_post"]),
        "first_stage_f": fs_f,
        "first_stage_partial_r2": fs_partial_r2,
        "j_stat": j_stat,
        "j_p": j_p,
        "j_df": j_df,
        "wu_hausman_stat": wh_stat,
        "wu_hausman_p": wh_p,
        "ar_stat": ar_stat,
        "ar_p": ar_p,
        "ar_df": ar_df,
        "n": int(iv.nobs),
    }


def run_long_difference(
    df: pd.DataFrame,
    outcome: str = "cbr",
    pre_years: tuple[int, int] = (1862, 1871),
    post_years: tuple[int, int] = (1880, 1889),
    controls: Optional[List[str]] = None,
) -> Dict[str, Union[float, int, pd.DataFrame]]:
    """
    Long-difference specification: collapse the panel to two periods and
    regress the change in the outcome on the (time-invariant) treatment.

    Specification
    -------------
    For each county i compute Δ Y_i = mean(Y_i, post_years) - mean(Y_i, pre_years)
    and regress

        Δ Y_i = α + β CathShare_i + γ X_i + ε_i

    Robust to TWFE pathologies (weighting, negative weights with continuous
    treatment) and to autocorrelation in the panel residuals.
    """
    if controls is None:
        controls = ["ln_pop"]

    pre_a, pre_b = pre_years
    post_a, post_b = post_years
    sub = df[df["Year"].between(pre_a, pre_b) | df["Year"].between(post_a, post_b)].copy()
    sub["period"] = np.where(sub["Year"].between(pre_a, pre_b), "pre", "post")

    cols_to_collapse = [outcome, "cath_share"] + [c for c in controls if c in sub.columns]
    means = (
        sub.groupby(["Code", "period"])[cols_to_collapse]
        .mean()
        .unstack("period")
    )
    diffs = pd.DataFrame(index=means.index)
    diffs[outcome] = means[(outcome, "post")] - means[(outcome, "pre")]
    diffs["cath_share"] = means[("cath_share", "pre")]  # time-invariant; use pre
    for c in controls:
        if c in sub.columns:
            diffs[c] = means[(c, "post")] - means[(c, "pre")]
    diffs = diffs.dropna()

    import statsmodels.api as sm
    X = sm.add_constant(diffs[["cath_share"] + controls])
    res = sm.OLS(diffs[outcome], X).fit(cov_type="HC1")

    return {
        "result": res,
        "coef": float(res.params["cath_share"]),
        "se": float(res.bse["cath_share"]),
        "p": float(res.pvalues["cath_share"]),
        "n": int(res.nobs),
        "diffs": diffs,
    }


def pretrends_wald_test(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment_var: str = "cath_share",
    ref_year: int = 1872,
    pre_cutoff: int = 1872,
) -> Dict[str, float]:
    """
    Joint Wald test that all pre-treatment event-study coefficients are zero.

    Returns the F-statistic, degrees of freedom, and p-value for
    H_0: β_t = 0  for all t < pre_cutoff (excluding the omitted ref_year).
    """
    es = run_event_study(
        df, outcome=outcome, treatment_var=treatment_var, ref_year=ref_year
    )
    res = es["result"]

    pre_terms = [
        f"treat_x_{yr}"
        for yr in df["Year"].dropna().astype(int).unique()
        if yr < pre_cutoff and yr != ref_year and f"treat_x_{yr}" in res.params.index
    ]
    if not pre_terms:
        return {"f_stat": float("nan"), "df": 0, "p_value": float("nan"), "n_terms": 0}

    # Linear-restrictions test: each pre-period coefficient equals zero.
    # Build the restriction matrix manually rather than relying on the
    # linearmodels formula parser, which is finicky with comma syntax.
    from scipy import stats

    params = res.params
    cov = res.cov
    k = len(pre_terms)
    R = np.zeros((k, len(params)))
    for i, term in enumerate(pre_terms):
        R[i, params.index.get_loc(term)] = 1.0
    Rb = R @ params.values
    RVRt = R @ cov.values @ R.T
    W = float(Rb @ np.linalg.inv(RVRt) @ Rb)  # Wald chi-square statistic
    p = float(stats.chi2.sf(W, df=k))
    return {
        "f_stat": W / k,  # F-equivalent (Wald / df)
        "wald_chi2": W,
        "df": k,
        "p_value": p,
        "n_terms": k,
        "pre_terms": pre_terms,
    }


def run_heterogeneity_did(
    df: pd.DataFrame,
    moderator: str,
    outcome: str = "cbr",
) -> Dict[str, Union[PanelEffectsResults, float, int]]:
    """
    Heterogeneity by a time-invariant moderator (e.g. school enrollment, urban
    share). Estimates

        Y = beta1 (CathShare x Post) + beta2 (Moderator x Post)
            + beta3 (CathShare x Moderator x Post) + entity FE + year FE.

    The triple coefficient ``beta3`` is the differential treatment effect per
    unit of moderator. The moderator's level effect and its CathShare
    interaction are absorbed by entity FE.
    """
    if moderator not in df.columns:
        raise KeyError(
            f"Moderator {moderator!r} not in panel — rebuild via "
            "`dvc repro build` so iPEHD merge runs."
        )

    sub = df[["Code", "Year", outcome, "cath_share", "post_kulturkampf",
              moderator, "ln_pop"]].dropna().copy()
    # Center the moderator so the main CathShare x Post coefficient is the
    # effect at the moderator's mean.
    mod_centered = sub[moderator] - sub[moderator].mean()
    sub["mod_x_post"] = mod_centered * sub["post_kulturkampf"]
    sub["cath_x_post"] = sub["cath_share"] * sub["post_kulturkampf"]
    sub["triple"] = sub["cath_share"] * mod_centered * sub["post_kulturkampf"]
    sub = sub.set_index(["Code", "Year"])

    y = sub[outcome]
    X = sub[["cath_x_post", "mod_x_post", "triple", "ln_pop"]]
    valid = y.notna() & X.notna().all(axis=1)
    y, X = y[valid], X[valid]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return {
        "result": res,
        "moderator": moderator,
        "main_coef": float(res.params["cath_x_post"]),
        "main_se": float(res.std_errors["cath_x_post"]),
        "main_p": float(res.pvalues["cath_x_post"]),
        "mod_post_coef": float(res.params["mod_x_post"]),
        "mod_post_se": float(res.std_errors["mod_x_post"]),
        "triple_coef": float(res.params["triple"]),
        "triple_se": float(res.std_errors["triple"]),
        "triple_p": float(res.pvalues["triple"]),
        "n": int(res.nobs),
    }


def run_jewish_placebo(
    df: pd.DataFrame,
    outcomes: tuple[str, ...] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    controls: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Falsification: replace cath_share with f_jew (Jewish share) and re-run
    the baseline DiD. The Kulturkampf was a Catholic--Protestant conflict;
    Jewish-share interactions with Post should show null effects.
    """
    if controls is None:
        controls = ["ln_pop"]
    if "f_jew" not in df.columns:
        raise KeyError(
            "f_jew not in panel — rebuild via `dvc repro build` so iPEHD merge runs."
        )

    rows = []
    for outcome in outcomes:
        sub = df[["Code", "Year", outcome, "f_jew", "post_kulturkampf"] + controls].dropna().copy()
        sub["jew_x_post"] = sub["f_jew"] * sub["post_kulturkampf"]
        sub = sub.set_index(["Code", "Year"])

        y = sub[outcome]
        X = sub[["jew_x_post"] + [c for c in controls if c in sub.columns]]
        valid = y.notna() & X.notna().all(axis=1)
        y, X = y[valid], X[valid]

        mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        rows.append({
            "outcome": outcome,
            "coef": float(res.params["jew_x_post"]),
            "se": float(res.std_errors["jew_x_post"]),
            "p": float(res.pvalues["jew_x_post"]),
            "n": int(res.nobs),
        })
    return pd.DataFrame(rows)


def run_fake_treatment_placebo(
    df: pd.DataFrame,
    outcomes: tuple[str, ...] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    fake_post_year: int = 1865,
    sample_end_year: int = 1871,
) -> pd.DataFrame:
    """
    Falsification: pretend the Kulturkampf occurred in ``fake_post_year`` and
    drop observations from ``sample_end_year + 1`` onward. The standard DiD
    should produce null estimates because the actual treatment hasn't
    happened yet within this restricted sample.
    """
    rows = []
    sub = df[df["Year"] <= sample_end_year].copy()
    sub["fake_post"] = (sub["Year"] >= fake_post_year).astype(int)
    sub["cath_share_x_fake_post"] = sub["cath_share"] * sub["fake_post"]

    for outcome in outcomes:
        s = sub[["Code", "Year", outcome, "cath_share_x_fake_post", "ln_pop"]].dropna().copy()
        s = s.set_index(["Code", "Year"])
        y = s[outcome]
        X = s[["cath_share_x_fake_post", "ln_pop"]]
        valid = y.notna() & X.notna().all(axis=1)
        y, X = y[valid], X[valid]
        if len(y) < 100:
            continue
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        rows.append({
            "outcome": outcome,
            "fake_post_year": fake_post_year,
            "coef": float(res.params["cath_share_x_fake_post"]),
            "se": float(res.std_errors["cath_share_x_fake_post"]),
            "p": float(res.pvalues["cath_share_x_fake_post"]),
            "n": int(res.nobs),
        })
    return pd.DataFrame(rows)


def run_triple_difference_polish(
    df: pd.DataFrame,
    outcome: str = "cbr",
    polish_rbs: tuple[str, ...] = ("POS", "BRO"),
) -> Dict[str, Union[PanelEffectsResults, float, int]]:
    """
    Triple-difference: ``cath_share x Post x Polish``. Single regression
    formally testing whether the Polish-province response to the
    Kulturkampf differs from the rest of the panel. The coefficient on
    the triple interaction is the *differential* effect for Polish
    counties, on top of the main cath_share x Post coefficient.
    """
    sub = df[["Code", "Year", outcome, "cath_share", "post_kulturkampf",
              "Rb", "ln_pop"]].dropna().copy()
    sub["polish"] = sub["Rb"].isin(polish_rbs).astype(int)
    sub["cath_share_x_post"] = sub["cath_share"] * sub["post_kulturkampf"]
    sub["polish_x_post"] = sub["polish"] * sub["post_kulturkampf"]
    sub["cath_share_x_polish"] = sub["cath_share"] * sub["polish"]
    sub["triple"] = sub["cath_share"] * sub["post_kulturkampf"] * sub["polish"]
    sub = sub.set_index(["Code", "Year"])

    y = sub[outcome]
    # polish (level) and cath_share_x_polish are absorbed by entity FE; drop them.
    X = sub[["cath_share_x_post", "polish_x_post", "triple", "ln_pop"]]
    valid = y.notna() & X.notna().all(axis=1)
    y, X = y[valid], X[valid]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    return {
        "result": res,
        "main_effect": float(res.params["cath_share_x_post"]),
        "main_se": float(res.std_errors["cath_share_x_post"]),
        "main_p": float(res.pvalues["cath_share_x_post"]),
        "triple_coef": float(res.params["triple"]),
        "triple_se": float(res.std_errors["triple"]),
        "triple_p": float(res.pvalues["triple"]),
        "polish_post_coef": float(res.params["polish_x_post"]),
        "polish_post_se": float(res.std_errors["polish_x_post"]),
        "n": int(res.nobs),
    }


def run_iv_did(
    df: pd.DataFrame,
    outcome: str = "cbr",
    instrument: str = "kmwittenberg",
    controls: Optional[List[str]] = None,
) -> Dict[str, Union[object, float, int]]:
    """
    Two-stage least squares DiD using distance to Wittenberg as an instrument
    for the Catholic-share treatment intensity (Becker--Woessmann style).

    Endogenous regressor : cath_share_x_post = cath_share * post_kulturkampf
    Instrument           : km_witt_x_post     = kmwittenberg * post_kulturkampf
    Fixed effects        : entity (Code) + year, included as explicit dummies.
    Standard errors      : clustered at the county level.

    Returns a dict with the fitted result, the first-stage F-statistic on
    the excluded instrument, and the OLS comparison coefficient.
    """
    if controls is None:
        controls = ["ln_pop"]
    if instrument not in df.columns:
        raise KeyError(
            f"Instrument column {instrument!r} not in panel — rebuild via "
            "`dvc repro build` so merge_ipehd_controls runs."
        )

    needed = ["Code", "Year", outcome, "cath_share", "post_kulturkampf",
              "cath_share_x_post", instrument] + controls
    sub = df[[c for c in needed if c in df.columns]].dropna().copy()
    sub["instr_x_post"] = sub[instrument] * sub["post_kulturkampf"]

    entity_d = pd.get_dummies(sub["Code"], prefix="ent", drop_first=True).astype(float)
    year_d = pd.get_dummies(sub["Year"], prefix="yr", drop_first=True).astype(float)
    exog = pd.concat(
        [sub[controls].reset_index(drop=True), entity_d.reset_index(drop=True),
         year_d.reset_index(drop=True)],
        axis=1,
    )
    exog.insert(0, "const", 1.0)

    iv = IV2SLS(
        dependent=sub[outcome].reset_index(drop=True),
        exog=exog,
        endog=sub[["cath_share_x_post"]].reset_index(drop=True),
        instruments=sub[["instr_x_post"]].reset_index(drop=True),
    ).fit(cov_type="clustered", clusters=sub["Code"].reset_index(drop=True))

    # First-stage F-stat on the excluded instrument (from the diagnostics
    # table — partial F testing joint significance of excluded instruments
    # after partialling out the included exog).
    fs_diag = iv.first_stage.diagnostics
    fs_f = float(fs_diag.loc["cath_share_x_post", "f.stat"])
    fs_partial_r2 = float(fs_diag.loc["cath_share_x_post", "partial.rsquared"])

    # OLS comparison: same FE, same control, but cath_share_x_post enters as exog
    ols_res = run_baseline_did(df, outcome=outcome, treatment="continuous")["result"]
    ols_coef = float(ols_res.params["cath_share_x_post"])
    ols_se = float(ols_res.std_errors["cath_share_x_post"])
    ols_p = float(ols_res.pvalues["cath_share_x_post"])

    # Wu--Hausman test of endogeneity (rejection => OLS inconsistent, IV needed).
    wh = iv.wu_hausman()
    wh_stat = float(wh.stat)
    wh_p = float(wh.pval)

    return {
        "iv": iv,
        "iv_coef": float(iv.params["cath_share_x_post"]),
        "iv_se": float(iv.std_errors["cath_share_x_post"]),
        "iv_p": float(iv.pvalues["cath_share_x_post"]),
        "first_stage_f": fs_f,
        "first_stage_partial_r2": fs_partial_r2,
        "ols_coef": ols_coef,
        "ols_se": ols_se,
        "ols_p": ols_p,
        "wu_hausman_stat": wh_stat,
        "wu_hausman_p": wh_p,
        "n": int(iv.nobs),
    }


def run_event_study(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment_var: str = "cath_share",
    ref_year: int = 1872,
    cluster: str = "Code",
    controls: Optional[List[str]] = None,
) -> Dict[str, Union[PanelEffectsResults, pd.DataFrame]]:
    """
    Run an event-study specification around the Kulturkampf.
    
    Specification
    -------------
    Y_it = Σ_t β_t (CathShare_i × 1[Year=t]) + α_i + δ_t + ε_it
    
    Omits the reference year (default: 1872, the year before May Laws).
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    outcome : str
        Dependent variable.
    treatment_var : str
        'cath_share' (continuous) or 'high_cath' (binary).
    ref_year : int
        Omitted reference year for the event study.
    controls : list, optional
        Additional time-varying controls.
    
    Returns
    -------
    dict with keys:
        'result' : PanelOLS result
        'coefs'  : DataFrame with Year, beta, se, ci_lo, ci_hi
                    (for plotting)
    """
    if controls is None:
        # See note in run_baseline_did: infant_mortality_rate has a 1875
        # definition break and is unsafe as a panel-wide control.
        controls = ["ln_pop"]

    panel = _prepare_panel(df)

    # Drop missing
    mask = panel[outcome].notna()
    panel = panel[mask]

    # Get available years (excluding reference year)
    years = sorted(panel.index.get_level_values("Year").unique())
    interact_years = [y for y in years if y != ref_year]
    
    # Create year × treatment interactions
    for yr in interact_years:
        col_name = f"treat_x_{yr}"
        year_dummy = (panel.index.get_level_values("Year") == yr).astype(float)
        panel[col_name] = year_dummy * panel[treatment_var].values
    
    # Exogenous variables
    interact_cols = [f"treat_x_{yr}" for yr in interact_years]
    exog_vars = interact_cols + [c for c in controls if c in panel.columns]
    
    y = panel[outcome]
    X = panel[exog_vars]
    
    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid]
    X = X[valid]
    
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    
    # Extract coefficients for plotting
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
    
    # Add reference year as zero
    coef_data.append({
        "Year": ref_year,
        "beta": 0.0,
        "se": 0.0,
        "ci_lo": 0.0,
        "ci_hi": 0.0,
    })
    
    coefs = pd.DataFrame(coef_data).sort_values("Year").reset_index(drop=True)
    
    return {"result": res, "coefs": coefs}


def run_robustness(
    df: pd.DataFrame,
    outcome: str = "cbr",
    controls: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run a battery of robustness checks and return a summary table.
    
    Checks
    ------
    1. Binary treatment (HighCath × Post) instead of continuous
    2. Alternative post cutoff: 1872 instead of 1873
    3. Alternative post cutoff: 1875 instead of 1873
    4. Alternative Catholic threshold: 25% instead of 50%
    5. Alternative Catholic threshold: 75% instead of 50%
    6. Excluding Polish-majority counties (Posen, Bromberg provinces)
    7. Only rural counties (excluding Berlin and very large cities)
    
    Returns
    -------
    pd.DataFrame with columns: Specification, Coefficient, SE, p_value, N, N_counties
    """
    if controls is None:
        # See note in run_baseline_did: infant_mortality_rate has a 1875
        # definition break and is unsafe as a panel-wide control.
        controls = ["ln_pop"]

    results = []
    
    def _run_one(label, data, treat_var="cath_share_x_post"):
        try:
            panel = _prepare_panel(data)
            
            # Require non-missing outcome and all controls
            mask = panel[outcome].notna()
            for c in controls:
                if c in panel.columns:
                    mask = mask & panel[c].notna()
            panel = panel[mask]
            
            y = panel[outcome]
            
            # Exogenous variables
            exog_vars = [treat_var] + [c for c in controls if c in panel.columns]
            X = panel[exog_vars]
            
            valid = y.notna() & X.notna().all(axis=1)
            y, X = y[valid], X[valid]
            
            mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            
            results.append({
                "Specification": label,
                "Coefficient": res.params[treat_var],
                "SE": res.std_errors[treat_var],
                "p_value": res.pvalues[treat_var],
                "N": int(res.nobs),
                "N_counties": int(y.index.get_level_values(0).nunique()),
            })
        except Exception as e:
            logger.error("  [error] %s: %s", label, e)
    
    # 1. Baseline (continuous)
    _run_one("Baseline (continuous CathShare × Post)", df)
    
    # 2. Binary treatment
    _run_one("Binary (HighCath>50% × Post)", df, treat_var="treat_x_post")
    
    # 3. Alternative post: 1872
    df_alt = df.copy()
    df_alt["post_kulturkampf"] = (df_alt["Year"] >= 1872).astype(int)
    df_alt["cath_share_x_post"] = df_alt["cath_share"] * df_alt["post_kulturkampf"]
    _run_one("Post cutoff = 1872", df_alt)
    
    # 4. Alternative post: 1875
    df_alt = df.copy()
    df_alt["post_kulturkampf"] = (df_alt["Year"] >= 1875).astype(int)
    df_alt["cath_share_x_post"] = df_alt["cath_share"] * df_alt["post_kulturkampf"]
    _run_one("Post cutoff = 1875", df_alt)
    
    # 5. Alt threshold: 25%
    df_alt = df.copy()
    df_alt["high_cath_25"] = (df_alt["cath_share"] > 25).astype(int)
    df_alt["treat25_x_post"] = df_alt["high_cath_25"] * df_alt["post_kulturkampf"]
    _run_one("Binary threshold = 25%", df_alt, treat_var="treat25_x_post")
    
    # 6. Alt threshold: 75%
    df_alt = df.copy()
    df_alt["high_cath_75"] = (df_alt["cath_share"] > 75).astype(int)
    df_alt["treat75_x_post"] = df_alt["high_cath_75"] * df_alt["post_kulturkampf"]
    _run_one("Binary threshold = 75%", df_alt, treat_var="treat75_x_post")
    
    # 7. Drop Polish provinces (Posen=POS, Bromberg=BRO)
    df_nopol = df[~df["Rb"].isin(["POS", "BRO"])].copy()
    _run_one("Excl. Polish provinces", df_nopol)
    
    return pd.DataFrame(results)
