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
from typing import Optional, Dict, List, Sequence, Union

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Set multi-index for linearmodels and ensure correct dtypes."""
    out = df.copy()
    out = out.set_index(["Code", "Year"])
    return out


def _county_baselines(
    panel: pd.DataFrame, variables: Sequence[str]
) -> Dict[str, Dict[int, float]]:
    """
    Return county-level baseline values for the requested variables, keyed
    by county Code. Picks each county's *first non-missing* observation,
    treating these characteristics as time-invariant pre-treatment
    measurements (the iPEHD merge uses 1871 cross-sectional values, which
    are the pre-Kulturkampf baseline).
    """
    out: Dict[str, Dict[int, float]] = {}
    df = panel.reset_index() if "Code" not in panel.columns else panel
    for var in variables:
        if var not in df.columns:
            continue
        first = (
            df.dropna(subset=[var])
              .sort_values(["Code", "Year"])
              .groupby("Code")[var].first()
        )
        out[var] = first.to_dict()
    return out


def run_baseline_did(
    df: pd.DataFrame,
    outcome: str = "cbr",
    treatment: str = "continuous",
    controls: Optional[List[str]] = None,
    cluster: str = "Code",
    fe_design: str = "twfe",
    two_way_cluster: bool = False,
    pretreatment_trends: Optional[Sequence[str]] = None,
    pretreatment_trends_form: str = "year_dummies",
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
        Additional time-varying controls. Default: ``[]`` (no controls).
        ``ln_pop`` was previously the default but is now opt-in: since
        every rate outcome (CBR, marriage rate, IMR, etc.) uses
        mid-year population as its denominator, adding $\\ln(\\text{Pop})$
        as a control on the right-hand side is mechanically correlated
        with the LHS and risks "bad-control" bias. Pass
        ``controls=["ln_pop"]`` explicitly to recover the old behaviour
        for sensitivity checks. See ``run_emigration_robustness`` (§8.10
        of DATA_APPENDIX) for a six-spec population/migration-control
        ladder.
    cluster : str
        Cluster variable for standard errors. Default: 'Code' (county).

    Returns
    -------
    dict with keys: 'result' (PanelOLS result), 'summary' (string)
    """
    if controls is None:
        controls = []

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

    # Pre-treatment-characteristic time trends (Bai 2009; Hsiao 2014):
    # adds X_baseline_i × <year-trend> for each baseline characteristic, so
    # counties with different baseline urbanisation/literacy/etc. are allowed
    # to follow different trajectories. Identification of beta then comes from
    # *deviations* from those trajectories at 1873.
    if pretreatment_trends:
        baselines = _county_baselines(panel, pretreatment_trends)
        years_arr = np.asarray(
            panel.index.get_level_values("Year"), dtype=float
        )
        codes_arr = np.asarray(panel.index.get_level_values("Code"))
        if pretreatment_trends_form == "year_dummies":
            year_index = pd.Series(years_arr, index=panel.index)
            year_dummies = pd.get_dummies(year_index, prefix="yr", drop_first=True)
            year_dummies.index = panel.index
            for var in pretreatment_trends:
                if var not in baselines:
                    continue
                baseline_obs = pd.Series(
                    [baselines[var].get(c, np.nan) for c in codes_arr],
                    index=panel.index,
                ).astype(float)
                inter = year_dummies.multiply(baseline_obs, axis=0)
                inter = inter.add_prefix(f"{var}_x_")
                X = pd.concat([X, inter], axis=1)
        elif pretreatment_trends_form == "linear":
            year_centered = years_arr - years_arr.mean()
            for var in pretreatment_trends:
                if var not in baselines:
                    continue
                baseline_obs = np.array(
                    [baselines[var].get(c, np.nan) for c in codes_arr], dtype=float
                )
                X[f"{var}_x_year"] = baseline_obs * year_centered
        else:
            raise ValueError(
                f"pretreatment_trends_form must be 'year_dummies' or 'linear'; "
                f"got {pretreatment_trends_form!r}"
            )

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


# Mapping of cutoff years to the headline Kulturkampf legislation that
# took effect in that year. Used by `run_kulturkampf_phase_sensitivity`
# below and rendered into the LaTeX table's column headers.
KULTURKAMPF_PHASE_LABELS: dict[int, str] = {
    1871: "Pre-Kulturkampf (placebo)",
    1872: "Jesuits Law / school inspection",
    1873: "May Laws (Maigesetze)",
    1874: "Civil Marriage Act (Prussia)",
    1875: "Brotkorb / Congregations Law / Reichszivilehe",
    1876: "Bishop expulsions / sede vacante",
}


def run_kulturkampf_phase_sensitivity(
    df: pd.DataFrame,
    outcomes: Sequence[str] = (
        "cbr", "legitimate_br", "gmfr_static_1871",
        "illegitimacy_ratio", "marriage_rate", "I_g",
    ),
    cutoffs: Sequence[int] = (1872, 1873, 1874, 1875, 1876),
    treatment: str = "continuous",
    fe_design: str = "twfe",
    placebo_cutoff: int | None = 1871,
) -> pd.DataFrame:
    """
    Treatment-cutoff sensitivity by Kulturkampf legislative phase.

    The headline ``post_kulturkampf`` indicator is defined at 1873 (May
    Laws). The Kulturkampf however unfolded across at least five
    distinct phases:

      * **1872** -- *Jesuitengesetz* (July 4, 1872) expels the Society
        of Jesus; the *Schulaufsichtsgesetz* (March 11, 1872) transfers
        school inspection to the state.
      * **1873** -- the *Maigesetze* (May 11-14, 1873) regulate
        Catholic clerical training, appointments and discipline; the
        *Kulturexamen* requires state certification of clergy.
      * **1874** -- the *preussisches Zivilehegesetz* (March 9, 1874)
        introduces mandatory state civil marriage in Prussia for the
        first time, *before* the Reich-wide law of 1875.
      * **1875** -- the *Personenstandsgesetz* (February 6, 1875)
        nationalises civil marriage and birth/death registration;
        the *Brotkorbgesetz* (April 22, 1875) suspends state subsidies
        to disobedient bishops; the *Klostergesetz* (May 31, 1875)
        dissolves most Catholic religious orders.
      * **1876** -- mass episcopal expulsions; by year-end nine of
        twelve Prussian bishoprics are *sede vacante*.

    For each outcome and each candidate cutoff this routine re-fits the
    baseline DiD ``Y = beta(CathShare x 1[Year >= cutoff]) + alpha_i +
    delta_t + epsilon`` and returns the coefficient, standard error,
    p-value, sample size and within R^2. The 1871 row is a placebo: a
    pre-Kulturkampf cutoff that should yield zero loading if the
    apparent post-1873 effect is not a continuation of a pre-existing
    trend.

    Interpretation. A reader wanting to know whether the marriage-rate
    effect is specifically about the 1874 Civil Marriage Act -- as
    opposed to the broader 1873 May Laws shock -- can read off the
    1874 column of the marriage_rate row and compare it to 1873. If
    the marriage effect peaks at 1874 (or strengthens monotonically
    from 1873 to 1874), that is direct evidence for the Civil Marriage
    Act as the operative channel. The other rows let a reader inspect
    whether the fertility outcomes co-move with the same cutoff or
    pick up a different phase (e.g. 1875 Brotkorbgesetz for legitimate
    BR, since this is when clerical authority over baptismal
    registration was disrupted).

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel (with `Year`, `Code`, `cath_share`, the chosen
        outcomes already constructed).
    outcomes : Sequence[str]
        Outcomes to test. Default covers the five headline rates plus
        the Hutterite-normalised marital-fertility index $I_g$.
    cutoffs : Sequence[int]
        Candidate post-treatment cutoff years. ``post_t = 1[Year >=
        cutoff]`` is rebuilt for each cutoff.
    treatment : str
        Passed through to ``run_baseline_did``. ``"continuous"`` (the
        default) reports the coefficient on ``cath_share * post`` --
        the headline DiD parameter.
    fe_design : str
        ``"twfe"`` (county + year FE) or ``"year_x_rb"`` (county + year
        x Regierungsbezirk FE). Default is the headline TWFE.
    placebo_cutoff : int or None
        If set (default 1871), an additional pre-Kulturkampf placebo
        row is appended for diagnostic purposes.

    Returns
    -------
    pd.DataFrame with one row per (outcome, cutoff) combination and
    columns ``outcome``, ``cutoff``, ``phase_label``, ``coef``, ``se``,
    ``p``, ``n``, ``r2_within``, ``placebo`` (bool).
    """
    if placebo_cutoff is not None and placebo_cutoff not in cutoffs:
        cutoffs = (placebo_cutoff,) + tuple(cutoffs)

    rows: list[dict] = []
    for cutoff in cutoffs:
        sub = df.copy()
        sub["post_kulturkampf"] = (sub["Year"] >= cutoff).astype(int)
        sub["cath_share_x_post"] = (
            sub["cath_share"] * sub["post_kulturkampf"]
        )
        # The binary treatment interaction is regenerated too so a
        # downstream caller that chose ``treatment="binary"`` gets the
        # right cutoff applied to the high_cath dummy.
        if "high_cath" in sub.columns:
            sub["treat_x_post"] = (
                sub["high_cath"] * sub["post_kulturkampf"]
            )

        for outcome in outcomes:
            if outcome not in sub.columns:
                logger.warning(
                    "outcome %r not on panel -- skipping cutoff=%s",
                    outcome, cutoff,
                )
                continue
            try:
                res = run_baseline_did(
                    sub, outcome=outcome, treatment=treatment,
                    fe_design=fe_design,
                )["result"]
                param = (
                    "cath_share_x_post"
                    if treatment == "continuous" else "treat_x_post"
                )
                rows.append({
                    "outcome": outcome,
                    "cutoff": int(cutoff),
                    "phase_label": KULTURKAMPF_PHASE_LABELS.get(
                        int(cutoff), f"Cutoff {cutoff}"
                    ),
                    "coef": float(res.params[param]),
                    "se": float(res.std_errors[param]),
                    "p": float(res.pvalues[param]),
                    "n": int(res.nobs),
                    "r2_within": float(res.rsquared_within),
                    "placebo": bool(
                        placebo_cutoff is not None
                        and cutoff == placebo_cutoff
                    ),
                })
            except Exception as exc:
                logger.error(
                    "cutoff=%s outcome=%s failed: %s",
                    cutoff, outcome, exc,
                )
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
        controls = []
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
        controls = []

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
    pretreatment_trends: Optional[Sequence[str]] = None,
    pretreatment_trends_form: str = "linear",
) -> Dict[str, float]:
    """
    Joint Wald test that all pre-treatment event-study coefficients are zero.

    Returns the F-statistic, degrees of freedom, and p-value for
    H_0: β_t = 0  for all t < pre_cutoff (excluding the omitted ref_year).

    The optional ``pretreatment_trends`` argument is forwarded to the
    underlying event-study so the test can be re-run conditional on
    pre-treatment-characteristic × year interactions, which is the right
    diagnostic to ask whether those trends *explain* the pre-trend rejection.
    """
    es = run_event_study(
        df, outcome=outcome, treatment_var=treatment_var, ref_year=ref_year,
        pretreatment_trends=pretreatment_trends,
        pretreatment_trends_form=pretreatment_trends_form,
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
              moderator]].dropna().copy()
    # Center the moderator so the main CathShare x Post coefficient is the
    # effect at the moderator's mean.
    mod_centered = sub[moderator] - sub[moderator].mean()
    sub["mod_x_post"] = mod_centered * sub["post_kulturkampf"]
    sub["cath_x_post"] = sub["cath_share"] * sub["post_kulturkampf"]
    sub["triple"] = sub["cath_share"] * mod_centered * sub["post_kulturkampf"]
    sub = sub.set_index(["Code", "Year"])

    y = sub[outcome]
    X = sub[["cath_x_post", "mod_x_post", "triple"]]
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


SUBREGION_DEFINITIONS = {
    "Polish": ("POS", "BRO"),
    "German Catholic": ("KOL", "KOB", "TRI", "AAC", "OPP", "MUN"),
    # Anything else not matched above is "Protestant (rest)"
}


def run_subregion_did(
    df: pd.DataFrame,
    outcome: str = "marriage_rate",
    n_boot: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """
    DiD coefficient on $\\mathrm{CathShare} \\times \\mathrm{Post}$ run
    separately on each of the three sub-region sub-samples used by the wild
    cluster bootstrap (Polish, German Catholic, Protestant rest). Returns a
    DataFrame with one row per region listing $\\hat\\beta$, asymptotic SE,
    asymptotic and wild bootstrap $p$-values, and the cluster count $G$.

    This is the regression form used in Table~\\ref{tab:wild_bootstrap} —
    use the output to colour the choropleth so every county in a sub-region
    receives that sub-region's $\\hat\\beta$.
    """
    from src.analysis.wild_bootstrap import wild_cluster_bootstrap

    polish_rbs = SUBREGION_DEFINITIONS["Polish"]
    german_rbs = SUBREGION_DEFINITIONS["German Catholic"]

    samples = {
        "Polish": df[df["Rb"].isin(polish_rbs)].copy(),
        "German Catholic": df[df["Rb"].isin(german_rbs)].copy(),
        "Protestant (rest)": df[~df["Rb"].isin(polish_rbs + german_rbs)].copy(),
    }

    rows = []
    for label, sub in samples.items():
        if sub.empty or sub["Code"].nunique() < 2:
            continue
        try:
            res = run_baseline_did(
                sub, outcome=outcome, treatment="continuous"
            )["result"]
            wild = wild_cluster_bootstrap(
                df, outcome=outcome,
                sample_filter=lambda d, _label=label: (
                    d["Rb"].isin(polish_rbs) if _label == "Polish"
                    else d["Rb"].isin(german_rbs) if _label == "German Catholic"
                    else ~d["Rb"].isin(polish_rbs + german_rbs)
                ),
                n_boot=n_boot, seed=seed,
            )
            rows.append({
                "subregion": label,
                "coef": float(res.params["cath_share_x_post"]),
                "se": float(res.std_errors["cath_share_x_post"]),
                "p_asymptotic": float(res.pvalues["cath_share_x_post"]),
                "p_wild": wild["p_value"],
                "n_clusters": int(sub["Code"].nunique()),
                "n_obs": int(res.nobs),
            })
        except Exception as exc:
            logger.warning("subregion %s/%s failed: %s", label, outcome, exc)
    return pd.DataFrame(rows)


def run_rb_specific_did(
    df: pd.DataFrame,
    outcome: str = "marriage_rate",
    min_counties: int = 5,
    min_cath_share_sd: float = 5.0,
) -> pd.DataFrame:
    """
    Estimate a Regierungsbezirk-specific DiD coefficient on
    $\\mathrm{CathShare} \\times \\mathrm{Post}$ for each Rb. Run as a single
    pooled regression with one Rb-specific treatment interaction per Rb,
    plus county and year fixed effects:

        Y_{it} = sum_r beta_r (CathShare_i x Post_t x 1[Rb_i = r])
                 + alpha_i + delta_t + gamma X_{it} + eps_{it}.

    Returns a DataFrame with one row per Rb listing beta, SE, p, sample
    size, and a flag indicating whether the Rb meets minimum-precision
    criteria (>= ``min_counties`` counties and >= ``min_cath_share_sd``
    standard deviation of cath_share within the Rb).

    Rbs that fail either criterion are still estimated but flagged
    "imprecise" so the choropleth can shade them grey rather than
    overstating the precision of a near-singular slope.
    """
    df = df.copy().sort_values(["Code", "Year"])
    rbs = sorted(df["Rb"].dropna().unique().tolist())

    # Precision flags
    rb_stats = (
        df.drop_duplicates("Code")
          .groupby("Rb")["cath_share"]
          .agg(["count", "std"])
          .rename(columns={"count": "n_counties", "std": "cath_share_sd"})
    )

    # Build Rb-specific treatment interactions
    panel = _prepare_panel(df)
    treat_cols = []
    for rb in rbs:
        col = f"treat_x_{rb}"
        panel[col] = (
            panel["cath_share_x_post"]
            * (panel["Rb"] == rb).astype(float)
        )
        treat_cols.append(col)

    exog_vars = treat_cols
    y = panel[outcome]
    X = panel[exog_vars]
    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid]
    X = X[valid]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for rb in rbs:
        col = f"treat_x_{rb}"
        if col not in res.params.index:
            continue
        stats = rb_stats.loc[rb] if rb in rb_stats.index else None
        n_counties = int(stats["n_counties"]) if stats is not None else 0
        sd = float(stats["cath_share_sd"]) if stats is not None and not pd.isna(stats["cath_share_sd"]) else 0.0
        precise = (n_counties >= min_counties) and (sd >= min_cath_share_sd)
        rows.append({
            "Rb": rb,
            "coef": float(res.params[col]),
            "se": float(res.std_errors[col]),
            "p": float(res.pvalues[col]),
            "n_counties": n_counties,
            "cath_share_sd": sd,
            "precise": precise,
        })
    return pd.DataFrame(rows).sort_values("coef")


def run_subsample_decomposition(
    df: pd.DataFrame,
    outcomes: tuple[str, ...] = ("cbr", "marriage_rate"),
) -> pd.DataFrame:
    """
    Decompose the headline DiD coefficient by sample. The full-panel
    coefficient mixes three population components:

    1. *1866 annexations* (~85 counties from Schleswig-Holstein, Hanover,
       Hesse-Kassel, Nassau, Frankfurt; mostly Protestant) which only enter
       the panel in 1867 and so cannot have a true 1862-1866 pre-trend
       observation.
    2. *Polish provinces* (Posen and Bromberg; ~24 counties) where Catholic
       share aligns with Polish ethnicity and the policy bundle was richer
       (Germanisation, Polenausweisungen).
    3. *Core German Catholic and Protestant counties* — the cleanest test of
       the religious-institutional disruption hypothesis.

    Reports the headline DiD coefficient and the pre-trends Wald chi-square
    on each cut, so the contribution of each component is transparent.
    """
    first_year = df.groupby("Code")["Year"].min()
    core_codes = first_year[first_year == int(df["Year"].min())].index

    samples = [
        ("Full panel", df),
        ("Core Prussia (excl. 1866 annexations)",
            df[df["Code"].isin(core_codes)]),
        ("No Polish provinces (POS, BRO)",
            df[~df["Rb"].isin(["POS", "BRO"])]),
        ("Core Prussia + No Polish",
            df[df["Code"].isin(core_codes) & ~df["Rb"].isin(["POS", "BRO"])]),
    ]

    rows = []
    for name, sub in samples:
        for outcome in outcomes:
            try:
                res = run_baseline_did(sub, outcome=outcome,
                                       treatment="continuous")["result"]
                wald = pretrends_wald_test(sub, outcome=outcome)
                rows.append({
                    "sample": name,
                    "outcome": outcome,
                    "coef": float(res.params["cath_share_x_post"]),
                    "se": float(res.std_errors["cath_share_x_post"]),
                    "p": float(res.pvalues["cath_share_x_post"]),
                    "n": int(res.nobs),
                    "n_counties": int(sub["Code"].nunique()),
                    "pretrends_chi2": wald["wald_chi2"],
                    "pretrends_p": wald["p_value"],
                })
            except Exception as exc:
                logger.warning("subsample %s/%s failed: %s", name, outcome, exc)
    return pd.DataFrame(rows)


def run_pretreatment_trends_robustness(
    df: pd.DataFrame,
    outcomes: tuple[str, ...] = ("cbr", "marriage_rate"),
    form: str = "year_dummies",
) -> pd.DataFrame:
    """
    Pretreatment-characteristic time-trend robustness (Bai 2009; Hsiao 2014).

    Estimates the headline DiD coefficient under five progressively more
    flexible specifications. Each adds an interaction between pre-treatment
    iPEHD characteristics (county-level constants from 1871) and either
    year fixed effects (the most flexible form) or a centred linear trend.
    The interaction lets counties with different baseline characteristics
    follow different trajectories, identifying the Kulturkampf effect from
    *deviations* from those trajectories rather than from the overall
    differential trend across treatment status.

    Specifications:
      (1) Baseline TWFE
      (2) + literacy (school1517) x trend
      (3) + literacy + urbanisation (f_urban) x trend
      (4) + lit + urban + Prussian citizenship (f_pruss) x trend
      (5) + lit + urban + pruss + Jewish share (f_jew) x trend
      (6) + lit + urban + pruss + jew + female-15-49 share x trend

    Note: f_jew x trend likely absorbs the differential dynamics of eastern
    Polish provinces (where Jewish share is high), so the gap between (4)
    and (5) is informative about how much of the Kulturkampf coefficient
    operates through the Polish-province channel. Spec (6) adds the share
    of women aged 15-49 in 1871 (women_share_15_49_1871, from POP1871) as
    a baseline -- this is the demographic age-structure analogue of the
    iPEHD socio-economic baselines and tests whether a county's *fertility
    capacity* (not just its religion) drives differential trends.
    """
    specs = [
        ("(1) Baseline (no pretreatment trends)", None),
        ("(2) + literacy x trend", ("school1517",)),
        ("(3) + lit + urban x trend", ("school1517", "f_urban")),
        ("(4) + lit + urban + pruss x trend", ("school1517", "f_urban", "f_pruss")),
        ("(5) + lit + urban + pruss + jew x trend",
            ("school1517", "f_urban", "f_pruss", "f_jew")),
        ("(6) + lit + urban + pruss + jew + women-15-49-share x trend",
            ("school1517", "f_urban", "f_pruss", "f_jew", "women_share_15_49_1871")),
        # New (7)-(9) add moderators from the post-May-2026 data merges.
        # (7) injects pre-treatment mobility (BIR1871's born-in-Kreis
        # share) as a moderator: addresses the concern that high-Catholic
        # Polish provinces had distinctively immobile populations whose
        # demography evolved on its own trajectory. (8) adds the 1849
        # student share -- a literally pre-treatment human-capital
        # baseline (`school1517` is 1871, contemporaneous with treatment
        # start). (9) adds the 1882 farm-size Gini and the 1876 income-
        # tax-per-capita moderator to allow Catholic-share-relevant
        # structural-economic gradients to follow their own trends.
        ("(7) + born-in-Kreis share x trend",
            ("school1517", "f_urban", "f_pruss", "f_jew",
             "women_share_15_49_1871", "born_in_kreis_share_1871")),
        ("(8) + 1849 student share x trend",
            ("school1517", "f_urban", "f_pruss", "f_jew",
             "women_share_15_49_1871", "born_in_kreis_share_1871",
             "attend_rate_1849_baseline")),
        ("(9) + land Gini 1882 + log income-tax-pc 1876 x trend",
            ("school1517", "f_urban", "f_pruss", "f_jew",
             "women_share_15_49_1871", "born_in_kreis_share_1871",
             "attend_rate_1849_baseline", "land_gini_1882",
             "ln_income_tax_pc_1876")),
    ]

    # Materialise the 1849 student-share baseline once so it lives on
    # the working frame for each call below. We use the same
    # 1849-attendance proxy as ``schooling_channel`` does.
    df = df.copy()
    if ("attend_rate_1849_baseline" not in df.columns
            and "edu1849_pub_ele_stud_m" in df.columns
            and "pop1849_tot" in df.columns):
        students = df["edu1849_pub_ele_stud_m"].fillna(0) + df["edu1849_pub_ele_stud_f"].fillna(0)
        df["attend_rate_1849_baseline"] = students / df["pop1849_tot"].replace(0, np.nan)

    rows = []
    for outcome in outcomes:
        for label, pt in specs:
            try:
                res = run_baseline_did(
                    df, outcome=outcome, treatment="continuous",
                    pretreatment_trends=pt,
                    pretreatment_trends_form=form,
                )["result"]
                rows.append({
                    "outcome": outcome,
                    "spec": label,
                    "coef": float(res.params["cath_share_x_post"]),
                    "se": float(res.std_errors["cath_share_x_post"]),
                    "p": float(res.pvalues["cath_share_x_post"]),
                    "n": int(res.nobs),
                })
            except Exception as exc:
                logger.warning("pretreatment trends %s/%s failed: %s",
                               outcome, label, exc)
    return pd.DataFrame(rows)


def run_emigration_robustness(
    df: pd.DataFrame,
    outcomes: tuple[str, ...] = ("cbr", "marriage_rate"),
) -> pd.DataFrame:
    """
    Robustness specifications addressing the post-1885 Polish-province
    emigration confound:

    (1) Baseline TWFE (no controls beyond entity + year FE).
    (2) + population growth rate as additional control.
    (3) + implied net migration rate (= pop_change - natural_increase,
        per 1{,}000 pop) as additional control. The migration variable is
        itself an outcome of the Kulturkampf, so this is a "bad-control"
        specification -- but if the headline coefficient survives, the
        result clearly is not just emigration mechanics.
    (4) Sample restricted to pre-1885, before the Bismarck-era
        Polenausweisungen and Settlement Commission. Cleanest cut.
    (1') Baseline, no controls, restricted to the measured-migration
        sub-sample (1862-1867 and 1872-1886 -- the years where Galloway
        VIT records outmig_rate / net_mig_rate). Diagnostic row: the
        difference between (1) and (1') is the *sample-composition*
        effect of dropping war years and post-expulsion years; the
        difference between (1') and (5)/(6) is the *migration-channel*
        effect of conditioning on measured migration. Without this row,
        readers naturally compare (1) to (5)/(6) directly and
        mis-attribute the sample effect to the migration control.
    (5) + *measured* out-migration rate from Galloway VIT (annual,
        per 1,000 pop). Cleaner than the implied identity in (3) but only
        available for years with VIT migration columns (1862-1867 and
        1872-1886; ~21 of 29 panel years).
    (6) + *measured* net migration rate (in - out, per 1,000 pop). Same
        coverage caveat as (5).
    """
    work = df.copy().sort_values(["Code", "Year"])
    work["pop_change"] = work.groupby("Code")["Poptot"].diff()
    work["pop_growth_rate"] = (
        work["pop_change"] / work.groupby("Code")["Poptot"].shift(1) * 1000.0
    )
    work["natural_increase"] = work["Birtot"] - work["Dthtot"]
    work["implied_migration"] = work["pop_change"] - work["natural_increase"]
    work["migration_rate"] = work["implied_migration"] / work["Poptot"] * 1000.0

    rows = []
    for outcome in outcomes:
        for label, controls, sample_filter in [
            ("(1) Baseline (no controls)", [], None),
            ("(2) + pop growth rate",      ["pop_growth_rate"], None),
            ("(3) + implied migration",    ["migration_rate"], None),
            ("(4) Restrict to pre-1885",   [], lambda d: d["Year"] < 1885),
            ("(1') Baseline, measured-mig sub-sample", [],
                lambda d: d["outmig_rate"].notna() & d["net_mig_rate"].notna()),
            ("(5) + measured outmig rate", ["outmig_rate"], None),
            ("(6) + measured net mig rate", ["net_mig_rate"], None),
        ]:
            sub = work if sample_filter is None else work[sample_filter(work)]
            try:
                res = run_baseline_did(
                    sub, outcome=outcome, treatment="continuous", controls=controls
                )["result"]
                coef = float(res.params["cath_share_x_post"])
                se = float(res.std_errors["cath_share_x_post"])
                p = float(res.pvalues["cath_share_x_post"])
                rows.append({
                    "outcome": outcome,
                    "spec": label,
                    "coef": coef,
                    "se": se,
                    "p": p,
                    "n": int(res.nobs),
                })
            except Exception as exc:
                logger.warning("emigration robustness %s/%s failed: %s",
                               outcome, label, exc)
    return pd.DataFrame(rows)


def run_count_marriage_did(
    df: pd.DataFrame,
    outcome: str = "Martot",
) -> pd.DataFrame:
    """
    Total marriages (count) as outcome rather than marriage *rate*. Eliminates
    the population denominator, so changes in count cannot be mechanical
    population-shrinkage artefacts. Returns the headline coefficient.

    Also reports legitimate births per marriage (Birlegtot / Martot), which
    is a marital-fertility intensive-margin measure that does not depend on
    population at all.
    """
    work = df.copy()
    work["bir_per_mar"] = work["Birlegtot"] / work["Martot"].replace(0, np.nan)

    rows = []
    for label, outcome_col in [
        ("Total marriages (count)", "Martot"),
        ("Births per marriage", "bir_per_mar"),
    ]:
        try:
            res = run_baseline_did(
                work, outcome=outcome_col, treatment="continuous",
            )["result"]
            rows.append({
                "outcome": label,
                "coef": float(res.params["cath_share_x_post"]),
                "se": float(res.std_errors["cath_share_x_post"]),
                "p": float(res.pvalues["cath_share_x_post"]),
                "n": int(res.nobs),
            })
        except Exception as exc:
            logger.warning("count regression %s failed: %s", label, exc)
    return pd.DataFrame(rows)


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
        controls = []
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
        s = sub[["Code", "Year", outcome, "cath_share_x_fake_post"]].dropna().copy()
        s = s.set_index(["Code", "Year"])
        y = s[outcome]
        X = s[["cath_share_x_fake_post"]]
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
              "Rb"]].dropna().copy()
    sub["polish"] = sub["Rb"].isin(polish_rbs).astype(int)
    sub["cath_share_x_post"] = sub["cath_share"] * sub["post_kulturkampf"]
    sub["polish_x_post"] = sub["polish"] * sub["post_kulturkampf"]
    sub["cath_share_x_polish"] = sub["cath_share"] * sub["polish"]
    sub["triple"] = sub["cath_share"] * sub["post_kulturkampf"] * sub["polish"]
    sub = sub.set_index(["Code", "Year"])

    y = sub[outcome]
    # polish (level) and cath_share_x_polish are absorbed by entity FE; drop them.
    X = sub[["cath_share_x_post", "polish_x_post", "triple"]]
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
        controls = []
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
    pretreatment_trends: Optional[Sequence[str]] = None,
    pretreatment_trends_form: str = "linear",
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
        controls = []

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

    # Optional: pretreatment-characteristic time-trend interactions
    if pretreatment_trends:
        baselines = _county_baselines(panel, pretreatment_trends)
        years_arr = np.asarray(
            panel.index.get_level_values("Year"), dtype=float
        )
        codes_arr = np.asarray(panel.index.get_level_values("Code"))
        if pretreatment_trends_form == "linear":
            year_centered = years_arr - years_arr.mean()
            for var in pretreatment_trends:
                if var not in baselines:
                    continue
                baseline_obs = np.array(
                    [baselines[var].get(c, np.nan) for c in codes_arr], dtype=float
                )
                X[f"{var}_x_year"] = baseline_obs * year_centered
        else:
            year_index = pd.Series(years_arr, index=panel.index)
            year_dummies = pd.get_dummies(year_index, prefix="yr", drop_first=True)
            year_dummies.index = panel.index
            for var in pretreatment_trends:
                if var not in baselines:
                    continue
                baseline_obs = pd.Series(
                    [baselines[var].get(c, np.nan) for c in codes_arr],
                    index=panel.index,
                ).astype(float)
                inter = year_dummies.multiply(baseline_obs, axis=0)
                X = pd.concat([X, inter.add_prefix(f"{var}_x_")], axis=1)

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
        controls = []

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
