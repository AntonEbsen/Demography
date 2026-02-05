"""
force_mortality.py

Python translation of the Stata do-file "Force_mortality":
- Scatter plots of mortality_rate vs age_id for DNK males across years
- Nonlinear Gompertz law with Makeham constant: m = a + b * exp(c * age)
- Log-linear Gompertz with Makeham = 0: ln(m) = intercept + slope * age
- Survival curve implied by Makeham=0 Gompertz

Designed to be imported into a Jupyter notebook, but also runnable as a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.optimize import curve_fit


XTICKS = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


REQUIRED_COLUMNS = {"isocode", "year", "sex_nr", "age_id", "mortality_rate"}


@dataclass(frozen=True)
class GompertzMakehamParams:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class LogLinearGompertzParams:
    intercept: float  # b_ln in your Stata code
    slope: float      # c_ln in your Stata code


def load_stata(path: str) -> pd.DataFrame:
    """Load a .dta file and validate required columns."""
    df = pd.read_stata(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return df


def subset_country_sex_year(
    df: pd.DataFrame,
    isocode: str,
    sex_nr: int,
    year: int,
) -> pd.DataFrame:
    """Filter and sort like Stata: if isocode==... & year==... & sex_nr==...; sort by age_id."""
    out = df[(df["isocode"] == isocode) & (df["sex_nr"] == sex_nr) & (df["year"] == year)].copy()
    out = out.sort_values("age_id")
    return out


def plot_scatter_mortality_age(
    df_sub: pd.DataFrame,
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Equivalent to Stata: sc mortality_rate age_id, xlabel(...)"""
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(df_sub["age_id"], df_sub["mortality_rate"])
    ax.set_xticks(XTICKS)
    ax.set_xlabel("age_id")
    ax.set_ylabel("mortality_rate")
    ax.set_title(title)
    return ax


def plot_joint_scatter_years(
    df: pd.DataFrame,
    isocode: str,
    sex_nr: int,
    years: Iterable[int],
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Equivalent to Stata joint graph: sc ... || sc ... || sc ... with legend labels."""
    if ax is None:
        _, ax = plt.subplots()

    for y in years:
        sub = subset_country_sex_year(df, isocode=isocode, sex_nr=sex_nr, year=y)
        ax.scatter(sub["age_id"], sub["mortality_rate"], label=str(y))

    ax.set_xticks(XTICKS)
    ax.set_xlabel("age_id")
    ax.set_ylabel("mortality_rate")
    ax.set_title(title)
    ax.legend()
    return ax


def gompertz_makeham(age: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """m(age) = a + b * exp(c * age)"""
    return a + b * np.exp(c * age)


def fit_gompertz_makeham_nl(
    df_sub: pd.DataFrame,
    initial: Tuple[float, float, float] = (0.005, 0.001, 0.1),
    nonnegative: bool = True,
    maxfev: int = 20000,
) -> GompertzMakehamParams:
    """
    Equivalent to Stata:
    nl (mortality_rate = {a} + {b} * exp({c} * age_id)), initial(a ... b ... c ...)
    """
    x = df_sub["age_id"].to_numpy(dtype=float)
    y = df_sub["mortality_rate"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if nonnegative:
        bounds = (0, np.inf)
    else:
        bounds = (-np.inf, np.inf)

    params, _cov = curve_fit(
        gompertz_makeham,
        x,
        y,
        p0=list(initial),
        bounds=bounds,
        maxfev=maxfev,
    )

    a, b, c = (float(params[0]), float(params[1]), float(params[2]))
    return GompertzMakehamParams(a=a, b=b, c=c)


def add_nl_prediction_column(
    df_sub: pd.DataFrame,
    params: GompertzMakehamParams,
    colname: str = "NL_predict_mort",
) -> pd.DataFrame:
    """Equivalent to gen/replace for NL predictions in Stata."""
    out = df_sub.copy()
    out[colname] = gompertz_makeham(out["age_id"].astype(float).to_numpy(), params.a, params.b, params.c)
    return out


def plot_line_vs_scatter(
    df_sub: pd.DataFrame,
    x_col: str,
    y_line_col: str,
    y_scatter_col: str,
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Equivalent to Stata: line pred age || sc actual age"""
    if ax is None:
        _, ax = plt.subplots()

    df_sub = df_sub.sort_values(x_col)
    ax.plot(df_sub[x_col], df_sub[y_line_col], label="Prediction")
    ax.scatter(df_sub[x_col], df_sub[y_scatter_col], label="Observed")

    ax.set_xticks(XTICKS)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_scatter_col)
    ax.set_title(title)
    ax.legend()
    return ax


def add_log_mortality(df: pd.DataFrame, colname: str = "lnmortality_rate") -> pd.DataFrame:
    """Equivalent to Stata: gen lnmortality_rate=ln(mortality_rate). Nonpositive -> missing."""
    out = df.copy()
    out[colname] = np.where(out["mortality_rate"] > 0, np.log(out["mortality_rate"]), np.nan)
    return out


def fit_loglinear_gompertz(
    df_sub: pd.DataFrame,
    log_col: str = "lnmortality_rate",
) -> Tuple[LogLinearGompertzParams, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Equivalent to Stata:
    reg lnmortality_rate age_id
    b_ln = _b[_cons]
    c_ln = _b[age_id]
    """
    d = df_sub.dropna(subset=[log_col]).copy()

    X = sm.add_constant(d["age_id"].astype(float))
    y = d[log_col].astype(float)

    model = sm.OLS(y, X).fit()
    intercept = float(model.params["const"])
    slope = float(model.params["age_id"])
    return LogLinearGompertzParams(intercept=intercept, slope=slope), model


def add_loglinear_predictions_for_years(
    df: pd.DataFrame,
    params_by_year: Dict[int, LogLinearGompertzParams],
    isocode: str,
    sex_nr: int,
    ln_pred_col: str = "ln_predict_mort",
    pred_col: str = "predict_mort",
) -> pd.DataFrame:
    """
    Equivalent to Stata:
    replace ln_predict_mort = b_ln_year + age_id*c_ln_year if ...
    gen predict_mort = exp(ln_predict_mort)
    """
    out = df.copy()
    out[ln_pred_col] = np.nan

    for year, params in params_by_year.items():
        mask = (out["isocode"] == isocode) & (out["sex_nr"] == sex_nr) & (out["year"] == year)
        out.loc[mask, ln_pred_col] = params.intercept + params.slope * out.loc[mask, "age_id"].astype(float)

    out[pred_col] = np.exp(out[ln_pred_col])
    return out


def survival_curve_makeham0(
    age: np.ndarray,
    intercept: float,
    slope: float,
    start_population: float = 1000.0,
) -> np.ndarray:
    """
    Equivalent to Stata:
    b_1950 = exp(b_ln_1950)
    l_curve = 1000*exp( (-b_1950/c_ln_1950)*(exp(c_ln_1950*age_id) - 1))
    where intercept=b_ln, slope=c_ln.
    """
    b = np.exp(intercept)
    return start_population * np.exp((-b / slope) * (np.exp(slope * age) - 1))


def add_survival_curve_column(
    df_sub: pd.DataFrame,
    params: LogLinearGompertzParams,
    colname: str = "l_curve",
    start_population: float = 1000.0,
) -> pd.DataFrame:
    """Compute l_curve for a subset (e.g., DNK males 1950)."""
    out = df_sub.copy()
    age = out["age_id"].astype(float).to_numpy()
    out[colname] = survival_curve_makeham0(age, params.intercept, params.slope, start_population=start_population)
    return out


def run_force_mortality_workflow(
    df: pd.DataFrame,
    isocode: str = "DNK",
    sex_nr: int = 3,
    years_scatter: Tuple[int, int, int] = (1910, 1950, 2010),
) -> dict:
    """
    Convenience function if you want a single call that returns estimates and key dataframes.
    Notebook-friendly: returns dict with params/models and predicted frames.
    """
    # Scatter subsets
    d = {y: subset_country_sex_year(df, isocode=isocode, sex_nr=sex_nr, year=y) for y in years_scatter}

    # Nonlinear fits (example: 1950 + 2010)
    nl_1950 = fit_gompertz_makeham_nl(d[1950])
    d1950_nl = add_nl_prediction_column(d[1950], nl_1950)

    nl_2010 = fit_gompertz_makeham_nl(d[2010])

    # Log-linear fits
    df_log = add_log_mortality(df)

    ll_1950_params, ll_1950_model = fit_loglinear_gompertz(subset_country_sex_year(df_log, isocode, sex_nr, 1950))
    ll_2010_params, ll_2010_model = fit_loglinear_gompertz(subset_country_sex_year(df_log, isocode, sex_nr, 2010))

    df_pred = add_loglinear_predictions_for_years(
        df_log,
        params_by_year={1950: ll_1950_params, 2010: ll_2010_params},
        isocode=isocode,
        sex_nr=sex_nr,
    )

    # Survival curve (1950)
    surv1950 = add_survival_curve_column(subset_country_sex_year(df_pred, isocode, sex_nr, 1950), ll_1950_params)

    return {
        "subsets": d,
        "nl_params_1950": nl_1950,
        "nl_params_2010": nl_2010,
        "d1950_nl": d1950_nl,
        "ll_params_1950": ll_1950_params,
        "ll_model_1950": ll_1950_model,
        "ll_params_2010": ll_2010_params,
        "ll_model_2010": ll_2010_model,
        "df_pred": df_pred,
        "survival_1950": surv1950,
    }


if __name__ == "__main__":
    # Minimal CLI usage example (edit path before running):
    path = r"PATH_TO_YOUR\HMD_5yr_mort_rates.dta"
    df0 = load_stata(path)
    results = run_force_mortality_workflow(df0)
    print("NL 1950:", results["nl_params_1950"])
    print("NL 2010:", results["nl_params_2010"])
    print("LL 1950:", results["ll_params_1950"])
    print("LL 2010:", results["ll_params_2010"])
