"""
latex_tables.py
===============
Generate JEH-style LaTeX tables for the Kulturkampf paper.

Each function writes a standalone `.tex` file that can be included in an
Overleaf document via ``\\input{path/to/table.tex}``. Tables use ``booktabs``
rules (\\toprule, \\midrule, \\bottomrule) — make sure your preamble has
``\\usepackage{booktabs}``. No other LaTeX packages required.

Significance stars follow the JEH convention:
    * p<0.10   ** p<0.05   *** p<0.01

Run as a script to regenerate all tables under
``exam_project2/outputs/tables/``::

    python -m src.analysis.latex_tables
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.analysis.channels import infant_mortality_analysis
from src.analysis.coale_indices import (
    aggregate_by_group_period as coale_aggregate,
    compute_coale_indices,
    did_on_indices as coale_did,
)
from src.analysis.cohort_translation import cohort_translation
from src.analysis.conley_se import spatial_did_se
from src.analysis.dcdh_diagnostic import diagnostic as dcdh_diagnostic
from src.analysis.honest_did import honest_did_bounds
from src.analysis.magnitudes import magnitude_decomposition
from src.analysis.multiple_testing import sharpened_q_values
from src.analysis.permutation_inference import permutation_p_value
from src.analysis.polish_german import polish_german_rollback
from src.analysis.regressions import (
    pretrends_wald_test,
    run_baseline_did,
    run_count_marriage_did,
    run_emigration_robustness,
    run_event_study,
    run_fake_treatment_placebo,
    run_heterogeneity_did,
    run_iv_did,
    run_iv_did_multi,
    run_jewish_placebo,
    run_long_difference,
    run_pretreatment_trends_robustness,
    run_robustness,
    run_start_year_sensitivity,
    run_subsample_decomposition,
    run_triple_difference_polish,
)
from src.analysis.rollback import rollback_event_study
from src.analysis.utils import safe_panel_ols
from src.analysis.variance_decomposition import variance_decomposition
from src.analysis.wild_bootstrap import wild_cluster_bootstrap
from src.data.centroids import load_centroids

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_panel.parquet"

# Display labels for outcomes and regressors. Edit here to retitle in tables.
OUTCOME_LABELS: dict[str, str] = {
    # Headline rates use the mid-year-interpolated denominator (standard
    # demographic convention, matching Galloway, Hammel & Lee 1994).
    # The label intentionally drops any qualifier -- this is what "the
    # crude birth rate" means.
    "cbr": "Crude birth rate",
    "legitimate_br": "Legit.\\ birth rate",
    "illegitimate_br": "Illegit.\\ birth rate",
    "illegitimacy_ratio": "Illegitimacy ratio",
    "marriage_rate": "Marriage rate",
    "infant_mortality_rate": "Infant mortality",
    "cath_marriage_share": "Catholic marriage share",
    # Princeton EFP / Coale indices. I_g is the Galloway-tradition
    # marital-fertility headline (Hutterite-normalised; Galloway, Hammel
    # & Lee 1994 use its unnormalised form, the GMFR). See
    # coale_indices.py and DATA_APPENDIX.md sec. 6.5.
    "I_f": "$I_f$ (overall fertility)",
    "I_g": "$I_g$ (marital fertility)",
    "I_h": "$I_h$ (illegitimate fertility)",
    "gmfr": "GMFR (per 1k married women)",
    # Deprecated -- kept for back-compat. Static-1871 GFR is superseded
    # by I_g for marital-fertility analysis.
    "gfr_static_1871": "GFR (1871 base, deprecated)",
    # Galloway carry-forward variants: same numerator, but denominator is
    # the previous December census carried forward unchanged in
    # inter-census years (i.e. raw Galloway `Poptot`). Used only in the
    # mid-year-vs-carry-forward robustness row of the headline DiD table.
    "cbr_carryforward": "CBR (Galloway carry-forward)",
    "legitimate_br_carryforward": "Legit.\\ BR (carry-forward)",
    "illegitimate_br_carryforward": "Illegit.\\ BR (carry-forward)",
    "marriage_rate_carryforward": "Marriage rate (carry-forward)",
}

REGRESSOR_LABELS: dict[str, str] = {
    "cath_share_x_post": r"CathShare $\times$ Post",
    "treat_x_post": r"HighCath $\times$ Post",
    "treat25_x_post": r"HighCath$_{25}$ $\times$ Post",
    "treat75_x_post": r"HighCath$_{75}$ $\times$ Post",
    "cath_x_enforcement": r"CathShare $\times$ Enforcement (1873--78)",
    "cath_x_rollback": r"CathShare $\times$ Rollback (1880--87)",
    "cath_x_postrollback": r"CathShare $\times$ Post-rollback (1888+)",
    "ln_pop": r"$\ln(\text{Population})$",
    "infant_mortality_rate": "Infant mortality rate",
}


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return r"$^{***}$"
    if p < 0.05:
        return r"$^{**}$"
    if p < 0.10:
        return r"$^{*}$"
    return ""


def _fmt_coef(coef: float, p: float, digits: int = 3) -> str:
    if pd.isna(coef):
        return ""
    return f"{coef:.{digits}f}{_stars(p)}"


def _fmt_se(se: float, digits: int = 3) -> str:
    if pd.isna(se):
        return ""
    return f"({se:.{digits}f})"


def _label(name: str) -> str:
    return REGRESSOR_LABELS.get(name, name.replace("_", r"\_"))


def _outcome_label(name: str) -> str:
    return OUTCOME_LABELS.get(name, name.replace("_", r"\_"))


def _latex_escape(s: str) -> str:
    """Escape LaTeX-special characters in free-text labels."""
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{").replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("×", r"$\times$")
    s = s.replace("≤", r"$\leq$")
    s = s.replace("≥", r"$\geq$")
    s = s.replace(">", r"$>$").replace("<", r"$<$")
    return s


def _wrap_table(
    body: str,
    *,
    caption: str,
    label: str,
    n_cols: int,
    notes: str,
) -> str:
    """Wrap a tabular body in JEH-style table scaffolding."""
    return (
        "% Auto-generated by src/analysis/latex_tables.py — do not edit by hand.\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\begin{minipage}{\\linewidth}\n"
        "\\vspace{0.5em}\n"
        f"\\footnotesize \\textit{{Notes:}} {notes}\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def summary_statistics_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """Means and SDs by treatment group × period."""
    out_path = out_path or TABLES_DIR / "summary_stats.tex"
    df = panel.copy()
    df["period"] = np.where(df["Year"] >= 1873, "Post (1873--90)", "Pre (1862--72)")
    df["group"] = np.where(df["high_cath"] == 1, "High Catholic", "Low Catholic")

    rows: list[str] = []
    for var in outcomes:
        agg = (
            df.groupby(["group", "period"])[var]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        cells: list[str] = [_outcome_label(var)]
        for grp in ("Low Catholic", "High Catholic"):
            for per in ("Pre (1862--72)", "Post (1873--90)"):
                row = agg[(agg["group"] == grp) & (agg["period"] == per)]
                if row.empty or pd.isna(row.iloc[0]["mean"]):
                    cells.extend(["", ""])
                else:
                    m, s = row.iloc[0]["mean"], row.iloc[0]["std"]
                    cells.append(f"{m:.2f}")
                    cells.append(f"[{s:.2f}]")
        rows.append(" & ".join(cells) + r" \\")

    n_low = int(df[df["group"] == "Low Catholic"]["Code"].nunique())
    n_high = int(df[df["group"] == "High Catholic"]["Code"].nunique())

    tabular = (
        "\\begin{tabular}{l*{8}{c}}\n"
        "\\toprule\n"
        " & \\multicolumn{4}{c}{Low Catholic ($\\le 50\\%$)} & "
        "\\multicolumn{4}{c}{High Catholic ($> 50\\%$)} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        " & \\multicolumn{2}{c}{Pre} & \\multicolumn{2}{c}{Post} & "
        "\\multicolumn{2}{c}{Pre} & \\multicolumn{2}{c}{Post} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
        " & Mean & SD & Mean & SD & Mean & SD & Mean & SD \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\midrule\n"
        f"Counties & \\multicolumn{{4}}{{c}}{{{n_low}}} & "
        f"\\multicolumn{{4}}{{c}}{{{n_high}}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption="Summary statistics by Catholic share and period",
        label="tab:summary_stats",
        n_cols=9,
        notes=(
            "Means and standard deviations (in brackets) computed across "
            "county--year cells. ``High Catholic'' counties have $>$50\\% "
            "Catholic population in 1871. ``Pre'' covers 1862--1872; ``Post'' "
            "covers 1873--1890. Birth and marriage rates are per 1,000 "
            "population; illegitimacy ratio is in percent."
        ),
    )
    _write(out_path, out)
    return out


def _did_column(panel: pd.DataFrame, outcome: str, fe_design: str) -> dict:
    """One column of the baseline_did table."""
    res = run_baseline_did(
        panel, outcome=outcome, treatment="continuous", fe_design=fe_design
    )["result"]
    return {
        "outcome": outcome,
        "fe_design": fe_design,
        "coef": res.params["cath_share_x_post"],
        "se": res.std_errors["cath_share_x_post"],
        "p": res.pvalues["cath_share_x_post"],
        "ln_pop_coef": res.params.get("ln_pop", np.nan),
        "ln_pop_se": res.std_errors.get("ln_pop", np.nan),
        "ln_pop_p": res.pvalues.get("ln_pop", np.nan),
        "n": int(res.nobs),
        "r2": float(res.rsquared_within),
    }


def baseline_did_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """Multi-outcome baseline DiD with TWFE and stricter Year x Rb FE columns.

    Headline rates (`cbr`, `legitimate_br`, `illegitimate_br`,
    `marriage_rate`) use the standard demographic CBR convention: the
    population denominator is linearly interpolated between consecutive
    December census anchors and evaluated at July 1 of each calendar year.
    The ``Galloway carry-forward robustness'' row reports the same
    coefficients using the raw Galloway `Poptot` (previous December
    census carried forward in inter-census years), so a reader can see
    how using the database "out of the box" differs from the proper
    mid-year convention.

    Includes a pre-trends Wald-$\\chi^2$ $p$-value row for each outcome,
    plus a one-line ``GFR comparison'' showing the same test on
    ``gfr_static_1871`` (births per 1{,}000 women aged 15--49 using the
    1871 census denominator). The GFR line addresses the standard
    demographic critique that CBR is mechanically affected by age
    structure -- a reader can see directly that the pre-trends
    conclusion is not driven by age-composition contamination of the
    headline outcome.
    """
    out_path = out_path or TABLES_DIR / "baseline_did.tex"

    cols_twfe = [_did_column(panel, o, "twfe") for o in outcomes]
    cols_strict = [_did_column(panel, o, "year_x_rb") for o in outcomes]
    cols = cols_twfe + cols_strict
    n_out = len(outcomes)
    n = len(cols)

    # Carry-forward (Galloway raw) robustness: rerun TWFE and Year x Rb
    # on the `_carryforward` rate variants -- denominator is the previous
    # December census carried forward unchanged in inter-census years
    # (i.e. raw Galloway `Poptot`). This is what one gets by using the
    # database "out of the box". Reported here so a reader can see how
    # the headline (proper-mid-year) coefficients differ from the raw
    # Galloway-tradition number; the headline rows themselves use the
    # mid-year convention (standard demographic CBR), not this one.
    _carryforward_map = {
        "cbr": "cbr_carryforward",
        "legitimate_br": "legitimate_br_carryforward",
        "illegitimate_br": "illegitimate_br_carryforward",
        "marriage_rate": "marriage_rate_carryforward",
    }
    cf_outcomes = [_carryforward_map.get(o, o) for o in outcomes]
    cols_cf_twfe = [_did_column(panel, o, "twfe") for o in cf_outcomes]
    cols_cf_strict = [_did_column(panel, o, "year_x_rb") for o in cf_outcomes]
    cols_cf = cols_cf_twfe + cols_cf_strict

    # Pre-trends Wald chi-squared per outcome (TWFE event study), and a
    # marital-fertility (I_g) comparison reported on a single auxiliary
    # line. I_g is the Galloway, Hammel & Lee (1994) headline outcome --
    # a Hutterite-normalised marital fertility rate that nets out
    # nuptiality. Reporting its pre-trends chi-squared here lets a
    # demography-aware reader confirm the pre-trends conclusion holds
    # under the Princeton EFP framework.
    pretrends_per_outcome = [
        pretrends_wald_test(panel, outcome=o) for o in outcomes
    ]
    pretrends_ig = pretrends_wald_test(panel, outcome="I_g")

    # Header: Panel A (TWFE) and Panel B (Year x Rb FE) spanning the columns
    cmidrules = (
        f"\\cmidrule(lr){{2-{n_out + 1}}}"
        f"\\cmidrule(lr){{{n_out + 2}-{n + 1}}}"
    )
    panel_header = (
        " & "
        + f"\\multicolumn{{{n_out}}}{{c}}{{Panel A: County + Year FE}}"
        + " & "
        + f"\\multicolumn{{{n_out}}}{{c}}{{Panel B: County + (Year $\\times$ Rb) FE}}"
        + r" \\"
        + "\n"
        + cmidrules
    )
    outcome_header = (
        " & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + " & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
    )
    col_nums = (
        " & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
    )

    coef_row = (
        _label("cath_share_x_post")
        + " & "
        + " & ".join(_fmt_coef(c["coef"], c["p"]) for c in cols)
        + r" \\"
    )
    se_row = (
        " & "
        + " & ".join(_fmt_se(c["se"]) for c in cols)
        + r" \\"
    )
    lnpop_coef = (
        _label("ln_pop")
        + " & "
        + " & ".join(_fmt_coef(c["ln_pop_coef"], c["ln_pop_p"]) for c in cols)
        + r" \\"
    )
    lnpop_se = (
        " & "
        + " & ".join(_fmt_se(c["ln_pop_se"]) for c in cols)
        + r" \\"
    )

    yes_twfe = " & ".join("Yes" for _ in cols_twfe)
    yes_strict = " & ".join("Yes" for _ in cols_strict)
    no_strict = " & ".join("--" for _ in cols_strict)

    # Pre-trends Wald p-value row: same value for both panels because the
    # test is run on the TWFE event-study; replicated across columns so a
    # reader scanning a single column sees it.
    pretrends_p_row = (
        "Pre-trends $\\chi^{2}$ $p$ & "
        + " & ".join(f"{pt['p_value']:.3f}" for pt in pretrends_per_outcome)
        + " & "
        + " & ".join(f"{pt['p_value']:.3f}" for pt in pretrends_per_outcome)
        + r" \\"
    )

    # Carry-forward (Galloway raw) robustness rows: same TWFE / Year x Rb
    # FE specification but using rate variables built from the raw
    # Galloway `Poptot` denominator (previous December census carried
    # forward in inter-census years). Two rows (coef + SE) so the table
    # stays compact rather than adding a third panel.
    carryforward_coef_row = (
        "\\quad CathShare $\\times$ Post (Galloway carry-forward) & "
        + " & ".join(_fmt_coef(c["coef"], c["p"]) for c in cols_cf)
        + r" \\"
    )
    carryforward_se_row = (
        " & "
        + " & ".join(_fmt_se(c["se"]) for c in cols_cf)
        + r" \\"
    )
    # One-line I_g pre-trends Wald comparison (spans full table width).
    pretrends_ig_row = (
        f"\\multicolumn{{{n + 1}}}{{l}}{{"
        f"\\textit{{Pre-trends Wald $\\chi^{{2}}$ on $I_g$ "
        f"(Coale marital fertility, Galloway-tradition headline)}}: "
        f"$\\chi^{{2}} = {pretrends_ig['wald_chi2']:.2f}$, "
        f"df $= {pretrends_ig['df']}$, "
        f"$p = {pretrends_ig['p_value']:.3f}$"
        f"}} \\\\"
    )

    tabular = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + panel_header + "\n"
        + outcome_header + "\n"
        + col_nums + "\n"
        + "\\midrule\n"
        + coef_row + "\n"
        + se_row + "\n\\addlinespace\n"
        + lnpop_coef + "\n"
        + lnpop_se + "\n"
        "\\midrule\n"
        + f"County FE & {yes_twfe} & {yes_strict} \\\\\n"
        + f"Year FE & {yes_twfe} & {no_strict} \\\\\n"
        + f"Year $\\times$ Rb FE & {' & '.join('--' for _ in cols_twfe)} & {yes_strict} \\\\\n"
        + "Observations & "
        + " & ".join(f"{c['n']:,}" for c in cols)
        + " \\\\\n"
        + "Within $R^{2}$ & "
        + " & ".join(f"{c['r2']:.3f}" for c in cols)
        + " \\\\\n"
        + "\\midrule\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Galloway carry-forward robustness}} "
        f"(rate $=$ count $/$ raw Galloway \\texttt{{Poptot}}; previous Dec.\\ census "
        f"carried forward in inter-census years)}} \\\\\n"
        + carryforward_coef_row + "\n"
        + carryforward_se_row + "\n"
        + "\\midrule\n"
        + pretrends_p_row + "\n"
        + "\\addlinespace\n"
        + pretrends_ig_row + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption="Baseline difference-in-differences: Kulturkampf and demographic outcomes",
        label="tab:baseline_did",
        n_cols=n + 1,
        notes=(
            "Two-way fixed-effects estimates of equation "
            "$Y_{it} = \\beta\\,(\\mathrm{CathShare}_i \\times \\mathrm{Post}_t) + "
            "\\alpha_i + \\delta_t + \\gamma X_{it} + \\varepsilon_{it}$, with "
            "$\\delta_t$ replaced by year~$\\times$~Regierungsbezirk fixed effects "
            "in Panel~B. Post is an indicator for $t \\geq 1873$. Standard errors "
            "clustered at the county level in parentheses. Birth and marriage rates "
            "per 1{,}000 \\emph{mid-year} population; illegitimacy ratio in percent. "
            "Mid-year population is constructed by linearly interpolating between "
            "consecutive December census anchors and evaluating at July 1 of each "
            "calendar year (standard demographic convention). The ``Galloway "
            "carry-forward robustness'' row reports the same coefficients using "
            "the raw Galloway \\texttt{Poptot}, which carries the previous "
            "December census forward unchanged in inter-census years and biases "
            "CBR upward by 1--3\\% in growing populations. The "
            "``Pre-trends $\\chi^{2}$ $p$'' row reports the joint Wald test that "
            "all event-study coefficients in the pre-1872 period equal zero "
            "(estimated separately on the TWFE event-study; identical $p$-value "
            "applies under both FE designs). The single-line ``$I_g$ comparison'' "
            "reports the same test on Coale's marital-fertility index "
            "(Hutterite-normalised legitimate births per married woman 15--49) -- "
            "the headline outcome in Galloway, Hammel \\& Lee (1994). The "
            "companion event-study figure \\texttt{fig5\\_event\\_study\\_cbr\\_ig.png} "
            "plots the CBR and $I_g$ event studies side by side. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def robustness_table(robustness_df: pd.DataFrame, out_path: Path | None = None) -> str:
    """Long-form robustness battery from ``run_robustness``."""
    out_path = out_path or TABLES_DIR / "robustness.tex"

    rows: list[str] = []
    for _, r in robustness_df.iterrows():
        rows.append(
            f"{_latex_escape(r['Specification'])} & "
            f"{_fmt_coef(r['Coefficient'], r['p_value'])} & "
            f"{_fmt_se(r['SE'])} & "
            f"{int(r['N']):,} & {int(r['N_counties']):,} \\\\"
        )

    tabular = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Specification & Coefficient & SE & $N$ & Counties \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption="Robustness: alternative specifications of the baseline DiD",
        label="tab:robustness",
        n_cols=5,
        notes=(
            "Each row is a separate two-way fixed-effects regression of crude birth "
            "rate on the listed treatment variable, controlling for $\\ln(\\mathrm{Pop})$ "
            "and infant mortality. Standard errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def channels_table(panel: pd.DataFrame, out_path: Path | None = None) -> str:
    """Mechanisms: illegitimacy + infant mortality."""
    out_path = out_path or TABLES_DIR / "channels.tex"

    illeg = safe_panel_ols(panel, "illegitimacy_ratio", ["cath_share_x_post", "ln_pop"])
    mort_panel = panel[panel["Year"] >= 1875].copy()
    mort_panel["post_rollback"] = (mort_panel["Year"] >= 1880).astype(int)
    mort_panel["cath_x_rollback"] = mort_panel["cath_share"] * mort_panel["post_rollback"]
    mort = safe_panel_ols(mort_panel, "infant_mortality_rate", ["cath_x_rollback", "ln_pop"])

    cols = [
        {
            "header": "Illegitimacy ratio",
            "treat_label": _label("cath_share_x_post"),
            "treat_coef": illeg.params["cath_share_x_post"],
            "treat_se": illeg.std_errors["cath_share_x_post"],
            "treat_p": illeg.pvalues["cath_share_x_post"],
            "ln_pop_coef": illeg.params["ln_pop"],
            "ln_pop_se": illeg.std_errors["ln_pop"],
            "ln_pop_p": illeg.pvalues["ln_pop"],
            "n": int(illeg.nobs),
            "r2": float(illeg.rsquared_within),
            "sample": "Full panel",
        },
        {
            "header": "Infant mortality",
            "treat_label": _label("cath_x_rollback"),
            "treat_coef": mort.params["cath_x_rollback"],
            "treat_se": mort.std_errors["cath_x_rollback"],
            "treat_p": mort.pvalues["cath_x_rollback"],
            "ln_pop_coef": mort.params["ln_pop"],
            "ln_pop_se": mort.std_errors["ln_pop"],
            "ln_pop_p": mort.pvalues["ln_pop"],
            "n": int(mort.nobs),
            "r2": float(mort.rsquared_within),
            "sample": "1875+ (def.\\ change)",
        },
    ]

    n = len(cols)
    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        "Dependent variable: & "
        + " & ".join(c["header"] for c in cols)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
    )
    # Each column has its own treat label. Render as two row-blocks.
    for i, c in enumerate(cols):
        body += f"{c['treat_label']} "
        for j in range(n):
            if j == i:
                body += f"& {_fmt_coef(c['treat_coef'], c['treat_p'])} "
            else:
                body += "& "
        body += r"\\" + "\n"
        for j in range(n):
            if j == i:
                body += f"& {_fmt_se(c['treat_se'])} "
            else:
                body += "& "
        body += r"\\" + "\n\\addlinespace\n"

    body += (
        _label("ln_pop")
        + " & "
        + " & ".join(_fmt_coef(c["ln_pop_coef"], c["ln_pop_p"]) for c in cols)
        + r" \\"
        + "\n & "
        + " & ".join(_fmt_se(c["ln_pop_se"]) for c in cols)
        + r" \\"
        + "\n\\midrule\n"
        + f"Sample & {' & '.join(c['sample'] for c in cols)} \\\\\n"
        + f"County FE & {' & '.join('Yes' for _ in cols)} \\\\\n"
        + f"Year FE & {' & '.join('Yes' for _ in cols)} \\\\\n"
        + "Observations & "
        + " & ".join(f"{c['n']:,}" for c in cols)
        + " \\\\\n"
        + "Within $R^{2}$ & "
        + " & ".join(f"{c['r2']:.3f}" for c in cols)
        + " \\\\\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption="Mechanisms: illegitimacy and infant mortality",
        label="tab:channels",
        n_cols=n + 1,
        notes=(
            "Column (1) tests whether illegitimate birth ratios rose in Catholic "
            "counties under the Kulturkampf using the full panel. Column (2) "
            "examines infant mortality during the rollback period (1880--87 vs "
            "1875--79); restricted to 1875+ because Galloway's infant mortality "
            "definition changes in 1875. Standard errors clustered at the county "
            "level. $^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def polish_german_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "I_g", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """Heterogeneity by sub-region (Polish vs German Catholic vs Protestant).

    A panel per outcome covering the full Coale--Watkins decomposition:
    CBR (overall fertility), $I_g$ (marital fertility -- the Galloway,
    Hammel & Lee 1994 headline), and marriage rate (nuptiality). Reading
    across panels reveals whether the sub-region divergence operates
    through marital fertility (within-marriage childbearing) or through
    marriage formation.
    """
    out_path = out_path or TABLES_DIR / "polish_german.tex"

    panels = {
        outcome: polish_german_rollback(
            panel.copy(), outcome=outcome, savepath=None
        )["results"]
        for outcome in outcomes
    }

    # Sub-region columns (same set in every panel; take from the first).
    cols = list(next(iter(panels.values())).keys())
    n = len(cols)

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        " & " + " & ".join(c for c in cols) + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
    )

    panel_labels = "ABCDEFGH"
    for letter, outcome in zip(panel_labels, outcomes):
        pg = panels[outcome]
        body += (
            f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel {letter}: "
            f"{_outcome_label(outcome)}}}}} \\\\\n"
            "\\addlinespace\n"
        )
        for key, label in [
            ("enforcement", _label("cath_x_enforcement")),
            ("rollback", _label("cath_x_rollback")),
            ("post_rollback", _label("cath_x_postrollback")),
        ]:
            body += (
                f"{label} & "
                + " & ".join(_fmt_coef(pg[c][key]["coef"], pg[c][key]["p"]) for c in cols)
                + r" \\" + "\n"
                + " & "
                + " & ".join(_fmt_se(pg[c][key]["se"]) for c in cols)
                + r" \\" + "\n\\addlinespace\n"
            )

    # Footer (county / year FE and counties) — identical across panels;
    # use the last panel's county counts (sub-regions are the same).
    last_pg = panels[outcomes[-1]]
    body += (
        "\\midrule\n"
        + f"County FE & {' & '.join('Yes' for _ in cols)} \\\\\n"
        + f"Year FE & {' & '.join('Yes' for _ in cols)} \\\\\n"
        + "Counties & "
        + " & ".join(f"{last_pg[c]['n_counties']:,}" for c in cols)
        + " \\\\\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Heterogeneity: Kulturkampf by sub-region across enforcement and "
            "rollback periods"
        ),
        label="tab:polish_german",
        n_cols=n + 1,
        notes=(
            "Each column estimates the same specification on a different "
            "sub-sample of provinces: Polish (Posen, Bromberg), German Catholic "
            "(Cologne, Koblenz, Trier, Aachen, Oppeln, M{\\\"u}nster), and the "
            "remaining (largely Protestant) provinces. Panel A reports the crude "
            "birth rate; Panel B reports the General Fertility Rate (births per "
            "1{,}000 women aged 15--49 using the 1871 census denominator), "
            "which addresses the demographic critique that CBR is mechanically "
            "affected by age structure. The qualitative sub-region pattern "
            "(Polish negative, German Catholic positive in enforcement; "
            "Protestant null) is robust across the two outcomes; magnitudes "
            "scale by roughly $1/0.25\\approx 4$ as expected from the women-15--49 "
            "population share. Standard errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def iv_results_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """OLS vs 2SLS estimates using distance to Wittenberg as an instrument."""
    out_path = out_path or TABLES_DIR / "iv_results.tex"

    cols = [run_iv_did(panel, outcome=o, instrument="kmwittenberg") for o in outcomes]
    n = len(cols)

    head = (
        "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
    )
    nums = (
        " & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
    )
    panel_a_label = (
        f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel A: OLS}}}} \\\\"
    )
    ols_coef = (
        _label("cath_share_x_post")
        + " & "
        + " & ".join(_fmt_coef(c["ols_coef"], c["ols_p"]) for c in cols)
        + r" \\"
    )
    ols_se = (
        " & "
        + " & ".join(_fmt_se(c["ols_se"]) for c in cols)
        + r" \\"
    )
    panel_b_label = (
        f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel B: 2SLS, instrument $=$ kmwittenberg $\\times$ Post}}}} \\\\"
    )
    iv_coef = (
        _label("cath_share_x_post")
        + " & "
        + " & ".join(_fmt_coef(c["iv_coef"], c["iv_p"]) for c in cols)
        + r" \\"
    )
    iv_se = (
        " & "
        + " & ".join(_fmt_se(c["iv_se"]) for c in cols)
        + r" \\"
    )

    tabular = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + head + "\n"
        + nums + "\n"
        "\\midrule\n"
        + panel_a_label + "\n"
        + ols_coef + "\n"
        + ols_se + "\n"
        "\\addlinespace\n"
        + panel_b_label + "\n"
        + iv_coef + "\n"
        + iv_se + "\n"
        "\\midrule\n"
        + "First-stage $F$ & "
        + " & ".join(f"{c['first_stage_f']:.1f}" for c in cols)
        + r" \\" + "\n"
        + "Partial $R^{2}$ (instrument) & "
        + " & ".join(f"{c['first_stage_partial_r2']:.3f}" for c in cols)
        + r" \\" + "\n"
        + "Wu--Hausman $p$ (endogeneity) & "
        + " & ".join(f"{c['wu_hausman_p']:.3f}" for c in cols)
        + r" \\" + "\n"
        + "County FE & "
        + " & ".join("Yes" for _ in cols)
        + r" \\" + "\n"
        + "Year FE & "
        + " & ".join("Yes" for _ in cols)
        + r" \\" + "\n"
        + "Observations & "
        + " & ".join(f"{c['n']:,}" for c in cols)
        + r" \\" + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption=(
            "Instrumental-variables estimates: distance to Wittenberg as an "
            "instrument for $\\mathrm{CathShare} \\times \\mathrm{Post}$"
        ),
        label="tab:iv_results",
        n_cols=n + 1,
        notes=(
            "Panel~A reproduces the OLS coefficient from Table~\\ref{tab:baseline_did} "
            "for comparison. Panel~B reports 2SLS estimates with the Becker--Woessmann "
            "instrument $\\mathrm{kmwittenberg}_i \\times \\mathrm{Post}_t$, where "
            "$\\mathrm{kmwittenberg}_i$ is the great-circle distance from each county "
            "to Wittenberg (the cradle of the Reformation). Sample shrinks because "
            "iPEHD coverage is $\\sim$90\\% of Galloway counties. "
            "First-stage $F$ tests joint significance of the excluded instrument; "
            "the Stock--Yogo (2005) critical value for ``10\\% maximal IV size'' is "
            "$F = 16.38$ with one instrument, so values above this threshold are "
            "considered strong. The Wu--Hausman test rejects the null that "
            "$\\mathrm{CathShare} \\times \\mathrm{Post}$ is exogenous in OLS, "
            "indicating that 2SLS is required for consistent estimates. Standard "
            "errors clustered at the county level. $^{*}\\,p<0.10$, $^{**}\\,p<0.05$, "
            "$^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def conley_robustness_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    cutoff_km: float = 200.0,
    out_path: Path | None = None,
) -> str:
    """Conley (1999) spatial HAC standard errors as a robustness column."""
    out_path = out_path or TABLES_DIR / "conley_robustness.tex"

    cols = []
    for o in outcomes:
        r = spatial_did_se(panel, outcome=o, cutoff_km=cutoff_km)
        cols.append({
            "outcome": o,
            "coef": r["coef"]["cath_share_x_post"],
            "cluster_se": r["cluster_se"]["cath_share_x_post"],
            "conley_se": r["conley_se"]["cath_share_x_post"],
            "n": r["n"],
        })

    n = len(cols)

    def _stars_from_se(coef: float, se: float) -> str:
        if se <= 0:
            return ""
        z = abs(coef / se)
        # Two-sided normal p-value via complementary error function
        p = math.erfc(z / math.sqrt(2.0))
        return _stars(p)

    coef_row_cluster = (
        _label("cath_share_x_post")
        + " & "
        + " & ".join(
            f"{c['coef']:.3f}{_stars_from_se(c['coef'], c['cluster_se'])}"
            for c in cols
        )
        + r" \\"
    )
    cluster_row = (
        " & "
        + " & ".join(f"({c['cluster_se']:.3f})" for c in cols)
        + r" \\"
    )
    conley_row = (
        " & "
        + " & ".join(f"[{c['conley_se']:.3f}]" for c in cols)
        + r" \\"
    )

    tabular = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + "Dependent variable: & "
        + " & ".join(_outcome_label(c["outcome"]) for c in cols)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
        + coef_row_cluster + "\n"
        + cluster_row + "\n"
        + conley_row + "\n"
        "\\midrule\n"
        + "County FE & "
        + " & ".join("Yes" for _ in cols)
        + r" \\" + "\n"
        + "Year FE & "
        + " & ".join("Yes" for _ in cols)
        + r" \\" + "\n"
        + "Observations & "
        + " & ".join(f"{c['n']:,}" for c in cols)
        + r" \\" + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption=(
            "Spatial robustness: Conley (1999) HAC standard errors with a "
            f"{int(cutoff_km)} km Bartlett cutoff"
        ),
        label="tab:conley_robustness",
        n_cols=n + 1,
        notes=(
            f"Same baseline DiD specification as Table~\\ref{{tab:baseline_did}}, "
            f"restricted to the {int(cols[0]['n']):,} county--year observations "
            "with available centroid coordinates (96.7\\% of the full panel). "
            "Standard errors clustered at the county level reported in "
            "parentheses; Conley (1999) spatial HAC standard errors with a "
            f"{int(cutoff_km)} km Bartlett kernel cutoff in brackets. The "
            "Conley correction allows arbitrary spatial correlation in the "
            "residuals up to the cutoff and decaying linearly to zero. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$ "
            "(stars based on the cluster-robust standard error)."
        ),
    )
    _write(out_path, out)
    return out


def headline_summary_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """One-stop headline table: TWFE / Year x Rb FE / 2SLS / long-diff side-by-side."""
    out_path = out_path or TABLES_DIR / "headline_summary.tex"

    twfe = [_did_column(panel, o, "twfe") for o in outcomes]
    strict = [_did_column(panel, o, "year_x_rb") for o in outcomes]
    trends = [_did_column(panel, o, "twfe_county_trends") for o in outcomes]
    iv = [run_iv_did(panel, outcome=o, instrument="kmwittenberg") for o in outcomes]
    longd = [run_long_difference(panel, outcome=o) for o in outcomes]
    pre = [pretrends_wald_test(panel, outcome=o) for o in outcomes]
    perm = [permutation_p_value(panel, outcome=o, n_permutations=1000, seed=42)
            for o in outcomes]

    qs = sharpened_q_values({o: c["p"] for o, c in zip(outcomes, twfe)})

    n = len(outcomes)

    def _row(label, vals_coef, vals_p):
        return (
            label
            + " & "
            + " & ".join(_fmt_coef(c, p) for c, p in zip(vals_coef, vals_p))
            + r" \\"
        )

    def _se_row(vals_se):
        return (
            " & "
            + " & ".join(_fmt_se(s) for s in vals_se)
            + r" \\"
        )

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel A: OLS with county and year FE (TWFE)}}}} \\\\\n"
        + _row(_label("cath_share_x_post"), [c["coef"] for c in twfe], [c["p"] for c in twfe]) + "\n"
        + _se_row([c["se"] for c in twfe]) + "\n"
        + "Anderson sharp.\\ $q$ & "
        + " & ".join(f"{qs[o]:.3f}" for o in outcomes)
        + r" \\" + "\n"
        + "\\addlinespace\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel B: OLS with county and (year $\\times$ Rb) FE}}}} \\\\\n"
        + _row(_label("cath_share_x_post"), [c["coef"] for c in strict], [c["p"] for c in strict]) + "\n"
        + _se_row([c["se"] for c in strict]) + "\n"
        + "\\addlinespace\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel C: OLS with county FE, year FE, and county-specific linear trends}}}} \\\\\n"
        + _row(_label("cath_share_x_post"), [c["coef"] for c in trends], [c["p"] for c in trends]) + "\n"
        + _se_row([c["se"] for c in trends]) + "\n"
        + "\\addlinespace\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel D: 2SLS, instrument $=$ kmwittenberg $\\times$ Post}}}} \\\\\n"
        + _row(_label("cath_share_x_post"), [c["iv_coef"] for c in iv], [c["iv_p"] for c in iv]) + "\n"
        + _se_row([c["iv_se"] for c in iv]) + "\n"
        + "First-stage $F$ & "
        + " & ".join(f"{c['first_stage_f']:.1f}" for c in iv)
        + r" \\" + "\n"
        + "\\addlinespace\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel E: Long-difference, 1862--71 vs 1880--89}}}} \\\\\n"
        + _row(r"CathShare", [c["coef"] for c in longd], [c["p"] for c in longd]) + "\n"
        + _se_row([c["se"] for c in longd]) + "\n"
        + "\\midrule\n"
        + "Permutation $p$-value (Panel A) & "
        + " & ".join(f"{p['p_value']:.3f}" for p in perm)
        + r" \\" + "\n"
        + "Pre-trends $\\chi^{2}$ $p$-value & "
        + " & ".join(f"{p['p_value']:.3f}" for p in pre)
        + r" \\" + "\n"
        + "Observations (A, B, C) & "
        + " & ".join(f"{c['n']:,}" for c in twfe)
        + r" \\" + "\n"
        + "Observations (D, IV) & "
        + " & ".join(f"{c['n']:,}" for c in iv)
        + r" \\" + "\n"
        + "Counties (E, long-diff) & "
        + " & ".join(f"{c['n']:,}" for c in longd)
        + r" \\" + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption="Headline estimates: the Kulturkampf and demographic outcomes",
        label="tab:headline_summary",
        n_cols=n + 1,
        notes=(
            "Each panel reports a different estimator of the Kulturkampf effect; "
            "all use $\\mathrm{CathShare} \\times \\mathrm{Post}$ as the treatment "
            "(Panel E omits the Post interaction since the panel has been "
            "long-differenced). Standard errors clustered at the county level "
            "in parentheses; long-difference uses HC1. Anderson (2008) sharpened "
            "$q$-values control the false-discovery rate across the four outcomes "
            "for the TWFE specification. The permutation $p$-value is the share "
            "of 1{,}000 cath\\_share random reassignments yielding a coefficient "
            "as extreme as the observed (Panel~A) coefficient. The pre-trends "
            "$\\chi^{2}$ tests the joint hypothesis that all event-study "
            "coefficients in the pre-1872 period equal zero (event study estimated "
            "separately, see Table~\\ref{tab:event_study}). "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def magnitudes_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """IV-implied vs observed differential change between high- and low-Catholic counties."""
    out_path = out_path or TABLES_DIR / "magnitudes.tex"
    df = magnitude_decomposition(panel, outcomes=outcomes)

    rows = []
    for _, r in df.iterrows():
        counterfactual = r["observed_gap"] - r["iv_implied"]
        rows.append(
            f"{_outcome_label(r['outcome'])} & "
            f"{r['delta_high']:+.3f} & "
            f"{r['delta_low']:+.3f} & "
            f"{r['observed_gap']:+.3f} & "
            f"{r['iv_implied']:+.3f} & "
            f"{counterfactual:+.3f} \\\\"
        )

    tabular = (
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        " & \\multicolumn{3}{c}{Observed change post $-$ pre} & "
        "\\multicolumn{2}{c}{Decomposition} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
        " & High-Cath & Low-Cath & Differential & "
        "IV-implied & Counterfactual \\\\\n"
        " & ($>$75\\%) & ($<$25\\%) & (1)$-$(2) & "
        "Kulturkampf & gap (3)$-$(4) \\\\\n"
        " & (1) & (2) & (3) & (4) & (5) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption=(
            "Magnitude decomposition: Kulturkampf-attributable change vs.\\ "
            "the observed differential between high- and low-Catholic counties"
        ),
        label="tab:magnitudes",
        n_cols=6,
        notes=(
            "Columns (1)--(2) report the change in the outcome between the "
            "1862--71 mean and the 1880--89 mean for counties above 75\\% "
            "Catholic and below 25\\% Catholic respectively. Column~(3) is the "
            "differential change. Column~(4) is the Kulturkampf-attributable "
            "differential implied by the 2SLS coefficient: "
            "$\\hat{\\beta}_{IV} \\times (\\bar{\\text{cath}}_{\\text{high}} - "
            "\\bar{\\text{cath}}_{\\text{low}})$. Column~(5) is the counterfactual "
            "differential absent the Kulturkampf, $(3) - (4)$. A negative "
            "Kulturkampf-attributable column with a positive observed differential "
            "(as for CBR) implies that secular pre-trends were widening the "
            "high--low gap and that the Kulturkampf partially offset, rather than "
            "drove, the observed pattern."
        ),
    )
    _write(out_path, out)
    return out


def pretrends_robustness_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """Sample-restriction sensitivity + Honest DiD breakdown M values."""
    out_path = out_path or TABLES_DIR / "pretrends_robustness.tex"
    n = len(outcomes)

    # Sample-restriction sensitivity
    start_years = (1862, 1865, 1867, 1869, 1871)
    sens = {o: run_start_year_sensitivity(panel, outcome=o, start_years=start_years)
            for o in outcomes}

    # Honest DiD breakdown M (average post effect)
    honest = {o: honest_did_bounds(panel, outcome=o, target="average") for o in outcomes}

    # Sample-sensitivity rows: one row per start year
    rows_sens = []
    for sy in start_years:
        cells = []
        for o in outcomes:
            row = sens[o][sens[o]["start_year"] == sy]
            if row.empty:
                cells.append("")
            else:
                r = row.iloc[0]
                cells.append(_fmt_coef(r["coef"], r["p"]))
        rows_sens.append(
            f"Start year $=$ {sy} & " + " & ".join(cells) + r" \\"
        )

    rows_sens_se = []
    for sy in start_years:
        cells = []
        for o in outcomes:
            row = sens[o][sens[o]["start_year"] == sy]
            cells.append(_fmt_se(row.iloc[0]["se"]) if not row.empty else "")
        rows_sens_se.append(" & " + " & ".join(cells) + r" \\")

    # Interleave coef and SE rows
    sens_block = "\n".join(
        line for pair in zip(rows_sens, rows_sens_se) for line in pair
    )

    honest_row = (
        "Honest DiD breakdown $M$ & "
        + " & ".join(
            f"{honest[o].breakdown_m:.2f}" if honest[o].breakdown_m != float("inf")
            else "$\\infty$"
            for o in outcomes
        )
        + r" \\"
    )

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel A: Sample-restriction sensitivity ($\\mathrm{{CathShare}} \\times \\mathrm{{Post}}$, TWFE)}}}} \\\\\n"
        + sens_block + "\n"
        + "\\midrule\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel B: Honest DiD bounds (Rambachan \\& Roth 2023)}}}} \\\\\n"
        + honest_row + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Pre-trends robustness: sample-restriction sensitivity and Honest DiD "
            "breakdown $M$"
        ),
        label="tab:pretrends_robustness",
        n_cols=n + 1,
        notes=(
            "Panel~A re-estimates the baseline TWFE specification on samples "
            "starting in increasingly later years; if the result is robust to "
            "non-zero pre-trends in the early panel, the coefficient should "
            "stabilise as the start year approaches 1872. Panel~B reports the "
            "breakdown $M$ from the simplified Rambachan \\& Roth (2023) "
            "smoothness restriction: the smallest $M$ at which the honest "
            "confidence interval for the average post-period treatment effect "
            "first contains zero. $M = 1$ means the post-period trend can change "
            "between adjacent years by as much as the worst pre-period change; "
            "small breakdown $M$ indicates a fragile result. Standard errors "
            "clustered at the county level in parentheses. $^{*}\\,p<0.10$, "
            "$^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def dcdh_diagnostic_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> str:
    """Negative-weights diagnostic for TWFE with continuous treatment."""
    out_path = out_path or TABLES_DIR / "dcdh_diagnostic.tex"
    diag = dcdh_diagnostic(panel).iloc[0]

    body = (
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Statistic & Value \\\\\n"
        "\\midrule\n"
        f"Total county-year observations & {int(diag['n_total']):,} \\\\\n"
        f"Observations with negative implicit weight & {int(diag['n_negative']):,} \\\\\n"
        f"Share with negative weight & {diag['share_negative']:.3f} \\\\\n"
        f"Sum of positive weights & {diag['sum_pos_weights']:+.4f} \\\\\n"
        f"Sum of negative weights & {diag['sum_neg_weights']:+.4f} \\\\\n"
        f"Ratio $|\\Sigma w_-| / \\Sigma w_+$ & "
        f"{diag['ratio_neg_pos']:.4f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "de Chaisemartin \\& D'Haultfoeuille negative-weights diagnostic "
            "for the TWFE coefficient on $\\mathrm{CathShare} \\times \\mathrm{Post}$"
        ),
        label="tab:dcdh_diagnostic",
        n_cols=2,
        notes=(
            "The TWFE coefficient on the continuous treatment intensity is a "
            "weighted average of unit-level effects with weights "
            "$w_{it} = D_{it}\\,\\tilde D_{it} / \\sum_{js} \\tilde D_{js}^{2}$, "
            "where $\\tilde D$ is the two-way within-transformed treatment. "
            "Negative weights occur for observations where $\\tilde D_{it} < 0$. "
            "The ratio of the absolute sum of negative weights to the sum of "
            "positive weights is the headline diagnostic: values below 0.05 "
            "indicate that heterogeneous-effects weighting is unlikely to bias "
            "the TWFE estimate; values above 0.20 strongly recommend the dCDH "
            "or Borusyak--Jaravel--Spiess estimators instead."
        ),
    )
    _write(out_path, out)
    return out


def variance_decomposition_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """R^2 of nested specifications: county FE / year FE / both / + treatment."""
    out_path = out_path or TABLES_DIR / "variance_decomposition.tex"
    df = variance_decomposition(panel, outcomes=outcomes)

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{_outcome_label(r['outcome'])} & "
            f"{r['r2_county']:.3f} & "
            f"{r['r2_year']:.3f} & "
            f"{r['r2_county_year']:.3f} & "
            f"{r['r2_full']:.3f} & "
            f"{r['marginal_treatment']:.4f} \\\\"
        )

    body = (
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        " & \\multicolumn{4}{c}{$R^{2}$ of nested specifications} & "
        "\\multicolumn{1}{c}{Marginal} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\n"
        "Dependent variable & County FE & Year FE & Both FE & "
        "$+$ Treatment & contribution \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Variance decomposition: $R^{2}$ of nested specifications and "
            "marginal contribution of the treatment"
        ),
        label="tab:variance_decomposition",
        n_cols=6,
        notes=(
            "Each cell reports the $R^{2}$ of the named specification. "
            "``Marginal contribution'' is the gain in $R^{2}$ from adding "
            "$\\mathrm{CathShare} \\times \\mathrm{Post}$ to the two-way FE "
            "specification. County FE absorb the bulk of the variation in "
            "fertility outcomes; the marginal contribution of the treatment is "
            "small in absolute terms but informative about identification: "
            "a near-zero marginal $R^{2}$ alongside a precisely estimated "
            "coefficient (as in the marriage-rate column) implies a real but "
            "narrow within-county effect."
        ),
    )
    _write(out_path, out)
    return out


def cohort_translation_table(
    panel: pd.DataFrame,
    iv_coef: float,
    out_path: Path | None = None,
) -> str:
    """Translate the IV CBR coefficient into TFR/CCF and missing-births terms."""
    out_path = out_path or TABLES_DIR / "cohort_translation.tex"
    df = cohort_translation(panel, iv_coef=iv_coef).iloc[0]

    rows = [
        ("Mean cath\\_share, high-Cath ($>$75\\%)", f"{df['high_cath_mean']:.1f}\\%"),
        ("Mean cath\\_share, low-Cath ($<$25\\%)", f"{df['low_cath_mean']:.1f}\\%"),
        ("Catholic-share contrast (high $-$ low)", f"{df['delta_cath']:.1f} pp"),
        ("\\addlinespace", None),
        ("IV coefficient on $\\mathrm{CathShare} \\times \\mathrm{Post}$ (CBR)",
         f"{df['iv_coef_cbr']:+.4f}"),
        ("Annual CBR effect, high-vs-low (per 1{,}000)",
         f"{df['annual_cbr_diff']:+.2f}"),
        ("Number of post-treatment years",
         f"{int(df['n_post_years'])}"),
        ("Cumulative birth deficit (per 1{,}000 over post period)",
         f"{df['cumulative_per_1000']:+.1f}"),
        ("\\addlinespace", None),
        ("Implied annual GFR effect (per 1{,}000 women 15--45)",
         f"{df['annual_gfr_diff']:+.1f}"),
        ("Implied TFR effect (period)",
         f"{df['tfr_diff']:+.3f}"),
        ("Implied CCF effect (cohort, partial overlap)",
         f"{df['ccf_diff']:+.3f}"),
    ]

    rendered_rows = []
    for label, value in rows:
        if value is None:
            rendered_rows.append("\\addlinespace")
        else:
            rendered_rows.append(f"{label} & {value} \\\\")

    body = (
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Quantity & Value \\\\\n"
        "\\midrule\n"
        + "\n".join(rendered_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Demographic translation of the IV CBR coefficient into total fertility "
            "rate (TFR) and completed cohort fertility (CCF) terms"
        ),
        label="tab:cohort_translation",
        n_cols=2,
        notes=(
            "Translation uses constant-share approximations: women aged 15--45 "
            "comprise $f_w = 0.22$ of total population, and the implied total "
            "fertility rate is $\\mathrm{GFR} \\times 30 / 1{,}000$ for a 30-year "
            "reproductive lifespan with a flat age-specific fertility schedule. "
            f"The {int(df['overlap_frac'] * 100)}\\% overlap factor scales the "
            "TFR effect to a cohort whose reproductive career intersects the "
            "Kulturkampf and rollback windows. Both translations are first-order "
            "approximations; precise cohort estimates require single-year "
            "age-specific fertility schedules unavailable in the Galloway panel."
        ),
    )
    _write(out_path, out)
    return out


def falsifications_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """Three falsification checks: Jewish-share placebo + fake-treatment + triple-diff."""
    out_path = out_path or TABLES_DIR / "falsifications.tex"
    n = len(outcomes)

    jewish = run_jewish_placebo(panel, outcomes=tuple(outcomes))
    fake = run_fake_treatment_placebo(panel, outcomes=tuple(outcomes))
    triples = {o: run_triple_difference_polish(panel, outcome=o) for o in outcomes}

    def _row(label, coefs, pvals):
        return (
            label
            + " & "
            + " & ".join(_fmt_coef(c, p) for c, p in zip(coefs, pvals))
            + r" \\"
        )

    def _se_row(ses):
        return " & " + " & ".join(_fmt_se(s) for s in ses) + r" \\"

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel A: Jewish-share placebo (replace cath\\_share with f\\_jew)}}}} \\\\\n"
        + _row(r"$f_{\mathrm{jew}}$ $\times$ Post",
               [r["coef"] for _, r in jewish.iterrows()],
               [r["p"] for _, r in jewish.iterrows()]) + "\n"
        + _se_row([r["se"] for _, r in jewish.iterrows()]) + "\n"
        + "\\addlinespace\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel B: Pre-1872 fake-treatment placebo (Post $=$ 1865, sample 1862--71)}}}} \\\\\n"
        + _row(r"CathShare $\times$ FakePost$_{1865}$",
               [r["coef"] for _, r in fake.iterrows()],
               [r["p"] for _, r in fake.iterrows()]) + "\n"
        + _se_row([r["se"] for _, r in fake.iterrows()]) + "\n"
        + "\\addlinespace\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel C: Triple-difference, formal Polish vs non-Polish heterogeneity}}}} \\\\\n"
        + _row(_label("cath_share_x_post"),
               [triples[o]["main_effect"] for o in outcomes],
               [triples[o]["main_p"] for o in outcomes]) + "\n"
        + _se_row([triples[o]["main_se"] for o in outcomes]) + "\n"
        + "\\addlinespace\n"
        + _row(r"CathShare $\times$ Post $\times$ Polish",
               [triples[o]["triple_coef"] for o in outcomes],
               [triples[o]["triple_p"] for o in outcomes]) + "\n"
        + _se_row([triples[o]["triple_se"] for o in outcomes]) + "\n"
        + "\\midrule\n"
        + "Observations (A, C) & "
        + " & ".join(f"{int(triples[o]['n']):,}" for o in outcomes)
        + r" \\" + "\n"
        + "Observations (B) & "
        + " & ".join(f"{int(r['n']):,}" for _, r in fake.iterrows())
        + r" \\" + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption="Falsifications: Jewish-share placebo, pre-1872 fake treatment, and Polish triple-difference",
        label="tab:falsifications",
        n_cols=n + 1,
        notes=(
            "Panel~A replaces $\\mathrm{CathShare}$ with the Jewish population "
            "share $f_{\\mathrm{jew}}$ as the treatment-intensity variable; the "
            "Kulturkampf was a Catholic--Protestant conflict, so a null "
            "coefficient is expected. Panel~B restricts the sample to pre-1872 "
            "and assigns a placebo Post indicator at 1865; a non-zero coefficient "
            "indicates a pre-existing trend that the baseline DiD would absorb "
            "into the Kulturkampf effect. Panel~C estimates the triple "
            "difference $\\mathrm{CathShare} \\times \\mathrm{Post} \\times "
            "\\mathrm{Polish}$ in a single regression: the triple coefficient is "
            "the *additional* effect for counties in Posen and Bromberg "
            "Regierungsbezirke. Standard errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def heterogeneity_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "marriage_rate", "I_g"),
    moderators: tuple[str, ...] = ("school1517", "f_urban"),
    out_path: Path | None = None,
) -> str:
    """Treatment effect interactions with iPEHD moderators (literacy, urban share).

    Default outcomes include CBR (overall fertility), marriage rate
    (nuptiality), and Coale's $I_g$ (marital fertility, Hutterite-
    normalised; the Galloway-tradition headline measure -- Galloway,
    Hammel & Lee 1994 use its unnormalised form, the GMFR). The trio
    spans the full Coale--Watkins decomposition: $I_f \\approx I_g \\cdot
    I_m + I_h(1-I_m)$, so the reader can see whether heterogeneity
    operates through *marital* fertility or through *nuptiality*.
    """
    out_path = out_path or TABLES_DIR / "heterogeneity.tex"

    moderator_labels = {
        "school1517": "School enrolment 15--17",
        "f_urban": "Urban population share",
        "f_litrate": "Literacy rate",
    }

    n = len(outcomes)
    blocks = []
    for moderator in moderators:
        results = {o: run_heterogeneity_did(panel, moderator=moderator, outcome=o)
                   for o in outcomes}
        block = (
            f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Moderator: "
            f"{moderator_labels.get(moderator, moderator)}}}}} \\\\\n"
            + _label("cath_share_x_post") + " (at moderator mean) & "
            + " & ".join(_fmt_coef(results[o]["main_coef"], results[o]["main_p"])
                         for o in outcomes)
            + r" \\" + "\n"
            + " & "
            + " & ".join(_fmt_se(results[o]["main_se"]) for o in outcomes)
            + r" \\" + "\n"
            + r"CathShare $\times$ Post $\times$ Moderator & "
            + " & ".join(_fmt_coef(results[o]["triple_coef"], results[o]["triple_p"])
                         for o in outcomes)
            + r" \\" + "\n"
            + " & "
            + " & ".join(_fmt_se(results[o]["triple_se"]) for o in outcomes)
            + r" \\" + "\n"
            + "\\addlinespace\n"
        )
        blocks.append(block)

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(blocks)
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Treatment-effect heterogeneity by iPEHD moderators "
            "(school enrolment, urban share)"
        ),
        label="tab:heterogeneity",
        n_cols=n + 1,
        notes=(
            "Each block reports the interacted DiD with a different time-invariant "
            "moderator (centred at its mean). The first row in each block is the "
            "Kulturkampf effect for a county at the mean of the moderator; the "
            "second row is the differential effect per unit of the moderator. "
            "Moderator level effects and CathShare $\\times$ Moderator are "
            "absorbed by entity fixed effects. Standard errors clustered at the "
            "county level. $^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def iv_overid_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "marriage_rate"),
    out_path: Path | None = None,
) -> str:
    """2SLS with Wittenberg + Bishop instruments and Wooldridge over-id test."""
    out_path = out_path or TABLES_DIR / "iv_overid.tex"
    centroids = load_centroids()
    panel_b = panel.merge(centroids[["Code", "km_bishop"]], on="Code", how="left")

    cols = [run_iv_did_multi(panel_b, outcome=o) for o in outcomes]
    n = len(cols)

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
        + _label("cath_share_x_post")
        + " & "
        + " & ".join(_fmt_coef(c["iv_coef"], c["iv_p"]) for c in cols)
        + r" \\" + "\n"
        + " & "
        + " & ".join(_fmt_se(c["iv_se"]) for c in cols)
        + r" \\" + "\n"
        + "\\midrule\n"
        + "First-stage $F$ & "
        + " & ".join(f"{c['first_stage_f']:.1f}" for c in cols)
        + r" \\" + "\n"
        + "Partial $R^{2}$ & "
        + " & ".join(f"{c['first_stage_partial_r2']:.3f}" for c in cols)
        + r" \\" + "\n"
        + "Wooldridge over-id $p$ & "
        + " & ".join(f"{c['j_p']:.3f}" for c in cols)
        + r" \\" + "\n"
        + "Wu--Hausman $p$ (endogeneity) & "
        + " & ".join(f"{c['wu_hausman_p']:.3f}" for c in cols)
        + r" \\" + "\n"
        + "Anderson--Rubin $p$ (weak-IV robust) & "
        + " & ".join(f"{c['ar_p']:.3f}" for c in cols)
        + r" \\" + "\n"
        + "Observations & "
        + " & ".join(f"{c['n']:,}" for c in cols)
        + r" \\" + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Multi-instrument 2SLS: distance to Wittenberg and distance to "
            "nearest Catholic bishop's seat as instruments for "
            "$\\mathrm{CathShare} \\times \\mathrm{Post}$"
        ),
        label="tab:iv_overid",
        n_cols=n + 1,
        notes=(
            "Both instruments enter the first stage as $\\mathrm{km}_z \\times "
            "\\mathrm{Post}$ for $z \\in \\{\\mathrm{Wittenberg}, "
            "\\mathrm{Bishop}\\}$. The Wooldridge $p$-value tests the "
            "over-identifying restriction (failure to reject is consistent with "
            "instrument exogeneity). The Stock--Yogo (2005) critical value for "
            "``10\\% maximal IV size'' with two instruments is $F = 19.93$, so "
            "all reported first-stage $F$ values are well above the strong-instrument "
            "threshold. The Wu--Hausman test rejects exogeneity (so 2SLS is "
            "required); the Anderson--Rubin $p$-value tests "
            "$H_0\\!: \\beta = 0$ in a way that is robust to weak instruments. "
            "Where AR and the standard 2SLS $p$-value disagree, AR is the more "
            "conservative (and weak-instrument-defensible) inference. Standard "
            "errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def wild_bootstrap_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "marriage_rate", "I_g"),
    out_path: Path | None = None,
) -> str:
    """Wild cluster bootstrap p-values across full panel and key sub-samples.

    The $I_g$ column (Coale's marital fertility index, Hutterite-
    normalised; the Galloway, Hammel & Lee 1994 tradition outcome) tests
    whether the small-cluster sub-region results survive when fertility
    is measured net of nuptiality. Because $I_g$ is dimensionless, its
    coefficients are not directly comparable to the per-1{,}000 CBR /
    marriage-rate columns; what *is* comparable across columns is the
    sign and the wild-bootstrap $p$-value.
    """
    out_path = out_path or TABLES_DIR / "wild_bootstrap.tex"

    samples = {
        "Full panel": None,
        "Polish provinces": lambda d: d["Rb"].isin(["POS", "BRO"]),
        "German Catholic prov.": lambda d: d["Rb"].isin(
            ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]
        ),
        "Protestant prov.\\ (rest)": lambda d: ~d["Rb"].isin(
            ["POS", "BRO", "KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]
        ),
    }

    n = len(outcomes)
    rows = []
    for label, sf in samples.items():
        cells = []
        n_clusters = None
        for o in outcomes:
            r = wild_cluster_bootstrap(panel, outcome=o, sample_filter=sf,
                                       n_boot=999, seed=42)
            n_clusters = r["n_clusters"]
            cells.append((r["beta_obs"], r["p_value"]))
        rows.append((label, n_clusters, cells))

    coef_lines = []
    for label, n_clu, cells in rows:
        coef_lines.append(
            f"{label} ($G = {n_clu}$) & "
            + " & ".join(_fmt_coef(b, p) for b, p in cells)
            + r" \\"
        )
        coef_lines.append(
            "\\quad wild $p$-value & "
            + " & ".join(f"[{p:.3f}]" for _, p in cells)
            + r" \\"
            + "\n\\addlinespace"
        )

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + " & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
        + "\n & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(coef_lines)
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Wild cluster bootstrap (Cameron, Gelbach \\& Miller 2008) "
            "$p$-values across sub-samples"
        ),
        label="tab:wild_bootstrap",
        n_cols=n + 1,
        notes=(
            "Each row reports the baseline DiD coefficient on $\\mathrm{CathShare} "
            "\\times \\mathrm{Post}$ for a sub-sample, with the wild cluster "
            "bootstrap $p$-value (999 Rademacher draws under the null) in "
            "brackets. Wild bootstrap delivers reliable inference even when the "
            "number of clusters $G$ is small (e.g.\\ the 24 Polish-province "
            "counties) where conventional cluster-robust standard errors are "
            "unreliable. The GFR column uses the General Fertility Rate "
            "(births per 1{,}000 women aged 15--49 in 1871) and addresses the "
            "standard demographic critique that CBR is mechanically affected by "
            "age structure. Because GFR has higher residual variance, the "
            "wild-bootstrap and asymptotic $p$-values diverge in the "
            "small-cluster sub-regions; the wild-bootstrap value is the correct "
            "small-sample inference. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$ "
            "stars are based on the wild bootstrap $p$-value."
        ),
    )
    _write(out_path, out)
    return out


def coale_decomposition_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> str:
    """
    Princeton-EFP-style fertility decomposition: I_f, I_g, I_h, marriage rate.

    Two panels: (A) group means by Catholic share x period, and (B) DiD on
    each component. The decomposition isolates whether the Kulturkampf
    operated through marital fertility (I_g) or nuptiality (marriage rate).
    """
    out_path = out_path or TABLES_DIR / "coale_decomposition.tex"
    panel_with = compute_coale_indices(panel)

    means = coale_aggregate(panel_with).set_index(["group", "period"])
    did = coale_did(panel_with).set_index("index")

    index_labels = {
        "I_f": "$I_f$ (overall fertility)",
        "I_g": "$I_g$ (marital fertility)",
        "I_h": "$I_h$ (illegitimate fertility)",
        "marriage_rate": "Marriage rate (per 1{,}000)",
    }

    # Panel A: pre/post means by group
    panel_a_rows = []
    for idx_name, idx_label in index_labels.items():
        if idx_name not in means.columns:
            continue
        cells = []
        for grp in ("Low Cath", "High Cath"):
            for per in ("Pre", "Post"):
                try:
                    val = means.loc[(grp, per), idx_name]
                    cells.append(f"{val:.3f}")
                except KeyError:
                    cells.append("")
        panel_a_rows.append(f"{idx_label} & " + " & ".join(cells) + r" \\")

    # Panel B: DiD coefficients (scientific-style precision since the
    # Hutterite-benchmarked indices vary by 10^{-4} per pp of cath_share).
    def _fmt_small(x: float, p: float, digits: int = 5) -> str:
        if pd.isna(x):
            return ""
        return f"{x:+.{digits}f}{_stars(p)}"

    panel_b_rows = []
    for idx_name, idx_label in index_labels.items():
        if idx_name not in did.index:
            continue
        r = did.loc[idx_name]
        digits = 3 if idx_name == "marriage_rate" else 5
        coef_str = _fmt_small(r["coef"], r["p"], digits=digits)
        se_str = (f"({r['se']:.{digits}f})" if not pd.isna(r["se"]) else "")
        panel_b_rows.append(
            f"{idx_label} & {coef_str} & {se_str} & {int(r['n']):,} \\\\"
        )

    body = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Low Catholic ($\\le$50\\%)} & "
        "\\multicolumn{2}{c}{High Catholic ($>$50\\%)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        " & Pre & Post & Pre & Post \\\\\n"
        "\\midrule\n"
        + f"\\multicolumn{{5}}{{l}}{{\\textit{{Panel A: Group means}}}} \\\\\n"
        + "\n".join(panel_a_rows)
        + "\n\\midrule\n"
        + "\\multicolumn{5}{l}{\\textit{Panel B: DiD coefficient on "
          "$\\mathrm{CathShare} \\times \\mathrm{Post}$ "
          "(coefficient, SE, observations)}} \\\\\n"
        + "Outcome & Coefficient & SE & $N$ & \\\\\n"
        + "\\cmidrule(lr){1-4}\n"
    )
    for row in panel_b_rows:
        body += row + "\n"
    body = body.rstrip("\n") + "\n\\bottomrule\n\\end{tabular}\n"

    out = _wrap_table(
        body,
        caption=(
            "Princeton fertility decomposition (Coale and Watkins 1986): "
            "marital fertility $I_g$ vs.\\ nuptiality (marriage rate)"
        ),
        label="tab:coale_decomposition",
        n_cols=5,
        notes=(
            "$I_f$, $I_g$, and $I_h$ are Hutterite-benchmarked Princeton EFP "
            "indices for overall, marital, and non-marital fertility. They "
            "are computed using the Coale-Demeny ``West'' female age "
            "distribution and an assumed nuptiality schedule (calibrated to a "
            "weighted-mean marriage prevalence of 64\\% among women 15--49, "
            "consistent with the eastern side of the Hajnal line). Because "
            "Galloway lacks age structure of women and married women, the "
            "absolute index levels depend on the calibration and should be "
            "read alongside the cross-county and pre/post differences. "
            "The Princeton I_m index is omitted because it is constant under "
            "the calibration; the observed marriage rate (Martot / Poptot per "
            "1{,}000) serves as the nuptiality marker. Panel~A reports group "
            "means; Panel~B reports the coefficient on $\\mathrm{CathShare} "
            "\\times \\mathrm{Post}$ from a county- and year-fixed-effects "
            "regression with $\\ln(\\mathrm{Pop})$ as a control. The "
            "demographic interpretation: $I_g$ shows no DiD effect, but the "
            "marriage rate shows a strong negative effect, implying that the "
            "Kulturkampf operated through marriage formation (nuptiality), "
            "not within-marriage childbearing. Standard errors clustered at "
            "the county level. $^{*}\\,p<0.10$, $^{**}\\,p<0.05$, "
            "$^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def emigration_robustness_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> str:
    """
    Address the post-1885 Polish-province emigration confound. Reports the
    headline DiD coefficient on cath_share x post under four specifications,
    separately for the full panel and the Polish sub-sample, plus the
    population-free intensive-margin outcomes (total marriages count, births
    per marriage).
    """
    out_path = out_path or TABLES_DIR / "emigration_robustness.tex"

    full = run_emigration_robustness(panel, outcomes=("cbr", "marriage_rate"))
    polish = run_emigration_robustness(
        panel[panel["Rb"].isin(["POS", "BRO"])],
        outcomes=("cbr", "marriage_rate"),
    )
    counts = run_count_marriage_did(panel)

    def _block(rows: pd.DataFrame, header_label: str) -> str:
        out = (
            f"\\multicolumn{{4}}{{l}}{{\\textit{{{header_label}}}}} \\\\\n"
        )
        # Group by spec; within each spec, two outcomes (cbr, marriage_rate)
        for spec in rows["spec"].drop_duplicates():
            sub = rows[rows["spec"] == spec]
            cbr_row = sub[sub["outcome"] == "cbr"].iloc[0] if len(sub[sub["outcome"] == "cbr"]) else None
            mar_row = sub[sub["outcome"] == "marriage_rate"].iloc[0] if len(sub[sub["outcome"] == "marriage_rate"]) else None
            cbr_str = (_fmt_coef(cbr_row["coef"], cbr_row["p"], digits=4) if cbr_row is not None else "")
            cbr_se = (_fmt_se(cbr_row["se"], digits=4) if cbr_row is not None else "")
            mar_str = (_fmt_coef(mar_row["coef"], mar_row["p"], digits=4) if mar_row is not None else "")
            mar_se = (_fmt_se(mar_row["se"], digits=4) if mar_row is not None else "")
            n_str = f"{int(cbr_row['n']):,}" if cbr_row is not None else f"{int(mar_row['n']):,}"
            out += f"{_latex_escape(spec)} & {cbr_str} & {mar_str} & {n_str} \\\\\n"
            out += f" & {cbr_se} & {mar_se} & \\\\\n"
        return out

    # Panel C: each row is one outcome with its own coefficient.
    # Format: outcome label | coefficient | SE | N -- same shape as Panels A/B.
    counts_block = ""
    for _, r in counts.iterrows():
        counts_block += (
            f"{_latex_escape(r['outcome'])} & "
            f"{_fmt_coef(r['coef'], r['p'], digits=4)} & "
            f"{_fmt_se(r['se'], digits=4)} & "
            f"{int(r['n']):,} \\\\\n"
        )
    panel_c_header = (
        "\\multicolumn{4}{l}{\\textit{Panel C: "
        "Population-free outcomes (one outcome per row)}} \\\\\n"
        " & Coefficient & SE & $N$ \\\\\n"
        "\\cmidrule(lr){1-4}\n"
    )

    body = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        " & CBR & Marriage rate & $N$ \\\\\n"
        " & (1) & (2) & \\\\\n"
        "\\midrule\n"
        + _block(full, "Panel A: Full panel")
        + "\\addlinespace\n"
        + _block(polish, "Panel B: Polish provinces (POS, BRO)")
        + "\\midrule\n"
        + panel_c_header
        + counts_block
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Emigration robustness: addressing the post-1885 "
            "$\\mathit{Polenausweisungen}$ confound"
        ),
        label="tab:emigration_robustness",
        n_cols=4,
        notes=(
            "All entries are the DiD coefficient (and clustered SE in "
            "parentheses) on $\\mathrm{CathShare} \\times \\mathrm{Post}$ from "
            "a county- and year-fixed-effects regression. Panel~A reports the "
            "headline specification on the full panel under four control sets: "
            "(1) baseline with $\\ln(\\mathrm{Pop})$ only; (2) adds the "
            "annual population growth rate; (3) adds an implied net migration "
            "rate (population change minus natural increase, per 1{,}000 "
            "population); (4) restricts the sample to $t < 1885$, before the "
            "Bismarck-era $\\mathit{Polenausweisungen}$ and the 1886 "
            "Settlement Commission. Panel~B repeats the four specifications "
            "on the Polish sub-sample (Posen and Bromberg). Panel~C reports "
            "outcomes that do not depend on the population denominator and "
            "therefore cannot be mechanical artefacts of out-migration. "
            "Standard errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def pretreatment_trends_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> str:
    """
    Pretreatment-characteristic time-trend robustness (Bai 2009, Hsiao 2014).

    Five-row table reporting the headline DiD coefficient on cath_share x post
    when progressively more iPEHD-1871 baseline characteristics are interacted
    with year fixed effects (linear-trend form for parsimony).
    """
    out_path = out_path or TABLES_DIR / "pretreatment_trends.tex"

    df = run_pretreatment_trends_robustness(
        panel, outcomes=("cbr", "marriage_rate"), form="linear",
    )

    rows: list[str] = []
    for spec in df["spec"].drop_duplicates():
        sub = df[df["spec"] == spec]
        cbr_sub = sub[sub["outcome"] == "cbr"]
        mar_sub = sub[sub["outcome"] == "marriage_rate"]
        cbr = cbr_sub.iloc[0] if len(cbr_sub) else None
        mar = mar_sub.iloc[0] if len(mar_sub) else None
        cbr_coef = _fmt_coef(cbr["coef"], cbr["p"], digits=4) if cbr is not None else ""
        cbr_se = _fmt_se(cbr["se"], digits=4) if cbr is not None else ""
        mar_coef = _fmt_coef(mar["coef"], mar["p"], digits=4) if mar is not None else ""
        mar_se = _fmt_se(mar["se"], digits=4) if mar is not None else ""
        n_val = int(cbr["n"]) if cbr is not None else int(mar["n"])
        rows.append(
            f"{_latex_escape(spec)} & {cbr_coef} & {mar_coef} & {n_val:,} \\\\"
        )
        rows.append(f" & {cbr_se} & {mar_se} & \\\\")

    body = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        " & CBR & Marriage rate & $N$ \\\\\n"
        " & (1) & (2) & \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Pretreatment-characteristic time-trend robustness "
            "(Bai 2009; Hsiao 2014)"
        ),
        label="tab:pretreatment_trends",
        n_cols=4,
        notes=(
            "Each row is a separate two-way fixed-effects regression of the "
            "outcome on $\\mathrm{CathShare} \\times \\mathrm{Post}$, "
            "$\\ln(\\mathrm{Pop})$, and the listed baseline iPEHD-1871 "
            "characteristics interacted with a centred linear time trend. "
            "The interactions allow counties with different pre-treatment "
            "literacy ($\\mathrm{school1517}$), urbanisation "
            "($f_{\\mathrm{urban}}$), Prussian-citizenship share "
            "($f_{\\mathrm{pruss}}$), and Jewish-population share "
            "($f_{\\mathrm{jew}}$) to follow different trajectories. The "
            "Kulturkampf coefficient is then identified from deviations from "
            "those trajectories at 1873. The marriage-rate coefficient is "
            "essentially unchanged when literacy, urbanisation, and Prussian "
            "citizenship are added (rows 2--4) but attenuates by ~50\\% when "
            "Jewish share is added (row 5), which absorbs differential "
            "dynamics in eastern provinces with high Jewish settlement and "
            "thus partially captures the same Polish-province channel "
            "documented in Tables~\\ref{tab:falsifications} and "
            "\\ref{tab:emigration_robustness}. Sample shrinks slightly "
            "because the iPEHD merge covers $\\sim$90\\% of Galloway "
            "counties. Standard errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def subsample_decomposition_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> str:
    """
    Decompose the headline DiD coefficient by sample composition. Reveals
    how much of the full-panel marriage-rate effect comes from the 1866
    annexed territories, the Polish provinces, and the core German
    Catholic--Protestant comparison.
    """
    out_path = out_path or TABLES_DIR / "subsample_decomposition.tex"
    df = run_subsample_decomposition(panel, outcomes=("cbr", "marriage_rate"))

    rows: list[str] = []
    for name in df["sample"].drop_duplicates():
        sub = df[df["sample"] == name]
        cbr_sub = sub[sub["outcome"] == "cbr"]
        mar_sub = sub[sub["outcome"] == "marriage_rate"]
        cbr = cbr_sub.iloc[0] if len(cbr_sub) else None
        mar = mar_sub.iloc[0] if len(mar_sub) else None
        cbr_coef = _fmt_coef(cbr["coef"], cbr["p"], digits=4) if cbr is not None else ""
        cbr_se = _fmt_se(cbr["se"], digits=4) if cbr is not None else ""
        mar_coef = _fmt_coef(mar["coef"], mar["p"], digits=4) if mar is not None else ""
        mar_se = _fmt_se(mar["se"], digits=4) if mar is not None else ""
        ref = cbr if cbr is not None else mar
        n_counties = int(ref["n_counties"]) if ref is not None else 0
        n_obs = int(ref["n"]) if ref is not None else 0
        cbr_chi = float(cbr["pretrends_chi2"]) if cbr is not None else float("nan")
        mar_chi = float(mar["pretrends_chi2"]) if mar is not None else float("nan")
        rows.append(
            f"{_latex_escape(name)} & {cbr_coef} & {mar_coef} & "
            f"{n_counties} & {n_obs:,} \\\\"
        )
        rows.append(f" & {cbr_se} & {mar_se} & & \\\\")
        rows.append(
            f"\\quad pre-trends $\\chi^{{2}}(10)$ & "
            f"{cbr_chi:.1f} & {mar_chi:.1f} & & \\\\"
            "\n\\addlinespace"
        )

    body = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        " & CBR & Marriage rate & Counties & $N$ \\\\\n"
        " & (1) & (2) & & \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Sample-composition decomposition: where does the headline "
            "marriage-rate effect come from?"
        ),
        label="tab:subsample_decomposition",
        n_cols=5,
        notes=(
            "Each block reports the headline DiD coefficient on "
            "$\\mathrm{CathShare} \\times \\mathrm{Post}$ and the joint Wald "
            "$\\chi^{2}$ for pre-1872 event-study coefficients on a different "
            "sample cut. ``Core Prussia'' restricts to the 304 counties "
            "present in 1862, excluding the ~85 counties annexed in 1866 "
            "(Schleswig-Holstein, Hanover, Hesse-Kassel, Nassau, Frankfurt) "
            "which only enter the panel in 1867 and so cannot have observed "
            "1862--1866 pre-trend coefficients. ``No Polish provinces'' "
            "excludes Posen and Bromberg ($\\sim$24 counties), where Catholic "
            "share aligns with Polish ethnicity and the 1885+ "
            "$\\mathit{Polenausweisungen}$ generated mechanical out-migration. "
            "The marriage-rate coefficient attenuates from $-0.0036^{***}$ on "
            "the full panel to $-0.0013^{*}$ on the cleanest cut (core "
            "Prussia + no Polish), implying that approximately 22\\% of the "
            "magnitude operates through the annexed territories and 36\\% "
            "through the Polish channel. The pre-trends $\\chi^{2}$ "
            "*increases* under core Prussia, indicating that the annexations "
            "were partially masking a stronger pre-trend in the original "
            "Prussian core. Standard errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def event_study_table(
    panel: pd.DataFrame,
    *,
    out_path: Path | None = None,
    use_rollback: bool = False,
) -> str:
    """Year-by-year event-study coefficients (companion to the figure)."""
    if use_rollback:
        out_path = out_path or TABLES_DIR / "event_study_rollback.tex"
        coefs = rollback_event_study(panel.copy(), outcome="cbr", savepath=None)["coefs"]
        caption = "Event study with Kulturkampf enforcement and rollback periods"
        label = "tab:event_study_rollback"
    else:
        out_path = out_path or TABLES_DIR / "event_study.tex"
        coefs = run_event_study(panel, outcome="cbr")["coefs"]
        caption = "Event-study coefficients on $\\mathrm{CathShare} \\times \\mathbb{1}[t]$"
        label = "tab:event_study"

    rows = []
    for _, r in coefs.iterrows():
        if r["Year"] == coefs.loc[coefs["beta"] == 0, "Year"].iloc[0] and r["beta"] == 0:
            rows.append(f"{int(r['Year'])} & --- & --- & --- \\\\")
        else:
            p = 2 * (1 - _normal_cdf(abs(r["beta"] / r["se"]))) if r["se"] > 0 else np.nan
            rows.append(
                f"{int(r['Year'])} & {_fmt_coef(r['beta'], p)} & {_fmt_se(r['se'])} & "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] \\\\"
            )

    tabular = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Year & Coefficient & SE & 95\\% CI \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption=caption,
        label=label,
        n_cols=4,
        notes=(
            "Each row reports the coefficient on $\\mathrm{CathShare}_i \\times "
            "\\mathbb{1}[\\text{Year}=t]$ from a single two-way fixed-effects "
            "regression with $\\ln(\\mathrm{Pop})$ as a control. Year 1872 "
            "(omitted) is the reference. Standard errors clustered at the county "
            "level. $^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def _normal_cdf(z: float) -> float:
    """Standard-normal CDF; avoids a scipy dependency for one call."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all(panel: pd.DataFrame, out_dir: Path = TABLES_DIR) -> Iterable[Path]:
    """Generate every table in the suite. Returns the paths written."""
    written: list[Path] = []
    written.append(out_dir / "headline_summary.tex")
    headline_summary_table(panel, out_path=written[-1])

    written.append(out_dir / "summary_stats.tex")
    summary_statistics_table(panel, out_path=written[-1])

    written.append(out_dir / "baseline_did.tex")
    baseline_did_table(panel, out_path=written[-1])

    written.append(out_dir / "robustness.tex")
    robustness_table(run_robustness(panel), out_path=written[-1])

    written.append(out_dir / "channels.tex")
    channels_table(panel, out_path=written[-1])

    written.append(out_dir / "polish_german.tex")
    polish_german_table(panel, out_path=written[-1])

    written.append(out_dir / "iv_results.tex")
    iv_results_table(panel, out_path=written[-1])

    written.append(out_dir / "magnitudes.tex")
    magnitudes_table(panel, out_path=written[-1])

    # Counterfactual figure: pairs with the magnitudes table.
    from src.visualization.plots import plot_counterfactual_paths
    iv_cbr = run_iv_did(panel, outcome="cbr", instrument="kmwittenberg")
    cf_path = FIGURES_DIR / "fig_counterfactual.png"
    cf_path.parent.mkdir(parents=True, exist_ok=True)
    plot_counterfactual_paths(panel, iv_coef=iv_cbr["iv_coef"], outcome="cbr",
                              savepath=str(cf_path))
    logger.info("Wrote %s", cf_path)

    written.append(out_dir / "cohort_translation.tex")
    cohort_translation_table(panel, iv_coef=iv_cbr["iv_coef"], out_path=written[-1])

    written.append(out_dir / "pretrends_robustness.tex")
    pretrends_robustness_table(panel, out_path=written[-1])

    written.append(out_dir / "dcdh_diagnostic.tex")
    dcdh_diagnostic_table(panel, out_path=written[-1])

    written.append(out_dir / "variance_decomposition.tex")
    variance_decomposition_table(panel, out_path=written[-1])

    written.append(out_dir / "falsifications.tex")
    falsifications_table(panel, out_path=written[-1])

    written.append(out_dir / "heterogeneity.tex")
    heterogeneity_table(panel, out_path=written[-1])

    written.append(out_dir / "iv_overid.tex")
    iv_overid_table(panel, out_path=written[-1])

    written.append(out_dir / "wild_bootstrap.tex")
    wild_bootstrap_table(panel, out_path=written[-1])

    written.append(out_dir / "coale_decomposition.tex")
    coale_decomposition_table(panel, out_path=written[-1])

    written.append(out_dir / "emigration_robustness.tex")
    emigration_robustness_table(panel, out_path=written[-1])

    written.append(out_dir / "pretreatment_trends.tex")
    pretreatment_trends_table(panel, out_path=written[-1])

    written.append(out_dir / "subsample_decomposition.tex")
    subsample_decomposition_table(panel, out_path=written[-1])

    # Lexis diagram pairs with the Coale decomposition: shows which cohorts'
    # reproductive careers intersect the Kulturkampf and rollback windows.
    from src.visualization.plots import (
        plot_lexis_diagram,
        plot_population_and_migration,
    )
    lexis_path = FIGURES_DIR / "fig_lexis.png"
    lexis_path.parent.mkdir(parents=True, exist_ok=True)
    plot_lexis_diagram(savepath=str(lexis_path))
    logger.info("Wrote %s", lexis_path)

    pop_mig_path = FIGURES_DIR / "fig_population_migration.png"
    plot_population_and_migration(panel, savepath=str(pop_mig_path))
    logger.info("Wrote %s", pop_mig_path)

    # Choropleth maps of sub-region treatment effects (Polish / German
    # Catholic / Protestant rest). Pairs with Table tab:wild_bootstrap.
    try:
        from src.visualization.maps import (
            load_prussia_shapefile, map_subregion_treatment_effects,
        )
        from src.analysis.regressions import run_subregion_did
        shp_path = (
            PROJECT_ROOT / "data" / "raw" / "gis_data"
            / "German_Empire_1871_v.1.0.shp"
        )
        gdf = load_prussia_shapefile(shp_path)
        for outcome, label in [
            ("marriage_rate", "Marriage rate"),
            ("cbr", "Crude birth rate"),
        ]:
            sr = run_subregion_did(panel, outcome=outcome)
            fname = FIGURES_DIR / f"map5_{outcome}_subregion_effects.png"
            map_subregion_treatment_effects(
                gdf, panel, sr, outcome_label=label, savepath=str(fname),
            )
            logger.info("Wrote %s", fname)
    except Exception as exc:
        logger.warning("Skipped sub-region choropleths: %s", exc)

    written.append(out_dir / "conley_robustness.tex")
    conley_robustness_table(panel, out_path=written[-1])

    written.append(out_dir / "event_study.tex")
    event_study_table(panel, out_path=written[-1], use_rollback=False)

    written.append(out_dir / "event_study_rollback.tex")
    event_study_table(panel, out_path=written[-1], use_rollback=True)

    # Side-by-side CBR vs I_g event-study figure: a single artefact for
    # the demography-aware reader. CBR captures overall fertility (the
    # broad Galloway 1994 Figure 1 measure); I_g is the Coale marital-
    # fertility index that nets out nuptiality (Galloway, Hammel & Lee
    # 1994 headline outcome).
    try:
        from src.visualization.plots import plot_event_study_cbr_ig
        es_cbr = run_event_study(panel, outcome="cbr")
        es_ig = run_event_study(panel, outcome="I_g")
        pre_cbr = pretrends_wald_test(panel, outcome="cbr")
        pre_ig = pretrends_wald_test(panel, outcome="I_g")
        es_path = FIGURES_DIR / "fig5_event_study_cbr_ig.png"
        es_path.parent.mkdir(parents=True, exist_ok=True)
        plot_event_study_cbr_ig(
            es_cbr["coefs"], es_ig["coefs"],
            pretrends_cbr=pre_cbr, pretrends_ig=pre_ig,
            savepath=str(es_path),
        )
        logger.info("Wrote %s", es_path)
    except Exception as exc:
        logger.warning("Skipped CBR/I_g event-study figure: %s", exc)

    return written


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if not PANEL_PATH.exists():
        raise SystemExit(
            f"Panel not found at {PANEL_PATH}. Run `dvc repro build` first."
        )
    panel = pd.read_parquet(PANEL_PATH)
    paths = generate_all(panel)
    logger.info("Generated %d tables in %s", len(list(paths)), TABLES_DIR)
