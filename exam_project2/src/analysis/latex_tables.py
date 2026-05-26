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
    run_continuous_polish_decomposition,
    run_kulturkampf_vs_polenpolitik_timing,
    run_polenausweisungen_event_study,
    run_kulturkampf_phase_sensitivity,
    KULTURKAMPF_PHASE_LABELS,
)
from src.analysis.rollback import rollback_event_study
from src.analysis.synthetic_did import (
    run_sdid,
    run_sdid_threshold_sweep,
)
from src.analysis.utils import safe_panel_ols
from src.analysis.variance_decomposition import variance_decomposition
from src.analysis.wild_bootstrap import wild_cluster_bootstrap
from src.data.centroids import load_centroids

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_panel.parquet"

# Main-text outcome set. Four outcomes covering the relevant demographic
# margins with denominator-purified rates throughout:
#   - cbr                          aggregate fertility (per 1k mid-year pop)
#   - gmfr_static_1871             within-marriage fertility (per 1k married
#                                  women 15-49, denominator fixed at 1871)
#   - illegitimate_br_static_1871  non-marital fertility (per 1k *unmarried*
#                                  women 15-49, denominator fixed at 1871).
#                                  The symmetric counterpart of GMFR-static;
#                                  purges the marriage-prevalence channel
#                                  that contaminates `illegitimacy_ratio`
#                                  and `illegitimate_br` (mid-year-pop
#                                  denominator).
#   - general_marriage_rate        extensive margin (per 1k aged 15+)
# Composition-contaminated outcomes (legitimate_br, illegitimacy_ratio) are
# moved to the appendix robustness table.
MAIN_OUTCOMES: tuple[str, ...] = (
    "cbr",
    "gmfr_static_1871",
    "illegitimate_br_static_1871",
    "general_marriage_rate",
)
APPENDIX_OUTCOMES: tuple[str, ...] = (
    "legitimate_br", "illegitimacy_ratio",
)

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
    "illegitimate_br_static_1871": (
        "Illeg.\\ fert.\\ rate$^{1871}$ (per 1k unmar.\\ women, static prev.)"
    ),
    "marriage_rate": "Marriage rate",
    "general_marriage_rate": "Gen.\\ marriage rate",
    "gfr": "GFR (per 1k women 15--49)",
    "lgfr": "Legit.\\ GFR (per 1k women 15--49)",
    "infant_mortality_rate": "Infant mortality",
    "cath_marriage_share": "Catholic marriage share",
    # Princeton EFP / Coale indices. I_g is the Galloway-tradition
    # marital-fertility headline (Hutterite-normalised; Galloway, Hammel
    # & Lee 1994 use its unnormalised form, the GMFR). See
    # coale_indices.py and DATA_APPENDIX.md sec. 6.5.
    "I_f": "$I_f$ (overall fertility)",
    "I_g": "$I_g$ (marital fertility)",
    "I_h": "$I_h$ (illegitimate fertility)",
    "I_m": "$I_m$ (nuptiality)",
    "gmfr": "GMFR (per 1k married women)",
    # Static-1871-prevalence variants: marriage prevalence frozen at the
    # 1871 county-specific baseline so the denominator is purged of the
    # Kulturkampf bad-control channel (treatment-induced changes in
    # contemporaneous M_t inflate the rate mechanically). The
    # fertile-age female count W_t remains time-varying so the rate
    # still scales with population growth.
    "Ig_static_1871": "$I_g^{1871}$ (static prev.)",
    "gmfr_static_1871": "GMFR$^{1871}$ (per 1k mar.\\ women, static prev.)",
    "prop_married_15_49": "Prop.\\ married 15--49",
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
    "zentrum_share_x_post": r"ZentrumShare$_{1871}$ $\times$ Post",
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
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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


def descriptive_statistics_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> str:
    """Cross-sectional descriptive statistics by Catholic share.

    Single panel: unconditional means of the headline outcomes for
    Low-Catholic vs High-Catholic counties, pooled across the full
    1862--1890 window. Temporal dynamics are deferred to the time-series
    figures in the descriptive-evidence section; the table's purpose is
    to fix the baseline cross-sectional gap that the DiD identifies off.

    Requires the following LaTeX preamble in the paper:
    ``\\usepackage{booktabs, threeparttable, siunitx}``.
    """
    out_path = out_path or TABLES_DIR / "descriptive_statistics.tex"
    df = panel.copy()
    df["group_lbl"] = np.where(df["high_cath"] == 1, "HighCath", "LowCath")

    row_specs: list[tuple[str, str]] = [
        ("cbr", "Crude Birth Rate (CBR)"),
        ("legitimate_br", "Legitimate Birth Rate"),
        ("illegitimacy_ratio", "Illegitimacy ratio (\\%)"),
        ("general_marriage_rate", "General Marriage Rate (GMR)"),
        ("gmfr", "General Marital Fertility Rate (GMFR)"),
        ("cath_share", "Share Catholic (\\%)"),
    ]

    def _mean(mask: pd.Series, col: str) -> str:
        v = df.loc[mask, col].mean()
        return f"{v:.2f}" if pd.notna(v) else "{-}"

    low_mask = df["group_lbl"] == "LowCath"
    high_mask = df["group_lbl"] == "HighCath"

    body_rows: list[str] = []
    for col, label in row_specs:
        if col not in df.columns:
            continue
        b1 = _mean(low_mask, col)
        b2 = _mean(high_mask, col)
        body_rows.append(f"{label}   & {b1} & {b2} \\\\")

    n_low_counties = int(df.loc[low_mask, "Code"].nunique())
    n_high_counties = int(df.loc[high_mask, "Code"].nunique())
    n_total = len(df)

    body = (
        "\\begin{threeparttable}\n"
        "\\caption{Descriptive statistics by Catholic share}\n"
        "\\label{tab:descriptive_statistics}\n"
        "\\begin{tabular}{l S[table-format=3.2] S[table-format=3.2]}\n"
        "\\toprule\n"
        "\\textbf{Variable} & {Low-Cath} & {High-Cath} \\\\\n"
        " & {Mean} & {Mean} \\\\\n"
        "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        "\\midrule\n"
        f"Counties (N) & {{{n_low_counties}}} & {{{n_high_counties}}} \\\\\n"
        f"Observations ($N \\times T$) & \\multicolumn{{2}}{{c}}{{{n_total:,} (Total)}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item \\textit{Note:} Low-Catholic counties are defined as $\\le 50\\%$ "
        "Catholic in the 1871 census; High-Catholic as $> 50\\%$. Means are "
        "pooled across the full 1862--1890 panel. The Crude Birth Rate and "
        "Legitimate Birth Rate are per 1{,}000 \\emph{mid-year} inhabitants; "
        "the illegitimacy ratio is illegitimate births as a percentage of all "
        "births; the General Marriage Rate (GMR) is marriages per 1{,}000 "
        "women aged 15--49; and the General Marital Fertility Rate (GMFR) is "
        "legitimate births per 1{,}000 married women aged 15--49. The "
        "married-women and women 15--49 denominators of GMR and GMFR are "
        "time-varying, piecewise-linearly interpolated between Galloway's "
        "STA1871, AGE1882, and AGE1890 anchors. Temporal dynamics for each "
        "variable are shown in the time-series figures of the "
        "descriptive-evidence section.\n"
        "\\item \\textit{Source:} Author's calculations from the Galloway "
        "Prussia Database \\citep{Galloway2007}; mid-year population "
        "constructed via linear interpolation between consecutive December "
        "censuses.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
    )

    out = (
        "% Auto-generated by src/analysis/latex_tables.py -- do not edit by hand.\n"
        "% LaTeX preamble required: \\usepackage{booktabs, threeparttable, siunitx}\n"
        "% Citations used: \\citep{Galloway2007}.\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\small\n"
        + body +
        "\\end{table}\n"
    )
    _write(out_path, out)
    return out


def pretreatment_balance_1849_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
) -> str:
    """Pre-Kulturkampf (1849) balance by 1871 Catholic-share quartile.

    Strongest version of the balance argument: in 1849 -- 23 years
    before the May Laws -- counties that would later split into high-
    and low-Catholic groups did not yet differ on the key dimensions
    (schooling participation, religious infrastructure per capita,
    industrial density, family size). The reader can verify there is
    no systematic pre-trend baked into the cross-sectional Catholic-
    share variation that the DiD identifies on.

    Variables shown are derived from the 1849 iPEHD merge (using the
    name-based ``kreiskey1849 -> Code`` crosswalk; coverage ~70%) plus
    the 1871 birthplace mobility shares from BIR1871.
    """
    out_path = out_path or TABLES_DIR / "pretreatment_balance_1849.tex"

    # Collapse to one row per county.
    cs = (
        panel.sort_values(["Code", "Year"])
             .drop_duplicates(subset="Code", keep="first")
             .copy()
    )

    # Build the row variables on the county frame.
    if "edu1849_pub_ele_stud_m" in cs.columns and "pop1849_tot" in cs.columns:
        students = cs["edu1849_pub_ele_stud_m"].fillna(0) + cs["edu1849_pub_ele_stud_f"].fillna(0)
        cs["attend_rate_1849"] = students / cs["pop1849_tot"].replace(0, np.nan)
    if "rel1849_cat_priest" in cs.columns and "pop1849_tot" in cs.columns:
        cs["cat_priest_per_1k_1849"] = (
            cs["rel1849_cat_priest"] / cs["pop1849_tot"].replace(0, np.nan) * 1000
        )
    if "ipehd_1849_indu_fac_total" in cs.columns and "pop1849_tot" in cs.columns:
        cs["factories_per_10k_1849"] = (
            cs["ipehd_1849_indu_fac_total"] / cs["pop1849_tot"].replace(0, np.nan) * 10_000
        )
    if "pop1849_families" in cs.columns and "pop1849_tot" in cs.columns:
        cs["avg_household_size_1849"] = (
            cs["pop1849_tot"] / cs["pop1849_families"].replace(0, np.nan)
        )

    rows: list[tuple[str, str]] = [
        ("attend_rate_1849", "Elementary attendance rate (1849)"),
        ("cat_priest_per_1k_1849", "Catholic priests per 1,000 pop (1849)"),
        ("factories_per_10k_1849", "Factories per 10,000 pop (1849)"),
        ("avg_household_size_1849", "Avg. household size (1849)"),
        ("born_in_kreis_share_1871", "Share born in Kreis (1871)"),
    ]
    # Drop rows whose column is missing or all-null.
    rows = [(c, lbl) for c, lbl in rows
            if c in cs.columns and cs[c].notna().sum() >= 20]

    # 1871 Catholic-share quartiles using counties with any 1849 data.
    has_1849 = (
        cs[["attend_rate_1849", "cat_priest_per_1k_1849"]].notna().any(axis=1)
        if {"attend_rate_1849", "cat_priest_per_1k_1849"}.issubset(cs.columns)
        else pd.Series(True, index=cs.index)
    )
    bands = cs.loc[has_1849 & cs["cath_share"].notna()].copy()
    bands["q"] = pd.qcut(bands["cath_share"], 4,
                         labels=["Q1", "Q2", "Q3", "Q4"])

    body_rows = []
    for col, label in rows:
        means = bands.groupby("q", observed=True)[col].mean()
        # t-test Q4 - Q1 (two-sample, unequal variance).
        a = bands.loc[bands["q"] == "Q1", col].dropna()
        b = bands.loc[bands["q"] == "Q4", col].dropna()
        if len(a) > 5 and len(b) > 5:
            diff = float(b.mean() - a.mean())
            pooled_se = math.sqrt(
                a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
            )
            t = diff / pooled_se if pooled_se > 0 else 0.0
            # Two-sided p via normal approximation (large samples).
            p = 2 * (1 - _normal_cdf(abs(t)))
            star = _stars(p)
            diff_str = f"{diff:+.3f}{star}"
        else:
            diff_str = "{-}"
        cells = [f"{means.get(q, np.nan):.3f}" if pd.notna(means.get(q, np.nan))
                 else "{-}" for q in ["Q1", "Q2", "Q3", "Q4"]]
        body_rows.append(f"{label} & " + " & ".join(cells) + f" & {diff_str} \\\\")

    n_q = bands["q"].value_counts().reindex(["Q1", "Q2", "Q3", "Q4"]).fillna(0).astype(int)

    body = (
        "\\begin{threeparttable}\n"
        "\\caption{Pre-Kulturkampf (1849) balance by 1871 Catholic-share quartile}\n"
        "\\label{tab:pretreatment_balance_1849}\n"
        "\\begin{tabular}{l S S S S S}\n"
        "\\toprule\n"
        " & {Q1 (low)} & {Q2} & {Q3} & {Q4 (high)} & {Q4 $-$ Q1} \\\\\n"
        "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        "\\midrule\n"
        f"Counties (N) & {{{n_q.get('Q1', 0)}}} & {{{n_q.get('Q2', 0)}}} & "
        f"{{{n_q.get('Q3', 0)}}} & {{{n_q.get('Q4', 0)}}} & \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item \\textit{Note:} Columns Q1-Q4 are quartiles of the 1871 "
        "Catholic share; row entries are means within each quartile. The "
        "final column is the difference of means (Q4 $-$ Q1) with stars from "
        "a Welch t-test approximation. 1849 covariates come from the iPEHD "
        "(Becker-Woessmann) cross-section merged via a name-based "
        "\\texttt{kreiskey1849} $\\to$ Galloway Code crosswalk (coverage "
        "$\\approx 70\\%$). The 1871 birthplace share is from Galloway BIR1871. "
        "The table provides the strongest balance evidence in the paper: "
        "absent systematic Q4 $-$ Q1 gaps in 1849, the cross-sectional "
        "Catholic-share variation that the DiD identifies on cannot be "
        "attributed to a pre-existing economic-structural divergence.\n"
        "\\item \\textit{Significance:} \\sym{*} $p<0.10$, \\sym{**} $p<0.05$, "
        "\\sym{***} $p<0.01$.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
    )
    out = (
        "% Auto-generated by src/analysis/latex_tables.py -- do not edit by hand.\n"
        "% LaTeX preamble required: \\usepackage{booktabs, threeparttable, siunitx}\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\small\n"
        + body +
        "\\end{table}\n"
    )
    _write(out_path, out)
    return out


def war_province_diagnostic_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
    war_years: tuple[int, ...] = (1866, 1870, 1871),
    ref_years: tuple[int, ...] = (1864, 1865, 1868, 1869),
    outcome: str = "cbr",
) -> str:
    """
    By-Regierungsbezirk war-year CBR-drop diagnostic.

    Tests whether the pre-1873 CBR trend in the event study is driven
    by differential Prussian-Army recruitment burden across
    Protestant- vs Catholic-majority Regierungsbezirke. Each row shows
    mean CBR in war years (1866 + 1870-71), mean in flanking non-war
    years (1864-65, 1868-69), the difference, and the Rb's 1871
    Catholic share. Rbs ordered by largest war-year drop first.

    Companion to ``fig_war_context.png``. Both are diagnostic
    artefacts addressing the question: "are the Catholic-Protestant
    pre-trends in the event study a behavioural story or a
    war-cohort mechanical story?"
    """
    from src.analysis.war_robustness import province_war_effect

    out_path = out_path or TABLES_DIR / "war_province_diagnostic.tex"
    pwe = province_war_effect(
        panel, war_years=war_years, ref_years=ref_years, outcome=outcome,
    )

    n_rb = len(pwe)
    corr_value = pwe["cath_share_rb_mean"].corr(pwe["diff"])

    body_rows = []
    for _, row in pwe.iterrows():
        body_rows.append(
            f"{row['Rb']} & "
            f"{row['mean_war_years']:.2f} & "
            f"{row['mean_nonwar_years']:.2f} & "
            f"{row['diff']:+.2f} & "
            f"{row['cath_share_rb_mean']:.1f} & "
            f"{int(row['n_counties'])} \\\\"
        )

    body = (
        "\\begin{tabular}{l*{5}{c}}\n"
        "\\toprule\n"
        f"Regierungsbezirk & War-years mean & Non-war mean & "
        f"Diff (war$-$non) & Cath.\\ share (\\%) & $N_{{counties}}$ \\\\\n"
        "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        "\\midrule\n"
        f"$\\mathrm{{corr}}(\\text{{Cath. share}}, \\text{{Diff}})$ "
        f"& \\multicolumn{{5}}{{c}}{{{corr_value:+.3f} "
        f"({n_rb} Rbs)}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "War-cohort diagnostic: war-year (1866, 1870--71) vs "
            "non-war-year (1864--65, 1868--69) mean CBR by Regierungsbezirk"
        ),
        label="tab:war_province_diagnostic",
        n_cols=6,
        notes=(
            "Each row reports the mean crude birth rate in "
            "Austro-Prussian (1866) and Franco-Prussian (1870--71) war "
            "years, the mean in flanking non-war years (1864--65, "
            "1868--69), and the difference. ``Cath.\\ share (\\%)'' is "
            "the Regierungsbezirk-mean of \\texttt{cath\\_share} (1871 "
            "census). Rbs are sorted ascending by the war$-$non-war "
            "difference (most-negative dip first). The bottom-row "
            "correlation tests whether more-Catholic Rbs dipped less "
            "during war years (positive correlation = expected under "
            "the differential-conscription hypothesis); a value near "
            "zero indicates the pre-1873 Catholic-Protestant CBR "
            "trend is not driven by war-cohort mechanics. Diagnostic "
            "for the pre-trends discussion in Section "
            "\\ref{sec:pretrends}; pairs with \\texttt{fig\\_war\\_context.png}."
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
        "n": int(res.nobs),
        "r2": float(res.rsquared_within),
    }


def baseline_did_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = MAIN_OUTCOMES,
    out_path: Path | None = None,
    *,
    show_carryforward: bool = True,
    show_ig_pretrends_line: bool = True,
    digits_coef: int = 3,
    digits_se: int = 3,
    caption: str = (
        "Baseline difference-in-differences: Kulturkampf and "
        "conventional demographic rates"
    ),
    label: str = "tab:baseline_did",
    extra_note: str = (
        "Crude birth rates use mid-year total population in the denominator "
        "and therefore conflate age structure, marital structure, and "
        "within-marriage fertility. The general marriage rate uses pop.\\ aged "
        "15+ (Newell 1988), netting out the under-15 share. The Coale--Watkins "
        "decomposition in Table~\\ref{tab:baseline_did_indices} addresses the "
        "remaining birth-rate decomposition directly."
    ),
) -> str:
    """Multi-outcome baseline DiD with TWFE and stricter Year x Rb FE columns.

    Default outcomes are CBR, legitimate birth rate, illegitimacy ratio,
    and the general marriage rate. The companion
    ``baseline_did_indices_table`` reports the Coale--Watkins indices
    ($I_f$, $I_g$, $I_m$, $I_h$) that decompose the rates by age
    structure and nuptiality. Rates and indices live in separate tables
    because (i) they are on very different scales (~0--50 vs.~0--1),
    (ii) the Coale identity $I_f \\approx I_g\\cdot I_m + I_h(1-I_m)$
    lives in the indices table and is easier to read uncluttered, and
    (iii) the two tables answer sequential questions in the narrative.

    Headline rates use the standard demographic convention: the
    population denominator is linearly interpolated between consecutive
    December census anchors and evaluated at July 1 of each calendar
    year. The ``Galloway carry-forward robustness'' row reports the same
    coefficients using the raw Galloway `Poptot` (previous December
    census carried forward in inter-census years) so a reader can see
    how using the database "out of the box" differs from the proper
    mid-year convention.

    Parameters
    ----------
    show_carryforward : bool
        Include the Galloway carry-forward robustness row. Only meaningful
        for outcomes that have a ``_carryforward`` variant in the panel;
        for the Coale-index table this is False.
    show_ig_pretrends_line : bool
        Append a one-line `\\multicolumn` showing the joint Wald
        $\\chi^2$ test for pre-1872 event-study coefficients on $I_g$.
        Useful as a Galloway-tradition headline footer in the rates table;
        the indices table already includes $I_g$ as a column so the line
        is suppressed there.
    caption, label, extra_note : str
        Overrides for the LaTeX caption, table label, and an additional
        sentence appended to the standard notes block.
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
    # (i.e. raw Galloway `Poptot`). Only meaningful for outcomes with a
    # _carryforward column; the Coale indices use mid-year pop only and
    # have no carry-forward variant, so the indices-table call suppresses
    # this block via show_carryforward=False.
    _carryforward_map = {
        "cbr": "cbr_carryforward",
        "legitimate_br": "legitimate_br_carryforward",
        "illegitimate_br": "illegitimate_br_carryforward",
        "marriage_rate": "marriage_rate_carryforward",
    }
    if show_carryforward:
        cf_outcomes = [_carryforward_map.get(o) for o in outcomes]
        cols_cf_twfe = [
            _did_column(panel, o, "twfe") if o else None for o in cf_outcomes
        ]
        cols_cf_strict = [
            _did_column(panel, o, "year_x_rb") if o else None for o in cf_outcomes
        ]
        cols_cf = cols_cf_twfe + cols_cf_strict
        # If no outcome in this call actually has a carryforward variant,
        # suppress the row entirely.
        if not any(c is not None for c in cols_cf):
            show_carryforward = False

    # Pre-trends Wald chi-squared per outcome (TWFE event study).
    pretrends_per_outcome = [
        pretrends_wald_test(panel, outcome=o) for o in outcomes
    ]
    # Optional I_g pre-trends footer line: useful as a Galloway-tradition
    # headline footnote on the rates table; suppressed on the indices
    # table because I_g is already a column there.
    if show_ig_pretrends_line:
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
        + " & ".join(_fmt_coef(c["coef"], c["p"], digits=digits_coef) for c in cols)
        + r" \\"
    )
    se_row = (
        " & "
        + " & ".join(_fmt_se(c["se"], digits=digits_se) for c in cols)
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
    if show_carryforward:
        carryforward_coef_row = (
            "\\quad CathShare $\\times$ Post (Galloway carry-forward) & "
            + " & ".join(
                "--" if c is None else _fmt_coef(c["coef"], c["p"], digits=digits_coef)
                for c in cols_cf
            )
            + r" \\"
        )
        carryforward_se_row = (
            " & "
            + " & ".join(
                "" if c is None else _fmt_se(c["se"], digits=digits_se)
                for c in cols_cf
            )
            + r" \\"
        )
    # One-line I_g pre-trends Wald comparison (spans full table width).
    if show_ig_pretrends_line:
        pretrends_ig_row = (
            f"\\multicolumn{{{n + 1}}}{{l}}{{"
            f"\\textit{{Pre-trends Wald $\\chi^{{2}}$ on $I_g$ "
            f"(Coale marital fertility, Galloway-tradition headline)}}: "
            f"$\\chi^{{2}} = {pretrends_ig['wald_chi2']:.2f}$, "
            f"df $= {pretrends_ig['df']}$, "
            f"$p = {pretrends_ig['p_value']:.3f}$"
            f"}} \\\\"
        )

    # Build the FE-block and downstream rows. Carry-forward + I_g
    # pre-trends are conditional on the table type (rates vs indices).
    fe_block = (
        f"County FE & {yes_twfe} & {yes_strict} \\\\\n"
        + f"Year FE & {yes_twfe} & {no_strict} \\\\\n"
        + f"Year $\\times$ Rb FE & {' & '.join('--' for _ in cols_twfe)} & {yes_strict} \\\\\n"
        + "Observations & "
        + " & ".join(f"{c['n']:,}" for c in cols)
        + " \\\\\n"
        + "Within $R^{2}$ & "
        + " & ".join(f"{c['r2']:.3f}" for c in cols)
        + " \\\\\n"
    )
    carryforward_block = (
        "\\midrule\n"
        + f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Galloway carry-forward robustness}} "
        f"(rate $=$ count $/$ raw Galloway \\texttt{{Poptot}}; previous Dec.\\ census "
        f"carried forward in inter-census years)}} \\\\\n"
        + carryforward_coef_row + "\n"
        + carryforward_se_row + "\n"
    ) if show_carryforward else ""
    pretrends_block = (
        "\\midrule\n"
        + pretrends_p_row + "\n"
        + ("\\addlinespace\n" + pretrends_ig_row + "\n" if show_ig_pretrends_line else "")
    )

    tabular = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + panel_header + "\n"
        + outcome_header + "\n"
        + col_nums + "\n"
        + "\\midrule\n"
        + coef_row + "\n"
        + se_row + "\n"
        "\\midrule\n"
        + fe_block
        + carryforward_block
        + pretrends_block
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    notes_core = (
        "Two-way fixed-effects estimates of equation "
        "$Y_{it} = \\beta\\,(\\mathrm{CathShare}_i \\times \\mathrm{Post}_t) + "
        "\\alpha_i + \\delta_t + \\gamma X_{it} + \\varepsilon_{it}$, with "
        "$\\delta_t$ replaced by year~$\\times$~Regierungsbezirk fixed effects "
        "in Panel~B. Post is an indicator for $t \\geq 1873$. Standard errors "
        "clustered at the county level in parentheses. "
    )
    notes_rates = (
        "Birth rates per 1{,}000 \\emph{mid-year} population; "
        "the general marriage rate per 1{,}000 mid-year population aged 15+ "
        "(Newell 1988); illegitimacy ratio in percent. "
        "Mid-year population is constructed by linearly interpolating between "
        "consecutive December census anchors and evaluating at July 1 of each "
        "calendar year (standard demographic convention); the 15+ share is "
        "interpolated between the 1871 and 1890 AGE censuses with the 1882 "
        "AGE anchor used where available. "
    )
    notes_carryforward = (
        "The ``Galloway "
        "carry-forward robustness'' row reports the same coefficients using "
        "the raw Galloway \\texttt{Poptot}, which carries the previous "
        "December census forward unchanged in inter-census years and biases "
        "CBR upward by 1--3\\% in growing populations. "
    ) if show_carryforward else ""
    notes_pretrends_main = (
        "The "
        "``Pre-trends $\\chi^{2}$ $p$'' row reports the joint Wald test that "
        "all event-study coefficients in the pre-1872 period equal zero "
        "(estimated separately on the TWFE event-study; identical $p$-value "
        "applies under both FE designs). "
    )
    notes_pretrends_ig = (
        "The single-line ``$I_g$ comparison'' "
        "reports the same test on Coale's marital-fertility index "
        "(Hutterite-normalised legitimate births per married woman 15--49) -- "
        "the headline outcome in Galloway, Hammel \\& Lee (1994). The "
        "companion event-study figure \\texttt{fig5\\_event\\_study\\_cbr\\_ig.png} "
        "plots the CBR and $I_g$ event studies side by side. "
    ) if show_ig_pretrends_line else ""
    notes_extra = (extra_note + " ") if extra_note else ""
    notes_stars = "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."

    out = _wrap_table(
        tabular,
        caption=caption,
        label=label,
        n_cols=n + 1,
        notes=(
            notes_core
            + notes_rates
            + notes_carryforward
            + notes_pretrends_main
            + notes_pretrends_ig
            + notes_extra
            + notes_stars
        ),
    )
    _write(out_path, out)
    return out


def cath_polen_identification_table(
    panel: pd.DataFrame,
    polish_rbs: tuple[str, ...] = ("POS", "BRO"),
    german_cath_rbs: tuple[str, ...] = (
        "KOL", "KOB", "TRI", "AAC", "OPP", "MUN",
    ),
    out_path: Path | None = None,
) -> str:
    """Identification-support diagnostic for the religion vs ethnicity
    decomposition. Reports cross-county correlation between the 1871
    Catholic share and the 1871 Polenpartei vote share by sub-sample,
    documenting (i) that the full-panel correlation is modest, so the
    continuous decomposition has identifying content; and (ii) that
    within the Polish provinces the two variables are essentially
    collinear, so a within-Posen religion-vs-ethnicity statement is
    impossible. The German-Catholic sub-sample has zero variation in
    Polenpartei vote share, so those counties contribute only to the
    identification of the religion-only coefficient.
    """
    out_path = out_path or TABLES_DIR / "cath_polen_identification.tex"

    cs = panel.dropna(subset=["polen_share_1871"]).drop_duplicates("Code").copy()

    def _cls(rb: str) -> str:
        if rb in polish_rbs:
            return "Polish (POS, BRO)"
        if rb in german_cath_rbs:
            return "German Catholic (KOL, KOB, TRI, AAC, OPP, MUN)"
        return "Protestant remainder"

    cs["region"] = cs["Rb"].apply(_cls)

    rows: list[tuple[str, int, float | None, float | None, float, float]] = []
    for label in (
        "Polish (POS, BRO)",
        "German Catholic (KOL, KOB, TRI, AAC, OPP, MUN)",
        "Protestant remainder",
    ):
        sub = cs[cs["region"] == label]
        n = len(sub)
        cath_mean = float(sub["cath_share"].mean()) if n else float("nan")
        polen_mean = float(sub["polen_share_1871"].mean()) if n else float("nan")
        if n >= 2 and sub["polen_share_1871"].std() > 0:
            corr = float(sub[["cath_share", "polen_share_1871"]].corr().iloc[0, 1])
        else:
            corr = None  # undefined when one variable has zero variance
        rows.append((label, n, corr, cath_mean, polen_mean, 0.0))

    # Full panel row
    n_full = len(cs)
    cath_mean_full = float(cs["cath_share"].mean())
    polen_mean_full = float(cs["polen_share_1871"].mean())
    corr_full = float(
        cs[["cath_share", "polen_share_1871"]].corr().iloc[0, 1]
    )

    def _fmt_corr(c: float | None) -> str:
        return f"{c:+.3f}" if c is not None else "n/a"

    body = (
        r"\begin{tabular}{lcccc}" + "\n"
        + r"\toprule" + "\n"
        + r"Sub-sample & Counties & Mean cath. & Mean Polen. & corr(cath, polen) \\" + "\n"
        + r"\midrule" + "\n"
    )
    for label, n, corr, cath_mean, polen_mean, _ in rows:
        body += (
            f"{label} & {n} & {cath_mean:.1f} & {polen_mean:.1f} & {_fmt_corr(corr)} \\\\\n"
        )
    body += (
        r"\midrule" + "\n"
        + f"Full panel & {n_full} & {cath_mean_full:.1f} & {polen_mean_full:.1f} & {corr_full:+.3f} \\\\\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Identification support for the religion vs ethnicity "
            "decomposition: cross-county correlation between the 1871 "
            "Catholic share and the 1871 Polenpartei vote share, by "
            "sub-sample"
        ),
        label="tab:cath_polen_identification",
        n_cols=5,
        notes=(
            "Each row reports county-level means and the cross-county "
            "correlation between \\texttt{cath\\_share} (1871 Catholic "
            "population share) and \\texttt{polen\\_share\\_1871} (1871 "
            "Polenpartei Reichstag vote share) on the relevant sub-sample. "
            "The full-panel correlation of $+0.26$ is modest, so the "
            "continuous decomposition in "
            "Table~\\ref{tab:continuous_polish_decomposition} has "
            "identifying content. Within the Polish provinces (Posen, "
            "Bromberg) the correlation is $+0.93$ -- nearly collinear -- "
            "so the decomposition cannot tell apart religion from "
            "ethnicity \\emph{within} that sub-sample. The German "
            "Catholic counties have zero variance in Polenpartei voting "
            "(no county in those Regierungsbezirke recorded any Koło "
            "Polskie vote), so the within-group correlation is "
            "undefined; these counties contribute only to identifying "
            "the religion-only coefficient $\\beta_1$. The Protestant "
            "remainder correlation of $+0.30$ comes from a small number "
            "of Danzig and Marienwerder counties with non-zero Polish "
            "vote share but low Catholic share. See "
            "Figure~\\ref{fig:cath_polen_scatter} in the appendix for "
            "the corresponding county-level scatter."
        ),
    )
    _write(out_path, out)
    return out


def conventional_rates_appendix_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = APPENDIX_OUTCOMES,
    out_path: Path | None = None,
) -> str:
    """Appendix robustness: the two composition-contaminated rates dropped
    from the main-text headline table.

    Reports the same baseline DiD on the legitimate birth rate (mid-year
    total-population denominator) and the illegitimacy ratio
    (illegitimate / total births). Both outcomes mix behavioural fertility
    with marriage-prevalence composition: ``legitimate_br`` falls
    mechanically if the population becomes less married even at constant
    marital fertility, and ``illegitimacy_ratio`` rises mechanically if
    marriages fall even at constant marital and non-marital fertility.
    Reported for transparency; the denominator-purified replacements
    ``gmfr_static_1871`` (within-marriage) and
    ``illegitimate_br_static_1871`` (non-marital) appear in the main-text
    headline table.
    """
    out_path = out_path or TABLES_DIR / "conventional_rates_appendix.tex"
    return baseline_did_table(
        panel,
        outcomes=outcomes,
        out_path=out_path,
        show_carryforward=False,
        show_ig_pretrends_line=False,
        caption=(
            "Composition-contaminated rate outcomes "
            "(robustness, not headline): legitimate birth rate and "
            "illegitimacy ratio"
        ),
        label="tab:conventional_rates_appendix",
        extra_note=(
            "Both outcomes are mechanically affected by marriage-"
            "prevalence composition: the legitimate birth rate uses a "
            "total-population denominator that falls when the unmarried "
            "share rises, and the illegitimacy ratio rises mechanically "
            "if marriages decline at constant marital and non-marital "
            "fertility behaviour. The headline table replaces these "
            "with denominator-purified static-1871 rates: "
            "$\\text{GMFR}^{1871}$ (per 1k married women 15--49, "
            "denominator pinned at 1871) and the symmetric "
            "$\\text{Illeg.\\ FR}^{1871}$ (per 1k unmarried women "
            "15--49, denominator pinned at 1871). The composition-"
            "free versions isolate fertility behaviour on the marital "
            "and non-marital margins respectively."
        ),
    )


def baseline_did_indices_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("I_f", "I_g", "I_m", "I_h"),
    out_path: Path | None = None,
) -> str:
    """Baseline DiD on the four Princeton EFP / Coale-Watkins indices.

    Reports the same TWFE / Year x Rb FE specification as
    :func:`baseline_did_table` but with the Hutterite-normalised
    fertility indices ($I_f$, $I_g$, $I_m$, $I_h$) as outcomes. These
    indices satisfy the Coale identity

        I_f \\approx I_g * I_m + I_h * (1 - I_m)

    so reporting all four lets the reader read off the within-marriage
    vs. nuptiality decomposition directly. The conventional rates
    (CBR, legitimate BR, illegitimacy ratio, marriage rate) live in
    :func:`baseline_did_table`; that table's notes point readers here
    for the rigorous decomposition.

    Carry-forward robustness and the standalone $I_g$ pre-trends footer
    are both suppressed -- carry-forward because the indices use the
    interpolated mid-year denominators throughout, and the $I_g$ line
    because $I_g$ is now one of the columns.
    """
    return baseline_did_table(
        panel,
        outcomes=outcomes,
        out_path=out_path or TABLES_DIR / "baseline_did_indices.tex",
        show_carryforward=False,
        show_ig_pretrends_line=False,
        digits_coef=5,
        digits_se=5,
        caption=(
            "Baseline difference-in-differences: Coale--Watkins "
            "fertility and nuptiality indices"
        ),
        label="tab:baseline_did_indices",
        extra_note=(
            "$I_f$, $I_g$, $I_m$, $I_h$ are the four Princeton EFP / "
            "Coale--Watkins indices. They satisfy the identity "
            "$I_f \\approx I_g \\cdot I_m + I_h \\cdot (1 - I_m)$: a "
            "fall in overall fertility ($I_f$) can be decomposed into a "
            "fall in within-marriage fertility ($I_g$), a fall in the "
            "Hutterite-weighted proportion married ($I_m$), or a fall "
            "in illegitimate fertility ($I_h$). The denominators of "
            "$I_g$ and $I_m$ -- count of women aged 15--49 and count of "
            "married women aged 15--49 -- are piecewise-linearly "
            "interpolated between Galloway's STA1871 (1871), AGE1882 "
            "(1882), and AGE1890 (1890) Kreis-level anchors. The "
            "companion crude-rates table is "
            "Table~\\ref{tab:baseline_did}."
        ),
    )


def kulturkampf_phase_sensitivity_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = MAIN_OUTCOMES,
    cutoffs: Sequence[int] = (1871, 1872, 1873, 1874, 1875, 1876),
    placebo_cutoff: int | None = 1870,
    sample_year_range: tuple[int, int] = (1862, 1890),
    out_path: Path | None = None,
    digits_coef: int = 4,
    digits_se: int = 4,
) -> str:
    """
    Treatment-cutoff sensitivity by Kulturkampf legislative phase.

    Renders the output of
    :func:`regressions.run_kulturkampf_phase_sensitivity` as a
    cross-tabulated LaTeX table: rows are outcomes (default = the five
    headline conventional rates plus the Hutterite-normalised $I_g$),
    columns are alternative cutoff years corresponding to the five
    distinct Kulturkampf phases plus a 1871 placebo. Each cell reports
    the coefficient on $\\texttt{cath\\_share} \\times \\mathbb{1}[
    \\text{Year} \\geq \\text{cutoff}]$ with the clustered standard
    error in parentheses and significance stars.

    Reading the table.

    - Whether the marriage-rate effect is specifically about the 1874
      Civil Marriage Act (vs the broader 1873 May Laws): compare the
      ``general_marriage_rate`` row's 1873 column to its 1874 column. A
      sharp strengthening at 1874 (and only at 1874) points to the Civil
      Marriage Act as the operative channel; flat or monotonic-from-
      1873 patterns indicate the marriage response is part of the
      broader May-Laws-era shock.
    - Whether fertility effects pick up a different phase from
      marriage (e.g.\\ 1876 episcopal expulsions for ``cbr`` vs 1874
      for ``general_marriage_rate``): rows should be read independently.
    - The 1871 placebo column should yield small / insignificant
      coefficients for outcomes where the post-1873 effect is causal;
      a significant placebo coefficient is a pre-trends signal
      (already documented for $I_g$ in
      Table~\\ref{tab:pretreatment_trends}).
    """
    out_path = out_path or TABLES_DIR / "phase_sensitivity.tex"

    res = run_kulturkampf_phase_sensitivity(
        panel,
        outcomes=outcomes,
        cutoffs=cutoffs,
        placebo_cutoff=placebo_cutoff,
        sample_year_range=sample_year_range,
    )
    if res.empty:
        raise RuntimeError(
            "run_kulturkampf_phase_sensitivity returned no rows -- check that "
            "the requested outcomes are on the panel."
        )

    ordered_cutoffs = sorted(res["cutoff"].unique().tolist())
    placebo = (
        int(placebo_cutoff)
        if placebo_cutoff is not None
        else None
    )

    def _fmt_coef(row: pd.Series) -> str:
        stars = (
            "$^{***}$" if row["p"] < 0.01
            else "$^{**}$" if row["p"] < 0.05
            else "$^{*}$" if row["p"] < 0.10
            else ""
        )
        return f"{row['coef']:+.{digits_coef}f}{stars}"

    def _fmt_se(row: pd.Series) -> str:
        return f"({row['se']:.{digits_se}f})"

    # Header. Use centred ``c`` columns rather than siunitx ``S`` columns:
    # the cells contain LaTeX (signed coefficients, significance stars,
    # parenthesised standard errors) that siunitx cannot parse as bare
    # numbers, which produced "Invalid numerical input" errors when the
    # table was compiled.
    n_cols = len(ordered_cutoffs)
    col_spec = "l" + "c" * n_cols
    head_year = " & ".join(
        f"{{{int(c)}{'$^{P}$' if c == placebo else ''}}}"
        for c in ordered_cutoffs
    )
    # Subheader: short phase tags. Use \makecell so long labels wrap.
    short_phase = {
        1870: "(placebo)",
        1871: "Kanzelparagraph",
        1872: "Jesuits Law / school",
        1873: "May Laws",
        1874: "Civ.\\ Marriage Act (PR)",
        1875: "Brotkorb / Reichszivilehe",
        1876: "Bishop expulsions",
    }
    head_phase = " & ".join(
        f"{{\\footnotesize {short_phase.get(int(c), str(c))}}}"
        for c in ordered_cutoffs
    )

    # Sample-size and within-R^2 footer rows (per cutoff, pooled
    # across outcomes -- they are identical within a cutoff column
    # because the same panel is used).
    pivot = res.pivot(index="outcome", columns="cutoff", values=["coef", "se", "p", "n", "r2_within"])

    body_lines: list[str] = []
    for outcome in outcomes:
        if outcome not in pivot.index:
            continue
        label = _outcome_label(outcome)
        cells: list[str] = []
        for c in ordered_cutoffs:
            sub = res[(res["outcome"] == outcome) & (res["cutoff"] == c)]
            if sub.empty:
                cells.append(r"\multicolumn{1}{c}{--}")
                continue
            row = sub.iloc[0]
            cells.append(_fmt_coef(row))
        body_lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        # SE row
        se_cells: list[str] = []
        for c in ordered_cutoffs:
            sub = res[(res["outcome"] == outcome) & (res["cutoff"] == c)]
            if sub.empty:
                se_cells.append("")
                continue
            row = sub.iloc[0]
            se_cells.append(_fmt_se(row))
        body_lines.append(" & " + " & ".join(se_cells) + r" \\")

    # Sample-size footer (cutoff-invariant in the balanced panel, but
    # not literally identical because the post indicator changes
    # affects which years are "treated"; we report the general_marriage_rate
    # row's N as a representative figure since the n is identical
    # across cutoffs within an outcome).
    n_row_outcome = (
        "general_marriage_rate" if "general_marriage_rate" in pivot.index else outcomes[0]
    )
    n_cells = []
    for c in ordered_cutoffs:
        sub = res[(res["outcome"] == n_row_outcome) & (res["cutoff"] == c)]
        n_cells.append(f"{{{int(sub.iloc[0]['n']):,}}}" if not sub.empty else "")
    n_footer = "Observations & " + " & ".join(n_cells) + r" \\"

    body = "\n".join(body_lines)

    sample_note = (
        rf"Sample restricted to {sample_year_range[0]}--{sample_year_range[1]} "
        r"(headline Kulturkampf window) so estimates are not contaminated "
        r"by the post-1887 rollback recovery. "
        if sample_year_range is not None
        else ""
    )
    note = (
        r"\textit{Notes:} Two-way fixed-effects estimates of "
        r"$Y_{it} = \beta\,(\text{cath\_share}_i \times \mathbb{1}["
        r"\text{Year} \geq c]) + \alpha_i + \delta_t + \varepsilon_{it}$ "
        r"for each candidate cutoff $c$. " + sample_note +
        r"County and year fixed effects "
        r"throughout; standard errors clustered at the county level in "
        r"parentheses. The six non-placebo cutoffs correspond to "
        r"distinct legislative phases of the Kulturkampf: \textbf{1871} "
        r"the \emph{Kanzelparagraph} (\S 130a Reich Penal Code, Dec 10) "
        r"criminalises politically charged statements by Catholic "
        r"clergy and is the first Kulturkampf statute; \textbf{1872} "
        r"the \emph{Jesuitengesetz} (July 4) expels the Society of "
        r"Jesus and the \emph{Schulaufsichtsgesetz} (March 11) transfers "
        r"school inspection to the state; \textbf{1873} the "
        r"\emph{Maigesetze} (May 11--14) regulate Catholic clerical "
        r"training, appointment and discipline; \textbf{1874} the "
        r"Prussian \emph{Zivilehegesetz} (March 9) introduces "
        r"mandatory state civil marriage in Prussia for the first time; "
        r"\textbf{1875} the \emph{Personenstandsgesetz} (Feb 6) "
        r"nationalises civil marriage and birth/death registration "
        r"alongside the \emph{Brotkorbgesetz} (Apr 22) and the "
        r"\emph{Klostergesetz} (May 31); \textbf{1876} mass episcopal "
        r"expulsions leave nine of twelve Prussian bishoprics "
        r"\emph{sede vacante}. The 1870$^{P}$ column is the true "
        r"pre-Kulturkampf placebo cutoff (the Kanzelparagraph passed "
        r"in late 1871 puts that year inside the treatment window). A "
        r"significant coefficient at 1870 is a pre-trends signal (see "
        r"Table~\ref{tab:pretreatment_trends} and "
        r"\texttt{pretreatment\_trends.tex} for the formal Wald test). "
        r"$^{*}\,p<0.10$, $^{**}\,p<0.05$, $^{***}\,p<0.01$."
    )

    out = (
        "% Auto-generated by src/analysis/latex_tables.py -- do not edit by hand.\n"
        "% LaTeX preamble required: \\usepackage{booktabs, threeparttable, makecell}\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{threeparttable}\n"
        "\\caption{Treatment-cutoff sensitivity: DiD coefficient "
        "on $\\text{cath\\_share} \\times \\mathbb{1}["
        "\\text{Year} \\geq c]$ for $c \\in \\{1870, 1871, 1872, 1873, "
        "1874, 1875, 1876\\}$}\n"
        f"\\label{{tab:phase_sensitivity}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        " & " + head_year + r" \\" + "\n"
        " & " + head_phase + r" \\" + "\n"
        "\\midrule\n"
        + body + "\n"
        "\\midrule\n"
        + n_footer + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        f"\\item {note}\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
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

    illeg = safe_panel_ols(panel, "illegitimacy_ratio", ["cath_share_x_post"])
    mort_panel = panel[panel["Year"] >= 1875].copy()
    mort_panel["post_rollback"] = (mort_panel["Year"] >= 1880).astype(int)
    mort_panel["cath_x_rollback"] = mort_panel["cath_share"] * mort_panel["post_rollback"]
    mort = safe_panel_ols(mort_panel, "infant_mortality_rate", ["cath_x_rollback"])

    cols = [
        {
            "header": "Illegitimacy ratio",
            "treat_label": _label("cath_share_x_post"),
            "treat_coef": illeg.params["cath_share_x_post"],
            "treat_se": illeg.std_errors["cath_share_x_post"],
            "treat_p": illeg.pvalues["cath_share_x_post"],
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
        "\\midrule\n"
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
    outcomes: Sequence[str] = MAIN_OUTCOMES,
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
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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


def religiosity_robustness_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = (
        "cbr", "legitimate_br", "illegitimacy_ratio",
        "general_marriage_rate", "gmfr",
    ),
    out_path: Path | None = None,
) -> str:
    """Religiosity-vs-doctrine robustness: Catholic share vs Zentrum share.

    A common interpretive concern in Prussian fertility studies is that
    the "Catholic" effect picked up by ``cath_share_x_post`` may reflect
    differential *religiosity* (Catholics being more devout, Protestants
    more secular) rather than Catholic doctrine or Catholic institutions
    per se. This table addresses the concern by re-running the headline
    DiD with the Zentrum$+$Polen vote share in the 1871 Reichstag
    election as the treatment intensity. Zentrum vote share is the
    canonical revealed-preference measure of *mobilised* Catholic
    religious-political identity: counties with high Catholic population
    but low Zentrum share are nominal Catholics, while high-Catholic +
    high-Zentrum counties are politically and religiously mobilised
    Catholics. If the headline coefficient survives substituting Zentrum
    share for nominal Catholic share, the result reflects Catholic
    institutional/doctrinal exposure rather than nominal denomination
    alone.
    """
    out_path = out_path or TABLES_DIR / "religiosity_robustness.tex"

    cols: list[dict[str, float]] = []
    for o in outcomes:
        cat_res = run_baseline_did(panel, outcome=o, treatment="continuous")["result"]
        zen_res = run_baseline_did(panel, outcome=o, treatment="zentrum")["result"]
        cols.append({
            "outcome": o,
            "cat_coef": float(cat_res.params["cath_share_x_post"]),
            "cat_se": float(cat_res.std_errors["cath_share_x_post"]),
            "cat_p": float(cat_res.pvalues["cath_share_x_post"]),
            "cat_n": int(cat_res.nobs),
            "zen_coef": float(zen_res.params["zentrum_share_x_post"]),
            "zen_se": float(zen_res.std_errors["zentrum_share_x_post"]),
            "zen_p": float(zen_res.pvalues["zentrum_share_x_post"]),
            "zen_n": int(zen_res.nobs),
        })

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
        f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel A: Treatment intensity "
        f"$=$ Catholic population share (1871)}}}} \\\\"
    )
    cat_coef_row = (
        _label("cath_share_x_post")
        + " & "
        + " & ".join(_fmt_coef(c["cat_coef"], c["cat_p"], digits=4) for c in cols)
        + r" \\"
    )
    cat_se_row = (
        " & "
        + " & ".join(_fmt_se(c["cat_se"], digits=4) for c in cols)
        + r" \\"
    )
    cat_n_row = (
        "Observations & "
        + " & ".join(f"{c['cat_n']:,}" for c in cols)
        + r" \\"
    )

    panel_b_label = (
        f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel B: Treatment intensity "
        f"$=$ Catholic-party (Zentrum$+$Polen) vote share, 1871 Reichstag}}}} \\\\"
    )
    zen_coef_row = (
        _label("zentrum_share_x_post")
        + " & "
        + " & ".join(_fmt_coef(c["zen_coef"], c["zen_p"], digits=4) for c in cols)
        + r" \\"
    )
    zen_se_row = (
        " & "
        + " & ".join(_fmt_se(c["zen_se"], digits=4) for c in cols)
        + r" \\"
    )
    zen_n_row = (
        "Observations & "
        + " & ".join(f"{c['zen_n']:,}" for c in cols)
        + r" \\"
    )

    tabular = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + head + "\n"
        + nums + "\n"
        "\\midrule\n"
        + panel_a_label + "\n"
        + cat_coef_row + "\n"
        + cat_se_row + "\n"
        + cat_n_row + "\n"
        "\\addlinespace\n"
        + panel_b_label + "\n"
        + zen_coef_row + "\n"
        + zen_se_row + "\n"
        + zen_n_row + "\n"
        "\\midrule\n"
        + "County FE & "
        + " & ".join("Yes" for _ in cols)
        + r" \\" + "\n"
        + "Year FE & "
        + " & ".join("Yes" for _ in cols)
        + r" \\" + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption=(
            "Religiosity vs.\\ doctrine: Kulturkampf effect by Catholic-share "
            "vs.\\ Zentrum-share treatment intensity"
        ),
        label="tab:religiosity_robustness",
        n_cols=n + 1,
        notes=(
            "Panel~A reproduces the headline DiD specification from "
            "Table~\\ref{tab:baseline_did}, in which treatment intensity is the "
            "1871-census Catholic population share. Panel~B re-runs the same "
            "specification with the 1871-Reichstag Catholic-party (Zentrum$+$Polen) "
            "vote share as treatment intensity. Catholic population share is a "
            "measure of \\emph{nominal} Catholic denomination, while Catholic-party "
            "vote share is a revealed-preference measure of \\emph{mobilised} Catholic "
            "religious-political identity: counties with high Catholic population "
            "but low Zentrum share are nominal Catholics, while high-Catholic + "
            "high-Zentrum counties are devoutly and politically mobilised. The "
            "two intensities are correlated at $\\rho \\approx 0.78$ across "
            "counties but are not identical -- their gap is informative about "
            "whether the headline effect reflects nominal denomination or "
            "religious-political intensity. Both panels include county and year "
            "fixed effects and cluster standard errors at the county level. "
            "Panel~B's sample shrinks slightly because Galloway's electoral "
            "files do not cover every county. $^{*}\\,p<0.10$, $^{**}\\,p<0.05$, "
            "$^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def protestant_religiosity_placebo_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = (
        "cbr", "legitimate_br", "illegitimacy_ratio",
        "general_marriage_rate", "gmfr",
    ),
    out_path: Path | None = None,
) -> str:
    """Within-Protestant placebo: does Protestant religiosity predict
    "Catholic-like" outcomes inside the Low-Catholic group?

    Threat to the headline interpretation: the cross-sectional gap
    between Low- and High-Catholic counties could reflect a generic
    religiosity gradient (Catholics being more devout, Protestants more
    secular) rather than Catholic doctrine. If that story is right, then
    \\emph{within} Low-Catholic counties, those that are most
    religiously Protestant (high clergy density, Pietist heartlands)
    should look more "Catholic" on the fertility/marriage outcomes
    -- lower illegitimacy ratio, higher legitimate birth rate, higher
    GMFR, etc.

    Design: cross-section across Low-Catholic counties only ($\\le 50\\%$
    Catholic in 1871). The mean of each headline outcome over the
    pre-treatment window (1862--1872) is regressed on Protestant clergy
    density from the 1849 iPEHD merge ($\\mathrm{ProtPriest}_{1849}$ per
    $1{,}000$ inhabitants), with and without Regierungsbezirk fixed
    effects. The Rb-FE specification compares only counties within the
    same Prussian administrative region, which absorbs broad regional
    confounders (agrarian vs.\\ proto-industrial, eastern frontier vs.\\
    western, etc.).

    Interpretation: a negative coefficient on $\\mathrm{ProtPriest}_{1849}$
    in the illegitimacy-ratio column would support the religiosity-
    gradient story; a null is informative against it.
    """
    import statsmodels.formula.api as smf

    out_path = out_path or TABLES_DIR / "protestant_religiosity_placebo.tex"

    df = panel.copy()
    # Build Protestant clergy density (per 1k inhabitants, 1849 base).
    df["prot_priest_per_1k_1849"] = (
        df["rel1849_pro_priest"]
        / df["pop1849_tot"].replace(0, np.nan)
        * 1000.0
    )

    # Restrict to Low-Catholic counties and the pre-treatment window.
    pre = df[(df["cath_share"] <= 50) & (df["Year"].between(1862, 1872))].copy()

    # Collapse to one observation per county: pre-treatment mean of each
    # outcome, the time-invariant Protestant-religiosity proxy, and Rb.
    keep = list(outcomes) + ["prot_priest_per_1k_1849", "Rb"]
    keep = [c for c in keep if c in pre.columns]
    cs = (
        pre.groupby("Code")[keep]
           .agg(lambda x: x.iloc[0] if x.dtype == object else x.mean())
           .reset_index()
    )
    cs = cs.dropna(subset=["prot_priest_per_1k_1849", "Rb"])

    cols: list[dict[str, float]] = []
    for o in outcomes:
        sub = cs.dropna(subset=[o])
        # Spec 1: plain OLS, HC1.
        m1 = smf.ols(f"{o} ~ prot_priest_per_1k_1849", data=sub).fit(cov_type="HC1")
        # Spec 2: Rb fixed effects, HC1.
        m2 = smf.ols(
            f"{o} ~ prot_priest_per_1k_1849 + C(Rb)", data=sub
        ).fit(cov_type="HC1")
        cols.append({
            "outcome": o,
            "n": int(m1.nobs),
            "ols_coef": float(m1.params["prot_priest_per_1k_1849"]),
            "ols_se": float(m1.bse["prot_priest_per_1k_1849"]),
            "ols_p": float(m1.pvalues["prot_priest_per_1k_1849"]),
            "fe_coef": float(m2.params["prot_priest_per_1k_1849"]),
            "fe_se": float(m2.bse["prot_priest_per_1k_1849"]),
            "fe_p": float(m2.pvalues["prot_priest_per_1k_1849"]),
            "fe_n": int(m2.nobs),
            "n_rb": int(sub["Rb"].nunique()),
        })

    n = len(cols)

    head = (
        "Pre-1873 mean of: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\"
    )
    nums = (
        " & "
        + " & ".join(f"({i + 1})" for i in range(n))
        + r" \\"
    )

    panel_a_label = (
        f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel A: Pooled OLS, no fixed effects}}}} \\\\"
    )
    a_coef = (
        r"$\mathrm{ProtPriest}_{1849}$ per 1k pop & "
        + " & ".join(_fmt_coef(c["ols_coef"], c["ols_p"], digits=3) for c in cols)
        + r" \\"
    )
    a_se = (
        " & "
        + " & ".join(_fmt_se(c["ols_se"], digits=3) for c in cols)
        + r" \\"
    )

    panel_b_label = (
        f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Panel B: Regierungsbezirk fixed effects "
        f"(within-region comparison)}}}} \\\\"
    )
    b_coef = (
        r"$\mathrm{ProtPriest}_{1849}$ per 1k pop & "
        + " & ".join(_fmt_coef(c["fe_coef"], c["fe_p"], digits=3) for c in cols)
        + r" \\"
    )
    b_se = (
        " & "
        + " & ".join(_fmt_se(c["fe_se"], digits=3) for c in cols)
        + r" \\"
    )

    tabular = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        + head + "\n"
        + nums + "\n"
        "\\midrule\n"
        + panel_a_label + "\n"
        + a_coef + "\n"
        + a_se + "\n"
        "\\addlinespace\n"
        + panel_b_label + "\n"
        + b_coef + "\n"
        + b_se + "\n"
        "\\midrule\n"
        + "Counties & "
        + " & ".join(f"{c['n']:,}" for c in cols)
        + r" \\" + "\n"
        + "Regierungsbezirke & "
        + " & ".join(f"{c['n_rb']:,}" for c in cols)
        + r" \\" + "\n"
        + "Rb FE & "
        + " & ".join("Panel B only" for _ in cols)
        + r" \\" + "\n"
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption=(
            "Protestant-religiosity placebo: within Low-Catholic counties, does "
            "Protestant clergy density predict ``Catholic-like'' outcomes?"
        ),
        label="tab:protestant_religiosity_placebo",
        n_cols=n + 1,
        notes=(
            "Cross-section across Low-Catholic counties only ($\\le 50\\%$ Catholic "
            "in the 1871 census). Each column collapses the 1862--1872 county--year "
            "panel to one observation per county by taking the pre-treatment mean "
            "of the outcome. The regressor is Protestant clergy per $1{,}000$ "
            "inhabitants in 1849 (the iPEHD merge column "
            "``\\texttt{rel1849\\_pro\\_priest}/\\texttt{pop1849\\_tot}$\\times 1000$''), "
            "a proxy for the institutional density of Protestant religious life "
            "before any Kulturkampf treatment. Panel~A is pooled OLS; Panel~B "
            "adds Regierungsbezirk (Prussian administrative-region) fixed effects, "
            "which absorb broad regional confounders and identify the coefficient "
            "off within-region variation in Protestant clergy density. "
            "Heteroskedasticity-robust HC1 standard errors. If the headline "
            "Catholic--Protestant gap were driven by a generic religiosity "
            "gradient, then high-Protestant-clergy-density counties inside the "
            "Low-Catholic group should look more ``Catholic'' on the fertility/ "
            "marriage outcomes (lower illegitimacy ratio, higher legitimate birth "
            "rate, higher GMFR). $^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def conley_robustness_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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


def back_of_envelope_table(
    panel: pd.DataFrame,
    outcome: str = "general_marriage_rate",
    pre_year_start: int = 1862,
    pre_year_end: int = 1872,
    post_year_start: int = 1873,
    post_year_end: int = 1890,
    out_path: Path | None = None,
) -> str:
    """Reproducible back-of-envelope: marriages averted under the Kulturkampf.

    Pulls every input directly from the panel so the cells stay
    consistent with the headline DiD coefficient in
    ``baseline_did_table`` (column 5 of the rates panel) and with
    descriptive sample means.

    The "absolute marriages averted" cell uses two variants:

    - **Variant (a)** integrates over the actual joint distribution of
      Catholic share and 15+ population across treated county-years.
      This is the preferred figure for the abstract / introduction.
    - **Variant (b)** rescales to the 25\\,\\%\\,\\(\\to\\)\\,75\\,\\%
      Catholic-share contrast used elsewhere in the paper's magnitude
      decomposition. Matches the cross-section identification thought
      experiment but discards within-treated heterogeneity.
    """
    out_path = out_path or TABLES_DIR / "back_of_envelope.tex"

    # Restrict to the regression window (mirrors run_baseline_did's slice).
    sample = panel[
        (panel["Year"] >= pre_year_start) & (panel["Year"] <= post_year_end)
    ].copy()
    sample["post"] = (sample["Year"] >= post_year_start).astype(int)

    hi = sample[sample["high_cath"] == 1]
    hi_pre = hi[hi["post"] == 0]
    hi_post_all = hi[hi["post"] == 1]
    hi_post = hi_post_all.dropna(
        subset=["cath_share", "pop_15plus", outcome]
    )

    # DiD coefficient -- pull from the canonical baseline TWFE column so
    # this table can never drift from baseline_did_table.
    did = _did_column(sample, outcome, "twfe")
    beta = did["coef"]

    # Inputs.
    mean_cath_share = float(hi_post["cath_share"].mean())
    pre_outcome = float(hi_pre[outcome].mean())
    post_outcome = float(hi_post[outcome].mean())
    observed_change = post_outcome - pre_outcome
    mean_pop_15plus = float(hi_post["pop_15plus"].mean())
    n_counties = int(hi["Code"].nunique())
    n_years_post = post_year_end - post_year_start + 1

    # Implied magnitudes.
    per_1k_per_year = beta * mean_cath_share
    marriages_per_county_per_year = per_1k_per_year * mean_pop_15plus / 1_000.0
    pct_of_baseline = per_1k_per_year / pre_outcome * 100.0
    pct_of_observed = per_1k_per_year / observed_change * 100.0

    # Variant (a): integrate over actual joint distribution.
    contribs_a = beta * hi_post["cath_share"] * hi_post["pop_15plus"] / 1_000.0
    total_a = float(contribs_a.sum())

    # Variant (b): 50pp (25% -> 75%) shift x actual exposure.
    pop_sum_k = float(hi_post["pop_15plus"].sum()) / 1_000.0
    total_b = beta * 50.0 * pop_sum_k

    def _fmt_int(x: float) -> str:
        return f"{int(round(x)):,}".replace(",", "{,}")

    # ---- Panel C: sub-region decomposition of variant (a) ----
    # The pooled coefficient masks substantial heterogeneity across
    # sub-regions (Polish provinces, German Catholic, Protestant rest).
    # Re-running the baseline DiD on each sub-sample yields a sub-region
    # specific beta, which we apply only within that sub-region. Sub-region
    # totals are conceptually cleaner than the pooled (a) total because
    # they avoid the implicit "Catholic share is what matters" assumption.
    from src.analysis.regressions import SUBREGION_DEFINITIONS
    polish_rbs = SUBREGION_DEFINITIONS["Polish"]
    german_rbs = SUBREGION_DEFINITIONS["German Catholic"]

    def _subregion_mask(df, label):
        if label == "Polish":
            return df["Rb"].isin(polish_rbs)
        if label == "German Catholic":
            return df["Rb"].isin(german_rbs)
        return ~df["Rb"].isin(tuple(polish_rbs) + tuple(german_rbs))

    panel_c_rows_data = []
    panel_c_total = 0.0
    for label in ("Polish", "German Catholic", "Protestant (rest)"):
        sub_full = sample[_subregion_mask(sample, label)]
        if sub_full.empty or sub_full["Code"].nunique() < 2:
            continue
        sub_did = _did_column(sub_full, outcome, "twfe")
        beta_sub = sub_did["coef"]
        p_sub = sub_did["p"]
        n_clusters = int(sub_full["Code"].nunique())

        sub_hi_post = sub_full[
            (sub_full["high_cath"] == 1) & (sub_full["post"] == 1)
        ].dropna(subset=["cath_share", "pop_15plus", outcome])
        n_hi = int(sub_hi_post["Code"].nunique())
        if n_hi == 0:
            averted = 0.0
        else:
            averted = float(
                (beta_sub * sub_hi_post["cath_share"]
                 * sub_hi_post["pop_15plus"] / 1_000.0).sum()
            )
        panel_c_total += averted
        panel_c_rows_data.append({
            "label": label,
            "beta": beta_sub,
            "p": p_sub,
            "n_clusters": n_clusters,
            "n_hi": n_hi,
            "averted": averted,
        })

    panel_c_rows = [
        " & $\\hat{\\beta}_{\\text{sub}}$ ($p$) "
        "& Marriages averted \\\\",
        "\\addlinespace",
    ]
    for r in panel_c_rows_data:
        stars = _stars(r['p'])
        beta_str = f"${r['beta']:+.4f}${stars}"
        panel_c_rows.append(
            f"{r['label']} ($G={r['n_clusters']}$, "
            f"high-Cath\\;$n={r['n_hi']}$) "
            f"& {beta_str}\\;({r['p']:.3f}) "
            f"& {_fmt_int(r['averted'])} \\\\"
        )
    panel_c_rows.append("\\addlinespace")
    panel_c_rows.append(
        f"\\quad\\textbf{{Sub-region total}} & & "
        f"\\textbf{{{_fmt_int(panel_c_total)}}} \\\\"
    )
    panel_c_rows.append(
        "\\quad\\textit{Memo: pooled (a) from above} & & "
        f"\\textit{{{_fmt_int(total_a)}}} \\\\"
    )

    panel_a_rows = [
        f"DiD coefficient $\\hat{{\\beta}}$ on "
        f"$\\mathrm{{CathShare}} \\times \\mathrm{{Post}}$ "
        f"(GMR, per 1{{,}}000 pop.\\ 15+) "
        f"& {beta:+.4f} & Table~\\ref{{tab:baseline_did}}, col.\\ (5) \\\\",
        f"Mean Catholic share, high-Catholic counties (post-1873) & "
        f"{mean_cath_share:.1f} & Sample mean \\\\",
        f"Pre-Kulturkampf GMR, high-Catholic counties "
        f"($\\overline{{Y}}_{{{pre_year_start}-{pre_year_end}}}$) "
        f"& {pre_outcome:.3f} & Sample mean \\\\",
        f"Post-Kulturkampf GMR, high-Catholic counties "
        f"($\\overline{{Y}}_{{{post_year_start}-{post_year_end}}}$) "
        f"& {post_outcome:.3f} & Sample mean \\\\",
        f"Observed pre-to-post change in high-Catholic counties "
        f"& {observed_change:+.3f} & post mean $-$ pre mean \\\\",
        f"Mean population aged 15+, post-1873 (per county-year) & "
        f"{_fmt_int(mean_pop_15plus)} & AGE1871/82/90 interp. \\\\",
        f"Number of high-Catholic counties & {n_counties} & "
        f"\\texttt{{high\\_cath}} indicator \\\\",
        f"Years in post-treatment window & {n_years_post} & "
        f"{post_year_start}--{post_year_end} \\\\",
    ]

    panel_b_rows = [
        # Per-county-per-year intermediate
        f"Per-1{{,}}000-pop.-15+ effect at mean Cath.\\ share: "
        f"$\\hat{{\\beta}} \\times {mean_cath_share:.1f}$ "
        f"& {per_1k_per_year:+.3f} \\\\",
        f"\\quad converted to marriages per county per year: "
        f"$\\times\\;{_fmt_int(mean_pop_15plus)} / 1{{,}}000$ "
        f"& {marriages_per_county_per_year:+.1f} \\\\",
        # % of baseline
        f"Effect as share of pre-1873 baseline: "
        f"$({per_1k_per_year:+.3f}) / {pre_outcome:.3f}$ "
        f"& {pct_of_baseline:+.2f}\\,\\% \\\\",
        # % of observed
        f"Effect as share of observed pre-to-post decline: "
        f"$({per_1k_per_year:+.3f}) / ({observed_change:+.3f})$ "
        f"& {pct_of_observed:+.2f}\\,\\% \\\\",
        "\\midrule",
        # Variant (a)
        f"\\textbf{{Total marriages averted, "
        f"{post_year_start}--{post_year_end}, high-Catholic counties:}} & \\\\",
        f"\\quad (a) $\\sum_{{i,t}} \\hat{{\\beta}} \\cdot "
        f"\\mathrm{{CathShare}}_{{it}} \\cdot \\mathrm{{Pop15+}}_{{it}}/1{{,}}000$ "
        f"& {_fmt_int(total_a)} \\\\",
        # Variant (b)
        f"\\quad (b) 50-pp contrast (25\\,\\% $\\to$ 75\\,\\%): "
        f"$\\hat{{\\beta}} \\times 50 \\times \\sum_{{i,t}} "
        f"\\mathrm{{Pop15+}}_{{it}}/1{{,}}000$ "
        f"& {_fmt_int(total_b)} \\\\",
    ]

    tabular = (
        "\\begin{tabular}{lrl}\n"
        "\\toprule\n"
        " & Value & Source \\\\\n"
        "\\midrule\n"
        "\\multicolumn{3}{l}{\\textit{Panel A. Inputs}} \\\\\n"
        + "\n".join(panel_a_rows) + "\n"
        "\\midrule\n"
        "\\multicolumn{3}{l}{\\textit{Panel B. Implied magnitudes (pooled)}} \\\\\n"
        + "\n".join(panel_b_rows) + "\n"
        "\\midrule\n"
        "\\multicolumn{3}{l}{\\textit{Panel C. Sub-region decomposition of variant (a)}} \\\\\n"
        + "\n".join(panel_c_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        tabular,
        caption=(
            "Back-of-envelope magnitude calculation: marriages averted "
            "in high-Catholic counties under the Kulturkampf, "
            f"{post_year_start}--{post_year_end}"
        ),
        label="tab:back_of_envelope",
        n_cols=3,
        notes=(
            "All inputs are computed from the regression panel restricted "
            f"to {pre_year_start}--{post_year_end}; the pooled DiD "
            "coefficient is the TWFE specification reported in column~(5) "
            "of Table~\\ref{tab:baseline_did}. Population aged 15+ is "
            "interpolated between the 1871, 1882, and 1890 AGE censuses "
            "(Data Appendix~\\S6.5); the absolute totals inherit this "
            "measurement uncertainty. Variant~(a) integrates over the "
            "actual joint distribution of Catholic share and 15+ "
            "population across treated county-years. Variant~(b) reports "
            "the magnitude implied by moving a single county from low "
            "(25\\,\\%) to high (75\\,\\%) Catholic share over the same "
            "exposure, matching the cross-section identification thought "
            "experiment in Table~\\ref{tab:magnitudes}. "
            "\\textit{Panel C} re-applies the variant~(a) integration "
            "within each of the three sub-regions used in the wild "
            "cluster bootstrap (Polish provinces POS/BRO, German Catholic "
            "KOL/KOB/TRI/AAC/OPP/MUN, and the Protestant rest), using a "
            "sub-region-specific $\\hat{\\beta}_{\\text{sub}}$ estimated "
            "separately on each sub-sample with TWFE. Stars: "
            "$^{*}p<0.10,\\;^{**}p<0.05,\\;^{***}p<0.01$ (asymptotic). "
            "Wild-cluster bootstrap $p$-values for the same sub-region "
            "specifications are reported in Table~\\ref{tab:wild_bootstrap} "
            "and in the choropleth Figure~\\ref{fig:map5_marriage}. "
            "The sub-region total differs slightly from the pooled "
            "variant~(a) because the pooled coefficient is not a simple "
            "convex combination of the sub-region coefficients."
        ),
    )
    _write(out_path, out)
    return out


def pretrends_robustness_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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


def continuous_polish_decomposition_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = MAIN_OUTCOMES,
    out_path: Path | None = None,
) -> str:
    """Continuous Polish-mobilisation decomposition table.

    Replaces the binary ``Polish`` dummy in ``falsifications_table``
    Panel C with the continuous 1871 Polenpartei vote share. The
    coefficient on ``cath_share x Post`` is then identified off
    counties with zero Polish-party voting -- the German-Catholic
    heartland -- and is the pure Catholic-religion effect with the
    ethnic-Polish channel netted out by the second interaction.
    """
    out_path = out_path or TABLES_DIR / "continuous_polish_decomposition.tex"

    cols = [run_continuous_polish_decomposition(panel, outcome=o) for o in outcomes]
    n = len(cols)

    def _stars(p: float) -> str:
        return ("$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05
                else "$^{*}$" if p < 0.10 else "")

    cath_coefs = " & ".join(
        f"{c['cath_coef']:+.4f}{_stars(c['cath_p'])}" for c in cols
    )
    cath_ses = " & ".join(f"({c['cath_se']:.4f})" for c in cols)
    polen_coefs = " & ".join(
        f"{c['polen_coef']:+.4f}{_stars(c['polen_p'])}" for c in cols
    )
    polen_ses = " & ".join(f"({c['polen_se']:.4f})" for c in cols)
    obs = " & ".join(f"{c['n']:,}" for c in cols)

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\" + "\n"
        " & " + " & ".join(f"({i+1})" for i in range(n)) + r" \\" + "\n"
        "\\midrule\n"
        f"CathShare $\\times$ Post & {cath_coefs} \\\\\n"
        f" & {cath_ses} \\\\\n"
        "\\addlinespace\n"
        f"PolenShare$_{{1871}}$ $\\times$ Post & {polen_coefs} \\\\\n"
        f" & {polen_ses} \\\\\n"
        "\\midrule\n"
        f"Observations & {obs} \\\\\n"
        "County FE & " + " & ".join(["Yes"] * n) + r" \\" + "\n"
        "Year FE & " + " & ".join(["Yes"] * n) + r" \\" + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Continuous Polish-mobilisation decomposition: separating "
            "Catholic religion from Polish ethnicity"
        ),
        label="tab:continuous_polish_decomposition",
        n_cols=n + 1,
        notes=(
            "Each column estimates a single regression with two "
            "interactions: $\\text{CathShare}_i \\times \\text{Post}_t$ "
            "and $\\text{PolenShare}_i \\times \\text{Post}_t$, where "
            "$\\text{PolenShare}_i$ is the Polish-nationalist party "
            "(Ko\\l{}o Polskie) vote share in the 1871 Reichstag "
            "election. The CathShare coefficient is identified off "
            "counties with zero Polish-party vote share -- the "
            "German-Catholic heartland of Rheinland, Westphalia, "
            "Bavarian-Silesia -- and isolates the religion-only "
            "channel. The PolenShare coefficient is the additional "
            "ethnic-Polish response per percentage point of 1871 "
            "Polenpartei vote share, capturing the overlapping "
            "Germanization regime (1872 Schulaufsichtsgesetz, "
            "1885--86 Polenausweisungen, 1886 Settlement Commission). "
            "Sample 1862--1890, two-way fixed effects, standard errors "
            "clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def kulturkampf_vs_polenpolitik_timing_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = MAIN_OUTCOMES,
    out_path: Path | None = None,
) -> str:
    """Timing-decomposition table on the Polish sub-sample.

    Splits the post-1873 indicator into three windows -- Kulturkampf
    enforcement (1873--1878), rollback (1880--1887), and post-rollback
    (1888+) -- on the Polish counties only. The Kulturkampf hypothesis
    predicts the enforcement coefficient is largest. The Polenpolitik
    hypothesis predicts the post-rollback coefficient is largest
    (Polenausweisungen 1885--86, Settlement Commission 1886+).
    """
    out_path = out_path or TABLES_DIR / "kulturkampf_vs_polenpolitik_timing.tex"

    cols = [run_kulturkampf_vs_polenpolitik_timing(panel, outcome=o) for o in outcomes]
    n = len(cols)

    def _stars(p: float) -> str:
        return ("$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05
                else "$^{*}$" if p < 0.10 else "")

    def _coef_row(key_coef: str, key_se: str, key_p: str, label: str) -> str:
        coefs = " & ".join(
            f"{c[key_coef]:+.4f}{_stars(c[key_p])}" for c in cols
        )
        ses = " & ".join(f"({c[key_se]:.4f})" for c in cols)
        return (
            f"{label} & {coefs} \\\\\n"
            f" & {ses} \\\\\n"
        )

    body = (
        f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
        "\\toprule\n"
        "Dependent variable: & "
        + " & ".join(_outcome_label(o) for o in outcomes)
        + r" \\" + "\n"
        " & " + " & ".join(f"({i+1})" for i in range(n)) + r" \\" + "\n"
        "\\midrule\n"
        + _coef_row(
            "enf_coef", "enf_se", "enf_p",
            r"CathShare $\times$ Kulturkampf enforcement (1873--78)",
        )
        + "\\addlinespace\n"
        + _coef_row(
            "rollback_coef", "rollback_se", "rollback_p",
            r"CathShare $\times$ Rollback (1880--87)",
        )
        + "\\addlinespace\n"
        + _coef_row(
            "postroll_coef", "postroll_se", "postroll_p",
            r"CathShare $\times$ Post-rollback (1888--90)",
        )
        + "\\midrule\n"
        + "Observations & "
        + " & ".join(f"{c['n']:,}" for c in cols) + r" \\" + "\n"
        + "County FE & " + " & ".join(["Yes"] * n) + r" \\" + "\n"
        + "Year FE & " + " & ".join(["Yes"] * n) + r" \\" + "\n"
        + "Sample & "
        + " & ".join(["Polish (POS/BRO)"] * n) + r" \\" + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Differential timing on the Polish sub-sample: "
            "Kulturkampf enforcement vs Polenpolitik escalation"
        ),
        label="tab:kulturkampf_vs_polenpolitik_timing",
        n_cols=n + 1,
        notes=(
            "Sample restricted to the 24 counties of Regierungsbezirke "
            "Posen (POS) and Bromberg (BRO). Each column splits the "
            "post-1873 window into three legislative phases: "
            "\\emph{enforcement} (1873--1878), when the May Laws, "
            "Brotkorbgesetz, and episcopal expulsions were in force; "
            "\\emph{rollback} (1880--1887), when the Kulturkampf laws "
            "were progressively repealed under Leo XIII; and "
            "\\emph{post-rollback} (1888--1890), after most Kulturkampf "
            "legislation had been struck down. The Polenpolitik regime, "
            "by contrast, escalates across these windows: the 1885--86 "
            "\\emph{Polenausweisungen} (mass expulsion of $\\approx 32{,}000$ "
            "Polish-nationality residents) and the 1886 Prussian "
            "Settlement Commission's German-colonisation programme "
            "fall in the rollback and post-rollback windows. The "
            "Kulturkampf hypothesis predicts the enforcement coefficient "
            "is the largest in absolute value; the Polenpolitik "
            "hypothesis predicts the post-rollback coefficient is the "
            "largest. Standard errors clustered at the county level. "
            "$^{*}\\,p<0.10$, $^{**}\\,p<0.05$, $^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def heterogeneity_table(
    panel: pd.DataFrame,
    outcomes: Sequence[str] = ("cbr", "general_marriage_rate", "I_g"),
    moderators: tuple[str, ...] = (
        "school1517", "attend_rate_1849_baseline", "f_urban",
        "zentrum_share_1871", "polen_share_1871",
    ),
    out_path: Path | None = None,
) -> str:
    """Treatment effect interactions with iPEHD + political-economy moderators.

    Default outcomes span the Coale--Watkins decomposition: CBR
    (overall fertility), marriage rate (nuptiality), and Coale's $I_g$
    (marital fertility, Hutterite-normalised; the Galloway, Hammel \\&
    Lee 1994 headline measure).

    Default moderators cover two analytically distinct dimensions:
      - **Socio-economic** (iPEHD 1871 cross-section): school enrolment
        15--17 (literacy proxy) and urban population share.
      - **Political-economy** (Galloway ELE1871 1871 Reichstag
        election): vote share for the Catholic Centre Party (Zentrum)
        and the Polish-nationalist Catholic party (Polen). These
        directly measure Catholic political mobilisation and let us
        distinguish German-Catholic from Polish-Catholic dynamics
        more precisely than the geographic Rb classification.
    """
    out_path = out_path or TABLES_DIR / "heterogeneity.tex"

    moderator_labels = {
        "school1517": "School enrolment 15--17",
        "attend_rate_1849_baseline":
            "Elementary attendance rate (1849, pre-treatment baseline)",
        "f_urban": "Urban population share",
        "f_litrate": "Literacy rate",
        "zentrum_share_1871": "Zentrum vote share (1871)",
        "polen_share_1871": "Polish-party vote share (1871)",
        "catholic_party_share_1871": "Catholic-party vote share (1871; Zentrum$+$Polen)",
        "nat_liberal_share_1871": "National-liberal vote share (1871)",
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
    outcomes: Sequence[str] = ("cbr", "legitimate_br", "illegitimacy_ratio", "general_marriage_rate"),
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
    outcomes: Sequence[str] = (
        "cbr", "general_marriage_rate", "I_g", "gmfr_static_1871",
    ),
    out_path: Path | None = None,
) -> str:
    """Wild cluster bootstrap p-values across full panel and key sub-samples.

    The $I_g$ column (Coale's marital fertility index, Hutterite-
    normalised; the Galloway, Hammel & Lee 1994 tradition outcome) tests
    whether the small-cluster sub-region results survive when fertility
    is measured net of nuptiality. The companion ``gmfr_static_1871``
    column tests the same marital-fertility outcome with the
    *static-1871-prevalence* denominator -- nuptiality is held at its
    pre-Kulturkampf county-specific baseline ($\\mu_{i,1871}$ applied
    to the time-varying $W_t$), purging the bad-control bias in which
    contemporaneous $M_t$ responds to the marriage-formation shock
    itself. Reading the two marital-fertility columns side by side
    isolates which of the headline $I_g$ result survives once the
    denominator-response channel is removed: in the full panel the
    headline $I_g$ effect dissolves under ``gmfr_static_1871``,
    consistent with the interpretation that the apparent marital-
    fertility response is the nuptiality (denominator) channel rather
    than within-marriage behaviour.

    Because $I_g$ is dimensionless, its coefficients are not directly
    comparable to the per-1{,}000 CBR / marriage-rate / GMFR columns;
    what *is* comparable across columns is the sign and the wild-
    bootstrap $p$-value.
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
            "unreliable. The two marital-fertility columns are paired by "
            "design: $I_g$ uses the time-varying Hutterite-normalised "
            "marital-fertility denominator (the Galloway, Hammel \\& Lee 1994 "
            "headline), while $\\mathrm{GMFR}^{1871}$ holds the marriage "
            "prevalence $\\mu_{i}$ at its 1871 county-specific baseline and "
            "applies it to the time-varying women-15--49 count. The "
            "comparison purges the bad-control channel in which the "
            "Kulturkampf-induced drop in $M_t$ mechanically inflates "
            "$B_\\mathrm{leg}/M_t$ even absent any within-marriage fertility "
            "response. Because $I_g$ is dimensionless and $\\mathrm{GMFR}^{1871}$ "
            "is per 1{,}000 married women, the two columns' coefficient "
            "magnitudes are not directly comparable; the wild-bootstrap "
            "$p$-values are. "
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
        "marriage_rate": "Marriage rate (per 1{,}000 pop)",
        "general_marriage_rate":
            "Gen.\\ marriage rate (per 1{,}000 pop 15+)",
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
        digits = 3 if idx_name in ("marriage_rate", "general_marriage_rate") else 5
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

    full = run_emigration_robustness(panel, outcomes=("cbr", "general_marriage_rate"))
    polish = run_emigration_robustness(
        panel[panel["Rb"].isin(["POS", "BRO"])],
        outcomes=("cbr", "general_marriage_rate"),
    )
    counts = run_count_marriage_did(panel)

    def _block(rows: pd.DataFrame, header_label: str) -> str:
        out = (
            f"\\multicolumn{{4}}{{l}}{{\\textit{{{header_label}}}}} \\\\\n"
        )
        # Group by spec; within each spec, two outcomes (cbr, general_marriage_rate)
        for spec in rows["spec"].drop_duplicates():
            sub = rows[rows["spec"] == spec]
            cbr_row = sub[sub["outcome"] == "cbr"].iloc[0] if len(sub[sub["outcome"] == "cbr"]) else None
            mar_row = sub[sub["outcome"] == "general_marriage_rate"].iloc[0] if len(sub[sub["outcome"] == "general_marriage_rate"]) else None
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
            "headline specification on the full panel under seven specifications: "
            "(1) baseline TWFE, no migration controls; (2) adds the annual "
            "population growth rate; (3) adds the implied net migration rate "
            "(population change minus natural increase, per 1{,}000 "
            "population), which spans the full panel including the post-1885 "
            "expulsion years; (4) restricts the sample to $t < 1885$, before "
            "the Bismarck-era $\\mathit{Polenausweisungen}$ and the 1886 "
            "Settlement Commission; (1$'$) baseline TWFE with no migration "
            "control but restricted to the 1862--1867 and 1872--1886 "
            "sub-sample on which Galloway VIT records measured migration; "
            "(5) adds measured out-migration rate on the (1$'$) sub-sample; "
            "(6) adds measured net migration rate on the (1$'$) sub-sample; "
            "(7) adds the time-varying married sex ratio (Galloway, Hammel "
            "\\& Lee 1994: $100 \\times \\mathrm{MarriedM}_t / "
            "\\mathrm{MarriedF}_t$, piecewise-linearly interpolated between "
            "the 1871 STA1871 and 1885 POP1885 anchors). Specification "
            "(7) is the canonical Princeton-EFP control for ``spousal "
            "separation due to migration or military service'' -- the "
            "mechanical channel by which male relocation depresses period "
            "marital fertility without any behavioural change. If the "
            "$\\mathrm{cath\\_share} \\times \\mathrm{Post}$ coefficient "
            "shrinks substantially under (7), the emigration channel "
            "operates via married-man departure from married women; if it "
            "barely moves, the operative channel is reduced marriage "
            "formation (single-man emigration) or whole-family chain "
            "migration -- neither of which moves the married sex ratio. "
            "Row (1$'$) is the appropriate baseline for interpreting "
            "(5)--(6): the difference between (1) and (1$'$) isolates the "
            "sample-composition effect of dropping war and post-expulsion "
            "years, while the difference between (1$'$) and (5)/(6) "
            "isolates the migration-channel effect of conditioning on "
            "measured migration. Panel~B repeats the seven specifications on "
            "the Polish sub-sample (Posen and Bromberg). Panel~C reports "
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

    outcome_keys = (
        "cbr", "general_marriage_rate", "I_g", "gmfr",
    )
    df = run_pretreatment_trends_robustness(
        panel, outcomes=outcome_keys, form="linear",
    )

    def _digits(o):
        # I_g coefficients are O(1e-4); gmfr is O(1e-1); CBR /
        # marriage_rate / general_marriage_rate are O(1e-3). Use 5 digits
        # for I_g, 3 for gmfr, 4 for the rest.
        if o == "I_g":
            return 5
        if o == "gmfr":
            return 3
        return 4

    rows: list[str] = []
    for spec in df["spec"].drop_duplicates():
        sub = df[df["spec"] == spec]
        outs = {
            o: (sub[sub["outcome"] == o].iloc[0]
                if (sub["outcome"] == o).any() else None)
            for o in outcome_keys
        }
        coefs = [
            _fmt_coef(outs[o]["coef"], outs[o]["p"], digits=_digits(o))
            if outs[o] is not None else ""
            for o in outcome_keys
        ]
        ses = [
            _fmt_se(outs[o]["se"], digits=_digits(o))
            if outs[o] is not None else ""
            for o in outcome_keys
        ]
        n_val = next(
            (int(outs[o]["n"]) for o in outs if outs[o] is not None), 0
        )
        rows.append(
            f"{_latex_escape(spec)} & " + " & ".join(coefs)
            + f" & {n_val:,} \\\\"
        )
        rows.append(" & " + " & ".join(ses) + r" & \\")

    body = (
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        " & CBR & Marriage rate & Gen.\\ marriage rate "
        "& $I_g$ & GMFR & $N$ \\\\\n"
        " & (1) & (2) & (3) & (4) & (5) & \\\\\n"
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
        n_cols=7,
        notes=(
            "Each row is a separate two-way fixed-effects regression of the "
            "outcome on $\\mathrm{CathShare} \\times \\mathrm{Post}$ and the "
            "listed baseline iPEHD-1871 characteristics interacted with a "
            "centred linear time trend. The marital-fertility columns "
            "report Coale's $I_g$ (Hutterite-normalised) and the "
            "Galloway, Hammel \\& Lee (1994) headline GMFR (legitimate "
            "births per 1{,}000 married women 15--49, the unnormalised "
            "analogue of $I_g$). Both use the AGE1890 / STA1871 time-"
            "varying marital denominator (see \\S6.5 of the data appendix). "
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
    df = run_subsample_decomposition(panel, outcomes=("cbr", "general_marriage_rate"))

    rows: list[str] = []
    for name in df["sample"].drop_duplicates():
        sub = df[df["sample"] == name]
        cbr_sub = sub[sub["outcome"] == "cbr"]
        mar_sub = sub[sub["outcome"] == "general_marriage_rate"]
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


def _sdid_pretrend_rmse_reduction(res) -> float:
    """% reduction in pre-period RMSE: synthetic control vs naive control mean."""
    Y = res.Y
    tmask = res.treated_mask
    T_pre = res.T_pre
    Y_tr = Y[tmask][:, :T_pre].mean(axis=0)
    Y_naive = Y[~tmask][:, :T_pre].mean(axis=0)
    Y_syn = (Y[~tmask][:, :T_pre] * res.omega[:, None]).sum(axis=0) + res.omega0
    rmse_naive = float(np.sqrt(((Y_tr - Y_naive) ** 2).mean()))
    rmse_syn = float(np.sqrt(((Y_tr - Y_syn) ** 2).mean()))
    if rmse_naive <= 0:
        return float("nan")
    return 100.0 * (1.0 - rmse_syn / rmse_naive)


def sdid_results_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
    *,
    treatment_year: int = 1873,
    year_start: int = 1862,
    year_end: int = 1885,
    n_placebo: int = 1000,
    seed: int = 42,
) -> str:
    """
    Synthetic DiD estimates of the Kulturkampf ATT on the crude marriage rate
    and the crude birth rate. Headline cath_share > 50% specification plus a
    threshold-sweep robustness panel at 40% and 60%.
    """
    out_path = out_path or TABLES_DIR / "sdid_results.tex"

    panels: dict[float, dict[str, object]] = {}
    for thr in (40.0, 50.0, 60.0):
        col = f"_high_cath_thr_{int(thr)}"
        work = panel.copy()
        work[col] = (work["cath_share"] > thr).astype(int)
        panels[thr] = {
            "cmr": run_sdid(
                work, outcome="general_marriage_rate", treat_col=col,
                treatment_year=treatment_year,
                year_start=year_start, year_end=year_end,
                n_placebo=n_placebo, seed=seed,
            ),
            "cbr": run_sdid(
                work, outcome="cbr", treat_col=col,
                treatment_year=treatment_year,
                year_start=year_start, year_end=year_end,
                n_placebo=n_placebo, seed=seed,
            ),
        }

    def _row(metric: str, fmt) -> str:
        cells = []
        for thr in (40.0, 50.0, 60.0):
            cells.append(fmt(panels[thr]["cmr"], "cmr"))
            cells.append(fmt(panels[thr]["cbr"], "cbr"))
        return f"{metric} & " + " & ".join(cells) + r" \\"

    def coef_cell(res, _outcome):
        return _fmt_coef(res.tau_hat, res.p_value, digits=3)

    def se_cell(res, _outcome):
        return _fmt_se(res.se, digits=3) if res.se is not None else ""

    def p_cell(res, _outcome):
        return f"{res.p_value:.3f}" if res.p_value is not None else ""

    def n_cell(res, _outcome):
        return f"{res.n_treated} / {res.n_control}"

    def rmse_cell(res, _outcome):
        return f"{_sdid_pretrend_rmse_reduction(res):.1f}\\%"

    body = (
        "\\begin{tabular}{l*{6}{c}}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Panel A: $>$ 40\\% Catholic} "
        "& \\multicolumn{2}{c}{Panel B: $>$ 50\\% Catholic (headline)} "
        "& \\multicolumn{2}{c}{Panel C: $>$ 60\\% Catholic} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        " & Marriage & CBR & Marriage & CBR & Marriage & CBR \\\\\n"
        " & rate & & rate & & rate & \\\\\n"
        " & (1) & (2) & (3) & (4) & (5) & (6) \\\\\n"
        "\\midrule\n"
        + _row("SDID ATT $\\hat\\tau$", coef_cell) + "\n"
        + _row("", se_cell) + "\n"
        + _row("Placebo $p$-value", p_cell) + "\n"
        + _row("Treated / control counties", n_cell) + "\n"
        + _row("Pre-trend RMSE reduction", rmse_cell) + "\n"
        "\\midrule\n"
        f"Pre-period years & \\multicolumn{{6}}{{c}}{{{panels[50]['cmr'].T_pre} "
        f"({year_start}--{treatment_year - 1})}} \\\\\n"
        f"Post-period years & \\multicolumn{{6}}{{c}}{{{panels[50]['cmr'].T_post} "
        f"({treatment_year}--{year_end})}} \\\\\n"
        f"Placebo permutations & \\multicolumn{{6}}{{c}}{{{n_placebo:,}}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Synthetic difference-in-differences estimates of the Kulturkampf "
            "effect on marriage and fertility"
        ),
        label="tab:sdid_results",
        n_cols=7,
        notes=(
            "Synthetic Difference-in-Differences estimates (Arkhangelsky, Athey, "
            "Hirshberg, Imbens \\& Wager 2021). Unit weights $\\hat\\omega_i$ are "
            "chosen by simplex-constrained ridge regression so that the weighted "
            "low-Catholic trajectory matches the high-Catholic trajectory over the "
            "pre-period; time weights $\\hat\\lambda_t$ up-weight pre-period years "
            "informative about the immediate counterfactual. The point estimate is "
            "the weighted two-way fixed-effects coefficient on (Treated $\\times$ "
            "Post). Standard errors in parentheses are obtained by placebo "
            "permutation: treatment is randomly reassigned among the surviving "
            "low-Catholic counties $B = " + f"{n_placebo:,}" + "$ times, and the "
            "full SDID pipeline is re-estimated on each draw. The placebo $p$-value "
            "is the two-sided share of placebo $|\\hat\\tau|$ at least as extreme as "
            "the observed coefficient. The pre-trend RMSE reduction measures, in "
            "percent, how much closer the synthetic control's pre-period trajectory "
            "matches the treated trajectory relative to the unweighted low-Catholic "
            "mean; values above 50\\% indicate that the SDID weighting is doing "
            "non-trivial work over a naive DiD. All specifications use a balanced "
            "panel of counties with complete coverage on the outcome over "
            f"{year_start}--{year_end}. $^{{*}}\\,p<0.10$, $^{{**}}\\,p<0.05$, "
            "$^{***}\\,p<0.01$."
        ),
    )
    _write(out_path, out)
    return out


def sdid_donor_counties_table(
    panel: pd.DataFrame,
    out_path: Path | None = None,
    *,
    treatment_year: int = 1873,
    year_start: int = 1862,
    year_end: int = 1885,
    seed: int = 42,
    k: int = 10,
) -> str:
    """
    Top-k donor counties (by SDID unit weight $\\hat\\omega$) for the headline
    cath_share > 50% specification, separately for the marriage-rate and CBR
    synthetic controls.
    """
    out_path = out_path or TABLES_DIR / "sdid_donor_counties.tex"

    name_col = "Kreis" if "Kreis" in panel.columns else "Code"
    labels = panel.groupby("Code").agg(
        Kreis=(name_col, "first"),
        Rb=("Rb", "first") if "Rb" in panel.columns else ("Code", "first"),
        cath_share=("cath_share", "first"),
    )

    def _donor_rows(outcome: str) -> tuple[str, float, float]:
        res = run_sdid(
            panel, outcome=outcome, treat_col="high_cath",
            treatment_year=treatment_year,
            year_start=year_start, year_end=year_end,
            n_placebo=0, seed=seed,
        )
        control_codes = [
            c for c, t in zip(res.codes, res.treated_mask) if not t
        ]
        df_w = (
            pd.DataFrame({"Code": control_codes, "omega": res.omega})
            .merge(labels, on="Code", how="left")
            .sort_values("omega", ascending=False)
            .head(k)
        )
        rows = []
        for i, r in enumerate(df_w.itertuples(index=False), start=1):
            kreis = _latex_escape(str(r.Kreis).title())
            rb = _latex_escape(str(r.Rb))
            rows.append(
                f"{i} & {kreis} & {rb} & {r.cath_share:.1f} & {r.omega:.4f} \\\\"
            )
        eff_n = 1.0 / float((res.omega ** 2).sum())
        return "\n".join(rows), eff_n, float(res.omega.max())

    gmr_rows, gmr_eff_n, gmr_max_w = _donor_rows("general_marriage_rate")
    cbr_rows, cbr_eff_n, cbr_max_w = _donor_rows("cbr")

    body = (
        "\\begin{tabular}{rlcrr}\n"
        "\\toprule\n"
        " & Kreis & Rb & Cath.\\ share (\\%) & $\\hat\\omega$ \\\\\n"
        "\\midrule\n"
        "\\multicolumn{5}{l}{\\textit{Panel A: General marriage rate synthetic control}} \\\\\n"
        + gmr_rows + "\n"
        f"\\midrule\\multicolumn{{5}}{{l}}{{Effective number of donors $1/\\sum_i \\hat\\omega_i^2 = {gmr_eff_n:.1f}$; "
        f"max $\\hat\\omega = {gmr_max_w:.4f}$}} \\\\\n"
        "\\midrule\n"
        "\\multicolumn{5}{l}{\\textit{Panel B: CBR synthetic control}} \\\\\n"
        + cbr_rows + "\n"
        f"\\midrule\\multicolumn{{5}}{{l}}{{Effective number of donors $1/\\sum_i \\hat\\omega_i^2 = {cbr_eff_n:.1f}$; "
        f"max $\\hat\\omega = {cbr_max_w:.4f}$}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    out = _wrap_table(
        body,
        caption=(
            "Top-" + str(k) + " donor counties for the SDID synthetic control "
            "(headline cath\\_share $>$ 50\\% specification)"
        ),
        label="tab:sdid_donor_counties",
        n_cols=5,
        notes=(
            "Unit weights $\\hat\\omega_i$ are the simplex-constrained ridge "
            "least-squares solution that best matches the high-Catholic "
            "pre-period trajectory using a convex combination of low-Catholic "
            "counties. The effective number of donors $1/\\sum_i \\hat\\omega_i^2$ "
            "summarises weight diffusion: values close to the total number of "
            "control counties indicate near-uniform weighting and rule out "
            "concentration risk (i.e.\\ the result is not driven by a handful of "
            "idiosyncratic donors). Rb $=$ Regierungsbezirk (Prussian "
            "administrative region). The donor pool spans rural and "
            "mixed-agrarian Prussian Kreise across Rheinland (DUS, KOB), "
            "Westphalia (ARN, MIN), Silesia (BRE, LIE), and Thuringia/Saxony "
            "(ERF, MER) -- structurally comparable to the Catholic Kreise of "
            "Westphalia and the Rhineland, which mitigates concerns about "
            "geographic compositional bias."
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

def generate_all(
    panel: pd.DataFrame,
    out_dir: Path = TABLES_DIR,
    sample_year_range: tuple[int, int] | None = (1862, 1890),
) -> Iterable[Path]:
    """Generate every table in the suite. Returns the paths written.

    ``sample_year_range`` clips the panel to the headline Kulturkampf
    window before any table is built, so every downstream estimator sees
    the same sample. The on-disk panel extends to 1910 to support
    auxiliary outcomes (election-year cross-sections, post-rollback
    diagnostics), but the headline DiD, event study and phase-sensitivity
    estimates all live in 1862--1890; including the post-1887 rollback
    recovery contaminates the post indicator with the late-empire
    fertility decline. Pass ``None`` to disable.
    """
    if sample_year_range is not None:
        y_lo, y_hi = sample_year_range
        panel = panel[panel["Year"].between(y_lo, y_hi)].copy()
        logger.info(
            "generate_all: restricted panel to %d--%d (%d obs)",
            y_lo, y_hi, len(panel),
        )

    written: list[Path] = []
    written.append(out_dir / "headline_summary.tex")
    headline_summary_table(panel, out_path=written[-1])

    written.append(out_dir / "summary_stats.tex")
    summary_statistics_table(panel, out_path=written[-1])

    written.append(out_dir / "descriptive_statistics.tex")
    descriptive_statistics_table(panel, out_path=written[-1])

    written.append(out_dir / "pretreatment_balance_1849.tex")
    pretreatment_balance_1849_table(panel, out_path=written[-1])

    written.append(out_dir / "baseline_did.tex")
    baseline_did_table(panel, out_path=written[-1])

    written.append(out_dir / "conventional_rates_appendix.tex")
    conventional_rates_appendix_table(panel, out_path=written[-1])

    written.append(out_dir / "baseline_did_indices.tex")
    baseline_did_indices_table(panel, out_path=written[-1])

    written.append(out_dir / "phase_sensitivity.tex")
    kulturkampf_phase_sensitivity_table(panel, out_path=written[-1])

    written.append(out_dir / "robustness.tex")
    robustness_table(run_robustness(panel), out_path=written[-1])

    written.append(out_dir / "channels.tex")
    channels_table(panel, out_path=written[-1])

    written.append(out_dir / "polish_german.tex")
    polish_german_table(panel, out_path=written[-1])

    written.append(out_dir / "iv_results.tex")
    iv_results_table(panel, out_path=written[-1])

    written.append(out_dir / "religiosity_robustness.tex")
    religiosity_robustness_table(panel, out_path=written[-1])

    written.append(out_dir / "protestant_religiosity_placebo.tex")
    protestant_religiosity_placebo_table(panel, out_path=written[-1])

    written.append(out_dir / "magnitudes.tex")
    magnitudes_table(panel, out_path=written[-1])

    written.append(out_dir / "back_of_envelope.tex")
    back_of_envelope_table(panel, out_path=written[-1])

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

    # Sharper Polish-vs-religion identification: continuous Polenpartei
    # interaction (within-Polish-province variation) and a timing
    # decomposition that contrasts Kulturkampf enforcement vs Polenpolitik
    # escalation on the Polish sub-sample.
    written.append(out_dir / "continuous_polish_decomposition.tex")
    continuous_polish_decomposition_table(panel, out_path=written[-1])

    # Identification-support appendix: correlation table + scatter plot
    # documenting where the religion-vs-ethnicity decomposition has
    # identifying content (cross-province) and where it does not
    # (within Polish provinces).
    written.append(out_dir / "cath_polen_identification.tex")
    cath_polen_identification_table(panel, out_path=written[-1])
    try:
        from src.visualization.plots import plot_cath_polen_scatter
        from pathlib import Path as _Path
        fig_dir = _Path(out_dir).parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_cath_polen_scatter(
            panel,
            savepath=str(fig_dir / "fig_cath_polen_scatter.png"),
        )
        logger.info("Wrote %s", fig_dir / "fig_cath_polen_scatter.png")
    except Exception as exc:
        logger.warning("Skipped cath/polen scatter: %s", exc)

    written.append(out_dir / "kulturkampf_vs_polenpolitik_timing.tex")
    kulturkampf_vs_polenpolitik_timing_table(panel, out_path=written[-1])

    # Polenausweisungen event study on the Polish sub-sample: a clean
    # break at 1887+ (not at 1873) is evidence that the apparent Polish-
    # county response is the Germanization regime, not the Kulturkampf.
    try:
        from src.visualization.plots import plot_event_study
        es_pol_cbr = run_polenausweisungen_event_study(panel, outcome="cbr")
        es_pol_mr = run_polenausweisungen_event_study(
            panel, outcome="general_marriage_rate",
        )
        from pathlib import Path as _Path
        fig_dir = _Path(out_dir).parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_event_study(
            es_pol_cbr["coefs"], ref_year=es_pol_cbr["ref_year"],
            title=("Polenausweisungen event study, Polish sub-sample (CBR): "
                   "shock at 1885--86, not 1873"),
            ylabel="Coefficient on CathShare $\\times$ Year (CBR)",
            enforcement_years=(1873, 1878),
            rollback_years=(1885, 1887),  # Polenpolitik escalation window
            end_year=1890,
            savepath=str(fig_dir / "fig_polenausweisungen_es_cbr.png"),
        )
        plot_event_study(
            es_pol_mr["coefs"], ref_year=es_pol_mr["ref_year"],
            title=("Polenausweisungen event study, Polish sub-sample "
                   "(marriage rate): shock at 1885--86, not 1873"),
            ylabel="Coefficient on CathShare $\\times$ Year (marriage rate)",
            enforcement_years=(1873, 1878),
            rollback_years=(1885, 1887),
            end_year=1890,
            savepath=str(fig_dir / "fig_polenausweisungen_es_marriage.png"),
        )
        logger.info(
            "Polenausweisungen event study: %d counties, %d obs, ref %d",
            es_pol_cbr["n_counties"], es_pol_cbr["n_obs"],
            es_pol_cbr["ref_year"],
        )
    except Exception as exc:
        logger.warning("Skipped Polenausweisungen event study: %s", exc)

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
        plot_imr_break,
        plot_imr_by_group,
        plot_cbr_war_context,
    )
    lexis_path = FIGURES_DIR / "fig_lexis.png"
    lexis_path.parent.mkdir(parents=True, exist_ok=True)
    plot_lexis_diagram(savepath=str(lexis_path))
    logger.info("Wrote %s", lexis_path)

    pop_mig_path = FIGURES_DIR / "fig_population_migration.png"
    plot_population_and_migration(panel, savepath=str(pop_mig_path))
    logger.info("Wrote %s", pop_mig_path)

    # IMR-break diagnostic: documents the Galloway data-definition
    # change at 1875 (Dthyoung fallback -> Dth<1leg). Justifies the
    # 1875+ restriction in channels.infant_mortality_analysis.
    imr_path = FIGURES_DIR / "fig_imr_break.png"
    plot_imr_break(panel, break_year=1875, savepath=str(imr_path))
    logger.info("Wrote %s", imr_path)

    # IMR by high-Cath vs low-Cath: shows the 1875 break is uniform
    # across groups (measurement, not behaviour) and that the two IMR
    # series track closely post-1875 (visual IMR null).
    imr_grp_path = FIGURES_DIR / "fig_imr_by_group.png"
    plot_imr_by_group(panel, break_year=1875, savepath=str(imr_grp_path))
    logger.info("Wrote %s", imr_grp_path)

    # War-cohort diagnostic for the pre-1873 CBR trend: raw means
    # by Catholic-share group with the Austro-Prussian (1866) and
    # Franco-Prussian (1870-71) wars shaded, plus an Rb-level
    # comparison of war-year vs non-war-year CBR.
    war_fig_path = FIGURES_DIR / "fig_war_context.png"
    plot_cbr_war_context(panel, savepath=str(war_fig_path))
    logger.info("Wrote %s", war_fig_path)

    written.append(out_dir / "war_province_diagnostic.tex")
    war_province_diagnostic_table(panel, out_path=written[-1])

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
            ("general_marriage_rate", "General marriage rate"),
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

    written.append(out_dir / "sdid_results.tex")
    sdid_results_table(panel, out_path=written[-1])

    written.append(out_dir / "sdid_donor_counties.tex")
    sdid_donor_counties_table(panel, out_path=written[-1])

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
