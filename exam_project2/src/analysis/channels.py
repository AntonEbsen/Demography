"""
channels.py
============
Mechanism channel analyses: illegitimacy, infant mortality, religious
infrastructure (Kulturkampf clerical impact), and Catholic schooling.
"""

import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def illegitimacy_analysis(df: pd.DataFrame):
    """
    Did the Kulturkampf affect illegitimacy rates in Catholic counties?

    Logic: Catholic parish oversight of sexuality and marriage weakened
    under the Kulturkampf. Civil marriage replaced church marriage in 1875.
    If institutional oversight mattered for enforcing marital norms,
    illegitimate births should rise in Catholic counties.
    """
    df = df.copy()

    logger.info("=" * 60)
    logger.info("ILLEGITIMACY CHANNEL")
    logger.info("=" * 60)

    logger.info("Mean illegitimacy ratio (%% of births):")
    for period_label, mask in [
        ("Pre-Kulturkampf (1875-1878)", (df["Year"] >= 1875) & (df["Year"] < 1879)),
        ("Rollback (1880-1887)", (df["Year"] >= 1880) & (df["Year"] <= 1887)),
    ]:
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        by_cath = sub.groupby("high_cath")["illegitimacy_ratio"].mean()
        logger.info("  %s:", period_label)
        logger.info("    Low Catholic (<=50%%):  %.2f%%", by_cath.get(0, np.nan))
        logger.info("    High Catholic (>50%%): %.2f%%", by_cath.get(1, np.nan))

    logger.info("DiD: Illegitimacy ratio ~ CathShare × Post")
    res = safe_panel_ols(df, "illegitimacy_ratio", ["cath_share_x_post"])
    coef = res.params["cath_share_x_post"]
    se = res.std_errors["cath_share_x_post"]
    pval = res.pvalues["cath_share_x_post"]
    logger.info("  β = %.4f (SE = %.4f, p = %.3f)", coef, se, pval)
    logger.info("  N = %d", int(res.nobs))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, "#C0392B"),
        ("Low Catholic (≤50%)", df["high_cath"] == 0, "#2471A3"),
    ]:
        trend = df[mask].groupby("Year")["illegitimacy_ratio"].mean()
        ax.plot(trend.index, trend.values, color=color, linewidth=2, label=label)

    ax.axvspan(1871, 1878, alpha=0.15, color="#E8DAEF",
               label="Kulturkampf (1871-78)")
    ax.axvline(1873, color="#7B1A1A", linestyle="--", linewidth=1.2, alpha=0.9,
               label="May Laws (treatment year)")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Illegitimate births (% of total)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return {"result": res, "fig": fig}


def infant_mortality_analysis(df: pd.DataFrame):
    """
    Did disruption of Catholic health services affect infant survival?

    The headline outcome `infant_mortality_rate` is total IMR (= total
    infant deaths / total live births x 1000), the standard demographic
    measure (Princeton EFP / HMD / Galloway, Hammel & Lee 1994
    convention). It is well-defined only from 1875 onwards because
    Galloway's illegitimate-infant-death column `Dth<1bas` does not
    appear in pre-1875 VIT files; we therefore restrict the analysis
    to 1875+ regardless. See fig_imr_break.png for the data-break
    diagnostic on the legitimate-only series.
    """
    df = df[df["Year"] >= 1875].copy()

    logger.info("=" * 60)
    logger.info("INFANT MORTALITY CHANNEL (1875+ only due to data break)")
    logger.info("=" * 60)

    df["post_rollback"] = (df["Year"] >= 1880).astype(int)
    df["cath_x_rollback"] = df["cath_share"] * df["post_rollback"]

    res = safe_panel_ols(df, "infant_mortality_rate", ["cath_x_rollback"])
    coef = res.params["cath_x_rollback"]
    se = res.std_errors["cath_x_rollback"]
    pval = res.pvalues["cath_x_rollback"]
    logger.info("DiD: Infant Mortality Rate ~ CathShare × (Year>=1880)")
    logger.info("  β = %.4f (SE = %.4f, p = %.3f)", coef, se, pval)
    logger.info("  N = %d", int(res.nobs))

    polish_rbs = ["POS", "BRO"]
    german_cath_rbs = ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]

    logger.info("By sub-region (rollback vs enforcement):")
    for label, mask in [
        ("Polish provinces", df["Rb"].isin(polish_rbs)),
        ("German Catholic provinces", df["Rb"].isin(german_cath_rbs)),
        ("Protestant provinces", ~df["Rb"].isin(polish_rbs + german_cath_rbs)),
    ]:
        sub = df[mask].copy()
        if sub["Code"].nunique() < 10:
            continue
        try:
            res_sub = safe_panel_ols(sub, "infant_mortality_rate", ["cath_x_rollback"])
            logger.info("  %s (%d counties): β = %.4f (SE = %.4f, p = %.3f)",
                        label, sub["Code"].nunique(),
                        res_sub.params["cath_x_rollback"],
                        res_sub.std_errors["cath_x_rollback"],
                        res_sub.pvalues["cath_x_rollback"])
        except Exception as e:
            logger.warning("  %s: failed (%s)", label, e)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, mask, color in [
        ("High Catholic (>50%)", df["high_cath"] == 1, "#C0392B"),
        ("Low Catholic (≤50%)", df["high_cath"] == 0, "#2471A3"),
    ]:
        trend = df[mask].groupby("Year")["infant_mortality_rate"].mean()
        ax.plot(trend.index, trend.values, color=color, linewidth=2, marker="o", markersize=4, label=label)

    # Kulturkampf enforcement 1871-78; IMR data only well-defined from 1875,
    # so the visible portion of the band starts at 1875 and the May Laws
    # vertical line (1873) is suppressed (off-screen).
    ax.axvspan(1875, 1878, alpha=0.15, color="#C0392B",
               label="Enforcement (1871-78, visible from 1875)")
    ax.axvspan(1880, 1887, alpha=0.15, color="#2471A3", label="Rollback")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Infant mortality rate (per 1,000 legitimate live births)", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return {"result": res, "fig": fig}


def religious_infrastructure_channel(df: pd.DataFrame):
    """
    Did the Kulturkampf shrink Catholic religious-sector employment?

    Logic: the May Laws (1873) imposed state supervision of clerical
    training and appointments, exiled bishops, and shut down many
    Catholic orders. The Anzeigegesetz (May 1873) required clergy to
    notify the state of any appointment. If enforcement bit, the per-
    capita Catholic clerical/religious-sector workforce should have
    declined in heavily Catholic counties between the 1849 baseline
    and the 1882 occupational census, more so than in Protestant
    counties.

    Outcomes
    --------
    - 1849: ``rel1849_cat_priest`` per 1,000 Catholics (from iPEHD).
    - 1882: ``rel_edu_emp_1882`` per 1,000 population (from GEL1882;
      the 1882 census category covers religion, education and
      instruction occupations -- a coarser superset of priests but
      the only available 1882 measure).

    These two are not identical concepts (priests in 1849 vs.
    religion-education employment in 1882), so we report them as
    paired cross-sectional regressions and a stacked long-difference
    DiD with denomination-relevant scaling. The DiD coefficient is
    the relevant Kulturkampf-channel parameter; the two raw regressions
    serve as transparency.
    """
    df = df.copy()
    logger.info("=" * 60)
    logger.info("RELIGIOUS-INFRASTRUCTURE CHANNEL (1849 -> 1882)")
    logger.info("=" * 60)

    # County-level frame (one row per Code).
    cs = (
        df.sort_values(["Code", "Year"])
          .drop_duplicates(subset="Code", keep="first")
          [[
              "Code", "Rb", "cath_share",
              "rel1849_cat_priest", "rel1849_pro_priest",
              "pop1849_tot",
              "rel_edu_emp_1882", "rel_edu_emp_per_1k_1882",
              "pop_1880_gel",
          ]]
          .copy()
    )

    # 1849 Catholic-priest density per 1,000 Catholics.
    cs["cat_pop_1849"] = cs["cath_share"] / 100 * cs["pop1849_tot"]
    cs["cat_priest_per_1k_cath_1849"] = (
        cs["rel1849_cat_priest"] / cs["cat_pop_1849"].replace(0, np.nan) * 1000
    )

    # 1882 religion-education employment per 1,000 Catholics. Using the
    # Catholic-population denominator (rather than total pop) controls
    # for the mechanical scaling of religious workers with denominator
    # size, isolating the per-Catholic intensity.
    cs["cat_pop_1880_est"] = cs["cath_share"] / 100 * cs["pop_1880_gel"]
    cs["rel_edu_emp_per_1k_cath_1882"] = (
        cs["rel_edu_emp_1882"] / cs["cat_pop_1880_est"].replace(0, np.nan) * 1000
    )

    n_1849 = cs["cat_priest_per_1k_cath_1849"].notna().sum()
    n_1882 = cs["rel_edu_emp_per_1k_cath_1882"].notna().sum()
    logger.info("Counties with 1849 priest density: %d", n_1849)
    logger.info("Counties with 1882 rel-edu employment density: %d", n_1882)

    # Cross-sectional regressions: density ~ cath_share + Rb FE.
    rb_dummies = pd.get_dummies(cs["Rb"], prefix="rb", drop_first=True).astype(float)

    def _fit(y_col: str) -> sm.regression.linear_model.RegressionResultsWrapper:
        mask = cs[y_col].notna() & cs["cath_share"].notna()
        y = np.log(cs.loc[mask, y_col].replace(0, np.nan)).dropna()
        idx = y.index
        X = pd.concat(
            [pd.Series(1.0, index=idx, name="const"),
             cs.loc[idx, "cath_share"],
             rb_dummies.loc[idx]],
            axis=1,
        ).astype(float)
        return sm.OLS(y, X).fit(cov_type="HC1")

    res_1849 = _fit("cat_priest_per_1k_cath_1849")
    res_1882 = _fit("rel_edu_emp_per_1k_cath_1882")
    logger.info("1849 log(priest density) ~ cath_share + Rb FE: "
                "β = %.5f (SE = %.5f, p = %.3f, N = %d)",
                res_1849.params["cath_share"], res_1849.bse["cath_share"],
                res_1849.pvalues["cath_share"], int(res_1849.nobs))
    logger.info("1882 log(rel-edu emp density) ~ cath_share + Rb FE: "
                "β = %.5f (SE = %.5f, p = %.3f, N = %d)",
                res_1882.params["cath_share"], res_1882.bse["cath_share"],
                res_1882.pvalues["cath_share"], int(res_1882.nobs))

    # Stacked long-difference DiD: pool 1849 and 1882, identify the
    # change in the cath_share-density gradient.
    stacked = []
    for year, y_col in [
        (1849, "cat_priest_per_1k_cath_1849"),
        (1882, "rel_edu_emp_per_1k_cath_1882"),
    ]:
        sub = cs[["Code", "Rb", "cath_share", y_col]].rename(
            columns={y_col: "rel_density"}
        )
        sub["year"] = year
        stacked.append(sub)
    stacked = pd.concat(stacked, ignore_index=True)
    stacked["post"] = (stacked["year"] == 1882).astype(int)
    stacked["cath_x_post"] = stacked["cath_share"] * stacked["post"]
    stacked["log_rel_density"] = np.log(stacked["rel_density"].replace(0, np.nan))

    stacked = stacked.dropna(subset=["log_rel_density", "cath_share"])
    yr_dummies = pd.get_dummies(stacked["year"], prefix="yr", drop_first=True).astype(float)
    rb_d2 = pd.get_dummies(stacked["Rb"], prefix="rb", drop_first=True).astype(float)
    X_did = pd.concat([
        pd.Series(1.0, index=stacked.index, name="const"),
        stacked["cath_share"], stacked["post"], stacked["cath_x_post"],
        yr_dummies, rb_d2,
    ], axis=1).astype(float)
    res_did = sm.OLS(stacked["log_rel_density"], X_did).fit(
        cov_type="cluster", cov_kwds={"groups": stacked["Code"]},
    )
    logger.info(
        "Long-difference DiD on log(religious density) — "
        "coefficient on cath_share × Post1882: β = %.5f (SE = %.5f, p = %.3f, N = %d)",
        res_did.params["cath_x_post"], res_did.bse["cath_x_post"],
        res_did.pvalues["cath_x_post"], int(res_did.nobs),
    )
    if res_did.params["cath_x_post"] < 0 and res_did.pvalues["cath_x_post"] < 0.10:
        logger.info("  -> Consistent with Kulturkampf reducing Catholic "
                    "religious-sector employment density.")
    else:
        logger.info("  -> Effect imprecisely estimated or in the unexpected "
                    "direction; see caveats on 1849 vs 1882 category comparability.")

    # Visualization: density-by-cath-share scatter for both years. We
    # restrict the plot to counties with cath_share >= 5% so the per-
    # Catholic denominator is well-defined; very-low-Catholic counties
    # produce mechanical outliers (tiny Catholic population in the
    # denominator) that obscure the gradient. The regression coefficients
    # reported above use the full sample on a log scale, where these
    # outliers are absorbed.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (year, y_col) in zip(axes, [
        (1849, "cat_priest_per_1k_cath_1849"),
        (1882, "rel_edu_emp_per_1k_cath_1882"),
    ]):
        plot_df = cs.loc[cs["cath_share"] >= 5, ["cath_share", y_col]].dropna()
        # Clip the visible y-range at the 95th percentile to suppress the
        # remaining handful of high-density outliers (mostly Stadt-Land
        # boundary cases).
        if len(plot_df) > 5:
            y95 = plot_df[y_col].quantile(0.95)
            ax.set_ylim(0, max(y95 * 1.2, 5))
        ax.scatter(plot_df["cath_share"], plot_df[y_col], alpha=0.5, s=20,
                   color="#7D3C98", edgecolor="white")
        if len(plot_df) > 5:
            xs = np.linspace(plot_df["cath_share"].min(),
                              plot_df["cath_share"].max(), 100)
            coef = np.polyfit(plot_df["cath_share"], plot_df[y_col], 1)
            ax.plot(xs, np.polyval(coef, xs), color="#1B4F72", linewidth=2)
        ax.set_xlabel("Catholic share (1871, %)")
        ax.set_ylabel("Religious workers per 1,000 Catholics")
        ax.grid(alpha=0.3)
    plt.tight_layout()

    return {
        "result_1849": res_1849,
        "result_1882": res_1882,
        "result_did": res_did,
        "fig": fig,
        "n_1849": int(n_1849),
        "n_1882": int(n_1882),
    }


def schooling_channel(df: pd.DataFrame):
    """
    Did the Kulturkampf reduce schooling in Catholic counties?

    Stacks three cross-sections of schooling participation:
        1849 (EDU1849)  -- pre-treatment
        1871 (iPEHD school1517) -- contemporaneous-treatment-start
        1886 (EDU1886)  -- post-treatment / enforcement endpoint

    The 1849 measure is elementary public-school students as a share
    of population. The 1871 measure (school1517) is the iPEHD-published
    age-15-17 school enrollment rate -- a literacy proxy rather than
    primary attendance, but it pins the trajectory. The 1886 measure
    is compulsory-school attendance (volksschule + private) as a
    share of compulsory-age population (6-14).

    The three series are not on a single denominator, so the
    canonical DiD compares 1849 -> 1886 (the only two periods on a
    nearly common participation-rate scale, both with population-share
    denominators) with 1871 reported as a midpoint diagnostic. The
    interaction ``cath_share × Post1886`` is the parameter of interest.

    Gender split: 1849 reports students by gender (m/f). EDU1886 does
    not, so the gender heterogeneity is presented for 1849 only as a
    placebo-style baseline (boys were the primary target of Catholic
    seminary suppression).
    """
    df = df.copy()
    logger.info("=" * 60)
    logger.info("SCHOOLING CHANNEL (1849 -> 1871 -> 1886)")
    logger.info("=" * 60)

    cs = (
        df.sort_values(["Code", "Year"])
          .drop_duplicates(subset="Code", keep="first")
          [[
              "Code", "Rb", "cath_share", "school1517",
              "edu1849_pub_ele_stud_m", "edu1849_pub_ele_stud_f",
              "pop1849_tot", "pop1849_m_tot", "pop1849_f_tot",
              "attend_public_1886", "attend_private_1886",
              "school_age_pop_1886", "attend_rate_1886",
              "teachers_1886", "pupils_per_teacher_1886",
          ]]
          .copy()
    )

    # 1849 elementary attendance rate (overall, m, f). The 1849 file
    # gives students by gender; the denominator is total male/female
    # population. Treat the gender-specific rates as participation
    # proxies rather than precise enrollment ratios (the 1849 census
    # did not publish school-age population).
    cs["edu1849_total_students"] = (
        cs["edu1849_pub_ele_stud_m"].fillna(0)
        + cs["edu1849_pub_ele_stud_f"].fillna(0)
    )
    cs["attend_rate_1849"] = (
        cs["edu1849_total_students"] / cs["pop1849_tot"].replace(0, np.nan)
    )
    cs["attend_rate_1849_m"] = (
        cs["edu1849_pub_ele_stud_m"] / cs["pop1849_m_tot"].replace(0, np.nan)
    )
    cs["attend_rate_1849_f"] = (
        cs["edu1849_pub_ele_stud_f"] / cs["pop1849_f_tot"].replace(0, np.nan)
    )

    for label, col in [
        ("1849 elementary attendance rate", "attend_rate_1849"),
        ("1871 age-15-17 enrollment rate (iPEHD)", "school1517"),
        ("1886 compulsory-school attendance rate", "attend_rate_1886"),
    ]:
        valid = cs[col].notna().sum()
        if valid > 0:
            logger.info("  %s: N = %d, mean = %.3f",
                        label, valid, cs[col].mean())

    # ------------------------------------------------------------------
    # Long-difference DiD on 1849 vs 1886 attendance rates.
    # ------------------------------------------------------------------
    stacked = []
    for year, y_col in [(1849, "attend_rate_1849"),
                        (1886, "attend_rate_1886")]:
        sub = cs[["Code", "Rb", "cath_share", y_col]].rename(
            columns={y_col: "attend_rate"}
        )
        sub["year"] = year
        stacked.append(sub)
    stacked = pd.concat(stacked, ignore_index=True)
    stacked["post"] = (stacked["year"] == 1886).astype(int)
    stacked["cath_x_post"] = stacked["cath_share"] * stacked["post"]
    stacked = stacked.dropna(subset=["attend_rate", "cath_share"])

    yr_d = pd.get_dummies(stacked["year"], prefix="yr", drop_first=True).astype(float)
    rb_d = pd.get_dummies(stacked["Rb"], prefix="rb", drop_first=True).astype(float)
    X = pd.concat([
        pd.Series(1.0, index=stacked.index, name="const"),
        stacked["cath_share"], stacked["post"], stacked["cath_x_post"],
        yr_d, rb_d,
    ], axis=1).astype(float)
    res = sm.OLS(stacked["attend_rate"], X).fit(
        cov_type="cluster", cov_kwds={"groups": stacked["Code"]},
    )
    logger.info(
        "Long-difference DiD on attendance rate (1849 vs 1886) — "
        "coefficient on cath_share × Post1886: "
        "β = %.6f (SE = %.6f, p = %.3f, N = %d)",
        res.params["cath_x_post"], res.bse["cath_x_post"],
        res.pvalues["cath_x_post"], int(res.nobs),
    )

    # ------------------------------------------------------------------
    # Gender-split 1849 baseline regression -- shows the *initial*
    # Catholic-share gradient by gender; this is descriptive only since
    # 1886 attendance is not split by gender.
    # ------------------------------------------------------------------
    for gender_col, label in [
        ("attend_rate_1849_m", "1849 boys' attendance"),
        ("attend_rate_1849_f", "1849 girls' attendance"),
    ]:
        mask = cs[gender_col].notna() & cs["cath_share"].notna()
        if mask.sum() < 30:
            continue
        rb_dd = pd.get_dummies(
            cs.loc[mask, "Rb"], prefix="rb", drop_first=True,
        ).astype(float)
        X_g = pd.concat([
            pd.Series(1.0, index=cs.loc[mask].index, name="const"),
            cs.loc[mask, "cath_share"],
            rb_dd,
        ], axis=1).astype(float)
        y_g = cs.loc[mask, gender_col]
        res_g = sm.OLS(y_g, X_g).fit(cov_type="HC1")
        logger.info("  %s ~ cath_share + Rb FE: β = %.6f (SE = %.6f, p = %.3f, N = %d)",
                    label, res_g.params["cath_share"],
                    res_g.bse["cath_share"],
                    res_g.pvalues["cath_share"], int(res_g.nobs))

    # ------------------------------------------------------------------
    # Plot the three-period trajectory by Catholic-share quartile.
    # ------------------------------------------------------------------
    cs["cath_quartile"] = pd.qcut(
        cs["cath_share"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
    )
    period_cols = [
        ("1849", "attend_rate_1849"),
        ("1871", "school1517"),
        ("1886", "attend_rate_1886"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for q, color in zip(["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                        ["#2471A3", "#5DADE2", "#E59866", "#C0392B"]):
        means = []
        for _, col in period_cols:
            means.append(cs.loc[cs["cath_quartile"] == q, col].mean())
        ax.plot([1849, 1871, 1886], means, marker="o", linewidth=2,
                color=color, label=f"Catholic share {q}")
    ax.axvspan(1871, 1878, alpha=0.15, color="#E8DAEF",
               label="Kulturkampf (1871-78)")
    ax.axvline(1873, color="#7B1A1A", linestyle="--", linewidth=1.2, alpha=0.9,
               label="May Laws (treatment year)")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Schooling rate (varies by period)", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return {"result_did": res, "fig": fig,
            "n_1849": int(cs["attend_rate_1849"].notna().sum()),
            "n_1886": int(cs["attend_rate_1886"].notna().sum())}
