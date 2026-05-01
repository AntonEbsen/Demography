import json
import os
from pathlib import Path

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.split("\n")]}

setup_template = """\
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Enable autoreload for development
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
DATA_RAW = project_root / "data" / "raw"
DATA_PROCESSED = project_root / "data" / "processed"
OUTPUTS = project_root / "outputs" / "figures"
OUTPUTS.mkdir(exist_ok=True, parents=True)

print("Setup complete. Outputs will be saved to:", OUTPUTS)
"""

# ==========================================
# 01_data_and_eda.ipynb
# ==========================================
nb1_cells = [
    md_cell("# Part 1: Data Preparation and Exploratory Analysis\n\n## The Kulturkampf and Catholic Fertility in Prussia\n\n**Research question:** Did Bismarck's anti-Catholic Kulturkampf legislation (1872–1878) affect the Catholic–Protestant fertility differential in Prussian counties?\n\nThis first notebook handles loading the raw Galloway Prussia Database (1861–1914), harmonizing variables across cross-sections, interpolating missing population data, and generating the core panel dataset used throughout the rest of the analysis."),
    md_cell("### 1. Environment Setup\nWe load the necessary paths, external libraries, and our custom `src.data` pipeline functions."),
    code_cell(setup_template + """
from src.data.load_data import load_rel1871, load_vit_panel, load_ipehd_master
from src.data.build_dataset import build_analysis_panel
from src.visualization.plots import plot_fertility_trends, plot_cath_distribution
"""),
    md_cell("### 2. Building the Analysis Panel\nWe join the time-invariant REL1871 religious census data (providing the baseline Catholic share for each county) with the annual VIT vital registration panel. We then compute crude birth rates, marriage rates, and other demographic outcomes."),
    code_cell("""\
panel = build_analysis_panel(
    data_dir=DATA_RAW,
    year_start=1862,
    year_end=1890,
    save=True,  # Saves to data/processed/analysis_panel.parquet
)
panel.head(10)
"""),
    md_cell("### 3. Descriptive Statistics\nLet's examine the raw means and standard deviations of our primary outcomes, split by pre- vs. post-Kulturkampf periods and by religious composition (High >50% Catholic vs. Low Catholic). This provides a foundational intuition before moving to rigorous econometric models."),
    code_cell("""\
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

for period, label in [(panel["Year"] < 1873, "Pre-Kulturkampf (1862-1872)"),
                       (panel["Year"] >= 1873, "Post-Kulturkampf (1873-1890)")]:
    sub = panel[period]
    print(f"\\n{label}:")
    print(f"  N = {len(sub)}, Counties = {sub['Code'].nunique()}")
    for var in ["cbr", "legitimate_br", "marriage_rate"]:
        if var in sub.columns and sub[var].notna().any():
            print(f"  {var}: mean={sub[var].mean():.2f}, sd={sub[var].std():.2f}")

print(f"\\n{'='*60}")
print("BY RELIGIOUS COMPOSITION (full sample)")
print(f"{'='*60}")
grouped = panel.groupby("high_cath").agg(
    n_counties=("Code", "nunique"),
    mean_cath_share=("cath_share", "mean"),
    mean_cbr=("cbr", "mean"),
    mean_marriage_rate=("marriage_rate", lambda x: x.mean() if x.notna().any() else np.nan),
).round(2)
grouped.index = ["Low Catholic (≤50%)", "High Catholic (>50%)"]
print(grouped)
"""),
    md_cell("### 4. Visualizing Demographics and the Treatment Variable\nThe Kulturkampf was a nationwide policy, but its localized intensity depended on the Catholic share of a county. We map out the distribution of Catholic shares to ensure there is sufficient variation for our continuous and binary treatment definitions."),
    code_cell("""\
fig, ax = plot_cath_distribution(
    panel,
    savepath=str(OUTPUTS / "fig1_cath_distribution.png"),
)
plt.show()
"""),
    md_cell("### 5. Fertility and Marriage Trends Over Time\nBefore running fixed-effects models, plotting raw trends is vital. Do high-Catholic counties have permanently higher fertility? Do their trajectories diverge after 1872?"),
    code_cell("""\
fig, ax = plot_fertility_trends(
    panel,
    outcome="cbr",
    ylabel="Crude birth rate (per 1,000)",
    title="Fertility trends: High- vs Low-Catholic counties",
    savepath=str(OUTPUTS / "fig2_fertility_trends.png"),
)
plt.show()

# Focus on legitimate births (only reliably available post-1875)
panel_post75 = panel[panel["Year"] >= 1875].copy()
fig, ax = plot_fertility_trends(
    panel_post75,
    outcome="legitimate_br",
    ylabel="Legitimate birth rate (per 1,000)",
    title="Legitimate fertility trends: High- vs Low-Catholic counties",
    savepath=str(OUTPUTS / "fig3_legit_fertility_trends.png"),
)
plt.show()

# Catholic marriage share
fig, ax = plot_fertility_trends(
    panel_post75,
    outcome="cath_marriage_share",
    ylabel="Catholic marriages (% of total)",
    title="Catholic marriage share over time",
    savepath=str(OUTPUTS / "fig4_cath_marriages.png"),
)
plt.show()
""")
]

# ==========================================
# 02_baseline_regressions.ipynb
# ==========================================
nb2_cells = [
    md_cell("# Part 2: Baseline Econometrics & Event Studies\n\nHaving constructed our panel dataset, this notebook implements our core empirical strategy: a Two-Way Fixed Effects (TWFE) Difference-in-Differences model to quantify the impact of the Kulturkampf on fertility outcomes in Catholic areas."),
    md_cell("### 1. Setup and Data Loading\nWe load the pre-compiled `.parquet` panel dataset from Part 1."),
    code_cell(setup_template + """
from src.analysis.regressions import run_baseline_did, run_event_study, run_robustness
from src.visualization.plots import plot_event_study, plot_robustness_table
from src.data.load_data import load_ipehd_master, load_rel1871

# Load pre-processed panel
panel = pd.read_parquet(DATA_PROCESSED / "analysis_panel.parquet")
"""),
    md_cell("### 2. Baseline Difference-in-Differences\nOur baseline specification relies on TWFE: $Y_{it} = \\beta (CathShare_i \\times Post_t) + \\alpha_i + \\delta_t + X_{it}'\\gamma + \\varepsilon_{it}$. We test both continuous treatment intensity (Catholic share percentage) and a binary treatment indicator (>50% Catholic)."),
    code_cell("""\
print("=" * 60)
print("BASELINE DiD: CBR ~ CathShare × Post")
print("=" * 60)
res_cont = run_baseline_did(panel, outcome="cbr", treatment="continuous")
print(res_cont["summary"])

print("=" * 60)
print("BASELINE DiD: CBR ~ HighCath × Post")
print("=" * 60)
res_bin = run_baseline_did(panel, outcome="cbr", treatment="binary")
print(res_bin["summary"])
"""),
    md_cell("We also check if this effect persists across other demographic outcomes, such as legitimate births and marriage rates. A change in crude birth rate could be mechanically driven by fewer marriages rather than altered marital fertility."),
    code_cell("""\
print("=" * 60)
print("DiD FOR ALTERNATIVE OUTCOMES")
print("=" * 60)
for outcome, label in [
    ("legitimate_br", "Legitimate birth rate"),
    ("marriage_rate", "Marriage rate"),
    ("illegitimacy_ratio", "Illegitimacy ratio"),
]:
    print(f"\\n--- {label} ---")
    try:
        res = run_baseline_did(panel, outcome=outcome, treatment="continuous")
        r = res["result"]
        treat_var = "cath_share_x_post"
        print(f"  Coef: {r.params[treat_var]:.4f} (SE: {r.std_errors[treat_var]:.4f}, p: {r.pvalues[treat_var]:.3f})")
    except Exception as e:
        print(f"  Error: {e}")
"""),
    md_cell("### 3. Event Study Design\nA fundamental assumption of DiD is parallel trends. We plot an event study interacting the Catholic share with year dummies to trace the dynamic effect over time and formally test for pre-existing trends prior to 1872."),
    code_cell("""\
print("=" * 60)
print("EVENT STUDY")
print("=" * 60)

es = run_event_study(panel, outcome="cbr", treatment_var="cath_share", ref_year=1872)
fig, ax = plot_event_study(
    es["coefs"],
    ref_year=1872,
    title="Event study: Catholic share × Year dummies on CBR",
    savepath=str(OUTPUTS / "fig5_event_study.png"),
)
plt.show()

# Formal pre-trend test
pre_coefs = es["coefs"][es["coefs"]["Year"] < 1872]
print(f"\\nPre-trend coefficients (before 1872):")
print(pre_coefs[["Year", "beta", "se"]].to_string(index=False))
"""),
    md_cell("### 4. Robustness Checks\nTo verify that our results are not artifacts of arbitrary methodological choices, we test alternative temporal cutoffs (1872, 1875), alternative treatment thresholds, and the exclusion of specific demographics like Polish-majority provinces."),
    code_cell("""\
print("=" * 60)
print("ROBUSTNESS CHECKS")
print("=" * 60)
rob = run_robustness(panel, outcome="cbr")
fig, ax = plot_robustness_table(rob, savepath=str(OUTPUTS / "fig6_robustness.png"))
plt.show()
"""),
    md_cell("### 5. Validation against iPEHD\nFinally, we cross-validate our key treatment variable (Catholic share) against the authoritative Becker-Woessmann (2009) iPEHD dataset to ensure our REL1871 data extraction is robust."),
    code_cell("""\
ipehd = load_ipehd_master(DATA_RAW / "ipehd_qje2009_master.dta")
rel = load_rel1871(DATA_RAW / "REL1871.XLS")

print(f"iPEHD f_cath: mean={ipehd['f_cath'].mean():.1f}, median={ipehd['f_cath'].median():.1f}, N={len(ipehd)}")
print(f"Galloway cath_share: mean={rel['cath_share'].mean():.1f}, median={rel['cath_share'].median():.1f}, N={len(rel)}")
""")
]

# ==========================================
# 03_extensions_and_mechanisms.ipynb
# ==========================================
nb3_cells = [
    md_cell("# Part 3: Mechanisms, Heterogeneity & Advanced Extensions\n\nHaving established a baseline effect, this notebook dives deeper into the *mechanisms* driving the Catholic fertility response. Was it an urban vs. rural phenomenon? Was it driven by Polish nationalism or purely religious friction?"),
    md_cell("### 1. Setup and Data Loading"),
    code_cell(setup_template + """
from src.analysis.exploratory import (
    heterogeneity_by_urbanization, polish_vs_german_catholics, fertility_convergence,
    marriage_to_birth_pipeline, dose_response_plot, infant_mortality_did,
)
from src.analysis.advanced import (
    rollback_event_study, illegitimacy_analysis, infant_mortality_analysis,
    franco_prussian_war_analysis, robustness_exclude_war, trend_adjusted_did,
    polish_german_rollback, placebo_test,
)

panel = pd.read_parquet(DATA_PROCESSED / "analysis_panel.parquet")
"""),
    md_cell("### 2. Heterogeneity Analysis\nThe Kulturkampf may have impacted urban Catholics (who faced modern state bureaucracy daily) differently from rural Catholics (where traditional parish structures remained resilient). We also separate the effect for Polish vs. German Catholics to untangle religious oppression from ethnic suppression."),
    code_cell("""\
print("=" * 60)
print("HETEROGENEITY: URBAN vs RURAL")
print("=" * 60)
het_results = heterogeneity_by_urbanization(panel, outcome="cbr")

print("\\n" + "=" * 60)
print("POLISH vs GERMAN CATHOLICS")
print("=" * 60)
pol_results = polish_vs_german_catholics(panel, outcome="cbr")
"""),
    md_cell("### 3. Fertility Convergence and Dose Response\nWas the Catholic fertility bump merely a delay in their demographic transition? We examine long-term convergence and estimate a non-linear dose-response curve based on varying intensities of Catholic populations."),
    code_cell("""\
fig_conv, conv_data = fertility_convergence(panel)
fig_conv.savefig(OUTPUTS / "fig7_convergence.png", dpi=300, bbox_inches="tight")
plt.show()

fig_dose, dose_data = dose_response_plot(panel, savepath=str(OUTPUTS / "fig8_dose_response.png"))
plt.show()
"""),
    md_cell("### 4. Mechanisms: Marriage Pipeline and Infant Mortality\nDemographically, a rise in the crude birth rate could result from earlier/more frequent marriages (the nuptiality channel) or a decline in infant mortality (survivor bias in registration). We test these specific channels."),
    code_cell("""\
lag_results = marriage_to_birth_pipeline(panel)
imr_result = infant_mortality_did(panel)

ill = illegitimacy_analysis(panel)
ill["fig"].savefig(OUTPUTS / "fig10_illegitimacy.png", dpi=300, bbox_inches="tight")
plt.show()

imr = infant_mortality_analysis(panel)
imr["fig"].savefig(OUTPUTS / "fig11_infant_mortality.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md_cell("### 5. Historical Confounders: The Franco-Prussian War & The Rollback\nThe Franco-Prussian war (1870-1871) caused severe demographic shocks precisely when our pre-trend window closes. Furthermore, Bismarck began 'rolling back' Kulturkampf laws in the late 1870s. We explicitly model both of these historical shocks to clean our estimates."),
    code_cell("""\
war = franco_prussian_war_analysis(panel)
war["fig"].savefig(OUTPUTS / "fig12_war_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# Robustness excluding the war years
war_rob = robustness_exclude_war(panel, outcome="cbr", war_years=(1870, 1871, 1872), ref_year=1869, savepath=str(OUTPUTS / "fig13_war_excluded_event_study.png"))
plt.show()

# Examining the Rollback period
rollback = rollback_event_study(panel, outcome="cbr", treatment_var="cath_share", ref_year=1872, savepath=str(OUTPUTS / "fig9_rollback_event_study.png"))
plt.show()
"""),
    md_cell("### 6. Final Robustness: Trend Adjustments & Placebo Tests\nTo conclude our empirical verification, we adjust for differential pre-trends explicitly and run placebo tests on arbitrary years to ensure our model isn't just picking up systemic noise."),
    code_cell("""\
ta_main = trend_adjusted_did(panel, outcome="cbr", trend_base_year=1862, exclude_war=False)
plc = placebo_test(panel, outcome="cbr", placebo_years=[1864, 1866, 1868, 1870, 1873, 1876, 1880, 1884], savepath=str(OUTPUTS / "fig15_placebo_test.png"))
plt.show()
""")
]

# ==========================================
# 04_spatial_analysis.ipynb
# ==========================================
nb4_cells = [
    md_cell("# Part 4: Spatial Analysis and Geographic Mapping\n\nStatistical tables only tell part of the story. In historical demography, geography matters immensely. This notebook leverages `geopandas` to map the spatial distribution of Catholics and the geographic footprint of the fertility response, allowing us to visually inspect for spatial clustering or omitted regional variables."),
    md_cell("### 1. Setup and Loading the Shapefile\nWe load the HGIS German Empire shapefile and merge it with our panel dataset."),
    code_cell(setup_template + """
from src.visualization.maps import (
    load_prussia_shapefile, map_catholic_share, map_fertility_change,
    map_polish_german_provinces, map_kulturkampf_residuals,
)

panel = pd.read_parquet(DATA_PROCESSED / "analysis_panel.parquet")

shp_path = DATA_RAW / "German_Empire_1871_v.1.0.shp"
gdf = load_prussia_shapefile(shp_path)
print(f"Loaded {len(gdf)} Prussian counties from shapefile")
"""),
    md_cell("### 2. Mapping the Treatment: Catholic Share\nFirst, we visualize our primary independent variable. This map highlights the deep religious divide in Prussia: the heavily Catholic Rhineland in the West and Polish territories in the East, contrasted against the Protestant Prussian core."),
    code_cell("""\
fig, ax = map_catholic_share(
    gdf, panel,
    savepath=str(OUTPUTS / "map1_catholic_share.png"),
)
plt.show()
"""),
    md_cell("### 3. Mapping the Outcome: Fertility Change\nNext, we calculate the naive difference in fertility between the pre-Kulturkampf era (1868-1872) and the post era (1878-1882) and map it. Darker areas experienced the largest fertility increases."),
    code_cell("""\
fig, ax = map_fertility_change(
    gdf, panel,
    pre_years=(1868, 1872),
    post_years=(1878, 1882),
    savepath=str(OUTPUTS / "map2_fertility_change.png"),
)
plt.show()
"""),
    md_cell("### 4. Polish vs. German Catholic Regions\nTo aid in our heterogeneity analysis from Notebook 3, we map out the specific Polish-majority provinces versus the rest of Prussia. This highlights the geographic overlap between Catholicism and Polish ethnicity in the Eastern provinces."),
    code_cell("""\
fig, ax = map_polish_german_provinces(
    gdf, panel,
    savepath=str(OUTPUTS / "map3_polish_german.png"),
)
plt.show()
"""),
    md_cell("### 5. Residual Mapping\nFinally, we map the residuals from our Difference-in-Differences model. If there is strong spatial autocorrelation in the residuals (e.g., all prediction errors are clustered in one specific region), it suggests an omitted variable bias tied to geography. A random scatter of residuals indicates a well-specified model."),
    code_cell("""\
fig, ax = map_kulturkampf_residuals(
    gdf, panel,
    pre_years=(1868, 1872),
    post_years=(1873, 1878),
    savepath=str(OUTPUTS / "map4_residuals.png"),
)
plt.show()
""")
]


notebooks_to_create = {
    "01_data_and_eda.ipynb": nb1_cells,
    "02_baseline_regressions.ipynb": nb2_cells,
    "03_extensions_and_mechanisms.ipynb": nb3_cells,
    "04_spatial_analysis.ipynb": nb4_cells,
}

out_dir = Path("c:/Users/Anton/Demography/exam_project2/notebooks")
for name, cells in notebooks_to_create.items():
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump(create_nb(cells), f, indent=1)
    print(f"Created {name}")

old_nb = out_dir / "exam_project2.ipynb"
if old_nb.exists():
    os.remove(old_nb)
    print("Deleted original monolithic notebook exam_project2.ipynb")
