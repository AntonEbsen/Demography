
# %% Cell 1 (Markdown)
# # The Kulturkampf and Catholic Fertility in Prussia 
# ### **Research question:** Did Bismarck's anti-Catholic Kulturkampf legislation (1872–1878) affect the Catholic–Protestant fertility differential in Prussian counties?
# ### **Data:** Galloway Prussia Database (1861–1914) + iPEHD (1871)

# %% Cell 2 (Markdown)
# # Cell 1: Setup

# %% Cell 3 (Code)
import sys
from pathlib import Path
 
# Add project root to path so we can import from src/
project_root = Path.cwd().parent  # assumes notebook is in notebooks/
sys.path.insert(0, str(project_root))
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from src.data.load_data import load_rel1871, load_vit_panel, load_ipehd_master
from src.data.build_dataset import build_analysis_panel
from src.analysis.regressions import run_baseline_did, run_event_study, run_robustness
from src.visualization.plots import (
    plot_fertility_trends, 
    plot_event_study, 
    plot_cath_distribution,
    plot_robustness_table,
)
 
# Paths
DATA_RAW = project_root / "data" / "raw"
OUTPUTS = project_root / "outputs"
OUTPUTS.mkdir(exist_ok=True)
 
print("Setup complete.")
print(f"Raw data directory: {DATA_RAW}")
print(f"Files found: {sorted([f.name for f in DATA_RAW.glob('*.XLSX')])[:10]}...")

# %% Cell 4 (Markdown)
# ### Cell 2: Build the analysis panel

# %% Cell 5 (Code)
panel = build_analysis_panel(
    data_dir=DATA_RAW,
    year_start=1862,
    year_end=1890,
    save=True,
)
 
panel.head(10)

# %% Cell 6 (Markdown)
# ### Cell 3: Descriptive statistics

# %% Cell 7 (Code)
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
 
# Summary by period
for period, label in [(panel["Year"] < 1873, "Pre-Kulturkampf (1862-1872)"),
                       (panel["Year"] >= 1873, "Post-Kulturkampf (1873-1890)")]:
    sub = panel[period]
    print(f"\n{label}:")
    print(f"  N = {len(sub)}, Counties = {sub['Code'].nunique()}")
    for var in ["cbr", "legitimate_br", "marriage_rate"]:
        if var in sub.columns and sub[var].notna().any():
            print(f"  {var}: mean={sub[var].mean():.2f}, sd={sub[var].std():.2f}")
 
# Summary by Catholic share group
print(f"\n{'='*60}")
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

# %% Cell 8 (Markdown)
# ### Cell 4: Distribution of Catholic shares

# %% Cell 9 (Code)
fig, ax = plot_cath_distribution(
    panel,
    savepath=str(OUTPUTS / "fig1_cath_distribution.png"),
)
plt.show()

# %% Cell 10 (Markdown)
# ### Cell 5: Fertility trends over time

# %% Cell 11 (Code)
fig, ax = plot_fertility_trends(
    panel,
    outcome="cbr",
    ylabel="Crude birth rate (per 1,000)",
    title="Fertility trends: High- vs Low-Catholic counties",
    savepath=str(OUTPUTS / "fig2_fertility_trends.png"),
)
plt.show()

# %% Cell 12 (Markdown)
# ### Cell 6: Fertility trends – legitimate births only
# (Only available from 1875 when detailed VIT files start)

# %% Cell 13 (Code)
panel_post75 = panel[panel["Year"] >= 1875].copy()
 
fig, ax = plot_fertility_trends(
    panel_post75,
    outcome="legitimate_br",
    ylabel="Legitimate birth rate (per 1,000)",
    title="Legitimate fertility trends: High- vs Low-Catholic counties",
    savepath=str(OUTPUTS / "fig3_legit_fertility_trends.png"),
)
plt.show()

# %% Cell 14 (Markdown)
# ### Cell 7: Marriage trends by religion
# ### Catholic marriage share only available from 1875

# %% Cell 15 (Code)
fig, ax = plot_fertility_trends(
    panel_post75,
    outcome="cath_marriage_share",
    ylabel="Catholic marriages (% of total)",
    title="Catholic marriage share over time",
    savepath=str(OUTPUTS / "fig4_cath_marriages.png"),
)
plt.show()

# %% Cell 16 (Markdown)
# ### Cell 8: Baseline DiD – Continuous treatment

# %% Cell 17 (Code)
print("=" * 60)
print("BASELINE DiD: CBR ~ CathShare × Post")
print("=" * 60)
 
res_cont = run_baseline_did(
    panel,
    outcome="cbr",
    treatment="continuous",
)
print(res_cont["summary"])


# %% Cell 18 (Markdown)
# ### Cell 9: Baseline DiD – Binary treatment

# %% Cell 19 (Code)
print("=" * 60)
print("BASELINE DiD: CBR ~ HighCath × Post")
print("=" * 60)
 
res_bin = run_baseline_did(
    panel,
    outcome="cbr",
    treatment="binary",
)
print(res_bin["summary"])

# %% Cell 20 (Markdown)
# ### Cell 10: DiD for other outcomes

# %% Cell 21 (Code)
print("=" * 60)
print("DiD FOR ALTERNATIVE OUTCOMES")
print("=" * 60)
 
for outcome, label in [
    ("legitimate_br", "Legitimate birth rate"),
    ("marriage_rate", "Marriage rate"),
    ("illegitimacy_ratio", "Illegitimacy ratio"),
]:
    print(f"\n--- {label} ---")
    try:
        res = run_baseline_did(panel, outcome=outcome, treatment="continuous")
        r = res["result"]
        treat_var = "cath_share_x_post"
        print(f"  Coef: {r.params[treat_var]:.4f} "
              f"(SE: {r.std_errors[treat_var]:.4f}, "
              f"p: {r.pvalues[treat_var]:.3f})")
    except Exception as e:
        print(f"  Error: {e}")

# %% Cell 22 (Markdown)
# ### Cell 11: Event study

# %% Cell 23 (Code)
print("=" * 60)
print("EVENT STUDY")
print("=" * 60)
 
es = run_event_study(
    panel,
    outcome="cbr",
    treatment_var="cath_share",
    ref_year=1872,
)
 
fig, ax = plot_event_study(
    es["coefs"],
    ref_year=1872,
    title="Event study: Catholic share × Year dummies on CBR",
    savepath=str(OUTPUTS / "fig5_event_study.png"),
)
plt.show()
 
# Print pre-trend test: are pre-1872 coefficients jointly zero?
pre_coefs = es["coefs"][es["coefs"]["Year"] < 1872]
print(f"\nPre-trend coefficients (before 1872):")
print(pre_coefs[["Year", "beta", "se"]].to_string(index=False))

# %% Cell 24 (Markdown)
# ### Cell 12: Robustness checks

# %% Cell 25 (Code)
print("=" * 60)
print("ROBUSTNESS CHECKS")
print("=" * 60)
 
rob = run_robustness(panel, outcome="cbr")
print(rob.to_string(index=False))
 
fig, ax = plot_robustness_table(
    rob,
    savepath=str(OUTPUTS / "fig6_robustness.png"),
)
plt.show()

# %% Cell 26 (Markdown)
# ### Cell 13: Validate against iPEHD

# %% Cell 27 (Code)
print("=" * 60)
print("VALIDATION: Compare Catholic shares with iPEHD")
print("=" * 60)
 
ipehd = load_ipehd_master(DATA_RAW / "ipehd_qje2009_master.dta")
rel = load_rel1871(DATA_RAW / "REL1871.XLSX")
 
# The two datasets use different county codes, so we compare distributions
print(f"iPEHD f_cath: mean={ipehd['f_cath'].mean():.1f}, "
      f"median={ipehd['f_cath'].median():.1f}, N={len(ipehd)}")
print(f"Galloway cath_share: mean={rel['cath_share'].mean():.1f}, "
      f"median={rel['cath_share'].median():.1f}, N={len(rel)}")
print("\nDistributions should be similar (not identical due to different")
print("sample definitions and Stadt/Land handling).")

# %% Cell 28 (Markdown)
# ### Cell 14: Setup exploratory analyses

# %% Cell 29 (Code)
from src.analysis.exploratory import (
    heterogeneity_by_urbanization,
    polish_vs_german_catholics,
    fertility_convergence,
    marriage_to_birth_pipeline,
    dose_response_plot,
    infant_mortality_did,
)

# %% Cell 30 (Markdown)
# ### Cell 15: Urban vs Rural heterogeneity

# %% Cell 31 (Code)
print("=" * 60)
print("HETEROGENEITY: URBAN vs RURAL")
print("=" * 60)
het_results = heterogeneity_by_urbanization(panel, outcome="cbr")

# %% Cell 32 (Markdown)
# ### Cell 16: Polish vs German Catholics

# %% Cell 33 (Code)
print("=" * 60)
print("POLISH vs GERMAN CATHOLICS")
print("=" * 60)
pol_results = polish_vs_german_catholics(panel, outcome="cbr")

# %% Cell 34 (Markdown)
# ### Cell 17: Fertility convergence

# %% Cell 35 (Code)
print("=" * 60)
print("FERTILITY CONVERGENCE")
print("=" * 60)
fig_conv, conv_data = fertility_convergence(panel)
fig_conv.savefig(OUTPUTS / "fig7_convergence.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Cell 36 (Markdown)
# ### Cell 18: Marriage to birth pipeline (lagged effects)

# %% Cell 37 (Code)
print("=" * 60)
print("MARRIAGE → BIRTH PIPELINE (LAGGED EFFECTS)")
print("=" * 60)
lag_results = marriage_to_birth_pipeline(panel)

# %% Cell 38 (Markdown)
# ### Cell 19: Dose-response plot

# %% Cell 39 (Code)
print("=" * 60)
print("DOSE-RESPONSE BY CATHOLIC SHARE BIN")
print("=" * 60)
fig_dose, dose_data = dose_response_plot(
    panel, savepath=str(OUTPUTS / "fig8_dose_response.png")
)
plt.show()

# %% Cell 40 (Markdown)
# ### Cell 20: Infant mortality channel

# %% Cell 41 (Code)
print("=" * 60)
print("INFANT MORTALITY CHANNEL")
print("=" * 60)
imr_result = infant_mortality_did(panel)

# %% Cell 42 (Markdown)
# ## Notes on identification
#  
# **What this design can show:**
# - Whether the Catholic–Protestant fertility differential changed around the Kulturkampf period - Whether the timing lines up with the legislation (event study)
# 
# **What this design CANNOT show:**
# - Clean causal effect: Catholic share is not randomly assigned. Counties differ in urbanisation, occupational structure, literacy, etc. - County + year fixed effects absorb time-invariant county differences and common shocks, but cannot rule out differential trends correlated with Catholic share
# 
# **Key threats to identification:**
# 1. The Kulturkampf was a *response* to Catholic political mobilisation, not a random shock
# 2. Catholic and Protestant counties may have been on different fertility trajectories for other reasons (industrialisation, urbanisation)
# 3. The event study pre-trends test is essential: if coefficients are already trending before 1872, the DiD is compromised
#  
# **This is fine for a course paper** — the examiner values transparent discussion of these limitations.

# %% Cell 43 (Markdown)
# ### Cell 21: Rollback event study

# %% Cell 44 (Code)
import importlib
import src.advanced
importlib.reload(src.advanced)
from src.analysis.advanced import (
    rollback_event_study,
    illegitimacy_analysis,
    infant_mortality_analysis,
    franco_prussian_war_analysis,
    robustness_exclude_war,
    trend_adjusted_did,
    polish_german_rollback,
    placebo_test,
)

print("=" * 60)
print("ROLLBACK EVENT STUDY")
print("=" * 60)
rollback = rollback_event_study(
    panel,
    outcome="cbr",
    treatment_var="cath_share",
    ref_year=1872,
    savepath=str(OUTPUTS / "fig9_rollback_event_study.png"),
)
plt.show()

# %% Cell 45 (Markdown)
# ### Cell 22: Illegitimacy analysis

# %% Cell 46 (Code)
ill = illegitimacy_analysis(panel)
ill["fig"].savefig(OUTPUTS / "fig10_illegitimacy.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Cell 47 (Markdown)
# ### Cell 23: Infant mortality channel

# %% Cell 48 (Code)
imr = infant_mortality_analysis(panel)
imr["fig"].savefig(OUTPUTS / "fig11_infant_mortality.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Cell 49 (Markdown)
# ### Cell 24: Franco-Prussian War

# %% Cell 50 (Code)
war = franco_prussian_war_analysis(panel)
war["fig"].savefig(OUTPUTS / "fig12_war_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Cell 51 (Markdown)
# ### Cell 25: Robustness — exclude war years

# %% Cell 52 (Code)
war_rob = robustness_exclude_war(
    panel,
    outcome="cbr",
    war_years=(1870, 1871, 1872),
    ref_year=1869,
    savepath=str(OUTPUTS / "fig13_war_excluded_event_study.png"),
)
plt.show()

# %% Cell 53 (Code)
war_rob_1868 = robustness_exclude_war(
    panel, ref_year=1868,
    savepath=str(OUTPUTS / "fig13b_ref1868.png"),
)

# %% Cell 54 (Markdown)
# ### Cell 26: Trend-adjusted DiD

# %% Cell 55 (Code)
# Main trend-adjusted estimate
ta_main = trend_adjusted_did(
    panel,
    outcome="cbr",
    trend_base_year=1862,
    exclude_war=False,
)

print("\n\n")

# Also run with war years excluded, as a further robustness check
ta_nowar = trend_adjusted_did(
    panel,
    outcome="cbr",
    trend_base_year=1862,
    exclude_war=True,
    war_years=(1870, 1871, 1872),
)

# %% Cell 56 (Markdown)
# ### Cell 27: Polish-German × Rollback interaction

# %% Cell 57 (Code)
pgr = polish_german_rollback(
    panel,
    outcome="cbr",
    savepath=str(OUTPUTS / "fig14_polish_german_rollback.png"),
)
plt.show()

# %% Cell 58 (Markdown)
# ### Cell 28: Placebo test

# %% Cell 59 (Code)
plc = placebo_test(
    panel,
    outcome="cbr",
    placebo_years=[1864, 1866, 1868, 1870, 1873, 1876, 1880, 1884],
    savepath=str(OUTPUTS / "fig15_placebo_test.png"),
)
plt.show()

# %% Cell 60 (Markdown)
# ### Cell 29: Load shapefile

# %% Cell 61 (Code)
from src.visualization.maps import (
    load_prussia_shapefile,
    map_catholic_share,
    map_fertility_change,
    map_polish_german_provinces,
    map_kulturkampf_residuals,
)

# Adjust path to match where you put the shapefile
shp_path = DATA_RAW / "German_Empire_1871_v.1.0.shp"
gdf = load_prussia_shapefile(shp_path)
print(f"Loaded {len(gdf)} Prussian counties from shapefile")

# %% Cell 62 (Markdown)
# ### Cell 30: Map 1 — Catholic share in 1871

# %% Cell 63 (Code)
fig, ax = map_catholic_share(
    gdf, panel,
    savepath=str(OUTPUTS / "map1_catholic_share.png"),
)
plt.show()

# %% Cell 64 (Markdown)
# ### Cell 31: Map 2 — Fertility change pre vs post Kulturkampf

# %% Cell 65 (Code)
fig, ax = map_fertility_change(
    gdf, panel,
    pre_years=(1868, 1872),
    post_years=(1878, 1882),
    savepath=str(OUTPUTS / "map2_fertility_change.png"),
)
plt.show()

# %% Cell 66 (Markdown)
# ### Cell 32: Map 3 — Polish vs German Catholic provinces

# %% Cell 67 (Code)
fig, ax = map_polish_german_provinces(
    gdf, panel,
    savepath=str(OUTPUTS / "map3_polish_german.png"),
)
plt.show()

# %% Cell 68 (Markdown)
# ### Cell 33: Map 4 — County-level residuals

# %% Cell 69 (Code)
fig, ax = map_kulturkampf_residuals(
    gdf, panel,
    pre_years=(1868, 1872),
    post_years=(1873, 1878),
    savepath=str(OUTPUTS / "map4_residuals.png"),
)
plt.show()

# %% Cell 70 (Code)

