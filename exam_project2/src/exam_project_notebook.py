"""
exam_project_notebook.py
========================
Template code for the Jupyter notebook.
Copy each section (marked with # %%) into a separate notebook cell.

Project: The Kulturkampf and Catholic Fertility in Prussia, 1862-1890
"""

# %% [markdown]
# # The Kulturkampf and Catholic Fertility in Prussia
# 
# **Research question:** Did Bismarck's anti-Catholic Kulturkampf legislation 
# (1872–1878) affect the Catholic–Protestant fertility differential in 
# Prussian counties?
#
# **Data:** Galloway Prussia Database (1861–1914) + iPEHD (1871)

# %% Cell 1: Setup
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
OUTPUTS = project_root / "outputs" / "figures"
OUTPUTS.mkdir(exist_ok=True)

print("Setup complete.")
print(f"Raw data directory: {DATA_RAW}")
print(f"Files found: {sorted([f.name for f in DATA_RAW.glob('*.XLS')])[:10]}...")


# %% Cell 2: Build the analysis panel
panel = build_analysis_panel(
    data_dir=DATA_RAW,
    year_start=1862,
    year_end=1890,
    save=True,
)

panel.head(10)


# %% Cell 3: Descriptive statistics
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


# %% Cell 4: Distribution of Catholic shares
fig, ax = plot_cath_distribution(
    panel,
    savepath=str(OUTPUTS / "fig1_cath_distribution.png"),
)
plt.show()


# %% Cell 5: Fertility trends over time
fig, ax = plot_fertility_trends(
    panel,
    outcome="cbr",
    ylabel="Crude birth rate (per 1,000)",
    title="Fertility trends: High- vs Low-Catholic counties",
    savepath=str(OUTPUTS / "fig2_fertility_trends.png"),
)
plt.show()


# %% Cell 6: Fertility trends – legitimate births only
# (Only available from 1875 when detailed VIT files start)
panel_post75 = panel[panel["Year"] >= 1875].copy()

fig, ax = plot_fertility_trends(
    panel_post75,
    outcome="legitimate_br",
    ylabel="Legitimate birth rate (per 1,000)",
    title="Legitimate fertility trends: High- vs Low-Catholic counties",
    savepath=str(OUTPUTS / "fig3_legit_fertility_trends.png"),
)
plt.show()


# %% Cell 7: Marriage trends by religion
# Catholic marriage share only available from 1875
fig, ax = plot_fertility_trends(
    panel_post75,
    outcome="cath_marriage_share",
    ylabel="Catholic marriages (% of total)",
    title="Catholic marriage share over time",
    savepath=str(OUTPUTS / "fig4_cath_marriages.png"),
)
plt.show()


# %% Cell 8: Baseline DiD – Continuous treatment
print("=" * 60)
print("BASELINE DiD: CBR ~ CathShare × Post")
print("=" * 60)

res_cont = run_baseline_did(
    panel,
    outcome="cbr",
    treatment="continuous",
)
print(res_cont["summary"])


# %% Cell 9: Baseline DiD – Binary treatment
print("=" * 60)
print("BASELINE DiD: CBR ~ HighCath × Post")
print("=" * 60)

res_bin = run_baseline_did(
    panel,
    outcome="cbr",
    treatment="binary",
)
print(res_bin["summary"])


# %% Cell 10: DiD for other outcomes
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


# %% Cell 11: Event study
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


# %% Cell 12: Robustness checks
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


# %% Cell 13: Validate against iPEHD
print("=" * 60)
print("VALIDATION: Compare Catholic shares with iPEHD")
print("=" * 60)

ipehd = load_ipehd_master(DATA_RAW / "ipehd_qje2009_master.dta")
rel = load_rel1871(DATA_RAW / "REL1871.XLS")

# The two datasets use different county codes, so we compare distributions
print(f"iPEHD f_cath: mean={ipehd['f_cath'].mean():.1f}, "
      f"median={ipehd['f_cath'].median():.1f}, N={len(ipehd)}")
print(f"Galloway cath_share: mean={rel['cath_share'].mean():.1f}, "
      f"median={rel['cath_share'].median():.1f}, N={len(rel)}")
print("\nDistributions should be similar (not identical due to different")
print("sample definitions and Stadt/Land handling).")


# %% [markdown]
# ## Notes on identification
# 
# **What this design can show:**
# - Whether the Catholic–Protestant fertility differential changed around 
#   the Kulturkampf period
# - Whether the timing lines up with the legislation (event study)
# 
# **What this design CANNOT show:**
# - Clean causal effect: Catholic share is not randomly assigned. Counties 
#   differ in urbanisation, occupational structure, literacy, etc.
# - County + year fixed effects absorb time-invariant county differences 
#   and common shocks, but cannot rule out differential trends correlated 
#   with Catholic share
# 
# **Key threats to identification:**
# 1. The Kulturkampf was a *response* to Catholic political mobilisation, 
#    not a random shock
# 2. Catholic and Protestant counties may have been on different fertility 
#    trajectories for other reasons (industrialisation, urbanisation)
# 3. The event study pre-trends test is essential: if coefficients are 
#    already trending before 1872, the DiD is compromised
# 
# **This is fine for a course paper** — the examiner values transparent 
# discussion of these limitations.
