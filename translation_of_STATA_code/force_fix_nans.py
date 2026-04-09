import json
import os

path = 'c:/Users/Anton/Demography/translation_of_STATA_code/notebooks/exam_project.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

reg_vars_str = "    reg_vars = ['TFR', 'treat_dummy', 'Year', 'IMR', 'SC6', 'REGDIST']\n"

for cell in nb['cells']:
    source = cell.get('source', [])
    source_str = "".join(source)
    
    if "### 7. Difference-in-Differences (DiD) Regression" in source_str:
        cell['source'] = [
            "import statsmodels.formula.api as smf\n",
            "\n",
            "### 7. Difference-in-Differences (DiD) Regression\n",
            "# FIX: We drop rows with missing values to ensure the clustering variable matches the fitted data length.\n",
            "reg_vars = ['TFR', 'treat_dummy', 'Year', 'IMR', 'SC6', 'REGDIST']\n",
            "df_reg = df_panel.dropna(subset=reg_vars).copy()\n",
            "df_reg['Year'] = df_reg['Year'].astype(int)\n",
            "\n",
            "model = smf.ols(\"TFR ~ treat_dummy * C(Year) + IMR + SC6\", data=df_reg)\n",
            "results = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg['REGDIST']})\n",
            "\n",
            "print(\"Difference-in-Differences Estimation Results (Clustered SEs):\")\n",
            "print(results.summary())\n",
            "\n",
            "print(\"\\n--- DiD Coefficients (Treatment impacts over time) ---\")\n",
            "print(results.params.filter(like='treat_dummy:'))\n"
        ]
    
    elif "### 9. Heterogeneity Analysis: Urban vs. Rural" in source_str:
        cell['source'] = [
            "### 9. Heterogeneity Analysis: Urban vs. Rural\n",
            "urban_types = ['textile', 'professional', 'transport', 'other urban', 'semi-professional']\n",
            "df_panel['is_urban'] = df_panel['TYPE'].isin(urban_types).astype(int)\n",
            "\n",
            "# Prepare clean subsets for Urban and Rural to avoid length mismatch errors\n",
            "reg_vars = ['TFR', 'treat_dummy', 'Year', 'IMR', 'SC6', 'REGDIST']\n",
            "df_urban = df_panel[df_panel['is_urban'] == 1].dropna(subset=reg_vars).copy()\n",
            "df_rural = df_panel[df_panel['is_urban'] == 0].dropna(subset=reg_vars).copy()\n",
            "\n",
            "model_urban = smf.ols(\"TFR ~ treat_dummy * C(Year) + IMR + SC6\", data=df_urban).fit(cov_type='cluster', cov_kwds={'groups': df_urban['REGDIST']})\n",
            "model_rural = smf.ols(\"TFR ~ treat_dummy * C(Year) + IMR + SC6\", data=df_rural).fit(cov_type='cluster', cov_kwds={'groups': df_rural['REGDIST']})\n",
            "\n",
            "print(\"DIFFERENCE-IN-DIFFERENCES: URBAN DISTRICTS\")\n",
            "print(model_urban.params.filter(like='treat_dummy:'))\n",
            "print(\"\\nDIFFERENCE-IN-DIFFERENCES: RURAL DISTRICTS\")\n",
            "print(model_rural.params.filter(like='treat_dummy:'))\n"
        ]

    elif "### 11. Robustness Check: Leave-One-County-Out" in source_str:
        cell['source'] = [
            "### 11. Robustness Check: Leave-One-County-Out\n",
            "reg_vars = ['TFR', 'treat_dummy', 'Year', 'IMR', 'SC6', 'REGDIST']\n",
            "top_counties = df_panel[df_panel['Year'] == 1851].groupby('REGCNTY')['F_TEX'].mean().sort_values(ascending=False).head(5).index.tolist()\n",
            "print(f\"Testing robustness by dropping: {top_counties}\")\n",
            "\n",
            "robustness_results = []\n",
            "for county in top_counties:\n",
            "    # Filter and drop NaNs for this iteration\n",
            "    df_sub = df_panel[df_panel['REGCNTY'] != county].dropna(subset=reg_vars).copy()\n",
            "    mod = smf.ols(\"TFR ~ treat_dummy * C(Year) + IMR + SC6\", data=df_sub).fit(cov_type='cluster', cov_kwds={'groups': df_sub['REGDIST']})\n",
            "    coef_1881 = mod.params.get('treat_dummy:C(Year)[T.1881]', None)\n",
            "    robustness_results.append({'Dropped': county, 'Beta_1881': coef_1881})\n",
            "\n",
            "print(\"--- Robustness Results (1881 Coefficients) ---\")\n",
            "print(pd.DataFrame(robustness_results))\n"
        ]

    elif "### 12. Mechanism Check: The Schooling Proxy" in source_str:
        cell['source'] = [
            "### 12. Mechanism Check: The Schooling Proxy\n",
            "mechanism_vars = ['C_TEACHER', 'treat_dummy', 'Year', 'IMR', 'SC6', 'REGDIST']\n",
            "df_mech = df_panel.dropna(subset=mechanism_vars).copy()\n",
            "\n",
            "model_schooling = smf.ols(\"C_TEACHER ~ treat_dummy * C(Year) + IMR + SC6\", data=df_mech)\n",
            "results_schooling = model_schooling.fit(cov_type='cluster', cov_kwds={'groups': df_mech['REGDIST']})\n",
            "\n",
            "print(\"MECHANISM CHECK: IMPACT ON SCHOOLING PROXY (C_TEACHER)\")\n",
            "print(results_schooling.params.filter(like='treat_dummy:'))\n"
        ]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with NaN bug fixes.")
