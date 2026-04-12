import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def append_did_analysis():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Markdown: Methodology and Harmonization
    methodology_md = (
        "## **5. Difference-in-Differences Analysis (1831-1881)**\n\n"
        "### **5.1 Geographic Harmonization Strategy**\n"
        "To bridge the spatial mismatch between the 1831 parish-level counts and the 1851-1881 Registration District panel, we implement a strict geographic rounding rule. \n\n"
        "**Rule:** `DISTRICT_ID = (REGDIST // 1000) * 1000`\n\n"
        "This transforms the 1831 data (Sub-District level) into unified Registration District identifiers. We aggregate the baseline metrics by taking the **maximum** of the treatment dummy and the **mean** of manufacturing intensity within each District ID.\n\n"
        "### **5.2 Econometric Specification**\n"
        "We estimate a **Two-Way Fixed Effects (TWFE)** model with Robust Standard Errors (HC3):\n"
        "$$TFR_{it} = \\alpha + \\sum\\beta_t (Treated_i \\times Year_t) + \\delta IMR_{it} + \\mu_i + \\epsilon_{it}$$\n\n"
        "Where:\n"
        "- **$\mu_i$** represents District Fixed Effects.\n"
        "- **$Treated_i$** is the 1831 baseline indicator.\n"
        "- **$HC3$** robust standard errors are applied to account for potential heteroscedasticity across regional units."
    )
    nb.cells.append(nbf.v4.new_markdown_cell(methodology_md))

    # 2. Code: DiD Pipeline
    did_code = (
        "import pandas as pd\n"
        "import glob\n"
        "import os\n"
        "import statsmodels.formula.api as smf\n"
        "import numpy as np\n\n"
        "### 1. Load Baseline (1831)\n"
        "baseline_path = '../data/processed/census_1831_cleaned_for_merge.csv'\n"
        "df_1831 = pd.read_csv(baseline_path)\n\n"
        "# Apply Geographic Harmonization Rule\n"
        "df_1831['DISTRICT_ID'] = (df_1831['REGDIST'] // 1000) * 1000\n\n"
        "# Aggregate to District Level\n"
        "df_1831_agg = df_1831.groupby('DISTRICT_ID').agg({\n"
        "    'is_treated_baseline': 'max',\n"
        "    'MANUFAC': 'mean'\n"
        "}).reset_index()\n\n"
        "### 2. Standardize and Reconstruct the 1851-1881 Panel\n"
        "raw_data_dir = '../data/raw/'\n"
        "panel_years = [1851, 1861, 1871, 1881]\n"
        "panel_dfs = []\n\n"
        "for year in panel_years:\n"
        "    files = glob.glob(os.path.join(raw_data_dir, f'*{year}*.xlsx'))\n"
        "    if not files: continue\n"
        "    \n"
        "    df_year = pd.read_excel(files[0])\n"
        "    df_year.columns = df_year.columns.str.upper() # Normalize for KeyErrors\n"
        "    df_year['YEAR'] = year\n"
        "    \n"
        "    # Map census-specific ID to unified DISTRICT_ID\n"
        "    cen_col = f'CEN_{year}'\n"
        "    if cen_col in df_year.columns:\n"
        "        df_year = df_year.rename(columns={cen_col: 'DISTRICT_ID'})\n"
        "    \n"
        "    panel_dfs.append(df_year)\n\n"
        "df_panel_rec = pd.concat(panel_dfs, ignore_index=True)\n\n"
        "### 3. Merge Baseline and Panel\n"
        "df_final = pd.merge(df_panel_rec, df_1831_agg, on='DISTRICT_ID', how='inner')\n\n"
        "print(f\"Merge Metrics:\\n - Total Observations: {len(df_final)}\\n - Unique Districts: {df_final['DISTRICT_ID'].nunique()}\")\n\n"
        "### 4. Execute TWFE Regression\n"
        "reg_vars = ['TFR', 'is_treated_baseline', 'YEAR', 'IMR', 'DISTRICT_ID']\n"
        "df_reg = df_final.dropna(subset=reg_vars).copy()\n\n"
        "model = smf.ols(\"TFR ~ C(is_treated_baseline) * C(YEAR) + IMR + C(DISTRICT_ID)\", data=df_reg)\n"
        "results = model.fit(cov_type='HC3')\n\n"
        "print(\"\\n--- Difference-in-Differences Regression Results ---\")\n"
        "print(results.summary())"
    )
    nb.cells.append(nbf.v4.new_code_cell(did_code))

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("DiD Analysis section successfully appended to notebook.")

if __name__ == "__main__":
    append_did_analysis()
