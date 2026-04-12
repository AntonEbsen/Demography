import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def append_intensity_did():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Markdown: Continuous DiD Rationale
    intensity_md = (
        "## **6. Continuous Difference-in-Differences (Intensity Model)**\n\n"
        "### **6.1 The Dosage Effect of Manufacturing**\n"
        "While Section 5 utilized a binary treatment indicator, this section implements an **Intensity Model**. We use the continuous 1831 manufacturing share (`Industrial_Ratio_1831`) to measure the 'dosage' of the Factory Acts across districts.\n\n"
        "This approach identifies whether districts with higher concentrations of manufacturing experienced a proportionally larger fertility response following the legislative shocks.\n\n"
        "### **6.2 Dynamic Interaction Terms**\n"
        "We construct year-specific interaction terms, using 1851 as the excluded reference year. This allows us to observe the evolving impact of industrial intensity across each decadal wave:\n"
        "- **`ratio_x_1861`**: Industrial Intensity $\\times$ 1861 wave.\n"
        "- **`ratio_x_1871`**: Industrial Intensity $\\times$ 1871 wave.\n"
        "- **`ratio_x_1881`**: Industrial Intensity $\\times$ 1881 wave."
    )
    nb.cells.append(nbf.v4.new_markdown_cell(intensity_md))

    # 2. Code: Intensity DiD Implementation
    intensity_code = (
        "import pandas as pd\n"
        "import glob\n"
        "import os\n"
        "import statsmodels.formula.api as smf\n"
        "import numpy as np\n\n"
        "### 1. Harmonize and Aggregate 1831 Intensity\n"
        "baseline_path = '../data/processed/census_1831_cleaned_for_merge.csv'\n"
        "df_1831 = pd.read_csv(baseline_path)\n"
        "df_1831['DISTRICT_ID'] = (df_1831['REGDIST'] // 1000) * 1000\n"
        "df_cont = df_1831.groupby('DISTRICT_ID')['Industrial_Ratio_1831'].mean().reset_index()\n\n"
        "### 2. Merge with Decadal Panel\n"
        "raw_data_dir = '../data/raw/'\n"
        "panel_years = [1851, 1861, 1871, 1881]\n"
        "panel_dfs = []\n\n"
        "for year in panel_years:\n"
        "    files = glob.glob(os.path.join(raw_data_dir, f'*{year}*.xlsx'))\n"
        "    if not files: continue\n"
        "    df_year = pd.read_excel(files[0])\n"
        "    df_year.columns = df_year.columns.str.upper()\n"
        "    df_year['YEAR'] = year\n"
        "    cen_col = f'CEN_{year}'\n"
        "    if cen_col in df_year.columns: df_year = df_year.rename(columns={cen_col: 'DISTRICT_ID'})\n"
        "    panel_dfs.append(df_year)\n\n"
        "df_p = pd.concat(panel_dfs, ignore_index=True)\n"
        "df_intensity = pd.merge(df_p, df_cont, on='DISTRICT_ID', how='inner')\n\n"
        "### 3. Feature Engineering: Interaction Terms\n"
        "for y in [1861, 1871, 1881]:\n"
        "    df_intensity[f'ratio_x_{y}'] = df_intensity['Industrial_Ratio_1831'] * (df_intensity['YEAR'] == y).astype(int)\n\n"
        "### 4. Execute TWFE Intensity Model\n"
        "reg_vars = ['TFR', 'Industrial_Ratio_1831', 'YEAR', 'IMR', 'DISTRICT_ID', 'ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']\n"
        "df_reg = df_intensity.dropna(subset=reg_vars).copy()\n\n"
        "model = smf.ols(\"TFR ~ C(YEAR) + ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + IMR + C(DISTRICT_ID)\", data=df_reg)\n"
        "results = model.fit(cov_type='HC3')\n\n"
        "print(\"--- Intensity DiD (Continuous Dosage) Results ---\")\n"
        "print(results.summary().tables[1]) # Focus on coefficients\n\n"
        "# Highlight Hypothesis Testing\n"
        "print(\"\\n--- Interpretation ---\")\n"
        "for var in ['ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']:\n"
        "    p = results.pvalues[var]\n"
        "    sig = \"***\" if p < 0.01 else \"**\" if p < 0.05 else \"*\" if p < 0.1 else \"\"\n"
        "    print(f\"{var}: P-value = {p:.4f} {sig}\")"
    )
    nb.cells.append(nbf.v4.new_code_cell(intensity_code))

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Intensity DiD section successfully appended to notebook.")

if __name__ == "__main__":
    append_intensity_did()
