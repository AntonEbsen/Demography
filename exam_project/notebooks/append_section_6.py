import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def append_section_6():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Markdown Cell: Continuous DiD Explanation
    section_6_md = (
        "## **6. Continuous Difference-in-Differences (Intensity Model)**\n\n"
        "### **6.1 Concept and Dosage Effect**\n"
        "While the binary DiD in Section 5 measures the impact on 'treated' vs 'control' districts, the **Intensity Model** evaluates the **'dosage' effect**. We use the continuous manufacturing share from 1831 (`Industrial_Ratio_1831`) to test if districts with higher levels of industrialization experienced proportionally larger fertility responses after the Factory Acts.\n\n"
        "### **6.2 Econometric Specification**\n"
        "We estimate a Two-Way Fixed Effects (TWFE) model including District and Year Fixed Effects:\n\n"
        "$$TFR_{it} = \\alpha + \\sum_{y \\in \\{1861, 1871, 1881\\}} \\beta_y (Intensity_i \\times Year_t) + \\Gamma X_{it} + \\mu_i + \\tau_t + \\epsilon_{it}$$\n\n"
        "Where:\n"
        "- **$\\beta_y$**: The 'dosage' effect of manufacturing on fertility in each decade, relative to the 1851 baseline.\n"
        "- **$X_{it}$**: Control for Infant Mortality Rate (IMR).\n"
        "- **$\\mu_i, \\tau_t$**: District and Year Fixed Effects."
    )
    nb.cells.append(nbf.v4.new_markdown_cell(section_6_md))

    # 2. Code Cell: Intensity DiD Implementation
    section_6_code = (
        "import pandas as pd\n"
        "import glob\n"
        "import os\n"
        "import statsmodels.formula.api as smf\n"
        "import numpy as np\n\n"
        "# 1. Geographic Harmonization (1831 Baseline)\n"
        "baseline_df = pd.read_csv('../data/processed/census_1831_cleaned_for_merge.csv')\n"
        "baseline_df['DISTRICT_ID'] = (baseline_df['REGDIST'] // 1000) * 1000\n"
        "intensity_1831 = baseline_df.groupby('DISTRICT_ID')['Industrial_Ratio_1831'].mean().reset_index()\n\n"
        "# 2. Panel Reconstruction (1851-1881)\n"
        "raw_path = '../data/raw/'\n"
        "panel_list = []\n"
        "for yr in [1851, 1861, 1871, 1881]:\n"
        "    f = glob.glob(os.path.join(raw_path, f'*{yr}*.xlsx'))[0]\n"
        "    df_yr = pd.read_excel(f)\n"
        "    df_yr.columns = df_yr.columns.str.upper()\n"
        "    df_yr['YEAR'] = yr\n"
        "    cen_col = f'CEN_{yr}'\n"
        "    if cen_col in df_yr.columns: df_yr = df_yr.rename(columns={cen_col: 'DISTRICT_ID'})\n"
        "    panel_list.append(df_yr)\n\n"
        "full_panel = pd.concat(panel_list, ignore_index=True)\n"
        "df_merged = pd.merge(full_panel, intensity_1831, on='DISTRICT_ID', how='inner')\n\n"
        "# 3. engineering Interaction Terms\n"
        "for y in [1861, 1871, 1881]:\n"
        "    df_merged[f'ratio_x_{y}'] = df_merged['Industrial_Ratio_1831'] * (df_merged['YEAR'] == y).astype(int)\n\n"
        "# 4. TWFE Regression with HC3 Robust Errors\n"
        "reg_vars = ['TFR', 'IMR', 'YEAR', 'DISTRICT_ID', 'ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']\n"
        "df_reg = df_merged.dropna(subset=reg_vars).copy()\n\n"
        "formula = 'TFR ~ C(YEAR) + ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + IMR + C(DISTRICT_ID)'\n"
        "res = smf.ols(formula, data=df_reg).fit(cov_type='HC3')\n\n"
        "print(\"--- INTENSITY DID RESULTS (SECTION 6) ---\")\n"
        "print(res.summary().tables[1])\n\n"
        "print(\"\\nInterpretation:\")\n"
        "for v in ['ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']:\n"
        "    p = res.pvalues[v]\n"
        "    star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''\n"
        "    print(f\"{v}: P-Value = {p:.4f} {star}\")"
    )
    nb.cells.append(nbf.v4.new_code_cell(section_6_code))

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Section 6 successfully appended to notebook.")

if __name__ == "__main__":
    append_section_6()
