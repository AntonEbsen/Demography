import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def create_base_notebook():
    nb = nbf.v4.new_notebook()
    
    # 1. Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("# Exam Project: The \"Cost of Quality\"\n## Legislative Shocks and the Fertility Transition"))
    
    # 2. Setup cell
    setup_code = """import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import glob
import os
from linearmodels import PanelOLS

def load_and_harmonize_data():
    \"\"\"
    Consolidated Geographic Harmonization & Data Loading Logic.
    Joins the 1831 Baseline with the 1851-1881 Panel and engineers DiD interaction terms.
    \"\"\"
    # 1. Load the Master Panel
    panel_path = '../data/processed/master_panel_data.csv'
    df = pd.read_csv(panel_path)
    df.columns = df.columns.str.upper()
    
    # 2. Load and Clean 1831 Baseline
    baseline_path = '../data/processed/census_1831_cleaned_for_merge.csv'
    df_1831 = pd.read_csv(baseline_path)
    df_1831['DISTRICT_ID'] = (df_1831['REGDIST'] // 1000) * 1000
    df_1831_agg = df_1831.groupby('DISTRICT_ID').agg({
        'is_treated_baseline': 'max',
        'Industrial_Ratio_1831': 'mean'
    }).reset_index()
    
    # 3. Harmonize IDs and Merge
    # We use 1851 CEN ID as the anchor for the Registration District panel
    df_final = pd.merge(df, df_1831_agg, left_on='CEN_1851', right_on='DISTRICT_ID', how='inner')
    
    # 4. Pre-calculate Interaction Terms for DiD and Intensity Models
    for yr in [1861, 1871, 1881]:
        # Binary DiD interactions
        df_final[f'treated_x_{yr}'] = ((df_final['is_treated_baseline'] == 1) & (df_final['YEAR'] == yr)).astype(int)
        # Continuous Intensity interactions
        df_final[f'ratio_x_{yr}'] = df_final['Industrial_Ratio_1831'] * (df_final['YEAR'] == yr).astype(int)
    
    return df_final

print(\"Shared utility functions loaded. Ready for analysis.\")"""
    nb.cells.append(nbf.v4.new_code_cell(setup_code))
    
    # 3. Load data
    load_code = "df_merged = load_and_harmonize_data()\nprint(f'Consolidated Dataset Loaded: {len(df_merged)} rows, {df_merged[\"DISTRICT_ID\"].nunique()} districts.')"
    nb.cells.append(nbf.v4.new_code_cell(load_code))
    
    # 4. Section 4: Regression Analysis
    nb.cells.append(nbf.v4.new_markdown_cell("### **4. Regression Analysis: Determinants of Mid-Victorian Fertility**"))

    # Adding Regressions 1-37 from the log (simplified/refactored)
    regressions = [
        # REG 1
        {"title": "Baseline Model: TFR ~ Child Labor", "code": "smf.ols('TFR ~ F_CL_1013 + IMR + C(YEAR)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 2 (Binary DiD)
        {"title": "TWFE Binary DiD", "code": "smf.ols('TFR ~ C(YEAR) + treated_x_1861 + treated_x_1871 + treated_x_1881 + IMR + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 3 (Continuous DiD)
        {"title": "Intensity Model (Industrial Dosage)", "code": "smf.ols('TFR ~ C(YEAR) + ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + IMR + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 8 (Triple Difference)
        {"title": "Triple Difference (DDD): Textile Heartlands", "code": "df_merged['is_heartland'] = (df_merged['Industrial_Ratio_1831'] > df_merged['Industrial_Ratio_1831'].median()).astype(int)\nsmf.ols('TFR ~ C(YEAR) * is_heartland + IMR + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 9 (Mechanism Test)
        {"title": "Mechanism Test: Direct Labor Impact", "code": "smf.ols('TFR ~ F_CL_1013 + IMR + C(YEAR) + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 11 (Teacher Density)
        {"title": "Impact on Educational Quality (Teacher Density)", "code": "smf.ols('C_TEACHER ~ ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + C(YEAR) + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 13 (Marriage Age)
        {"title": "Marriage Market Response (F_SMAM)", "code": "smf.ols('F_SMAM ~ ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + C(YEAR) + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 15 (Marital Fertility)
        {"title": "Intensive Margin: Marital Fertility (TMFR)", "code": "smf.ols('TMFR ~ ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + IMR + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 18 (Housing Stress)
        {"title": "Housing Stress: Boarders and Density", "code": "smf.ols('TFR ~ BOARD + POP_DENS + ratio_x_1871 + IMR + C(YEAR) + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 23 (Teen Substitution)
        {"title": "Age Substitution: 10-13 vs 14-18 Brackets", "code": "smf.ols('TFR ~ F_CL_1013 + F_CL_1418 + IMR + C(YEAR) + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 25 (Placebo)
        {"title": "Placebo Test: Impact on Blindness (HC1)", "code": "smf.ols('HC1 ~ ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + C(YEAR) + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
        # REG 36 (Gendered Labor)
        {"title": "Gendered Impact: Boys vs Girls Child Labor", "code": "smf.ols('TFR ~ F_CL_1013 + M_CL_1013 + IMR + C(YEAR) + C(DISTRICT_ID)', data=df_merged).fit(cov_type='HC3').summary()"},
    ]
    
    for reg in regressions:
        nb.cells.append(nbf.v4.new_markdown_cell(f"#### **{reg['title']}**"))
        nb.cells.append(nbf.v4.new_code_cell(reg['code']))

    # 5. Advanced Spatial Econometrics Section
    nb.cells.append(nbf.v4.new_markdown_cell("## **7. Advanced Spatial Econometrics**"))
    spatial_code = """from libpysal.weights import Queen
from esda.moran import Moran
from splot.esda import plot_local_autocorrelation
from spreg import ML_Lag

# Standardize data for 1881 cross-section
df_1881 = df_merged[df_merged['YEAR'] == 1881].copy()

# Load boundaries (assuming gdf is available or using district mapping)
# For this demonstration, we focus on the econometric specification
print(\"Spatial Econometrics setup complete. Refer to scripts/spatial_regression.py for full mapping pipeline.\")"""
    nb.cells.append(nbf.v4.new_code_cell(spatial_code))

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_base_notebook()
    print("Notebook successfully reconstructed with consolidated logic.")
