import pandas as pd
import glob
import os
import statsmodels.formula.api as smf
import numpy as np

def perform_continuous_did():
    # 1. Load Baseline (1831)
    baseline_path = 'exam_project/data/processed/census_1831_cleaned_for_merge.csv'
    df_1831 = pd.read_csv(baseline_path)
    
    # Geographic Harmonization (REGDIST -> DISTRICT_ID)
    df_1831['DISTRICT_ID'] = (df_1831['REGDIST'] // 1000) * 1000
    
    # Aggregate Continuous Intensity (Mean ratio per District)
    df_1831_agg = df_1831.groupby('DISTRICT_ID')['Industrial_Ratio_1831'].mean().reset_index()
    
    # 2. Reconstruct Panel (1851-1881)
    raw_data_dir = 'exam_project/data/raw/'
    panel_years = [1851, 1861, 1871, 1881]
    panel_dfs = []
    
    for year in panel_years:
        files = glob.glob(os.path.join(raw_data_dir, f'*{year}*.xlsx'))
        if not files: continue
        
        df_year = pd.read_excel(files[0])
        df_year.columns = df_year.columns.str.upper()
        df_year['YEAR'] = year
        
        cen_col = f'CEN_{year}'
        if cen_col in df_year.columns:
            df_year = df_year.rename(columns={cen_col: 'DISTRICT_ID'})
        panel_dfs.append(df_year)
        
    df_panel = pd.concat(panel_dfs, ignore_index=True)
    
    # 3. Merge
    df_merged = pd.merge(df_panel, df_1831_agg, on='DISTRICT_ID', how='inner')
    
    # 4. Construct Interaction Terms (Continuous DiD)
    # Reference Year is 1851 (implicitly handled by omitting it)
    df_merged['ratio_x_1861'] = df_merged['Industrial_Ratio_1831'] * (df_merged['YEAR'] == 1861).astype(int)
    df_merged['ratio_x_1871'] = df_merged['Industrial_Ratio_1831'] * (df_merged['YEAR'] == 1871).astype(int)
    df_merged['ratio_x_1881'] = df_merged['Industrial_Ratio_1831'] * (df_merged['YEAR'] == 1881).astype(int)
    
    # 5. TWFE Intensity Model
    reg_vars = ['TFR', 'Industrial_Ratio_1831', 'YEAR', 'IMR', 'DISTRICT_ID', 'ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']
    df_reg = df_merged.dropna(subset=reg_vars).copy()
    
    formula = "TFR ~ C(YEAR) + ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + IMR + C(DISTRICT_ID)"
    model = smf.ols(formula, data=df_reg)
    results = model.fit(cov_type='HC3')
    
    print("\n--- Continuous Difference-in-Differences (Intensity) Summary ---")
    print(results.summary())
    
    # Specifically print coefficients and symbols
    print("\nHighlighting Key Parameters:")
    for interaction in ['ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']:
        if interaction in results.params:
            coef = results.params[interaction]
            p_val = results.pvalues[interaction]
            print(f"- {interaction}: Coef = {coef:.4f}, P-value = {p_val:.4f}")
            
    return df_reg, results

if __name__ == "__main__":
    df, res = perform_continuous_did()
