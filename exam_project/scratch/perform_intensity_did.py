import pandas as pd
import glob
import os
import statsmodels.formula.api as smf
import numpy as np

def perform_intensity_did():
    # 1. Load Baseline (1831)
    baseline_path = 'exam_project/data/processed/census_1831_cleaned_for_merge.csv'
    df_1831 = pd.read_csv(baseline_path)
    
    # Harmonize IDs (Sub-District REGDIST -> District level DISTRICT_ID)
    df_1831['DISTRICT_ID'] = (df_1831['REGDIST'] // 1000) * 1000
    
    # Aggregate Continuous Industrial Intensity (Mean Industrial Ratio per District)
    df_1831_agg = df_1831.groupby('DISTRICT_ID')['Industrial_Ratio_1831'].mean().reset_index()
    
    # 2. Reconstruct decadal Panel (1851, 1861, 1871, 1881)
    # Using relative paths from the root to ensure standard script execution
    raw_data_dir = 'exam_project/data/raw/'
    panel_years = [1851, 1861, 1871, 1881]
    dfs = []
    
    for year in panel_years:
        files = glob.glob(os.path.join(raw_data_dir, f'PopulationsPast_census_data_{year}.xlsx'))
        if not files: 
            # Try flexible globbing if exact name fails
            files = glob.glob(os.path.join(raw_data_dir, f'*{year}*.xlsx'))
        
        if files:
            df_year = pd.read_excel(files[0])
            df_year.columns = df_year.columns.str.upper()
            df_year['YEAR'] = year
            # Rename CEN_YYYY to DISTRICT_ID
            cen_col = f'CEN_{year}'
            if cen_col in df_year.columns:
                df_year = df_year.rename(columns={cen_col: 'DISTRICT_ID'})
            dfs.append(df_year)
    
    df_panel = pd.concat(dfs, ignore_index=True)
    
    # 3. Merge Panel with 1831 Industrial Intensity
    df_merged = pd.merge(df_panel, df_1831_agg, on='DISTRICT_ID', how='inner')
    
    # 4. Feature Engineering: Interaction Terms (Intensity x Post-Shock Decades)
    # 1851 is the omitted reference category
    df_merged['ratio_x_1861'] = df_merged['Industrial_Ratio_1831'] * (df_merged['YEAR'] == 1861).astype(int)
    df_merged['ratio_x_1871'] = df_merged['Industrial_Ratio_1831'] * (df_merged['YEAR'] == 1871).astype(int)
    df_merged['ratio_x_1881'] = df_merged['Industrial_Ratio_1831'] * (df_merged['YEAR'] == 1881).astype(int)
    
    # 5. Clean for Regression (Drop NaNs in key variables)
    reg_vars = ['TFR', 'IMR', 'Industrial_Ratio_1831', 'YEAR', 'DISTRICT_ID', 
                'ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']
    df_reg = df_merged.dropna(subset=reg_vars).copy()
    
    # 6. Execute TWFE Intensity Model (OLS with HC3 Robust Errors)
    # C(DISTRICT_ID) handles District Fixed Effects
    # C(YEAR) handles Year Fixed Effects
    formula = "TFR ~ C(YEAR) + ratio_x_1861 + ratio_x_1871 + ratio_x_1881 + IMR + C(DISTRICT_ID)"
    model = smf.ols(formula, data=df_reg)
    results = model.fit(cov_type='HC3')
    
    print("\n" + "="*80)
    print("CONTINUOUS DIFFERENCE-IN-DIFFERENCES (INTENSITY MODEL) SUMMARY")
    print("="*80)
    # Print the interaction coefficients specifically
    print(results.summary().tables[1])
    print("\nSignificance Highlights:")
    for var in ['ratio_x_1861', 'ratio_x_1871', 'ratio_x_1881']:
        coef = results.params[var]
        pval = results.pvalues[var]
        star = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{var}: Coef={coef:.4f}, P-Value={pval:.4f} {star}")
        
    return df_reg, results

if __name__ == "__main__":
    df, res = perform_intensity_did()
