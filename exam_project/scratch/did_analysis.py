import pandas as pd
import glob
import os
import statsmodels.formula.api as smf
import numpy as np

def perform_did_analysis():
    # 1. Load Baseline (1831)
    baseline_path = 'exam_project/data/processed/census_1831_cleaned_for_merge.csv'
    df_1831 = pd.read_csv(baseline_path)
    
    # Harmonization Logic (Source: Prompt Requirement)
    # The 1831 data is at the Sub-District level. 
    # We round down to the nearest 1000 to match the 1851-1881 District IDs.
    df_1831['DISTRICT_ID'] = (df_1831['REGDIST'] // 1000) * 1000
    
    # Aggregate baseline to District level
    # We take the max of treatment (if any sub-district is treated, the district is)
    # and the mean of manufacturing intensity.
    df_1831_agg = df_1831.groupby('DISTRICT_ID').agg({
        'is_treated_baseline': 'max',
        'MANUFAC': 'mean'
    }).reset_index()
    
    # 2. Load Panel (1851-1881)
    raw_data_dir = 'exam_project/data/raw/'
    panel_years = [1851, 1861, 1871, 1881]
    panel_dfs = []
    
    for year in panel_years:
        files = glob.glob(os.path.join(raw_data_dir, f'*{year}*.xlsx'))
        if not files:
            print(f"Warning: No file found for year {year}")
            continue
            
        df_year = pd.read_excel(files[0])
        # Force column names to uppercase to avoid KeyErrors
        df_year.columns = df_year.columns.str.upper()
        
        # Standardize Year and ID
        df_year['YEAR'] = year
        
        # Identify the District ID column (CEN_1851, CEN_1861, etc.)
        cen_col = f'CEN_{year}'
        if cen_col in df_year.columns:
            df_year = df_year.rename(columns={cen_col: 'DISTRICT_ID'})
        else:
            # Fallback if column names vary (e.g. CEN1851)
            found = False
            for col in df_year.columns:
                if 'CEN' in col and str(year) in col:
                    df_year = df_year.rename(columns={col: 'DISTRICT_ID'})
                    found = True
                    break
            if not found:
                print(f"Error: Could not find District ID column in {year} data.")
                continue
                
        panel_dfs.append(df_year)
        
    df_panel = pd.concat(panel_dfs, ignore_index=True)
    
    # 3. Merge Baseline and Panel
    # Inner join as requested to ensure matched observations
    df_final = pd.merge(df_panel, df_1831_agg, on='DISTRICT_ID', how='inner')
    
    print(f"Merge successful. Final observation count: {len(df_final)}")
    print(f"Number of unique Districts: {df_final['DISTRICT_ID'].nunique()}")
    
    # 4. Econometric Model (TWFE)
    # Model: TFR ~ Treatment * Year + Controls + District Fixed Effects
    # We use HC3 Robust Standard Errors as requested.
    
    # Drop rows with NaNs in core variables to ensure statsmodels doesn't fail
    reg_vars = ['TFR', 'is_treated_baseline', 'YEAR', 'IMR', 'DISTRICT_ID']
    df_reg = df_final.dropna(subset=reg_vars).copy()
    
    model = smf.ols("TFR ~ C(is_treated_baseline) * C(YEAR) + IMR + C(DISTRICT_ID)", data=df_reg)
    results = model.fit(cov_type='HC3')
    
    print("\n--- Difference-in-Differences (TWFE) Summary ---")
    print(results.summary())
    
    return df_final, results

if __name__ == "__main__":
    df, res = perform_did_analysis()
