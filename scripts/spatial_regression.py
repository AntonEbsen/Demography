import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from spreg import ML_Lag
import numpy as np
from pathlib import Path
import os

def run_spatial_regression():
    print("--- EXECUTING SPATIAL LAG MODEL (SLM) ---")
    
    root = Path(__file__).parents[1]
    geo_path = root / 'exam_project' / 'data' / 'raw' / 'data1881_0.geojson'
    data_path = root / 'exam_project' / 'data' / 'processed' / 'master_panel_data.csv'
    
    if not geo_path.exists() or not data_path.exists():
        print("Data missing for SLM.")
        return

    # 1. Load and Merge
    gdf = gpd.read_file(geo_path)
    # Drop existing outcome/predictor columns to avoid collision with CSV merge
    cols_to_drop = ['TFR', 'F_CL_1013', 'IMR', 'HOUSE_SERV']
    gdf = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns])
    
    df = pd.read_csv(data_path)
    # Normalize join keys for robust matching
    gdf['REGDIST_MATCH'] = gdf['REGDIST'].astype(str).str.upper().str.strip()
    df_1881 = df[df['Year'] == 1881].copy()
    df_1881['REGDIST_MATCH'] = df_1881['REGDIST'].astype(str).str.upper().str.strip()
    
    merged = gdf.merge(df_1881, on='REGDIST_MATCH', how='inner')
    # Filter for non-null regression variables
    reg_vars = ['TFR', 'F_CL_1013', 'IMR', 'HOUSE_SERV']
    merged = merged.dropna(subset=reg_vars)
    
    # 2. Spatial Weights Matrix
    w = Queen.from_dataframe(merged)
    w.transform = 'r'
    
    # 3. Prepare Variables
    y = merged['TFR'].values.reshape(-1, 1)
    # X includes Child Labor, IMR, and Middle Class proxy
    x = merged[['F_CL_1013', 'IMR', 'HOUSE_SERV']].values
    
    # 4. Fit Maximum Likelihood Spatial Lag Model
    # This accounts for the dependency where Your TFR depends on Your Neighbors' TFR
    model = ML_Lag(y, x, w=w, name_y='TFR', name_x=['F_CL_1013', 'IMR', 'HOUSE_SERV'])
    
    # 5. Output Results Summary
    print(model.summary)
    
    # Save results to a text file for the Research Brief
    output_path = root / 'exam_project' / 'outputs' / 'spatial' / 'slm_results_1881.txt'
    with open(output_path, 'w') as f:
        f.write(model.summary)
        
    print(f"SLM Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_spatial_regression()
