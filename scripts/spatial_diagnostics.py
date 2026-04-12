import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parents[1] / "exam_project" / "src"))
from utils.spatial import calculate_spatial_weights, run_global_moran, plot_lisa_clusters

def run_diagnostics():
    """
    Runs spatial diagnostics comparing 1851 and 1881.
    """
    print("--- STARTING SPATIAL CONTAGION ANALYSIS ---")
    
    root = Path(__file__).parents[1]
    output_dir = root / 'exam_project' / 'outputs' / 'spatial'
    os.makedirs(output_dir, exist_ok=True)
    
    processed_csv = root / 'exam_project' / 'data' / 'processed' / 'master_panel_data.csv'
    if not processed_csv.exists():
        print(f"Error: {processed_csv} not found. Run data processing first.")
        return
        
    df = pd.read_csv(processed_csv)
    
    for year in [1851, 1881]:
        print(f"Processing Year: {year}")
        geo_file = root / 'exam_project' / 'data' / 'raw' / f'data{year}_0.geojson'
        
        if not geo_file.exists():
            print(f"Skipping {year}: GeoJSON missing.")
            continue
            
        gdf = gpd.read_file(geo_file)
        # Drop columns that exist in the CSV to avoid _x/_y collisions
        cols_to_drop = ['TFR', 'F_TEX', 'TMFR', 'IMR', 'F_CL_1013']
        gdf = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns])
        
        # Normalize join keys for robust matching
        gdf['REGDIST_MATCH'] = gdf['REGDIST'].astype(str).str.upper().str.strip()
        year_df = df[df['Year'] == year].copy()
        year_df['REGDIST_MATCH'] = year_df['REGDIST'].astype(str).str.upper().str.strip()
        
        merged = gdf.merge(year_df, on='REGDIST_MATCH', how='inner')
        merged = merged.dropna(subset=['TFR'])
        
        # 1. Spatial Weights (Queen Contiguity)
        w = calculate_spatial_weights(merged)
        
        # 2. Global Moran's I (Robust check for spatial dependency)
        moran = run_global_moran(merged, 'TFR', w)
        print(f"  [{year}] Global Moran's I: {moran['I']:.4f} (p={moran['p-value']:.4f})")
        
        # 3. LISA Cluster Mapping
        plot_lisa_clusters(merged, 'TFR', w, output_path=output_dir / f'lisa_tfr_{year}.png')
        
    print(f"--- SPATIAL DIAGNOSTICS COMPLETE. Maps saved to: {output_dir} ---")

if __name__ == "__main__":
    run_diagnostics()
