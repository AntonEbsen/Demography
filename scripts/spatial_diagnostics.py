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
    Runs spatial diagnostics for the 1851 baseline.
    """
    print("Running Spatial Diagnostics (1851)...")
    
    # Paths
    root = Path(__file__).parents[1]
    raw_geo = root / 'exam_project' / 'data' / 'raw' / 'data1851_0.geojson'
    processed_csv = root / 'exam_project' / 'data' / 'processed' / 'master_panel_data.csv'
    output_dir = root / 'exam_project' / 'outputs' / 'spatial'
    os.makedirs(output_dir, exist_ok=True)
    
    if not raw_geo.exists() or not processed_csv.exists():
        print("Data files missing. Aborting.")
        return
        
    # Load and Merge
    gdf = gpd.read_file(raw_geo).drop(columns=['TFR'], errors='ignore')
    df = pd.read_csv(processed_csv)
    df_1851 = df[df['Year'] == 1851]
    
    # Merge
    merged = gdf.merge(df_1851, on='REGDIST')
    merged = merged.dropna(subset=['TFR'])
    
    # Weights
    w = calculate_spatial_weights(merged)
    
    # Global Moran
    moran_results = run_global_moran(merged, 'TFR', w)
    print(f"Global Moran's I (TFR): {moran_results['I']:.4f}")
    print(f"P-value: {moran_results['p-value']:.4f}")
    
    # LISA Plots
    plot_lisa_clusters(merged, 'TFR', w, output_path=output_dir / 'lisa_tfr_1851.png')
    plot_lisa_clusters(merged, 'F_TEX', w, output_path=output_dir / 'lisa_textiles_1851.png')
    
    print(f"Diagnostics complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    run_diagnostics()
