import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_boundary_stability_analysis():
    """
    Identifies registration districts with significant boundary changes (area/centroid)
    between 1851 and 1881.
    """
    logger.info("Initializing Boundary Stability Analysis...")
    
    root = Path(__file__).parents[1]
    raw_dir = root / 'exam_project' / 'data' / 'raw'
    output_dir = root / 'exam_project' / 'outputs' / 'diagnostics'
    os.makedirs(output_dir, exist_ok=True)
    
    years = [1851, 1861, 1871, 1881]
    results = []
    
    # Baseline (1851)
    logger.info("Loading baseline geometry (1851)...")
    base_geo_path = raw_dir / "data1851_0.geojson"
    if not base_geo_path.exists():
        logger.error("Baseline GeoJSON (1851) missing.")
        return
        
    gdf_base = gpd.read_file(base_geo_path)
    if gdf_base.crs is None or gdf_base.crs.to_epsg() != 27700:
        gdf_base = gdf_base.to_crs(epsg=27700)
    
    gdf_base['base_area'] = gdf_base.geometry.area
    gdf_base['base_centroid'] = gdf_base.geometry.centroid
    
    gdf_base = gdf_base.drop_duplicates(subset=['REGDIST'])
    base_info = gdf_base.set_index('REGDIST')[['base_area', 'base_centroid']].to_dict('index')
    
    for year in years:
        if year == 1851: continue
        
        logger.info(f"Analyzing {year} boundaries...")
        geo_path = raw_dir / f"data{year}_0.geojson"
        if not geo_path.exists(): continue
            
        gdf_year = gpd.read_file(geo_path).to_crs(epsg=27700)
        
        for _, row in gdf_year.iterrows():
            reg_name = row['REGDIST']
            if reg_name in base_info:
                current_area = row.geometry.area
                current_centroid = row.geometry.centroid
                
                area_change_pct = abs(current_area - base_info[reg_name]['base_area']) / base_info[reg_name]['base_area'] * 100
                centroid_shift_m = current_centroid.distance(base_info[reg_name]['base_centroid'])
                
                results.append({
                    'REGDIST': reg_name,
                    'Year': year,
                    'Area_Change_Pct': area_change_pct,
                    'Centroid_Shift_M': centroid_shift_m,
                    'Unstable': area_change_pct > 5
                })
                
    df_results = pd.DataFrame(results)
    report_path = output_dir / 'boundary_stability_report.csv'
    df_results.to_csv(report_path, index=False)
    
    logger.info(f"Stability report generated with {len(df_results)} comparisons.")
    logger.info(f"Significant changes detected in {len(df_results[df_results['Unstable']])} district-years.")
    
    return report_path

if __name__ == "__main__":
    run_boundary_stability_analysis()
