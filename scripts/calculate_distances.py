import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core Industrial & Policy Hubs (British National Grid EPSG:27700)
HUBS = {
    'London': (530000, 180000),
    'Manchester': (380000, 398000),
    'Cardiff': (318000, 176000),
    'Edinburgh': (325000, 673000)
}

def calculate_hub_distances():
    """
    Calculates Euclidean distance from each Registration District to the nearest major hub.
    """
    logger.info("Starting GIS Hub Distance calculation...")
    
    root = Path(__file__).parents[1]
    geo_path = root / 'exam_project' / 'data' / 'raw' / 'data1851_0.geojson'
    processed_csv = root / 'exam_project' / 'data' / 'processed' / 'master_panel_data.csv'
    
    if not geo_path.exists() or not processed_csv.exists():
        logger.error("Missing GeoJSON or Processed CSV.")
        return
        
    # Load GeoJSON (assume it's in or can be converted to EPSG:27700)
    gdf = gpd.read_file(geo_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 27700:
        logger.info("Reprojecting to British National Grid (EPSG:27700)...")
        gdf = gdf.to_crs(epsg=27700)
        
    # Create Hub points
    hub_points = [Point(xy) for xy in HUBS.values()]
    
    # Calculate Distances
    logger.info("Computing minimum distance to hubs for each district...")
    # Get centroids
    gdf['centroid'] = gdf.geometry.centroid
    
    # Distance function
    def min_dist(centroid):
        return min([centroid.distance(hub) for hub in hub_points]) / 1000 # Convert to km
        
    gdf['dist_to_hub_km'] = gdf['centroid'].apply(min_dist)
    
    # Merge back to panel data
    df_panel = pd.read_csv(processed_csv)
    dist_map = gdf.set_index('REGDIST')['dist_to_hub_km'].to_dict()
    
    df_panel['dist_to_hub'] = df_panel['REGDIST'].map(dist_map)
    
    # Save back
    df_panel.to_csv(processed_csv, index=False)
    logger.info(f"Updated {processed_csv.name} with 'dist_to_hub' column.")

if __name__ == "__main__":
    calculate_hub_distances()
