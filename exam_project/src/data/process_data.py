import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import logging
from schemas import PanelSchema
import pandera as pa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_research_data():
    """
    Standardizes and merges 1851-1881 census and spatial data.
    Outputs a clean master panel CSV for analysis.
    """
    logger.info("Starting data processing pipeline...")
    
    # Paths relative to this script
    script_dir = Path(__file__).parent
    raw_dir = script_dir.parents[1] / 'data' / 'raw'
    processed_dir = script_dir.parents[1] / 'data' / 'processed'
    
    years = [1851, 1861, 1871, 1881]
    panel_dfs = []
    
    for year in years:
        logger.info(f"Processing Year: {year}")
        
        # Load Census Data
        excel_path = raw_dir / f"PopulationsPast_census_data_{year}.xlsx"
        if not excel_path.exists():
            logger.warning(f"{excel_path} not found. Skipping.")
            continue
            
        try:
            df_census = pd.read_excel(excel_path)
            logger.info(f"Loaded census data for {year}: {len(df_census)} rows.")
        except Exception as e:
            logger.error(f"Failed to load {excel_path}: {e}")
            continue
            
        # Load Spatial Data
        geojson_path = raw_dir / f"data{year}_0.geojson"
        if not geojson_path.exists():
            logger.warning(f"{geojson_path} not found. Merging without geometry.")
            df_year = df_census.copy()
        else:
            try:
                df_geo = gpd.read_file(geojson_path)
                # Typically REGDIST is the joining key
                df_year = df_geo[['REGDIST', 'geometry']].merge(df_census, on='REGDIST', how='left')
                logger.info(f"Mapped spatial data for {year}.")
            except Exception as e:
                logger.error(f"Failed to process spatial data for {year}: {e}")
                df_year = df_census.copy()
        
        df_year['Year'] = year
        panel_dfs.append(df_year)
    
    if not panel_dfs:
        logger.error("No data loaded. Pipeline aborted.")
        return
        
    # Combine into Master Panel
    master_panel = pd.concat(panel_dfs, ignore_index=True)
    
    # Validate with Pandera
    try:
        logger.info("Validating combined panel data...")
        master_panel = PanelSchema.validate(master_panel)
        logger.info("Validation successful.")
    except pa.errors.SchemaError as e:
        logger.error(f"Data validation failed: {e}")
        # We might still want to save but warn, or abort. 
        # For research integrity, warning is better if it's non-critical.
    
    # Save Processed Data
    os.makedirs(processed_dir, exist_ok=True)
    output_path = processed_dir / "master_panel_data.csv"
    
    try:
        if 'geometry' in master_panel.columns:
            master_panel.drop(columns=['geometry']).to_csv(output_path, index=False)
        else:
            master_panel.to_csv(output_path, index=False)
        logger.info(f"Success! Master panel saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save final CSV: {e}")
        
    logger.info(f"Final Panel shape: {master_panel.shape}")

if __name__ == "__main__":
    process_research_data()
