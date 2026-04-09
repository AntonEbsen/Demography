import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_fertility_animation():
    """
    Generates a time-lapse GIF of the fertility transition across England and Wales (1851-1881).
    """
    logger.info("Initializing Map Animation sequence...")
    
    root = Path(__file__).parents[1]
    raw_dir = root / 'exam_project' / 'data' / 'raw'
    processed_csv = root / 'exam_project' / 'data' / 'processed' / 'master_panel_data.csv'
    output_dir = root / 'exam_project' / 'outputs' / 'animations'
    temp_dir = root / 'exam_project' / 'outputs' / 'temp_frames'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    if not processed_csv.exists():
        logger.error("Master panel data not found. Run data pipeline first.")
        return
        
    df_panel = pd.read_csv(processed_csv)
    years = [1851, 1861, 1871, 1881]
    
    # Global scale for fixed comparability
    vmin, vmax = df_panel['TFR'].min(), df_panel['TFR'].max()
    
    frame_paths = []
    
    for year in years:
        logger.info(f"Generating frame for {year}...")
        geo_path = raw_dir / f"data{year}_0.geojson"
        if not geo_path.exists():
            logger.warning(f"GeoJSON for {year} missing at {geo_path}. Skipping frame.")
            continue
            
        gdf = gpd.read_file(geo_path).drop(columns=['TFR'], errors='ignore')
        year_data = df_panel[df_panel['Year'] == year]
        merged = gdf.merge(year_data, on='REGDIST', how='left')
        
        fig, ax = plt.subplots(figsize=(10, 12), facecolor='#0e1117')
        merged.plot(
            column='TFR', 
            cmap='magma', 
            vmin=vmin, 
            vmax=vmax, 
            ax=ax, 
            edgecolor='black', 
            linewidth=0.1,
            missing_kwds={'color': '#2c2c2c'}
        )
        ax.set_title(f"The British Fertility Transition: {year}", fontsize=20, color='white', pad=20)
        ax.axis('off')
        
        # Add a custom colorbar
        sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Total Fertility Rate (TFR)', color='white')
        
        frame_path = temp_dir / f"frame_{year}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
        plt.close()
        frame_paths.append(frame_path)
        
    # Compile GIF
    if frame_paths:
        logger.info("Compiling frames into GIF...")
        images = [imageio.imread(str(p)) for p in frame_paths]
        # Duplicate last frame for a "pause" at the end
        images.extend([images[-1]] * 10)
        
        gif_path = output_dir / 'fertility_decline_1851_1881.gif'
        imageio.mimsave(str(gif_path), images, fps=1)
        logger.info(f"Animation successfully saved to {gif_path}")
        
    # Cleanup temp frames
    for p in frame_paths:
        os.remove(p)
    os.rmdir(temp_dir)

if __name__ == "__main__":
    create_fertility_animation()
