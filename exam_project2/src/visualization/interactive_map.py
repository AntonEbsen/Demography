import pandas as pd
import geopandas as gpd
import plotly.express as px
from pathlib import Path
import os
import json

def generate_interactive_map():
    print("Generating Interactive WebGL Map...")
    
    # Paths
    project_root = Path(__file__).parents[3]
    geo_path = project_root / "exam_project2" / "data" / "raw" / "galloway_data" / "data1851_0.geojson"
    data_path = project_root / "exam_project2" / "data" / "processed" / "analysis_panel.parquet"
    output_dir = project_root / "kulturkampf_site" / "public" / "assets"
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = output_dir / "interactive_map.html"
    
    try:
        # We wrap in try-except in case of missing raw data on this machine
        if not geo_path.exists() or not data_path.exists():
            print("Required data files not found. Creating a placeholder interactive map.")
            with open(out_file, "w") as f:
                f.write("<html><body><h3>Interactive map unavailable (raw data missing)</h3></body></html>")
            return
            
        gdf = gpd.read_file(geo_path)
        df = pd.read_parquet(data_path)
        
        # Filter for the pre-kulturkampf baseline year
        df_1871 = df[df['year'] == 1871].copy()
        
        # Standardize matching keys
        gdf['REGDIST_MATCH'] = gdf['REGDIST'].astype(str).str.upper().str.strip()
        df_1871['REGDIST_MATCH'] = df_1871['id'].astype(str).str.upper().str.strip()
        
        merged = gdf.merge(df_1871, on='REGDIST_MATCH', how='inner')
        merged = merged.to_crs("EPSG:4326")
        
        # Create Plotly Mapbox Choropleth
        fig = px.choropleth_mapbox(
            merged,
            geojson=json.loads(merged.geometry.to_json()),
            locations=merged.index,
            color="cath_share",
            hover_name="REGDIST",
            hover_data=["fertility_rate", "ln_pop"],
            color_continuous_scale="Reds",
            mapbox_style="carto-darkmatter",
            zoom=4.5,
            center={"lat": 52.5, "lon": 13.4},
            opacity=0.7,
            title="Interactive Map: Catholic Share in 1871 Prussia"
        )
        
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor="#0b0c0e", font_color="#e0e0e0")
        fig.write_html(str(out_file))
        print(f"Successfully wrote interactive map to {out_file}")
        
    except Exception as e:
        print(f"Failed to generate map: {e}")

if __name__ == "__main__":
    generate_interactive_map()
