import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def cleanup_and_polish():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    unique_cells = []
    seen_sources = set()
    
    for cell in nb.cells:
        # Avoid duplicate markdown interpretation cells
        source_clean = cell.source.strip()
        if cell.cell_type == 'markdown' and '#### **Result Interpretation**' in source_clean:
            if source_clean in seen_sources:
                continue
            seen_sources.add(source_clean)
        
        # Replace the placeholder spatial cell with the full robust implementation
        if cell.cell_type == 'code' and 'ML_Lag' in cell.source:
            cell.source = """import geopandas as gpd
from libpysal.weights import Queen
from spreg import ML_Lag

def run_advanced_slm(df_merged):
    print(\"--- EXECUTING SPATIAL LAG MODEL (SLM) ---\")
    
    # 1. Load Geometries
    geo_path = '../data/raw/data1881_0.geojson'
    if not os.path.exists(geo_path):
        print(\"GeoJSON missing. Skipping SLM simulation.\")
        return
        
    gdf = gpd.read_file(geo_path)
    
    # 2. Harmonize and Merge
    # Standardize REGDIST for matching
    gdf['REGDIST_MATCH'] = gdf['REGDIST'].astype(str).str.upper().str.strip()
    df_1881 = df_merged[df_merged['YEAR'] == 1881].copy()
    df_1881['REGDIST_MATCH'] = df_1881['REGDIST'].astype(str).str.upper().str.strip()
    
    merged = gdf.merge(df_1881, on='REGDIST_MATCH', how='inner')
    
    # 3. Spatial Weights (Queen Contiguity)
    w = Queen.from_dataframe(merged)
    w.transform = 'r'
    
    # 4. Fit Spatial Lag Model
    # y: TFR
    # X: Child Labor, IMR, Middle Class (HOUSE_SERV)
    y = merged['TFR'].values.reshape(-1, 1)
    x = merged[['F_CL_1013', 'IMR', 'HOUSE_SERV']].values
    
    model = ML_Lag(y, x, w=w, name_y='TFR', name_x=['F_CL_1013', 'IMR', 'HOUSE_SERV'])
    print(model.summary)

# Run the SLM on the consolidated panel (1881 subset)
run_advanced_slm(df_merged)"""
        
        unique_cells.append(cell)

    nb.cells = unique_cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook cleanup complete: Duplicates removed and SLM integration finalized.")

if __name__ == "__main__":
    cleanup_and_polish()
