import nbformat as nbf
from pathlib import Path

def patch_notebook():
    nb_path = Path('exam_project/notebooks/exam_project.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Re-inject Section 7 cells starting from the marker
    new_cells = []
    
    # 7.1 Preparation
    new_cells.append(nbf.v4.new_markdown_cell("# 7. Advanced Spatial Econometrics (SLM)"))
    new_cells.append(nbf.v4.new_code_cell(
        "# 7.1 Data Preparation for Spatial Analysis\n"
        "# FIX: Using 'Year' (Title Case) to resolve KeyError\n"
        "geo_path = '../data/raw/data1881_0.geojson'\n"
        "gdf = gpd.read_file(geo_path)\n"
        "\n"
        "# Normalize join keys for robust spatial matching\n"
        "gdf['REGDIST_MATCH'] = gdf['REGDIST'].astype(str).str.upper().str.strip()\n"
        "temp_factory = df_factory.copy()\n"
        "temp_factory['REGDIST_MATCH'] = temp_factory['REGDIST'].astype(str).str.upper().str.strip()\n"
        "\n"
        "# Extract cross-section (1881)\n"
        "df_1881 = temp_factory[temp_factory['Year'] == 1881].copy()\n"
        "\n"
        "# Drop baseline columns from GDF to prevent merge collisions (_x/_y)\n"
        "cols_to_drop = ['TFR', 'F_CL_1013', 'IMR', 'HOUSE_SERV', 'F_TEX']\n"
        "gdf = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns])\n"
        "\n"
        "# Final Spatial Merge\n"
        "spatial_df = gdf.merge(df_1881, on='REGDIST_MATCH', how='inner')\n"
        "spatial_df = spatial_df.dropna(subset=['TFR'])\n"
        "\n"
        "print(f\"Spatial dataset prepared with {len(spatial_df)} districts matched.\")"
    ))

    # 7.2 Global Moran
    new_cells.append(nbf.v4.new_markdown_cell("#### **7.2 Global Moran's I: Robust Spatial Correlation Check**"))
    new_cells.append(nbf.v4.new_code_cell(
        "# Define spatial weights using Queen contiguity\n"
        "w = Queen.from_dataframe(spatial_df, silence_warnings=True)\n"
        "w.transform = 'r'\n"
        "\n"
        "# Moran's I test for TFR clustering\n"
        "mi = Moran(spatial_df['TFR'], w)\n"
        "print(f\"Global Moran's I (TFR 1881): {mi.I:.4f}\")\n"
        "print(f\"P-value (Randomization): {mi.p_sim:.4f}\")"
    ))

    # 7.3 LISA
    new_cells.append(nbf.v4.new_markdown_cell("#### **7.3 Local LISA Cluster Mapping**"))
    new_cells.append(nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 10))\n"
        "lisa_cluster(Moran_Local(spatial_df['TFR'], w), spatial_df, p=0.05, ax=ax)\n"
        "plt.title(\"LISA Cluster Map: Victorian Fertility Hotspots (1881)\")\n"
        "plt.show()"
    ))

    # 7.4 SLM
    new_cells.append(nbf.v4.new_markdown_cell("#### **7.4 Spatial Lag Model (SLM) Estimation**\n"
                                          "We test if child labor effects persistent after accounting for regional spillover ($\rho$)."))
    new_cells.append(nbf.v4.new_code_cell(
        "# Preparing dependent and independent variables\n"
        "y = spatial_df['TFR'].values.reshape(-1, 1)\n"
        "# Using Child Labor (10-13) and IMR as primary predictors\n"
        "X = spatial_df[['F_CL_1013', 'IMR']].values\n"
        "\n"
        "# Fit ML Spatial Lag\n"
        "slm_model = spreg.ML_Lag(y, X, w=w, name_y='TFR', name_x=['F_CL_1013', 'IMR'])\n"
        "print(slm_model.summary)"
    ))

    # Find where Section 7 starts and replace everything from there onwards
    # Or just append/replace. Since I appended Section 7 previously at the end.
    # I'll look for the first Section 7 cell.
    
    start_index = -1
    for i, cell in enumerate(nb.cells):
        if "# 7." in cell.source:
            start_index = i
            break
            
    if start_index != -1:
        nb.cells = nb.cells[:start_index] + new_cells
    else:
        nb.cells.extend(new_cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook Section 7 patched successfully.")

if __name__ == '__main__':
    patch_notebook()
