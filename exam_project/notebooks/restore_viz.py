import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def restore_viz():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    viz_md = nbf.v4.new_markdown_cell("### **3.1 Regional Analysis: Registration Counties**\nBefore proceeding to formal regressions, we visualize the regional distribution of our core variables across Registration Counties (REGCNTY) to identify spatial clusters of textile intensity and fertility.")
    
    viz_code = """import seaborn as sns
import matplotlib.pyplot as plt

# Since Registration Districts are historical, we visualize by Registration County (REGCNTY)
# to identify regional clusters of Textile and Fertility transition.
# We filter for the 1851 baseline to match the original research identification.

# Prepare the data subset
county_data = df_merged[df_merged['YEAR'] == 1851].groupby('REGCNTY')[['TFR', 'F_TEX']].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plotting Textile Intensity by County
sns.barplot(data=county_data.sort_values('F_TEX', ascending=False).head(15), 
            y='REGCNTY', x='F_TEX', ax=ax1, palette='viridis')
ax1.set_title('Top 15 Textile-Intensive Counties (1851)')
ax1.set_xlabel('Mean Female Textile Worker %')

# Plotting TFR by County to check visual correlation
sns.barplot(data=county_data.sort_values('TFR', ascending=False).head(15), 
            y='REGCNTY', x='TFR', ax=ax2, palette='magma')
ax2.set_title('Top 15 Highest Fertility Counties (1851)')
ax2.set_xlabel('Mean TFR')

plt.tight_layout()
plt.show()

print("Note: registration counties provide a robust high-level view of the textile heartlands.")"""
    viz_cell = nbf.v4.new_code_cell(viz_code)
    
    # Insertion point: after Section 3 (Data and Variables)
    # We find "### **3. Data and Variables**" or the loading cell
    new_cells = []
    inserted = False
    for i, cell in enumerate(nb.cells):
        new_cells.append(cell)
        if not inserted and cell.cell_type == 'code' and 'df_merged = load_and_harmonize_data()' in cell.source:
            new_cells.append(viz_md)
            new_cells.append(viz_cell)
            inserted = True
            
    if not inserted:
        # Fallback to the start
        nb.cells.insert(3, viz_md)
        nb.cells.insert(4, viz_cell)
    else:
        nb.cells = new_cells
        
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Geospatial visualization successfully restored to the notebook.")

if __name__ == "__main__":
    restore_viz()
