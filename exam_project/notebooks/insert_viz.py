import json
from pathlib import Path

def insert_visualization():
    nb_path = Path('exam_project/notebooks/exam_project.ipynb')
    if not nb_path.exists():
        print(f"Error: {nb_path} not found.")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Define the new cells
    markdown_cell = {
        "cell_type": "markdown",
        "id": "f8d2a1b9",
        "metadata": {},
        "source": [
            "### **Figure 3: National Demographic Trends**\n",
            "To better understand the aggregate relationship, we visualize the national mean of TFR alongside the female child labor participation rate for ages 10-13. This dual-axis plot highlights the divergent paths of fertility (slight rise/stability) and child labor (sharp decline) during the peak enforcement period of the Factory Acts."
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "b1e7c3a4",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Aggregate your RSD-level panel data by Census Year\n",
            "# We use the mean to represent the national 'average' experience\n",
            "national_trends = df_panel.groupby('Year').agg({\n",
            "    'TFR': 'mean',\n",
            "    'F_CL_1013': 'mean'\n",
            "}).reset_index()\n",
            "\n",
            "# 2. Setup the figure with a dual y-axis\n",
            "fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='#fdfcf0')\n",
            "ax1.set_facecolor('#fdfcf0')\n",
            "\n",
            "color_tfr = '#2c3e50'   # Dark slate\n",
            "color_cl = '#c0392b'    # Victorian red\n",
            "\n",
            "# --- Plot National TFR (Left Axis) ---\n",
            "line1 = ax1.plot(national_trends['Year'], national_trends['TFR'], \n",
            "                 color=color_tfr, marker='o', linewidth=3, label='National TFR')\n",
            "ax1.set_xlabel('Census Year', fontsize=12, fontweight='bold')\n",
            "ax1.set_ylabel('Total Fertility Rate (TFR)', color=color_tfr, fontsize=12, fontweight='bold')\n",
            "ax1.tick_params(axis='y', labelcolor=color_tfr)\n",
            "ax1.set_ylim(4, 5) # Adjusted to visualize the subtle mid-Victorian peak\n",
            "\n",
            "# --- Plot National Child Labor (Right Axis) ---\n",
            "ax2 = ax1.twinx()\n",
            "line2 = ax2.plot(national_trends['Year'], national_trends['F_CL_1013'], \n",
            "                 color=color_cl, linestyle='--', marker='s', linewidth=3, label='Female Child Labor (10-13)')\n",
            "ax2.set_ylabel('% Female Child Labor (10-13)', color=color_cl, fontsize=12, fontweight='bold')\n",
            "ax2.tick_params(axis='y', labelcolor=color_cl)\n",
            "\n",
            "# --- Title and Legend ---\n",
            "plt.title('National Trends: TFR and Female Child Labor (1851-1881)', fontsize=16, fontweight='bold', pad=20)\n",
            "lines = line1 + line2\n",
            "labels = [l.get_label() for l in lines]\n",
            "ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)\n",
            "\n",
            "# --- Historical Annotation: Factory Act Enforcement ---\n",
            "ax1.annotate('1860s: Intensified\\nFactory Act Inspection', \n",
            "             xy=(1861, 4.5), xytext=(1855, 4.7), \n",
            "             arrowprops=dict(facecolor='black', shrink=0.05, width=1))\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }

    # Find where Figure 2's markdown is
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and '### **Figure 2:' in "".join(cell['source']):
            target_idx = i + 1
            break

    if target_idx != -1:
        nb['cells'].insert(target_idx, markdown_cell)
        nb['cells'].insert(target_idx + 1, code_cell)
        print(f"Inserted cells at index {target_idx}")
    else:
        # Fallback to end if not found
        nb['cells'].append(markdown_cell)
        nb['cells'].append(code_cell)
        print("Figure 2 not found; appended cells to the end.")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook update complete.")

if __name__ == '__main__':
    insert_visualization()
