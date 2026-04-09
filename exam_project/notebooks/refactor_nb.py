import json
from pathlib import Path

def refactor_notebook():
    nb_path = Path('exam_project/notebooks/exam_project.ipynb')
    if not nb_path.exists():
        print(f"Error: {nb_path} not found.")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = "".join(cell['source'])
        
        # 1. Update Imports
        if 'import os' in source and 'import sys' in source:
            new_source = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Add root project directory to path for modular imports\n",
                "notebook_path = Path.cwd()\n",
                "root_dir = notebook_path if (notebook_path / 'src').exists() else notebook_path.parent\n",
                "if str(root_dir) not in sys.path:\n",
                "    sys.path.append(str(root_dir))\n",
                "\n",
                "import pandas as pd\n",
                "import geopandas as gpd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import statsmodels.api as sm\n",
                "import statsmodels.formula.api as smf\n",
                "from src.utils import plotting, econometrics\n"
            ]
            cell['source'] = new_source
            print(f"Updated Imports in cell {i}")

        # 2. Replace legacy data loading
        if 'factory_cols =' in source and 'read_excel' in source:
            new_source = [
                "### LOAD PROCESSED MASTER PANEL DATA\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "\n",
                "processed_data_path = Path('../data/processed/master_panel_data.csv')\n",
                "if not processed_data_path.exists():\n",
                "    processed_data_path = Path('exam_project/data/processed/master_panel_data.csv')\n",
                "\n",
                "print(f\"Loading data from: {processed_data_path}\")\n",
                "df_panel = pd.read_csv(processed_data_path)\n",
                "df_factory = df_panel\n",
                "print(f\"Loaded {len(df_panel)} rows with {len(df_panel.columns)} columns.\")\n",
                "df_panel.head()\n"
            ]
            cell['source'] = new_source
            print(f"Updated Data Loading in cell {i}")

        # 3. Fix Clustered OLS Calls
        if "model.fit(cov_type='cluster'" in source:
            # Simple string replacement for the common pattern found in the notebook
            source = source.replace(
                "results = model.fit(cov_type='cluster', cov_kwds={'groups': df_panel['REGDIST']})",
                "results = econometrics.run_clustered_ols(df_panel, f'{outcome} ~ treat_dummy + Year + REGDIST', 'REGDIST')"
            )
            # More general replacement for direct fit calls
            if "model.fit(cov_type='cluster'" in source:
                 source = source.replace("results = model.fit(cov_type='cluster',", "# results = model.fit(cov_type='cluster',")
                 # This is a bit coarse, let's just use the modular one where it makes sense.
            
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.split('\n')]
            print(f"Patched manual OLS fitting in cell {i}")

        # 4. Fix NameErrors (df_reg -> df_panel)
        if 'df_reg' in source:
             cell['source'] = [line.replace('df_reg', 'df_panel') for line in cell['source']]
             print(f"Fixed df_reg reference in cell {i}")

        # 5. Clear outputs to prevent confusion with old errors
        if 'outputs' in cell:
            cell['outputs'] = []

    # Save refactored notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Refactoring complete.")

if __name__ == '__main__':
    refactor_notebook()
