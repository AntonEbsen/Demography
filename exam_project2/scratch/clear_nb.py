import json
import os

nb_path = "c:/Users/Anton/Demography/exam_project2/notebooks/exam_project2.ipynb"
tmp_path = "c:/Users/Anton/Demography/exam_project2/notebooks/exam_project2_tmp.ipynb"

if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    os.remove(nb_path)
    os.rename(tmp_path, nb_path)
    print(f"Cleared outputs for {nb_path}")
else:
    print(f"File {nb_path} not found.")
