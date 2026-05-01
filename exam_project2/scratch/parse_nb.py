import json

nb_path = "c:/Users/Anton/Demography/exam_project2/notebooks/exam_project2.ipynb"
out_path = "c:/Users/Anton/Demography/exam_project2/scratch/notebook_source.py"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(out_path, 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            f.write(f"\n# %% Cell {i+1} (Code)\n")
            f.write("".join(cell.get('source', [])))
            f.write("\n")
        elif cell.get('cell_type') == 'markdown':
            f.write(f"\n# %% Cell {i+1} (Markdown)\n")
            for line in cell.get('source', []):
                f.write(f"# {line}")
            f.write("\n")
