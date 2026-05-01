import os
from pathlib import Path

replacements = {
    "from src.load_data": "from src.data.load_data",
    "from src.build_dataset": "from src.data.build_dataset",
    "from src.merge_ipehd": "from src.data.merge_ipehd",
    "from src.regressions": "from src.analysis.regressions",
    "from src.exploratory": "from src.analysis.exploratory",
    "from src.advanced": "from src.analysis.advanced",
    "from src.plots": "from src.visualization.plots",
    "from src.maps": "from src.visualization.maps",
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

# Update src files
src_dir = Path("c:/Users/Anton/Demography/exam_project2/src")
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".py"):
            process_file(os.path.join(root, file))

# Update notebook
notebook_path = Path("c:/Users/Anton/Demography/exam_project2/notebooks/exam_project.ipynb")
if notebook_path.exists():
    process_file(notebook_path)

print("Done.")
