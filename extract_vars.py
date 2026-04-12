import nbformat as nbf
import re
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def extract_metadata():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    all_vars = set()
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'smf.ols' in cell.source:
            # Match word chars in formulas
            match = re.search(r'formula\w* = "(.*)"', cell.source)
            if match:
                formula = match.group(1)
                found = re.findall(r'\b[A-Za-z]\w+\b', formula)
                for v in found:
                    if v not in ['Intercept', 'C', 'YEAR', 'DISTRICT_ID', 'T', 'is_high_sc1', 'is_middle_class']:
                        all_vars.add(v)
    
    print("EXTRACTED VARIABLES:")
    print("\n".join(sorted(list(all_vars))))

if __name__ == "__main__":
    extract_metadata()
