import nbformat as nbf
import re
import os

log_path = 'regression_outputs.log'
notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def smart_restore():
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Split by the separator
    regressions = re.split(r'--- REGRESSION \d+ \(CELL \d+\) ---', log_content)[1:]
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    # We want to insert these after Section 4
    new_cells = []
    section_found = False
    
    main_load_done = False
    
    for cell in nb.cells:
        new_cells.append(cell)
        if cell.cell_type == 'markdown' and '### **4. Regression Analysis' in cell.source:
            section_found = True
            for i, reg_block in enumerate(regressions):
                source_match = re.search(r'SOURCE:\n(.*?)\n(?:OUTPUT:|==================================================)', reg_block, re.DOTALL)
                if source_match:
                    source = source_match.group(1).strip()
                    # Refactor to use df_merged instead of local loading
                    # Remove redundant imports and loading
                    source = re.sub(r'import pandas as pd.*?(?=formula|model|reg)', '', source, flags=re.DOTALL)
                    source = re.sub(r'baseline_path = .*?(?=formula|model|reg)', '', source, flags=re.DOTALL)
                    source = re.sub(r'df_panel = .*?(?=formula|model|reg)', '', source, flags=re.DOTALL)
                    source = source.strip()
                    
                    if not source: continue
                    
                    # Add interpretation heading if missing
                    title_match = re.search(r'#\s*(.*)', source)
                    title = title_match.group(1) if title_match else f"Specification {i+1}"
                    
                    new_cells.append(nbf.v4.new_markdown_cell(f"#### **{title}**"))
                    new_cells.append(nbf.v4.new_code_cell(source))

    nb.cells = new_cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Smart-restored {len(regressions)} regressions into the notebook.")

if __name__ == "__main__":
    smart_restore()
