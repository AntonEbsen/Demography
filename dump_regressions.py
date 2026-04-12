import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def dump_results():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    log_content = []
    reg_count = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and ('smf.ols' in cell.source or 'PanelOLS' in cell.source):
            reg_count += 1
            log_content.append(f"--- REGRESSION {reg_count} (CELL {i}) ---")
            log_content.append(f"SOURCE:\n{cell.source[:200]}...")
            if cell.outputs:
                for out in cell.outputs:
                    if 'text' in out:
                        log_content.append(f"OUTPUT:\n{out['text'][:1000]}")
                    elif 'data' in out and 'text/plain' in out['data']:
                        log_content.append(f"OUTPUT:\n{out['data']['text/plain'][:1000]}")
            else:
                log_content.append("NO OUTPUT FOUND")
            log_content.append("\n" + "="*50 + "\n")
            
    with open('regression_outputs.log', 'w', encoding='utf-8') as f:
        f.write("\n".join(log_content))
    print(f"Dumped {reg_count} regression summaries to regression_outputs.log")

if __name__ == "__main__":
    dump_results()
