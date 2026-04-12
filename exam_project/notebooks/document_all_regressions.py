import nbformat as nbf
import re
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def formula_to_latex(formula):
    # Basic conversion of smf formulas to cleaner LaTeX
    formula = formula.replace('~', '= \\alpha + ')
    formula = formula.replace('C(YEAR)', '\\sum \\delta_t Year_t')
    formula = formula.replace('C(DISTRICT_ID)', '\\mu_i')
    formula = formula.replace(' + ', ' + ')
    
    # Replace variable names with cleaner versions
    replacements = {
        'TFR': 'TFR_{it}',
        'IMR': '\\gamma IMR_{it}',
        'ratio_x_1861': '\\beta_{1861} Intensity_i',
        'ratio_x_1871': '\\beta_{1871} Intensity_i',
        'ratio_x_1881': '\\beta_{1881} Intensity_i',
        'F_CL_1013': '\\theta ChildLabor_{it}',
        'M_CL_1013': '\\sigma MaleLabor_{it}',
        'HOUSE_SERV': '\\phi Servants_{it}',
        'F_SMAM': 'SMAM_{it}',
        'F_CEL_4554': 'CelibacyRate_{it}',
        'SC1': 'UpperClass_i',
        'C_TEACHER': 'TeacherDensity_i',
    }
    
    for k, v in replacements.items():
        # Using a lambda to avoid backslash issues in replacement string
        formula = re.sub(rf'\b{k}\b', lambda m: v, formula)
    
    return f"$${formula} + \\epsilon_{{it}}$$"

def document_regressions():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    new_cells = []
    
    for i, cell in enumerate(nb.cells):
        # We look for regressions that aren't already documented in a previous cell
        if cell.cell_type == 'code' and ('smf.ols' in cell.source or 'PanelOLS' in cell.source):
            source = cell.source
            # Extract first comment as Title/Hypothesis
            comment_match = re.search(r'#\s*(.*)', source)
            title = comment_match.group(1) if comment_match else "Unnamed Specification"
            
            # Extract formula string
            formula_match = re.search(r'formula\w* = "(.*)"', source)
            formula_str = formula_match.group(1) if formula_match else "Unknown Formula"
            latex_formula = formula_to_latex(formula_str)
            
            # Generate Markdown
            md_content = f"### **{title}**\n\n"
            md_content += f"**Hypothesis:** We test whether the variables in this specification explain the cross-sectional or longitudinal variation in fertility related to the project's core research goals.\n\n"
            md_content += f"**Econometric Specification:**\n{latex_formula}\n\n"
            md_content += f"**Motivation:** This specific control or interaction term isolates unobserved shocks or tests for specific structural breaks (e.g., skill premiums, marriage postponement, or gender-specific labor trends)."
            
            # Check if previous cell is already markdown with the same title to avoid duplicates
            already_documented = False
            if i > 0 and nb.cells[i-1].cell_type == 'markdown' and title in nb.cells[i-1].source:
                already_documented = True
            
            if not already_documented:
                new_cells.append(nbf.v4.new_markdown_cell(md_content))
            
        new_cells.append(cell)
    
    nb.cells = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook successfully documented with theoretical cells for all regressions.")

if __name__ == "__main__":
    document_regressions()
