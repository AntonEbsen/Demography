import nbformat as nbf
import os

notebook_path = 'exam_project/notebooks/exam_project.ipynb'

def enrich_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Methodology: Geographic Harmonization Rationale
    # Insert before Section 5.1 (Cell 32)
    harmonization_md = (
        "### **5.0 Methodology: Geographic Harmonization and Administrative Rationale**\n\n"
        "A central challenge in 19th-century British cliometrics is the shifting administrative boundaries. The 1831 baseline data is recorded at the **Sub-District** level, while the 1851-1881 panel is organized around **Registration Districts (RDs)**.\n\n"
        "**Unification Rule:** To ensure a stable geographic panel, we apply the `thousand-block` grouping rule:\n"
        "$$DISTRICT\\_ID = (REGDIST // 1000) * 1000$$\n\n"
        "This mapping relies on the hierarchical coding used by the Census Office, where sub-districts within the same parent district share the same thousand-block prefix. Summing or averaging variables (like manufacturing shares) at this level allows us to observe the fertility transition in consistent spatial units over 50 years."
    )
    nb.cells.insert(32, nbf.v4.new_markdown_cell(harmonization_md))

    # Identify new indices (approximate +1 for harmonization cell)
    # Binary DiD Theory
    binary_theory_md = (
        "### **5.2 The Identification Strategy: Parallel Trends**\n\n"
        "The **Difference-in-Differences (DiD)** estimator relies on the **Parallel Trends Assumption**: the hypothesis that in the absence of the Factory Acts, the Total Fertility Rate (TFR) in textile-industrialized 'treated' districts would have evolved following the same trend as the agricultural 'control' districts.\n\n"
        "By interacting the treatment status with post-shock year dummies, we isolate the dynamic impact of the legislation. $\\beta_{post}$ represents the divergence in fertility trajectories attributable to the legislative shock, controlling for time-invariant regional characteristics."
    )
    nb.cells.insert(34, nbf.v4.new_markdown_cell(binary_theory_md))

    # Intensity Model Theory
    intensity_theory_md = (
        "### **6.1.1 The Dosage Effect: Continuous Legislative Exposure**\n\n"
        "Binary DiD (Treated vs. Control) assumes a uniform impact within the treatment group. However, the **Intensity Model** accommodates the reality that legislative exposure was heterogenous. Factors like the density of mills and the prevalence of child-heavy manufacturing meant that some districts received a higher 'dosage' of the law's restrictions.\n\n"
        "We use the **1831 Industrial Ratio** as a baseline 'dosage' measure. This pre-treatment intensity serves as a continuous weight, where the coefficients $\\beta_y$ measure how much an additional percentage point of industrialization accelerated the fertility transition in decade $y$ relative to 1851."
    )
    nb.cells.insert(40, nbf.v4.new_markdown_cell(intensity_theory_md))

    # Technical Diagnostics: TWFE, HC3, and IMR
    technical_md = (
        "### **8. Technical Note: Econometric Robustness and Controls**\n\n"
        "#### **8.1 Two-Way Fixed Effects (TWFE)**\n"
        "Our models utilize **Two-Way Fixed Effects** to mitigate omitted variable bias:\n"
        "- **District Fixed Effects ($C(DISTRICT\\_ID)$)**: Controls for unobserved, time-invariant regional factors such as local culture, soil quality, or religious density.\n"
        "- **Year Fixed Effects ($C(YEAR)$)**: Controls for common national shocks, such as changes in grain prices, the Crimean War, or national educational reforms.\n\n"
        "#### **8.2 Robust Inference (HC3)**\n"
        "Because our data is clustered geographically, simple OLS standard errors would likely be biased downward due to heteroscedasticity. We utilize **HC3 Robust Standard Errors**, which provide conservative inference even when regional variances are non-constant.\n\n"
        "#### **8.3 Demographic Control: The Replacement Effect**\n"
        "We include the **Infant Mortality Rate (IMR)** as a critical control. In high-mortality regimes, families may engage in 'replacement fertility' (birthing more children to reach a target surviving family size). By controlling for IMR, we isolate the behavioral shift in fertility caused by the changing *price* of child quality, rather than shifts in mortality."
    )
    nb.cells.append(nbf.v4.new_markdown_cell(technical_md))

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook successfully enriched with deep theoretical content.")

if __name__ == "__main__":
    enrich_notebook()
