# The "Cost of Quality"
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AntonEbsen/Demography/main?filepath=notebooks%2Fexam_project.ipynb)

## Legislative Shocks to Child Labor and the Fertility Transition in 19th-Century Britain

This repository contains the analysis and data for my research into the impact of the **1833 and 1844 Factory Acts** on the British fertility transition.

---

## 📝 Abstract
While the broad decline in British fertility is well-documented for the late 19th century, this project investigates whether earlier legislative restrictions on child labor and the introduction of mandatory schooling for factory children acted as a catalyst for the "Quantity-Quality" (Q-Q) trade-off.

Using a **Difference-in-Differences (DiD)** design, I compare Registered Sub-Districts (RSDs) with high textile industry intensity (the "treatment" group) against agricultural control districts. The study leverages census data from **Populations Past** (1851–1891) and baseline data from 1831.

### Key Hypotheses
1. **The Opportunity Cost Shock**: Restrictions on child work hours reduced the economic benefit of high fertility.
2. **The Schooling Mandate**: Mandatory education increased the implicit "price" of child quality.
3. **Fertility Divergence**: Textile-heavy districts should experience a sharper TFR decline following these legislative shocks compared to controls.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/         # Original census XLSX and GeoJSON boundary files
│   ├── processed/   # Cleaned panel data and merged spatial datasets
│   └── data_dictionary.md # scientific codebook for variables
├── notebooks/
│   └── exam_project.ipynb  # Primary Python analysis notebook
└── README.md        # This file
```

---

## 📊 Methodology & Insights

The analysis is performed in Python using a registration-district-level panel.

- **Econometrics**: OLS and Fixed Effects (District & Year) regressions.
- **Geospatial**: `geopandas` is used to map fertility and industrial patterns across Britain.
- **Key Variables**:
    - `TFR`: Total Fertility Rate (Outcome)
    - `F_TEX`: Textile Industry Intensity (Treatment Proxy)
    - `F_CL_1013`: Female Child Labor aged 10-13 (Mechanism)
    - `IMR`: Infant Mortality Rate (Control)

---

## 🚀 Getting Started

### Prerequisites
You will need a Python environment with the following libraries:
- `pandas`
- `geopandas`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `openpyxl` (for reading Excel files)

### Running the Analysis
1. Clone the repository.
2. Ensure the raw data is present in `data/raw/`.
3. Open `notebooks/exam_project.ipynb` in your preferred Jupyter environment (VS Code, JupyterLab, etc.).
4. Run all cells to reproduce the findings and visualizations.

---

Data is sourced from **Populations Past** (University of Cambridge), providing spatially digitized census data for England and Wales. 

---

## 📜 Citation

If you use this data or code in your research, please cite it as:

> Ebsen, A. (2026). The Demographic Transition in Victorian England: A Spatial Econometric Approach. GitHub Repository. https://github.com/AntonEbsen/Demography

Alternatively, use the **"Cite this repository"** button in the sidebar or refer to the [CITATION.cff](./CITATION.cff) file.

## 📖 Data Documentation
For detailed variable definitions and data provenance, see the [Data Dictionary](./data/data_dictionary.md).
