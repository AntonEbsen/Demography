# Demography Research Framework

Welcome to the documentation for the **Cost of Quality** research project. This repository contains a fully automated pipeline for historical demographic analysis, with a focus on 19th-century British fertility transitions.

## 🚀 Key Features
- **Automated Pipeline**: Clean, merge, and validate census and textile data.
*   **Spatial Diagnostics**: Moran's I and LISA cluster analysis.
- **Advanced Econometrics**: Clustered OLS, Oster Sensitivity, and DiD.
*   **Interactive Dashboard**: Explore the data through Streamlit maps and policy simulators.

## 🛠️ Quick Start
To install dependencies:
```bash
make install
```

To run the full pipeline:
```bash
make data
```

To launch the dashboard:
```bash
streamlit run exam_project/src/app.py
```
