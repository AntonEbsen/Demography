# The Kulturkampf and Catholic Fertility in Prussia

This project analyzes whether Bismarck's anti-Catholic Kulturkampf legislation (1872–1878) affected the Catholic–Protestant fertility differential in Prussian counties. It leverages the Galloway Prussia Database (1861–1914) and the iPEHD (1871) dataset.

📑 **[Data Appendix](DATA_APPENDIX.md)** — variable definitions, formulas, sources, sample-construction rules, and estimation specifications.

## Project Structure

- `data/`: Contains raw data and processed datasets.
- `notebooks/`: Jupyter notebooks for interactive analysis.
- `src/`: Python source code organized by concern:
  - `data/`: Loading and harmonising raw files.
  - `analysis/`: Running regressions and exploratory data analysis.
  - `visualization/`: Plotting trends and spatial mapping.
- `outputs/`: Output figures and tables from the analysis.
- `reports/`: Research reports or project write-ups.
- `scratch/`: For temporary scripts and explorations.

## Setup

It is recommended to run this project in a dedicated Python environment.

```bash
pip install -r requirements.txt
```

You can execute the entire pipeline by running the cells in `notebooks/exam_project2.ipynb` or by exploring the `src/exam_project_notebook.py` file.
