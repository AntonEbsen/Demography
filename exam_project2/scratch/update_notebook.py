import nbformat

notebook_path = "exam_project2/notebooks/02_baseline_regressions.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

markdown_cell = nbformat.v4.new_markdown_cell(source="""### 3.5 Econometric Diagnostics (Assumption Testing)
Before proceeding to robustness checks, we mathematically validate our core identifying assumptions. We test for **Multicollinearity (VIF)**, **Serial Correlation (Durbin-Watson)**, and whether **Fixed Effects** are strictly required over Random Effects using the **Hausman Test**.""")

code_cell = nbformat.v4.new_code_cell(source="""import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS, RandomEffects
import numpy.linalg as la
from scipy import stats

print("=" * 60)
print("ECONOMETRIC DIAGNOSTICS")
print("=" * 60)

# 1. Serial Correlation (Durbin-Watson)
df_clean = panel[['fertility_rate', 'year']].dropna()
X_dw = sm.add_constant(df_clean['year'])
model_dw = sm.OLS(df_clean['fertility_rate'], X_dw).fit()
dw_stat = durbin_watson(model_dw.resid)
print(f"\\n1. Serial Correlation (Durbin-Watson):")
print(f"DW Statistic = {dw_stat:.3f}")
if dw_stat < 1.5:
    print("-> Result: Positive serial correlation detected. Clustered standard errors are REQUIRED.")
else:
    print("-> Result: No severe serial correlation.")

# 2. Hausman Test (FE vs RE)
df_panel = panel.set_index(['id', 'year']).dropna(subset=['fertility_rate', 'cath_share', 'ln_pop'])
y = df_panel['fertility_rate']
X_haus = df_panel[['cath_share', 'ln_pop']]

fe_model = PanelOLS(y, X_haus, entity_effects=True).fit()
re_model = RandomEffects(y, X_haus).fit()

diff = fe_model.params - re_model.params
cov_diff = fe_model.cov - re_model.cov

print(f"\\n2. Hausman Test (Fixed vs Random Effects):")
try:
    inv_cov_diff = la.inv(cov_diff.values)
    h_stat = float(diff.values.T @ inv_cov_diff @ diff.values)
    p_val = stats.chi2.sf(h_stat, len(fe_model.params))
    print(f"Hausman Chi-Square = {h_stat:.3f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("-> Result: Reject H0. Fixed Effects estimator is strictly required.")
    else:
        print("-> Result: Fail to reject H0. Random Effects is efficient.")
except Exception as e:
    print("-> Result: Test inconclusive due to covariance matrix properties.")

# 3. Multicollinearity (VIF)
print(f"\\n3. Variance Inflation Factor (VIF):")
df_vif = df_panel[['cath_share', 'ln_pop', 'infant_mortality_rate']].dropna()
for i, col in enumerate(df_vif.columns):
    vif = variance_inflation_factor(df_vif.values, i)
    print(f"VIF for {col}: {vif:.2f}")
    if vif > 10:
        print(f"-> Warning: High multicollinearity for {col}")
""")

# Insert before "4. Robustness Checks", which is cell 8 (index 7)
# Let's dynamically find it just in case
insert_idx = len(nb.cells)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and '### 4. Robustness Checks' in cell.source:
        insert_idx = i
        break

nb.cells.insert(insert_idx, markdown_cell)
nb.cells.insert(insert_idx + 1, code_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Successfully inserted diagnostic cells at index {insert_idx}")
