import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

def calculate_vif(df, features):
    """
    Computes Variance Inflation Factor to check for multicollinearity.
    """
    X = df[features]
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data[vif_data['Variable'] != 'const']

def prepare_did_sample(df, treatment_var, baseline_year):
    """
    Standardizes treatment identification and prepares data for DiD analysis.
    Assumes treatment is based on baseline year status.
    """
    median_val = df[df['Year'] == baseline_year][treatment_var].median()
    treatment_districts = df[(df['Year'] == baseline_year) & (df[treatment_var] > median_val)]['REGDIST'].unique()
    
    df['Treatment_Group'] = df['REGDIST'].isin(treatment_districts).map({True: 'High Intensity', False: 'Low Intensity'})
    df['treat_dummy'] = (df['Treatment_Group'] == 'High Intensity').astype(int)
    
    return df

def run_clustered_ols(df, formula, cluster_col):
    """
    Runs an OLS regression with clustered standard errors.
    Robustly handles missing values by aligning groups with sampled rows.
    """
    model = smf.ols(formula, data=df)
    # Statsmodels might drop rows with NaNs. We must align the group labels.
    results = model.fit(cov_type='cluster', 
                        cov_kwds={'groups': df.loc[model.data.row_labels, cluster_col]})
    return results

def save_results_to_latex(results_list, output_path, title="Regression Results", column_names=None):
    """
    Exports a list of statsmodels results to a LaTeX table.
    Useful for research papers.
    """
    from statsmodels.iolib.summary2 import summary_col
    
    df_results = summary_col(results_list, 
                             stars=True, 
                             float_format='%0.3f',
                             model_names=column_names if column_names else [f"({i+1})" for i in range(len(results_list))],
                             info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                        'R2': lambda x: "{:.3f}".format(x.rsquared)})
    
    latex_str = df_results.as_latex()
    
    # Add a custom title/caption if needed
    with open(output_path, "w") as f:
        f.write("% " + title + "\n")
        f.write(latex_str)
    
    print(f"LaTeX tables saved to {output_path}")
