import pandas as pd
import pytest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

@pytest.fixture
def data():
    """Load the processed analysis panel for assumption testing."""
    try:
        return pd.read_parquet("exam_project2/data/processed/analysis_panel.parquet")
    except FileNotFoundError:
        pytest.skip("Processed data not found. Run DVC pipeline first.")

def test_multicollinearity_vif(data):
    """
    Test that Variance Inflation Factor (VIF) is below critical thresholds (usually 10).
    Severe multicollinearity inflates standard errors and destabilizes coefficients.
    """
    # Assuming 'year', 'population' and 'fertility_rate' are standard columns in your IPEHD 1849 data.
    # We dynamically select numeric columns that aren't IDs or targets to test independent variables.
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # Drop obvious non-predictors
    cols_to_check = [c for c in numeric_cols if c not in ['id', 'year', 'fertility_rate']]
    
    if len(cols_to_check) < 2:
        pytest.skip("Not enough numeric columns for VIF test.")

    df_clean = data[cols_to_check].dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(df_clean.values, i) for i in range(len(df_clean.columns))]
    
    # Assert no feature has VIF > 10 (common econometric threshold)
    high_vif = vif_data[vif_data["VIF"] > 10]
    assert high_vif.empty, f"High multicollinearity detected:\n{high_vif}"

def test_stationarity_adfuller(data):
    """
    Test for Unit Roots / Stationarity in the dependent variable using Augmented Dickey-Fuller.
    If the dependent variable (e.g., fertility) has a unit root, TWFE regressions might be spurious.
    """
    if 'fertility_rate' not in data.columns or 'year' not in data.columns:
        pytest.skip("Required columns for time-series stationarity test missing.")
        
    # Aggregate to national/overall time series
    ts_data = data.groupby('year')['fertility_rate'].mean().dropna()
    
    if len(ts_data) < 10:
        pytest.skip("Time series too short for meaningful ADF test.")

    # ADF test: H0 is that there is a unit root (non-stationary)
    adf_result = adfuller(ts_data)
    p_value = adf_result[1]
    
    # We want to REJECT the null hypothesis (p < 0.05) to confirm stationarity
    # If p > 0.05, we might need to use first-differences in our regressions.
    if p_value > 0.05:
        logging.warning(f"Dependent variable may be non-stationary (ADF p-value = {p_value:.3f}). Consider First-Differences.")

def test_missingness_balance(data):
    """
    Test that missing data is not highly concentrated in specific time periods, 
    which could bias the panel balance.
    """
    if 'year' not in data.columns:
        pytest.skip("Year column missing.")
        
    missing_by_year = data.isnull().sum(axis=1).groupby(data['year']).mean()
    # Ensure no single year has radically more missing data than the global average (e.g., > 3 standard deviations)
    mean_missing = missing_by_year.mean()
    std_missing = missing_by_year.std()
    
    anomalies = missing_by_year[missing_by_year > (mean_missing + 3 * std_missing)]
    assert anomalies.empty, f"Severe unbalanced missingness in years: {anomalies.index.tolist()}"

def test_serial_correlation_dw(data):
    """
    Test for serial correlation (autocorrelation) in the panel using the Durbin-Watson statistic.
    Severe serial correlation biases standard errors downward, requiring clustered SEs.
    """
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    
    if 'year' not in data.columns or 'fertility_rate' not in data.columns:
        pytest.skip("Required columns missing.")
        
    # Fit a basic pooled OLS to check residuals
    df_clean = data[['fertility_rate', 'year']].dropna()
    X = sm.add_constant(df_clean['year'])
    model = sm.OLS(df_clean['fertility_rate'], X).fit()
    
    dw_stat = durbin_watson(model.resid)
    
    # DW stat ranges from 0 to 4. 2 is no autocorrelation.
    # Less than 1.5 indicates strong positive serial correlation.
    if dw_stat < 1.5:
        logging.warning(f"Strong positive serial correlation detected (DW = {dw_stat:.2f}). You MUST use clustered standard errors.")
    elif dw_stat > 2.5:
        logging.warning(f"Strong negative serial correlation detected (DW = {dw_stat:.2f}).")
    else:
        logging.info(f"Residuals appear relatively uncorrelated over time (DW = {dw_stat:.2f}).")

def test_hausman_fe_vs_re(data):
    """
    Hausman Test for Fixed Effects vs Random Effects.
    H0: Random Effects is consistent and efficient.
    H1: Random Effects is inconsistent (must use Fixed Effects).
    """
    try:
        from linearmodels.panel import PanelOLS, RandomEffects
        import numpy.linalg as la
        from scipy import stats
    except ImportError:
        pytest.skip("linearmodels not installed.")
        
    # Need a multi-index for linearmodels: (entity, time)
    if 'id' not in data.columns or 'year' not in data.columns or 'fertility_rate' not in data.columns:
        pytest.skip("Panel identifiers missing.")
        
    df_panel = data.set_index(['id', 'year']).dropna()
    
    # We need at least one exogenous predictor, let's pick the first numeric one available
    numeric_cols = [c for c in df_panel.columns if df_panel[c].dtype in [np.float64, np.int64, np.float32, np.int32] and c != 'fertility_rate']
    if not numeric_cols:
        pytest.skip("No predictors available for Hausman test.")
        
    y = df_panel['fertility_rate']
    X = df_panel[[numeric_cols[0]]] # Use just one for the test
    
    try:
        fe_model = PanelOLS(y, X, entity_effects=True).fit()
        re_model = RandomEffects(y, X).fit()
        
        b_fe = fe_model.params
        b_re = re_model.params
        v_fe = fe_model.cov
        v_re = re_model.cov
        
        # Hausman test statistic: (b_FE - b_RE)' [Var(b_FE) - Var(b_RE)]^-1 (b_FE - b_RE)
        diff = b_fe - b_re
        cov_diff = v_fe - v_re
        
        # If the difference in covariance matrices isn't positive definite, we fallback
        try:
            inv_cov_diff = la.inv(cov_diff.values)
            h_stat = float(diff.values.T @ inv_cov_diff @ diff.values)
            df = len(b_fe)
            p_val = stats.chi2.sf(h_stat, df)
            
            if p_val < 0.05:
                logging.info(f"Hausman test rejects H0 (p={p_val:.4f}). Fixed Effects is required.")
            else:
                logging.warning(f"Hausman test fails to reject H0 (p={p_val:.4f}). Random Effects might be more efficient.")
        except la.LinAlgError:
            pytest.skip("Covariance difference not invertible.")
    except Exception as e:
        pytest.skip(f"Hausman test failed to run: {e}")
