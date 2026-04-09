import numpy as np
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

def calculate_oster_delta(y, treatment, controls, df, R_max=0.7):
    """
    Implements a simplified version of Oster's (2019) Delta.
    Estimates the degree of selection on unobservables relative to observables 
    needed to explain away the treatment effect.
    """
    logger.info("Executing Oster's Sensitivity analysis...")
    
    # 1. Short Model (Treatment only)
    X_short = sm.add_constant(df[treatment])
    model_short = sm.OLS(y, X_short).fit()
    beta_short = model_short.params[treatment]
    r2_short = model_short.rsquared
    
    # 2. Long Model (Full controls)
    X_long = sm.add_constant(df[[treatment] + controls])
    model_long = sm.OLS(y, X_long).fit()
    beta_long = model_long.params[treatment]
    r2_long = model_long.rsquared
    
    # 3. Calculate Delta (simplified selection ratio)
    # This estimate follows the logic: how much stronger do unobservables need to be 
    # than observables to drive beta to 0.
    if beta_long == beta_short:
        delta = np.inf
    else:
        # Based on Oster (2019) JBES
        numerator = beta_long * (r2_long - r2_short)
        denominator = (beta_short - beta_long) * (R_max - r2_long)
        delta = (beta_long / (beta_short - beta_long)) * ((r2_long - r2_short) / (R_max - r2_long))
        
    logger.info(f"Oster's Delta calculated: {delta:.4f}")
    
    return {
        'beta_short': beta_short,
        'beta_long': beta_long,
        'r2_short': r2_short,
        'r2_long': r2_long,
        'delta': delta,
        'interpretation': "Highly Robust" if abs(delta) > 1 else "Sensitive"
    }
