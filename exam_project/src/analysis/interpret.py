import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap
from pathlib import Path
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ml_interpretation():
    """
    Trains a Random Forest models and uses SHAP to explain feature importance for fertility.
    """
    logger.info("Starting Machine Learning Interpretation (SHAP)...")
    
    root = Path(__file__).parents[3]
    processed_csv = root / 'exam_project' / 'data' / 'processed' / 'master_panel_data.csv'
    output_dir = root / 'exam_project' / 'outputs' / 'ml'
    os.makedirs(output_dir, exist_ok=True)
    
    if not processed_csv.exists():
        logger.error("Dataset missing.")
        return
        
    df = pd.read_csv(processed_csv)
    
    # Feature Engineering
    # We focus on the core theoretical drivers
    features = ['Year', 'F_TEX', 'IMR', 'F_CL_1013']
    target = 'TFR'
    
    df_clean = df[features + [target]].dropna()
    X = df_clean[features]
    y = df_clean[target]
    
    logger.info(f"Training Random Forest on {len(df_clean)} observations...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # SHAP Explanation
    logger.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance: Drivers of Fertility (1851-1881)")
    
    plot_path = output_dir / 'shap_summary_plot.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"SHAP analysis complete. Summary plot saved to {plot_path}")
    
    # Save a numeric importance report
    importance = pd.DataFrame({
        'Feature': features,
        'Mean_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_SHAP', ascending=False)
    
    importance.to_csv(output_dir / 'feature_importance_report.csv', index=False)

if __name__ == "__main__":
    run_ml_interpretation()
