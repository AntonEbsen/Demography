import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_audit():
    """
    Generates a comprehensive data audit for raw and processed datasets.
    """
    logger.info("Initializing Data Audit...")
    
    root = Path(__file__).parents[3] # To root
    output_dir = root / 'exam_project' / 'outputs' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    
    # Files to audit
    processed_path = root / 'exam_project' / 'data' / 'processed' / 'master_panel_data.csv'
    raw_files = list((root / 'exam_project' / 'data' / 'raw').glob('*.xlsx'))
    
    # 1. Audit Processed Panel
    if processed_path.exists():
        logger.info(f"Auditing processed panel: {processed_path.name}")
        df = pd.read_csv(processed_path)
        
        # Missingness Matrix
        plt.figure(figsize=(12, 8))
        msno.matrix(df)
        plt.title("Missingness Matrix: Master Panel Data")
        plt.savefig(output_dir / 'missingness_matrix_processed.png')
        plt.close()
        
        # Distribution of Core Variables
        core_vars = ['TFR', 'F_TEX', 'IMR', 'F_CL_1013']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        for i, var in enumerate(core_vars):
            if var in df.columns:
                sns.histplot(df[var].dropna(), kde=True, ax=axes[i], color='teal')
                axes[i].set_title(f"Distribution of {var}")
        plt.tight_layout()
        plt.savefig(output_dir / 'variable_distributions_processed.png')
        plt.close()
        
        # Integrity Summary
        with open(output_dir / 'data_integrity_summary.txt', 'w') as f:
            f.write("=== DATA INTEGRITY SUMMARY ===\n")
            f.write(f"Total Observations: {len(df)}\n")
            f.write(f"Registration Districts: {df['REGDIST'].nunique()}\n")
            f.write(f"Years Covered: {df['Year'].unique().tolist()}\n")
            f.write("\n=== MISSING DATA (COUNTS) ===\n")
            f.write(df.isnull().sum().to_string())
            f.write("\n\n=== DESCRIPTIVE STATISTICS ===\n")
            f.write(df[core_vars].describe().to_string())
            
        logger.info("Processed data audit complete.")

    # 2. Audit Raw Data (Quick check on row counts)
    raw_summary = []
    for f_path in raw_files:
        try:
            temp_df = pd.read_excel(f_path)
            raw_summary.append({
                'File': f_path.name,
                'Rows': len(temp_df),
                'Cols': len(temp_df.columns)
            })
        except Exception as e:
            logger.error(f"Could not audit {f_path.name}: {e}")
            
    if raw_summary:
        pd.DataFrame(raw_summary).to_csv(output_dir / 'raw_data_manifest.csv', index=False)
        logger.info("Raw data manifest generated.")

if __name__ == "__main__":
    run_data_audit()
