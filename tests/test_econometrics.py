import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parents[1] / "exam_project" / "src" / "utils"))
from econometrics import calculate_vif, prepare_did_sample

def test_calculate_vif():
    # Create a simple dataset with multicollinearity
    data = {
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],  # Perfect collinearity with x1
        'x3': [5, 1, 4, 2, 3]
    }
    df = pd.DataFrame(data)
    
    # x1 and x2 should have high VIF
    vif_df = calculate_vif(df, ['x1', 'x2', 'x3'])
    assert 'x1' in vif_df['Variable'].values
    assert 'x2' in vif_df['Variable'].values
    # Check if VIF is high (inf for perfect collinearity)
    assert vif_df[vif_df['Variable'] == 'x1']['VIF'].iloc[0] > 10

def test_prepare_did_sample():
    data = {
        'REGDIST': ['A', 'A', 'B', 'B'],
        'Year': [1851, 1861, 1851, 1861],
        'F_TEX': [10, 12, 50, 55]  # B is high intensity
    }
    df = pd.DataFrame(data)
    
    df_prepared = prepare_did_sample(df, 'F_TEX', 1851)
    
    assert 'Treatment_Group' in df_prepared.columns
    assert 'treat_dummy' in df_prepared.columns
    
    # Check that district B is High Intensity
    assert df_prepared[df_prepared['REGDIST'] == 'B']['Treatment_Group'].iloc[0] == 'High Intensity'
    assert df_prepared[df_prepared['REGDIST'] == 'B']['treat_dummy'].iloc[0] == 1
    
    # Check that district A is Low Intensity
    assert df_prepared[df_prepared['REGDIST'] == 'A']['Treatment_Group'].iloc[0] == 'Low Intensity'
    assert df_prepared[df_prepared['REGDIST'] == 'A']['treat_dummy'].iloc[0] == 0
