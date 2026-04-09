import pandas as pd
import pytest
import sys
from pathlib import Path
import pandera as pa

# Add src to path for imports
sys.path.append(str(Path(__file__).parents[1] / "exam_project" / "src" / "data"))
from schemas import PanelSchema

def test_panel_schema_valid():
    df = pd.DataFrame({
        'REGDIST': ['London', 'Manchester'],
        'Year': [1851, 1861],
        'TFR': [4.5, 3.8],
        'IMR': [150.0, 140.0],
        'F_TEX': [10.0, 45.0],
        'F_CL_1013': [5.0, 20.0]
    })
    # This should pass
    validated_df = PanelSchema.validate(df)
    assert validated_df is not None

def test_panel_schema_invalid_tfr():
    df = pd.DataFrame({
        'REGDIST': ['London'],
        'Year': [1851],
        'TFR': [15.0],  # Out of range (max 10)
        'IMR': [150.0],
        'F_TEX': [10.0],
        'F_CL_1013': [5.0]
    })
    # This should raise SchemaError
    with pytest.raises(pa.errors.SchemaError):
        PanelSchema.validate(df)

def test_panel_schema_missing_year():
    df = pd.DataFrame({
        'REGDIST': ['London'],
        'TFR': [4.5],
        'IMR': [150.0],
        'F_TEX': [10.0],
        'F_CL_1013': [5.0]
    })
    # This should raise SchemaError because 'Year' is missing
    with pytest.raises(pa.errors.SchemaError):
        PanelSchema.validate(df)
