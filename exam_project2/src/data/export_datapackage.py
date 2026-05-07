import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def generate_frictionless_datapackage():
    data_path = DATA_PROCESSED / "analysis_panel.parquet"
    output_path = DATA_PROCESSED / "datapackage.json"
    
    if not data_path.exists():
        logging.warning(f"Data file {data_path} not found. Cannot generate Data Package.")
        return

    df = pd.read_parquet(data_path)
    
    # Define Frictionless Data Package spec
    # This makes the data automatically parseable by repositories like ICPSR and Zenodo
    datapackage = {
        "name": "prussian-kulturkampf-demographics",
        "title": "Prussian Kulturkampf Demographic Panel (1849-1880)",
        "description": "A county-level panel dataset harmonizing the IPEHD 1849 census data with Prussian Vital Statistics.",
        "licenses": [{"name": "CC-BY-4.0"}],
        "profile": "tabular-data-package",
        "resources": [
            {
                "name": "analysis_panel",
                "path": "analysis_panel.parquet",
                "profile": "tabular-data-resource",
                "format": "parquet",
                "schema": {
                    "fields": []
                }
            }
        ]
    }
    
    # Auto-infer field types from pandas to DDI/Frictionless standard
    type_mapping = {
        'int64': 'integer', 'int32': 'integer',
        'float64': 'number', 'float32': 'number',
        'object': 'string', 'bool': 'boolean',
        'datetime64[ns]': 'datetime'
    }
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        frictionless_type = type_mapping.get(dtype, 'string')
        
        field = {
            "name": col,
            "type": frictionless_type,
            "description": f"Variable {col} extracted from archival sources." # Ideally mapped from your data_dictionary
        }
        datapackage["resources"][0]["schema"]["fields"].append(field)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datapackage, f, indent=2)
        
    logging.info(f"Frictionless Data Package generated successfully at {output_path}")

if __name__ == "__main__":
    generate_frictionless_datapackage()
