import logging
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

PANEL_SCHEMA = DataFrameSchema(
    {
        "Code": Column(int, Check.ge(0), nullable=False),
        "Year": Column(int, Check.in_range(1820, 1910), nullable=False),
        "Poptot": Column(float, Check.gt(0), nullable=True),
        "cbr": Column(float, Check.in_range(0, 100), nullable=True),
        "cath_share": Column(float, Check.in_range(0, 100), nullable=True),
        "post_kulturkampf": Column(int, Check.isin([0, 1]), nullable=True),
    },
    strict=False,
    unique=["Code", "Year"],
)


def audit_schema():
    data_path = DATA_PROCESSED / "analysis_panel.parquet"
    if not data_path.exists():
        logging.warning(f"Data file {data_path} not found. Skipping audit.")
        return

    df = pd.read_parquet(data_path)

    try:
        PANEL_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        logging.error(f"Schema validation failed:\n{e.failure_cases}")
        raise

    dup_count = df.duplicated(["Code", "Year"]).sum()
    if dup_count:
        raise AssertionError(
            f"Panel key (Code, Year) is not unique: {dup_count} duplicate rows."
        )

    critical_cols = ["year", "fertility_rate", "mortality_rate", "population"]
    for col in critical_cols:
        if col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            if null_pct > 0.15:
                logging.warning(f"Column '{col}' has {null_pct:.1%} missing values.")
            else:
                logging.info(f"Column '{col}' completeness check passed.")

    logging.info("Schema audit completed successfully. Data integrity verified.")


if __name__ == "__main__":
    audit_schema()
