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
        # Migration rates: nullable (only ~21 of 29 panel years have migration
        # data); generous bounds since some Galloway counties record extreme
        # gross flows in single years.
        "outmig_rate": Column(float, Check.in_range(-50, 500), nullable=True),
        "inmig_rate":  Column(float, Check.in_range(-50, 500), nullable=True),
        "net_mig_rate": Column(float, Check.in_range(-500, 500), nullable=True),
        # 1871 age structure: women-15-49 share of total population, in %.
        "women_share_15_49_1871": Column(float, Check.in_range(0, 100), nullable=True),
        # General fertility rate (births per 1,000 women aged 15-49 in 1871).
        # Bounded loosely; demographically extreme values still flag bad rows.
        "gfr_static_1871": Column(float, Check.in_range(0, 500), nullable=True),
        # Mid-year population: linear interpolation between consecutive
        # December census anchors evaluated at July 1. Headline `cbr`,
        # `legitimate_br`, `illegitimate_br`, `marriage_rate`, and
        # migration rates use this as their denominator (standard
        # demographic convention). See compute_midyear_population() for
        # the construction.
        "Poptot_midyear": Column(float, Check.gt(0), nullable=True),
        # Galloway carry-forward variant (= previous December census
        # carried forward in inter-census years), retained as the
        # robustness row in the headline DiD table.
        "cbr_carryforward": Column(float, Check.in_range(0, 100), nullable=True),
        # Coale Princeton EFP indices. I_g is the Galloway-tradition
        # marital-fertility headline (Hutterite-normalised; ~0.7 in 1871
        # Prussia per Coale 1986). Bound generously so flag-survivor
        # boundary-reform residuals are still caught by the schema.
        "I_f": Column(float, Check.in_range(0, 1.2), nullable=True),
        "I_g": Column(float, Check.in_range(0, 1.5), nullable=True),
        "gmfr": Column(float, Check.in_range(0, 600), nullable=True),
        # Headline IMR: total infant deaths / total live births x 1000.
        # 1875+ only (Galloway's Dth<1bas column starts in 1875).
        "infant_mortality_rate": Column(
            float, Check.in_range(0, 600), nullable=True
        ),
        # Legitimate-only IMR, retained as a diagnostic for the 1875
        # data-break figure.
        "infant_mortality_rate_leg": Column(
            float, Check.in_range(0, 600), nullable=True
        ),
        # 1871 Reichstag election vote shares (Galloway ELE1871). All
        # per-cent of valid votes; 1871 cross-section, time-invariant
        # after merge. ~85% Kreis coverage.
        "zentrum_share_1871": Column(float, Check.in_range(0, 100), nullable=True),
        "polen_share_1871": Column(float, Check.in_range(0, 100), nullable=True),
        "catholic_party_share_1871": Column(
            float, Check.in_range(0, 200), nullable=True,
        ),
        # Time-varying election shares: carry-forward of the most
        # recent Reichstag election (1871, 1874, 1878, 1881, 1884,
        # 1887, 1890). Used to measure Catholic political mobilisation
        # as a Kulturkampf outcome.
        "zentrum_share_current": Column(float, Check.in_range(0, 100), nullable=True),
        "polen_share_current": Column(float, Check.in_range(0, 100), nullable=True),
        "catholic_party_share_current": Column(
            float, Check.in_range(0, 200), nullable=True,
        ),
        # Time-varying urban share from Galloway URB1875/80/85/90,
        # linearly interpolated between anchors. NaN pre-1875.
        "urban_share_current": Column(float, Check.in_range(0, 100), nullable=True),
        # 1886 schooling cross-section (EDU1886). All derived rates;
        # time-invariant after merge. ~85% Kreis coverage.
        "volksschule_share_1886": Column(float, Check.in_range(0, 150), nullable=True),
        "private_school_share_1886": Column(float, Check.in_range(0, 100), nullable=True),
        "schooling_gap_1886": Column(float, Check.in_range(-50, 100), nullable=True),
        "teachers_per_1000_pupils_1886": Column(
            float, Check.in_range(0, 200), nullable=True
        ),
        "teacher_income_1886": Column(float, Check.gt(0), nullable=True),
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
