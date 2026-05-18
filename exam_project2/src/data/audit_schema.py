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
        # 1886 schooling cross-section (EDU1886). Time-invariant after
        # merge. ~96% Kreis coverage (453 of 392 by Code; some duplicates
        # in source). Used by channels.schooling_channel() for the
        # 1849->1886 long-difference DiD on attendance rates.
        # `school_age_pop_1886`, `attend_public_1886`, `attend_private_1886`,
        # `teachers_1886` are raw counts; `attend_rate_1886` is a fraction
        # in [0, 1.1] (slightly >1 in a few Kreise because attendance is
        # measured across all schools, including out-of-Kreis pupils);
        # `teacher_income_1886` is *total* annual Volksschule-teacher
        # income in Marks (not per teacher).
        "school_age_pop_1886": Column(float, Check.ge(0), nullable=True),
        "attend_public_1886": Column(float, Check.ge(0), nullable=True),
        "attend_private_1886": Column(float, Check.ge(0), nullable=True),
        "attend_rate_1886": Column(float, Check.in_range(0, 2), nullable=True),
        "teachers_1886": Column(float, Check.ge(0), nullable=True),
        "teacher_income_1886": Column(float, Check.gt(0), nullable=True),
        "pupils_per_teacher_1886": Column(
            float, Check.in_range(0, 500), nullable=True
        ),
        # 1849 elementary-school attendance rate (= 1849 students /
        # 1849 total population, both sexes). Continuous, genuinely
        # pre-treatment (23 years before the May Laws); the right
        # moderator for testing whether the Kulturkampf shock
        # interacted with baseline literacy / human-capital intensity.
        # ~71% Kreis coverage (constrained by 1849 iPEHD crosswalk).
        "attend_rate_1849_baseline": Column(
            float, Check.in_range(0, 0.5), nullable=True
        ),
        # STA1871 marriage prevalence (= Marriedover15f / Popover15f).
        # Feeds the proper Coale I_g recalibration in
        # compute_coale_indices(use_sta1871=True): the Princeton EFP
        # framework otherwise applies a Prussia-wide constant married-
        # share schedule, which kills cross-county variation in the
        # marital-fertility denominator.
        "married_share_over15_f_1871": Column(
            float, Check.in_range(0, 1), nullable=True
        ),
        # Implied count of married women aged 15-49, used as the GMFR
        # denominator. Time-varying via linear interpolation between
        # the 1871 anchor (STA1871-derived) and the 1890 anchor
        # (AGE1890's `Age15-49marriedf`); pop-scaled extrapolation for
        # pre-1871 years.
        "married_women_15_49": Column(
            float, Check.gt(0), nullable=True
        ),
        # Companion to `married_women_15_49`: time-varying count of all
        # women 15-49, anchored at POP1871 (1871) and AGE1890 (1890)
        # with linear interpolation in between.
        "women_15_49": Column(float, Check.gt(0), nullable=True),
        # AGE1890 cross-section direct columns (Galloway's own age-by-
        # marital tabulation). Time-invariant after merge (one value
        # per Kreis); used as the 1890 anchor for `women_15_49` and
        # `married_women_15_49` interpolation.
        "women_15_49_1890": Column(float, Check.gt(0), nullable=True),
        "married_women_15_49_1890": Column(
            float, Check.gt(0), nullable=True
        ),
        "married_share_15_49_f_1890": Column(
            float, Check.in_range(0, 1), nullable=True
        ),
        # AGE1890 internal calibration ratios (used to extract 15-49
        # counts from AGE1882's coarse 0-19 / 20-69 / 70+ bins).
        "r_w_15_49_in_popf_1890": Column(
            float, Check.in_range(0, 1), nullable=True
        ),
        "r_m_15_49_in_marriedf_1890": Column(
            float, Check.in_range(0, 1), nullable=True
        ),
        # AGE1882 cross-section (raw coarse totals + the derived 15-49
        # anchors). The 15-49 anchors are estimates: pop_1882f and
        # marriedf_1882 multiplied by the AGE1890 within-Kreis ratios
        # r_w_15_49_in_popf_1890 and r_m_15_49_in_marriedf_1890. This
        # is the second anchor in the 1871-1882-1890 piecewise linear
        # interpolation of women_15_49 and married_women_15_49 in
        # compute_coale_indices.
        "pop_1882f": Column(float, Check.gt(0), nullable=True),
        "marriedf_1882": Column(float, Check.gt(0), nullable=True),
        "women_15_49_1882": Column(float, Check.gt(0), nullable=True),
        "married_women_15_49_1882": Column(
            float, Check.gt(0), nullable=True
        ),
        "married_share_15_49_f_1882": Column(
            float, Check.in_range(0, 1), nullable=True
        ),
        # Pop 15+ anchors and the time-varying derived series.
        # `pop_15plus_1871` is exact (STA1871's Popover15m+f);
        # `pop_15plus_1890` uses 5/6 of AGE1890's Age14-19 plus
        # Age20-49 / Age50-69 / Age70+ (within-bin approximation for
        # ages 15-19). `pop_15plus` is the per-Kreis-per-year derived
        # count, anchored at the two cross-sections with linear
        # interpolation between and pop-scaled extrapolation pre-1871.
        "pop_15plus_1871": Column(float, Check.gt(0), nullable=True),
        "pop_15plus_1890": Column(float, Check.gt(0), nullable=True),
        "pop_15plus": Column(float, Check.gt(0), nullable=True),
        # General marriage rate (Newell 1988): marriages per 1,000 mid-
        # year population aged 15+. Strips out under-15 population from
        # the crude marriage rate's denominator -- a "marriageable-age"
        # rate. Generous upper bound: extreme single-year boundary-
        # reform spikes can push the rate well above the typical
        # 15-25 per 1k.
        "general_marriage_rate": Column(
            float, Check.in_range(0, 200), nullable=True
        ),
        # Legitimate general fertility rate (legitimate births per
        # 1,000 women 15-49). Marital-style headline rate using the
        # proper age-restricted denominator; complements CBR.
        "lgfr": Column(float, Check.in_range(0, 600), nullable=True),
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
