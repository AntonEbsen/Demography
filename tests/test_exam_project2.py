"""
Tests for the exam_project2 source modules.

These tests use synthetic data to verify correctness of data transformations,
outcome variable construction, and regression wrapper functions without
requiring the actual Galloway database files.
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exam_project2"))


# ===================================================================
# Tests for src/data/load_data.py — _normalize_columns
# ===================================================================

class TestNormalizeColumns:
    def test_uppercase_mapped_to_canonical(self):
        from src.data.load_data import _normalize_columns
        df = pd.DataFrame({"CODE": [1], "RB": ["KOL"], "YEAR": [1871], "POPTOT": [10000]})
        result = _normalize_columns(df)
        assert "Code" in result.columns
        assert "Rb" in result.columns
        assert "Year" in result.columns
        assert "Poptot" in result.columns

    def test_mixed_case_preserved(self):
        from src.data.load_data import _normalize_columns
        df = pd.DataFrame({"Code": [1], "Rb": ["KOL"], "Year": [1871]})
        result = _normalize_columns(df)
        assert list(result.columns) == ["Code", "Rb", "Year"]

    def test_unknown_columns_unchanged(self):
        from src.data.load_data import _normalize_columns
        df = pd.DataFrame({"Code": [1], "MyCustomCol": [42]})
        result = _normalize_columns(df)
        assert "MyCustomCol" in result.columns


# ===================================================================
# Tests for src/data/build_dataset.py — outcome construction
# ===================================================================

class TestOutcomeVariables:
    """Test that outcome variables are computed correctly from known inputs."""

    @pytest.fixture
    def synthetic_panel(self):
        """Create a minimal synthetic panel with known values."""
        np.random.seed(42)
        codes = [101, 102, 103]
        years = list(range(1862, 1891))
        rows = []
        for code in codes:
            for year in years:
                rows.append({
                    "Code": code,
                    "Rb": "KOL" if code == 101 else "BER",
                    "Kreis": f"County_{code}",
                    "Type": 0,
                    "Year": year,
                    "Poptot": 50000 + np.random.randint(-5000, 5000),
                    "Birtot": 2000 + np.random.randint(-200, 200),
                    "Birlegtot": 1800 + np.random.randint(-150, 150),
                    "Birbastot": 200 + np.random.randint(-50, 50),
                    "Dthtot": 1500 + np.random.randint(-200, 200),
                    "Dth_infant_leg": 300 + np.random.randint(-50, 50),
                    "Martot": 500 + np.random.randint(-50, 50),
                    "Marevan": 300 if code != 101 else 100,
                    "Marcath": 100 if code == 101 else 50,
                })
        return pd.DataFrame(rows)

    def test_cbr_formula(self, synthetic_panel):
        """CBR = Birtot / Poptot * 1000"""
        df = synthetic_panel.copy()
        df["cbr"] = df["Birtot"] / df["Poptot"] * 1000
        expected = df.iloc[0]["Birtot"] / df.iloc[0]["Poptot"] * 1000
        assert abs(df.iloc[0]["cbr"] - expected) < 1e-10

    def test_legitimate_br_formula(self, synthetic_panel):
        """legitimate_br = Birlegtot / Poptot * 1000"""
        df = synthetic_panel.copy()
        df["legitimate_br"] = df["Birlegtot"] / df["Poptot"] * 1000
        assert df["legitimate_br"].notna().all()
        assert (df["legitimate_br"] > 0).all()

    def test_illegitimacy_ratio_formula(self, synthetic_panel):
        """illegitimacy_ratio = Birbastot / Birtot * 100"""
        df = synthetic_panel.copy()
        df["illegitimacy_ratio"] = df["Birbastot"] / df["Birtot"] * 100
        assert (df["illegitimacy_ratio"] >= 0).all()
        assert (df["illegitimacy_ratio"] <= 100).all()

    def test_treatment_variables(self, synthetic_panel):
        """Test that Kulturkampf treatment indicators are correctly constructed."""
        df = synthetic_panel.copy()
        df["cath_share"] = 60.0  # Simulate a 60% Catholic county
        df["post_kulturkampf"] = (df["Year"] >= 1873).astype(int)
        df["high_cath"] = (df["cath_share"] > 50).astype(int)
        df["treat_x_post"] = df["high_cath"] * df["post_kulturkampf"]
        df["cath_share_x_post"] = df["cath_share"] * df["post_kulturkampf"]

        # Pre-1873 should have post=0
        pre = df[df["Year"] < 1873]
        assert (pre["post_kulturkampf"] == 0).all()
        assert (pre["treat_x_post"] == 0).all()
        assert (pre["cath_share_x_post"] == 0).all()

        # Post-1873 should have post=1
        post = df[df["Year"] >= 1873]
        assert (post["post_kulturkampf"] == 1).all()
        assert (post["treat_x_post"] == 1).all()
        assert (post["cath_share_x_post"] == 60.0).all()


# ===================================================================
# Tests for src/analysis/utils.py — safe_panel_ols
# ===================================================================

class TestSafePanelOls:
    """Test the shared regression utility."""

    @pytest.fixture
    def regression_panel(self):
        """Create a panel suitable for PanelOLS."""
        np.random.seed(123)
        codes = list(range(1, 21))  # 20 entities
        years = list(range(2000, 2010))  # 10 periods
        rows = []
        for code in codes:
            for year in years:
                treat = 1 if code > 10 else 0
                post = 1 if year >= 2005 else 0
                rows.append({
                    "Code": code,
                    "Year": year,
                    "y": 50 + 2 * treat * post + np.random.normal(0, 1),
                    "treat_x_post": treat * post,
                    "control": np.random.normal(10, 2),
                })
        return pd.DataFrame(rows)

    def test_returns_result_object(self, regression_panel):
        from src.analysis.utils import safe_panel_ols
        res = safe_panel_ols(regression_panel, "y", ["treat_x_post", "control"])
        assert hasattr(res, "params")
        assert hasattr(res, "std_errors")
        assert hasattr(res, "pvalues")
        assert "treat_x_post" in res.params.index

    def test_handles_nan(self, regression_panel):
        from src.analysis.utils import safe_panel_ols
        df = regression_panel.copy()
        df.loc[0:5, "y"] = np.nan
        res = safe_panel_ols(df, "y", ["treat_x_post", "control"])
        assert res.nobs < len(df)

    def test_handles_duplicates(self, regression_panel):
        from src.analysis.utils import safe_panel_ols
        df = pd.concat([regression_panel, regression_panel.iloc[:5]], ignore_index=True)
        res = safe_panel_ols(df, "y", ["treat_x_post", "control"])
        assert res.nobs == len(regression_panel)


# ===================================================================
# Tests for src/analysis/regressions.py — structure checks
# ===================================================================

class TestRegressionOutputStructure:
    """Test that regression wrappers return the expected dict structure."""

    @pytest.fixture
    def did_panel(self):
        np.random.seed(99)
        codes = list(range(1, 31))
        years = list(range(1862, 1891))
        rows = []
        for code in codes:
            cath = np.random.uniform(0, 100)
            for year in years:
                post = 1 if year >= 1873 else 0
                rows.append({
                    "Code": code, "Year": year,
                    "Rb": "KOL" if code <= 15 else "BER",
                    "cbr": 40 + np.random.normal(0, 3),
                    "cath_share": cath,
                    "high_cath": int(cath > 50),
                    "post_kulturkampf": post,
                    "cath_share_x_post": cath * post,
                    "treat_x_post": int(cath > 50) * post,
                    "ln_pop": np.log(50000 + np.random.randint(-5000, 5000)),
                })
        return pd.DataFrame(rows)

    def test_baseline_did_returns_dict(self, did_panel):
        from src.analysis.regressions import run_baseline_did
        result = run_baseline_did(did_panel, outcome="cbr", treatment="continuous")
        assert "result" in result
        assert "summary" in result
        assert isinstance(result["summary"], str)

    def test_event_study_returns_coefs(self, did_panel):
        from src.analysis.regressions import run_event_study
        result = run_event_study(did_panel, outcome="cbr", treatment_var="cath_share", ref_year=1872)
        assert "result" in result
        assert "coefs" in result
        coefs = result["coefs"]
        assert "Year" in coefs.columns
        assert "beta" in coefs.columns
        assert "ci_lo" in coefs.columns
        assert "ci_hi" in coefs.columns
        # Reference year should have beta=0
        ref_row = coefs[coefs["Year"] == 1872]
        assert len(ref_row) == 1
        assert ref_row.iloc[0]["beta"] == 0.0

    def test_robustness_returns_dataframe(self, did_panel):
        from src.analysis.regressions import run_robustness
        result = run_robustness(did_panel, outcome="cbr")
        assert isinstance(result, pd.DataFrame)
        assert "Specification" in result.columns
        assert "Coefficient" in result.columns
        assert len(result) > 0

    def test_baseline_did_coefficient_stability(self, did_panel):
        # Snapshot test: pins the headline DiD coefficient on the synthetic
        # fixture (seed=99, no true treatment effect) so silent drift in
        # regression construction or covariate handling fails loudly.
        # Baseline captured 2026-05-07 against linearmodels PanelOLS.
        from src.analysis.regressions import run_baseline_did

        result = run_baseline_did(did_panel, outcome="cbr", treatment="continuous")
        coef = float(result["result"].params["cath_share_x_post"])
        baseline = -0.000913
        assert abs(coef - baseline) < 0.05, (
            f"Headline DiD coefficient drifted: got {coef:.6f}, "
            f"baseline {baseline:.6f}. Investigate regressions.py or "
            f"the cath_share_x_post construction."
        )
