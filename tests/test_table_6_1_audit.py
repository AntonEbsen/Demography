"""Focused tests, independent of the repository's wider legacy test suite."""

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "table_audit", Path(__file__).resolve().parents[1] / "scripts/audit_table_6_1.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TableAuditTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(
            {
                "Code": [1, 1, 1],
                "Year": [1866, 1867, 1873],
                "women_15_49": [100.0, 100.0, 100.0],
                "married_women_15_49": [50.0, 60.0, 80.0],
                "Birbastot": [2.0, 2.0, 2.0],
                "cbr_flag": [False, False, True],
            }
        )

    def test_actual_year_field_controls_window(self):
        sample = audit.select_paper_sample(self.fixture())
        self.assertEqual(list(sample.Year), [1867, 1873])

    def test_dynamic_denominator_and_birth_mask(self):
        sample = audit.select_paper_sample(self.fixture())
        self.assertEqual(sample.loc[1, "nonmarital_paper_b5"], 50.0)
        self.assertTrue(pd.isna(sample.loc[2, "nonmarital_paper_b5"]))

    def test_duplicates_fail(self):
        frame = self.fixture()
        with self.assertRaisesRegex(ValueError, "Non-unique"):
            audit.select_paper_sample(pd.concat([frame, frame]))

    def test_manifest_changes_fail(self):
        for actual in [{}, {"x": "changed"}, {"x": "ok", "extra": "ok"}]:
            with self.assertRaises(ValueError):
                audit.verify_inputs(actual, {"x": "ok"})

    def test_print_precision_not_exact_unrounded_claim(self):
        self.assertTrue(audit.displayed_match(-0.010924639, -0.011))
        self.assertFalse(audit.displayed_match(-0.0104, -0.011))


if __name__ == "__main__":
    unittest.main()
