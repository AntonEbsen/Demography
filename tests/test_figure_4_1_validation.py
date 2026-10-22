"""Source-level validation checks; never modify the historical workbooks."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exam_project2"))
from src.validation.figure_4_1 import (
    canonical,
    find_workbook,
    paper_checks,
    prepare_sample,
    summarise,
)


class FigureValidationTests(unittest.TestCase):
    def test_workbook_variants_must_agree(self):
        root = Path("example_sources")
        variants = [root / "VIT1875.XLS", root / "VIT1875.xlsx"]
        with patch.object(Path, "iterdir", return_value=variants):
            with patch(
                "src.validation.figure_4_1.pd.read_excel", return_value=pd.DataFrame({"Code": [1]})
            ):
                self.assertEqual(find_workbook(root, "VIT1875").suffix, ".xlsx")
            with patch(
                "src.validation.figure_4_1.pd.read_excel",
                side_effect=[pd.DataFrame({"Code": [1]}), pd.DataFrame({"Code": [2]})],
            ):
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    find_workbook(root, "VIT1875")
            with self.assertRaisesRegex(ValueError, "Missing"):
                find_workbook(root, "VIT1900")

    def test_column_case_does_not_change_source_meaning(self):
        result = canonical(pd.DataFrame({"CODE": [1], "YEAR": [1871]}), ["Code", "Year"])
        self.assertEqual(list(result.columns), ["Code", "Year"])
        with self.assertRaisesRegex(ValueError, "Missing required"):
            canonical(pd.DataFrame({"Code": [1]}), ["Pop"])

    def test_county_weighting_threshold_and_endpoints(self):
        result = summarise(pd.DataFrame({"Code": [1, 2, 3, 4], "cath_share": [0, 20, 50, 80]}))
        self.assertEqual(result["low_catholic_le_50"], 3)
        self.assertEqual(result["high_catholic_gt_50"], 1)
        self.assertEqual(result["counties_between_20_and_80_inclusive"], 3)
        self.assertEqual(result["unweighted_mean_percent"], 37.5)
        self.assertEqual(sum(result["histogram_counts"]), 4)

    def test_duplicate_and_missing_counties_are_not_silently_counted(self):
        with self.assertRaisesRegex(ValueError, "one observation"):
            summarise(pd.DataFrame({"Code": [1, 1], "cath_share": [10, 10]}))
        with self.assertRaisesRegex(ValueError, "Invalid"):
            summarise(pd.DataFrame({"Code": [1], "cath_share": [np.nan]}))

    def test_actual_year_filter_excludes_mislabelled_record_without_relabelling(self):
        religion = pd.DataFrame(
            {
                "Code": [1],
                "Type": [0],
                "Kreis": ["Example"],
                "Pop": [100],
                "Relcathm": [20],
                "Relcathf": [30],
            }
        )
        vital = pd.DataFrame({"Code": [1] * 25, "Type": [0] * 25, "Year": list(range(1866, 1891))})
        counties, diagnostics = prepare_sample(religion, vital)
        self.assertEqual(len(counties), 1)
        self.assertEqual(diagnostics["matched_county_years"], 24)
        self.assertEqual(diagnostics["excluded_out_of_window_records"], [{"Code": 1, "Year": 1866}])
        self.assertEqual(vital.iloc[0]["Year"], 1866)

    def test_invalid_population_is_not_replaced_with_zero(self):
        religion = pd.DataFrame(
            {
                "Code": [1],
                "Type": [0],
                "Kreis": ["Example"],
                "Pop": [0],
                "Relcathm": [0],
                "Relcathf": [0],
            }
        )
        vital = pd.DataFrame({"Code": [1], "Type": [0], "Year": [1871]})
        with self.assertRaisesRegex(ValueError, "population"):
            prepare_sample(religion, vital)

    def test_prose_and_figure_claims_are_checked_at_reported_precision(self):
        metrics = {
            "counties": 392,
            "low_catholic_le_50": 262,
            "high_catholic_gt_50": 130,
            "unweighted_mean_percent": 34.009,
            "unweighted_median_percent": 13.332,
        }
        diagnostics = {"matched_county_years": 9261, "counties_without_all_24_years": 33}
        self.assertTrue(all(paper_checks(metrics, diagnostics).values()))
        metrics["counties"] = 393
        self.assertFalse(paper_checks(metrics, diagnostics)["figure_counties_392"])


if __name__ == "__main__":
    unittest.main()
