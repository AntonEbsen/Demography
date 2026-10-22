"""Wrapper safeguards; these synthetic fixtures are not historical evidence."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "reproduce", Path(__file__).resolve().parents[1] / "scripts/reproduce_figure_4_1.py"
)
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class ReproductionPackageTests(unittest.TestCase):
    def test_exact_counts_and_discrepancy_cannot_drift(self):
        reproduce.compare_report(
            {"count": 392, "share": 23.4693877551}, {"count": 392, "share": 23.469387755102}
        )
        for actual in [{"count": 393, "share": 23.469387755102}, {"count": 392, "share": 22.0}]:
            with self.assertRaisesRegex(ValueError, "Mismatch|mismatch"):
                reproduce.compare_report(actual, {"count": 392, "share": 23.469387755102})

    def test_missing_fields_or_histogram_bins_fail(self):
        for actual in [{}, {"bins": [1]}, {"bins": [1, 3]}]:
            with self.assertRaises(ValueError):
                reproduce.compare_report(actual, {"bins": [1, 2]})

    def test_nonfinite_values_and_boolean_counts_fail(self):
        for actual, expected in [(float("nan"), 1.0), (float("inf"), 1.0), (True, 1)]:
            with self.assertRaises(ValueError):
                reproduce.compare_report(actual, expected)

    def test_missing_or_changed_input_fails_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xls"
            source.write_bytes(b"fixture")
            digest = reproduce.sha256(source)
            reproduce.verify_inputs(root, {source.name: digest})
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                reproduce.verify_inputs(root, {source.name: digest})
            with self.assertRaisesRegex(ValueError, "missing"):
                reproduce.verify_inputs(root, {"absent.xls": digest})

    def test_output_cannot_overwrite_inputs_or_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "raw"
            data.mkdir()
            with self.assertRaisesRegex(ValueError, "separate"):
                reproduce.choose_output(data / "results", data)
            output = Path(directory) / "output"
            reproduce.choose_output(output, data)
            (output / "report.json").write_text("fixture")
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                reproduce.choose_output(output, data)


if __name__ == "__main__":
    unittest.main()
