"""One-command, isolated reproduction of Figure 4.1 only (Python 3.12).

Never downloads or changes source data. Creates a dedicated virtual environment,
installs the locked runtime, runs the edge-case tests and actual-source check,
then compares every report field with the recorded reference. No regressions run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "exam_project2"
REFERENCE = PROJECT / "outputs/validation/figure_4_1"
LOCK = ROOT / "reproduction/figure-4-1/requirements.lock.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(data_dir: Path, expected: dict) -> None:
    """Require the recorded bytes, not just filenames or cell counts."""
    problems = []
    for name, digest in expected.items():
        if Path(name).name != name:
            raise ValueError("Input manifest must contain filenames, not paths")
        path = data_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256(path) != digest:
            problems.append(f"SHA-256 mismatch: {name}")
    if problems:
        raise ValueError("Source preflight failed: " + "; ".join(problems))


def compare_report(actual, expected, location: str = "report") -> None:
    """Exact categories/counts; absolute 1e-10 tolerance for floating summaries."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Changed report fields at {location}")
        for key, value in expected.items():
            compare_report(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Changed report list at {location}")
        for index, value in enumerate(expected):
            compare_report(actual[index], value, f"{location}[{index}]")
    elif isinstance(expected, float):
        if type(actual) not in (int, float) or not math.isclose(
            actual, expected, rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"Numerical mismatch at {location}: {actual!r} != {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Mismatch at {location}: {actual!r} != {expected!r}")


def choose_output(requested: Path | None, data_dir: Path) -> Path:
    if requested is None:
        runs = ROOT / "reproduction-runs"
        runs.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="figure-4-1-", dir=runs))
    output = requested.resolve()
    if output == ROOT or output == PROJECT or output.is_relative_to(data_dir.resolve()):
        raise ValueError("Output must be separate from repository roots and raw inputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty; previous results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/raw/galloway_data")
    parser.add_argument(
        "--output-dir", type=Path, help="New/empty directory; default is a unique run directory"
    )
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 12):
            raise ValueError("Use Python 3.12 for this locked reproduction (Windows: py -3.12)")
        data_dir = args.data_dir.resolve()
        expected_manifest = json.loads(
            (REFERENCE / "input_manifest.json").read_text(encoding="utf-8")
        )
        expected_report = json.loads((REFERENCE / "report.json").read_text(encoding="utf-8"))
        verify_inputs(data_dir, expected_manifest["raw_inputs_sha256"])
        output = choose_output(args.output_dir, data_dir)
        environment = ROOT / ".venv-figure-4-1"
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(environment)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(LOCK)],
            check=True,
        )
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        subprocess.run(
            [
                str(python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_figure*.py",
                "-v",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "src.validation.figure_4_1",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output),
            ],
            cwd=PROJECT,
            check=True,
        )
        actual = json.loads((output / "report.json").read_text(encoding="utf-8"))
        compare_report(actual, expected_report)
        verify_inputs(data_dir, expected_manifest["raw_inputs_sha256"])
        verification = {
            "scope": (
                "Figure 4.1 and related descriptive/sample counts only; "
                "not full-paper replication"
            ),
            "all_reference_report_fields_match": True,
            "reference_input_files_matched": len(expected_manifest["raw_inputs_sha256"]),
            "input_bytes_unchanged": True,
            "float_absolute_tolerance": 1e-10,
            "requirements_sha256": sha256(LOCK),
            "reference_report_canonical_sha256": hashlib.sha256(
                json.dumps(expected_report, sort_keys=True).encode()
            ).hexdigest(),
            "figure_comparison": "30 numerical histogram bins, not a pixel-identical PNG hash",
            "known_discrepancy_retained": actual["known_prose_discrepancy"],
        }
        (output / "verification.json").write_text(
            json.dumps(verification, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"VERIFIED: all reference report fields and 29 source hashes match. Results: {output}"
        )
        print(
            "The approximately-22% prose discrepancy remains documented. "
            "No regression or causal claim is validated."
        )
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"REPRODUCTION FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
