"""
Regression test for the iPEHD <-> Galloway county crosswalk.

The crosswalk maps Galloway county codes to iPEHD kreiskey1871 keys via a
three-stage pipeline (direct match, manual lookup, name match) and currently
covers ~88% of the 393 Galloway Type-0 counties. A silent regression here
would corrupt every Catholic-share-based regression downstream.

Skips when raw archival data is not present (DVC-tracked, not in CI by default).
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GALLOWAY_DIR = REPO_ROOT / "exam_project2" / "data" / "raw" / "galloway_data"
IPEHD_PATH = (
    REPO_ROOT / "exam_project2" / "data" / "raw" / "ipehd_data" / "ipehd_qje2009_master.dta"
)

MIN_COVERAGE = 0.85
EXPECTED_TOTAL_TYPE0 = 393


def _find_rel_file() -> Path | None:
    for ext in (".xlsx", ".XLS", ".xls"):
        p = GALLOWAY_DIR / f"REL1871{ext}"
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def crosswalk_inputs():
    rel_path = _find_rel_file()
    if rel_path is None or not IPEHD_PATH.exists():
        pytest.skip("Raw Galloway/iPEHD data not present; run `dvc pull` to fetch.")
    return IPEHD_PATH, rel_path


def test_crosswalk_coverage_does_not_regress(crosswalk_inputs):
    import sys

    sys.path.insert(0, str(REPO_ROOT / "exam_project2"))
    from src.data.merge_ipehd import build_crosswalk

    ipehd_path, rel_path = crosswalk_inputs
    crosswalk = build_crosswalk(ipehd_path, rel_path, verbose=False)

    coverage = len(crosswalk) / EXPECTED_TOTAL_TYPE0
    assert coverage >= MIN_COVERAGE, (
        f"Crosswalk coverage regressed: {len(crosswalk)}/{EXPECTED_TOTAL_TYPE0} "
        f"= {coverage:.1%} (floor: {MIN_COVERAGE:.0%}). "
        "Investigate _MANUAL_CROSSWALK or the name-matching step in merge_ipehd.py."
    )

    assert crosswalk["Code"].is_unique, "Crosswalk emits duplicate Galloway Codes."
